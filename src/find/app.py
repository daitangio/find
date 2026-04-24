#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sqlite3

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# from typing import Any

from flask import Flask, g, redirect, render_template, request, url_for, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from jinja2 import DictLoader

from find import utils


DB_PATH = os.environ.get("SEARCH_DB", os.path.join(os.environ.get("HOME"), ".find.db"))


app = Flask(__name__)


@app.context_processor
def inject_find_version() -> dict[str, str]:
    return {"find_version": utils.get_version()}

# -------------------------
# Rate Limiting (DDoS protection 1)
# -------------------------
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["400 per day", "30 per hour"],
    storage_uri="memory://",
)


# -------------------------
# Query Complexity Limits (DDoS protection 2)
# -------------------------
MAX_QUERY_LENGTH = 150
MAX_QUERY_TERMS = 12
SEARCH_TIMEOUT_SECONDS = 1.1  # Max time for a search query

# Thread pool for timeout-protected search operations (DDoS protection 3)
_search_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="search")
# Unused for the meantime
# LINK_BOOST_WEIGHT = float(os.environ.get("LINK_BOOST_WEIGHT", "0.05"))
# LINK_BOOST_CAP = int(os.environ.get("LINK_BOOST_CAP", "20"))

###########################################################


# -------------------------
# DB helpers
# -------------------------
def get_db() -> sqlite3.Connection:
    if "db" not in g:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"Cannot continue: database {DB_PATH} not found")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        # Slightly nicer defaults
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        g.db = conn
    return g.db


def extract_meta_refresh(html: str) -> str | None:
    """Extract the redirect URL from a meta http-equiv refresh tag."""
    if not html:
        return None
    # Match <meta http-equiv="refresh" content="0; URL=..." />
    pattern = r'<meta[^>]+http-equiv=["\']?refresh["\']?[^>]+content=["\']?\d+;\s*url=([^"\'>\s]+)["\']?[^>]*>'
    match = re.search(pattern, html, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


@app.teardown_appcontext
def close_db(_exc: Exception | None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


def _pluralize(value: int, unit: str) -> str:
    suffix = "" if value == 1 else "s"
    return f"{value} {unit}{suffix}"


def _format_relative_delta(seconds: int) -> str:
    if seconds < 60:
        return "just now"

    units = (
        ("year", 365 * 24 * 60 * 60),
        ("month", 30 * 24 * 60 * 60),
        ("day", 24 * 60 * 60),
        ("hour", 60 * 60),
        ("minute", 60),
    )
    for unit, unit_seconds in units:
        if seconds >= unit_seconds:
            return _pluralize(seconds // unit_seconds, unit)
    return "just now"


def find_format_date(value: str | None, now: datetime | None = None) -> str:
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return ""

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)

    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None or now.utcoffset() is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    diff_seconds = int((now - parsed).total_seconds())
    if diff_seconds < 0:
        relative = _format_relative_delta(abs(diff_seconds))
        return "just now" if relative == "just now" else f"in {relative}"

    relative = _format_relative_delta(diff_seconds)
    return "just now" if relative == "just now" else f"{relative} ago"


# Used in the template
app.add_template_filter(find_format_date, "find_format_date")


def format_post_date(value: str | None) -> str:
    return find_format_date(value)


app.add_template_filter(format_post_date, "format_post_date")


# -------------------------
# Search logic
# -------------------------
@dataclass
class SearchResult:
    id: int
    url: str | None
    title: str | None
    snippet: str
    rank: int
    status_code: int
    post_date: str | None


def search_pages(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    offset: int = 0,
    order_by: str = "rank",
) -> tuple[list[SearchResult], int]:
    """
    Uses FTS5 with bm25 ranking, inbound-link boost, and snippet generation.
    GG: New boost score function need to be studied because added value is unclear

    bm25(pages_fts) * (
            1.0 + (? * MIN(COALESCE(inbound.inbound, 0), ?))
          ) AS score,
          + LINK_BOOST_WEIGHT, LINK_BOOST_CAP
    """
    # Count total hits
    total = conn.execute(
        "SELECT COUNT(*) AS c FROM pages_fts WHERE pages_fts MATCH ?;",
        (query,),
    ).fetchone()["c"]

    order_clause = "score ASC"
    if order_by == "date":
        order_clause = "p.post_date IS NULL ASC, p.post_date DESC, score ASC"

    rows = conn.execute(
        f"""
        WITH inbound AS (
          SELECT to_page_id, COUNT(DISTINCT from_page_id) AS inbound
          FROM links
          WHERE to_page_id IS NOT NULL
          GROUP BY to_page_id
        )
        SELECT
          p.id,
          p.url,
          p.title,
          snippet(pages_fts, 1, '<mark>', '</mark>', ' … ', 12) AS snippet,          
          bm25(pages_fts) as score,
          p.status_code,
          p.post_date
        FROM pages_fts
        JOIN pages p ON p.id = pages_fts.rowid
        LEFT JOIN inbound ON inbound.to_page_id = p.id
        WHERE pages_fts MATCH ?
        ORDER BY {order_clause}
        LIMIT ? OFFSET ?;
        """,
        (query, limit, offset),
    ).fetchall()

    results = []
    for r in rows:
        status_code = int(r["status_code"])
        # For dead links (404) we add the url to the title
        # because the title often is not very useful
        # it is just a web server error in the most luck cases
        if status_code == 404:
            page_title = r["url"] + " / " + r["title"]
        else:
            page_title = r["title"]

        score = float(r["score"])
        results.append(
            SearchResult(
                id=int(r["id"]),
                url=r["url"],
                title=page_title,
                snippet=r["snippet"] or "",
                rank=int(math.floor(10 * -1 * score)),
                status_code=int(r["status_code"]),
                post_date=r["post_date"],
            )
        )
    return results, int(total)


# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("home.html", title="Home")


def parse_search_query(q: str) -> str:
    """Transform site:something into url:"something" for FTS5 queries."""
    # Match site:domain or site:"domain"
    # Replace with url:"domain"
    q = re.sub(r'site:(["\']?)([^\s"\']+)\1', r'url:"\2"', q, flags=re.IGNORECASE)
    return q


def _search_pages_threaded(
    query: str, limit: int, offset: int, order_by: str
) -> tuple[list["SearchResult"], int]:
    """Thread-safe wrapper that creates its own DB connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        return search_pages(conn, query, limit, offset, order_by)
    finally:
        conn.close()


@app.route("/search")
@limiter.limit("20 per minute")  # Stricter limit for search endpoint
def search():
    q = (request.args.get("q") or "").strip()

    # Enforce query complexity limits
    if len(q) > MAX_QUERY_LENGTH:
        abort(400, description="Query too long (max 200 characters)")
    if len(q.split()) > MAX_QUERY_TERMS:
        abort(400, description="Too many search terms (max 10)")

    limit = min(int(request.args.get("limit", 10)), 50)
    offset = max(int(request.args.get("offset", 0)), 0)
    order_by = request.args.get("orderBy", "rank")
    if order_by not in {"rank", "date"}:
        order_by = "rank"

    if not q:
        return redirect(url_for("home"))

    # Transform site:something queries to url:"something"
    fts_query = parse_search_query(q)

    # Execute search with timeout protection (thread-safe)
    try:
        future = _search_executor.submit(
            _search_pages_threaded, fts_query, limit, offset, order_by
        )
        results, total = future.result(timeout=SEARCH_TIMEOUT_SECONDS)
        return render_template(
            "search.html",
            title=f"Search: {q}",
            q=q,
            results=results,
            total=total,
            limit=limit,
            offset=offset,
            order_by=order_by,
            max=max,
        )
    except FuturesTimeoutError:
        abort(504, description="Search timed out. Try a simpler query.")


# Disabled page cache
# @app.route("/page/<int:page_id>")
# def page(page_id: int):
#     conn = get_db()
#     row = conn.execute(
#         "SELECT id, url, title, html FROM pages WHERE id = ?;", (page_id,)
#     ).fetchone()
#     if row is None:
#         app.logger.info(f"Page not found {page_id}")
#         abort(404)
#     back_q = (request.args.get("q") or "").strip()
#     # Extract meta refresh redirect URL if present
#     meta_refresh_url = extract_meta_refresh(row["html"]) if row["html"] else None
#     if meta_refresh_url:
#         app.logger.info(
#             f"page_id {page_id} Meta redirect found. Source:{row['html']}: Redirect:{meta_refresh_url}"
#         )
#     return render_template(
#         "page.html",
#         title=row["title"] or f"Page #{page_id}",
#         page=row,
#         back_q=back_q,
#         meta_refresh_url=meta_refresh_url,
#     )


# -------------------------
# UI templates (inline to keep it small)
# -------------------------
BASE_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{{ title }}</title>
  <style>
    /* @see: https://thecascade.dev/article/least-amount-of-css/ */
    html {
    color-scheme: light dark;
    }

    body {
    font-family: system-ui;
    font-size: 1.25rem;
    line-height: 1.5;
    }

    img,
    svg,
    video {
    max-width: 100%;
    display: block;
    }

    main {
    max-width: min(70ch, 100% - 4rem);
    margin-inline: auto;
    }


    .right { text-align: right; }
    .tip   { font-size: .92rem; }
  </style>
</head>
<body>
  <img src="https://gioorgi.com/logos/vic20-anim.gif">
  <h1><a href="{{ url_for('home') }}">Find</a></h1>
  {% block body %}{% endblock %}
  <footer class="muted right"><a href="https://github.com/daitangio/find">Find {{ find_version }}</a></footer>
</body>
</html>
"""

HOME_HTML = """
{% extends "base.html" %}
{% block body %}
<form action="{{ url_for('search') }}" method="get">
  <input type="text" name="q" placeholder="Search..." value="{{ q|default('') }}" autofocus>
  <button type="submit">Search</button>
</form>
<p class="tip">
Wellcome to Find, a small search application for all gioorgi.com sites.
<br>
Some Tips:
<br>
<ul>
<li>Use FTS queries like <code>sqlite OR postgres</code>, <code>title:foo</code>, phrases like <code>"exact phrase"</code>
<li><a href="/search?q=url%3A%228bit.gioorgi.com%22">Search 8bit computers site only</a>
<li>Google-like <a href="/search?q=site:8bit.gioorgi.com">site:8bit.gioorgi.com syntax</a> is supported.
</ul>
<p>
Note: a rate limiter is active by default.
</p>
</p>
{% endblock %}
"""

SEARCH_HTML = """
{% extends "base.html" %}
{% block body %}
<form action="{{ url_for('search') }}" method="get">
  <input type="text" name="q" placeholder="Search..." value="{{ q }}" autofocus>
  <button type="submit">Search</button>
</form>

{% if q and total == 0 %}
  <p>No results.</p>
{% endif %}

{% if total > 0 %}
  <p class="muted">{{ total }} result(s). Showing Page {{ 1+(offset//10) }} of {{ 1+ (total // 10)}}.</p>
  <p class="muted">
    Sort:
    {% if order_by == "date" %}
      <a href="{{ url_for('search', q=q, offset=0, limit=limit, orderBy='rank') }}">relevance</a>
      <strong>date</strong>
    {% else %}
      <strong>relevance</strong>
      <a href="{{ url_for('search', q=q, offset=0, limit=limit, orderBy='date') }}">date</a>
    {% endif %}
  </p>

  {% for r in results %}
    <div class="result">
      <div>
        [ Score {{r.rank}}] <a title="Score {{r.rank}} Basic." href="{{ r.url }}"><strong>{{ r.title or ("Page #" ~ r.id) }}</strong>
        {% set formatted_post_date = r.post_date|find_format_date %}
        {% if formatted_post_date %}
          <dx class="muted">{{ formatted_post_date }}</dx>
        {% endif %}</a>

      </div>

      <div>{{ r.snippet|safe }}</div>
    </div>
  {% endfor %}

  <div style="margin-top: 1rem;">
    {% if offset > 0 %}
      <a href="{{ url_for('search', q=q, offset=max(offset-limit,0), limit=limit, orderBy=order_by) }}">← Prev</a>
    {% endif %}
    {% if offset + limit < total %}
      <span style="display:inline-block; width: 1rem;"></span>
      <a href="{{ url_for('search', q=q, offset=offset+limit, limit=limit, orderBy=order_by) }}">Next →</a>
    {% endif %}
  </div>
{% endif %}
{% endblock %}
"""

PAGE_HTML = """
{% extends "base.html" %}
{% block body %}
  <p><a href="{{ url_for('search', q=back_q) }}">← back to results</a></p>
  <h2>{{ page.title or ("Page #" ~ page.id) }}</h2>
  {% if page.url %}
    <div class="muted">{{ page.url }}</div>
  {% endif %}
  {% if meta_refresh_url %}
  <div style="background: #fff3cd; border: 1px solid #ffc107; padding: 1rem; margin: 1rem 0; border-radius: 5px;">
    <strong>⚠️ This page contains a redirect to:</strong>
    <a href="{{ meta_refresh_url }}">{{ meta_refresh_url }}</a>
  </div>
  {% else %}
    <div>{{ page.html|safe }}</div>  
  {% endif %}
  <hr/>
  
{% endblock %}
"""

# Register inline templates with Flask
app.jinja_loader = DictLoader(
    {
        "base.html": BASE_HTML,
        "home.html": HOME_HTML,
        "search.html": SEARCH_HTML,
        "page.html": PAGE_HTML,
    }
)



def web_run():
    # Run: python app.py
    app.run(host="0.0.0.0", port=7001)
