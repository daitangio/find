#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

from flask import Flask, g, redirect, render_template, request, url_for, abort

from jinja2 import DictLoader

DB_PATH = os.environ.get("SEARCH_DB", os.path.join(os.environ.get("HOME"), ".find.db"))
LINK_BOOST_WEIGHT = float(os.environ.get("LINK_BOOST_WEIGHT", "0.05"))
LINK_BOOST_CAP = int(os.environ.get("LINK_BOOST_CAP", "20"))

app = Flask(__name__)


# -------------------------
# DB helpers
# -------------------------
def get_db() -> sqlite3.Connection:
    if "db" not in g:
        if not os.path.exists(DB_PATH):
            raise Exception(f"Cannot continue: database {DB_PATH} not found")
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        # Slightly nicer defaults
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(_exc: Exception | None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


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
    basic_rank: int
    status_code: int


def search_pages(
    conn: sqlite3.Connection, query: str, limit: int = 10, offset: int = 0
) -> tuple[list[SearchResult], int]:
    """
    Uses FTS5 with bm25 ranking, inbound-link boost, and snippet generation.
    GG: New boost score function need to be studied because added value is unclear
    """
    # Count total hits
    total = conn.execute(
        "SELECT COUNT(*) AS c FROM pages_fts WHERE pages_fts MATCH ?;",
        (query,),
    ).fetchone()["c"]

    rows = conn.execute(
        """
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
          bm25(pages_fts) * (
            1.0 + (? * MIN(COALESCE(inbound.inbound, 0), ?))
          ) AS score,
          bm25(pages_fts) as basic_score,
          p.status_code
        FROM pages_fts
        JOIN pages p ON p.id = pages_fts.rowid
        LEFT JOIN inbound ON inbound.to_page_id = p.id
        WHERE pages_fts MATCH ?
        ORDER BY score ASC
        LIMIT ? OFFSET ?;
        """,
        (LINK_BOOST_WEIGHT, LINK_BOOST_CAP, query, limit, offset),
    ).fetchall()

    results = []
    for r in rows:
        status_code = int(r["status_code"])
        # For dead links (404) we add the url to the title because the title often is not very useful
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
                basic_rank=int(math.floor(10 * -1 * float(r["basic_score"]))),
                status_code=int(r["status_code"]),
            )
        )
    return results, int(total)


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
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 2rem; }
    input[type=text] { width: min(720px, 95vw); padding: .6rem; }
    button { padding: .6rem 1rem; }
    .result { margin: 1rem 0; padding: 1rem; border: 1px solid #ddd; border-radius: 10px; }
    .muted { color: #666; font-size: .92rem; }
    .tip   { font-size: .92rem; }
    mark { background: #ffef8a; }
    a { text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <img src="https://gioorgi.com/logos/vic20-anim.gif">
  <h1><a href="{{ url_for('home') }}">Find</a></h1>
  {% block body %}{% endblock %}
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
<p class="tip">Tips:</code>
<br>
<ul>
<li><a href="/search?q=%22dead+link%22">Search "dead link" to find all the dead link</a>
<li><a href="/search?q=url%3A%228bit.gioorgi.com%22">Search 8bit computers site only</a>
</ul>
<p>Use FTS queries like <code>sqlite OR postgres</code>, <code>title:foo</code> (if you store it), phrases like <code>"exact phrase"</p>
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

  {% for r in results %}
    <div class="result">
      <div>
        Score {{r.rank}} Basic. Score: {{r.basic_rank}} <a title="Score {{r.rank}} Basic. Score: {{r.basic_rank}}" href="{{ r.url }}"><strong>{{ r.title or ("Page #" ~ r.id) }}</strong></a>
        {% if r.url %}
          <a href="{{ url_for('page', page_id=r.id) }}"><div class="muted">Cached {{ ("Page #" ~ r.id) }}</div></a>        
        {% endif %}
      </div>

      <div>{{ r.snippet|safe }}</div>
    </div>
  {% endfor %}

  <div style="margin-top: 1rem;">
    {% if offset > 0 %}
      <a href="{{ url_for('search', q=q, offset=max(offset-limit,0), limit=limit) }}">← Prev</a>
    {% endif %}
    {% if offset + limit < total %}
      <span style="display:inline-block; width: 1rem;"></span>
      <a href="{{ url_for('search', q=q, offset=offset+limit, limit=limit) }}">Next →</a>
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
  <hr/>
  <div>{{ page.html|safe }}</div>
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


# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    conn = get_db()
    return render_template("home.html", title="Home")


@app.route("/search")
def search():
    conn = get_db()
    q = (request.args.get("q") or "").strip()
    limit = min(int(request.args.get("limit", 10)), 50)
    offset = max(int(request.args.get("offset", 0)), 0)

    if not q:
        return redirect(url_for("home"))

    results, total = search_pages(conn, q, limit=limit, offset=offset)
    return render_template(
        "search.html",
        title=f"Search: {q}",
        q=q,
        results=results,
        total=total,
        limit=limit,
        offset=offset,
        max=max,
    )


@app.route("/page/<int:page_id>")
def page(page_id: int):
    conn = get_db()
    row = conn.execute(
        "SELECT id, url, title, html FROM pages WHERE id = ?;", (page_id,)
    ).fetchone()
    if row is None:
        abort(404)
    back_q = (request.args.get("q") or "").strip()
    return render_template(
        "page.html",
        title=row["title"] or f"Page #{page_id}",
        page=row,
        back_q=back_q,
    )


def web_run():
    # Run: python app.py
    app.run(host="127.0.0.1", port=5000, debug=True)
