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

from find import utils

DB_PATH = os.environ.get("SEARCH_DB", os.path.join(os.environ.get("HOME"), ".find.db"))
FIND_SHOW_CACHED_PAGE = "FIND_SHOW_CACHED_PAGE"


app = Flask(__name__)


@app.context_processor
def inject_find_config() -> dict[str, str | bool]:
    return {
        "find_version": utils.get_version(),
        "find_show_cached_page": app.config[FIND_SHOW_CACHED_PAGE],
    }


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
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 50
SEARCH_TIMEOUT_SECONDS = 1.1  # Max time for a search query

# GG Set up weights
BM25_TITLE_WEIGHT = 5.0
BM25_TEXT_WEIGHT = 1.0
BM25_URL_WEIGHT = 2.5


# Thread pool for timeout-protected search operations (DDoS protection 3)
_search_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="search")
# Unused for the meantime
# LINK_BOOST_WEIGHT = float(os.environ.get("LINK_BOOST_WEIGHT", "0.05"))
# LINK_BOOST_CAP = int(os.environ.get("LINK_BOOST_CAP", "20"))

###########################################################


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled", "enable"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled", "disable"}:
        return False

    raise RuntimeError(f"{name} must be enabled or disabled, got {value!r}")


app.config[FIND_SHOW_CACHED_PAGE] = _env_flag(FIND_SHOW_CACHED_PAGE, default=False)


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

    # Use the same FTS columns for snippets and BM25, but weight title matches higher
    # because they are usually stronger relevance signals than body or URL matches.
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
          snippet(pages_fts, -1, '<mark>', '</mark>', ' … ', 12) AS snippet,
          bm25(pages_fts, ?, ?, ?) as score,
          p.status_code,
          p.post_date
        FROM pages_fts
        JOIN pages p ON p.id = pages_fts.rowid
        LEFT JOIN inbound ON inbound.to_page_id = p.id
        WHERE pages_fts MATCH ?
        ORDER BY {order_clause}
        LIMIT ? OFFSET ?;
        """,
        (
            BM25_TITLE_WEIGHT,
            BM25_TEXT_WEIGHT,
            BM25_URL_WEIGHT,
            query,
            limit,
            offset,
        ),
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
                rank=nice_score(score),
                status_code=int(r["status_code"]),
                post_date=r["post_date"],
            )
        )
    return results, int(total)


def nice_score(bmscore: float) -> float:
    """
     GG We want a limited rank value
    """
    return float(round(10 * -1 * bmscore,8))    

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("home.html", title="Home")


_FTS_COLUMNS = {"title", "text", "url"}
_FTS_OPERATORS = {"AND", "OR", "NOT"}
_FTS_TOKEN_RE = re.compile(r'"[^"]*"|\'[^\']*\'|\S+')
_FTS_SAFE_BARE_RE = re.compile(r"[\w]+(?:\*)?", re.UNICODE)


def _quote_fts_value(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _format_fts_value(value: str) -> str:
    value = _strip_quotes(value)
    if _FTS_SAFE_BARE_RE.fullmatch(value) and value != "*":
        return value
    return _quote_fts_value(value)


def _parse_search_token(token: str) -> str:
    if token.upper() in _FTS_OPERATORS:
        return token.upper()

    if token.startswith(("'", '"')):
        return _quote_fts_value(_strip_quotes(token))

    if ":" in token:
        column, value = token.split(":", 1)
        column_lower = column.lower()
        if column_lower == "site":
            return f"url:{_quote_fts_value(_strip_quotes(value))}"
        if column_lower in _FTS_COLUMNS and value:
            return f"{column_lower}:{_format_fts_value(value)}"

    return _format_fts_value(token)


def parse_search_query(q: str) -> str:
    """Return an FTS5-safe query while preserving supported search operators."""
    tokens = [
        _parse_search_token(match.group(0)) for match in _FTS_TOKEN_RE.finditer(q)
    ]
    return " ".join(tokens)


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


def _parse_int_arg(
    name: str,
    default: int,
    min_value: int,
    max_value: int | None = None,
) -> int:
    raw_value = request.args.get(name)
    if raw_value is None or raw_value == "":
        return default

    try:
        value = int(raw_value)
    except ValueError:
        abort(400, description=f"{name} must be an integer")

    if value < min_value:
        abort(400, description=f"{name} must be at least {min_value}")
    if max_value is not None and value > max_value:
        abort(400, description=f"{name} must be at most {max_value}")
    return value


@app.route("/search")
@limiter.limit("20 per minute")  # Stricter limit for search endpoint
def search():
    q = (request.args.get("q") or "").strip()

    # Enforce query complexity limits
    if len(q) > MAX_QUERY_LENGTH:
        abort(400, description=f"Query too long (max {MAX_QUERY_LENGTH} characters)")
    if len(q.split()) > MAX_QUERY_TERMS:
        abort(400, description=f"Too many search terms (max {MAX_QUERY_TERMS})")

    limit = _parse_int_arg("limit", DEFAULT_SEARCH_LIMIT, 1, MAX_SEARCH_LIMIT)
    offset = _parse_int_arg("offset", 0, 0)
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
        )
    except FuturesTimeoutError:
        abort(504, description="Search timed out. Try a simpler query.")
    except sqlite3.OperationalError:
        app.logger.exception("Invalid FTS query after parsing: %r", fts_query)
        abort(400, description="Invalid search query.")


def page(page_id: int):
    conn = get_db()
    row = conn.execute(
        "SELECT id, url, title, html FROM pages WHERE id = ?;", (page_id,)
    ).fetchone()
    if row is None:
        app.logger.info(f"Page not found {page_id}")
        abort(404)
    back_q = (request.args.get("q") or "").strip()
    # Extract meta refresh redirect URL if present
    meta_refresh_url = extract_meta_refresh(row["html"]) if row["html"] else None
    if meta_refresh_url:
        app.logger.info(
            f"page_id {page_id} Meta redirect found. Source:{row['html']}: Redirect:{meta_refresh_url}"
        )
    return render_template(
        "page.html",
        title=row["title"] or f"Page #{page_id}",
        page=row,
        back_q=back_q,
        meta_refresh_url=meta_refresh_url,
    )


if app.config[FIND_SHOW_CACHED_PAGE]:
    app.add_url_rule("/page/<int:page_id>", endpoint="page", view_func=page)


def web_run():
    # Run: python app.py
    app.run(host="0.0.0.0", port=7001)
