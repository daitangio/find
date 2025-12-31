#!/usr/bin/env python3
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any

from flask import Flask, g, redirect, render_template, request, url_for, abort

from jinja2 import DictLoader

DB_PATH = os.environ.get("SEARCH_DB", os.path.join(os.environ.get("HOME"),".find.db"))

app = Flask(__name__)

# -------------------------
# DB helpers
# -------------------------
def get_db() -> sqlite3.Connection:
    if "db" not in g:
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

def fts5_available(conn: sqlite3.Connection) -> bool:
    # Quick sanity check: will fail if SQLite not compiled with FTS5
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5test USING fts5(x);")
        conn.execute("DROP TABLE __fts5test;")
        return True
    except sqlite3.OperationalError:
        return False

# -------------------------
# Search logic
# -------------------------
@dataclass
class SearchResult:
    id: int
    url: str | None
    title: str | None
    snippet: str

def search_pages(conn: sqlite3.Connection, query: str, limit: int = 10, offset: int = 0) -> tuple[list[SearchResult], int]:
    """
    Uses FTS5 with bm25 ranking and snippet generation.
    """
    # Count total hits
    total = conn.execute(
        "SELECT COUNT(*) AS c FROM pages_fts WHERE pages_fts MATCH ?;",
        (query,),
    ).fetchone()["c"]

    rows = conn.execute(
        """
        SELECT
          p.id,
          p.url,
          p.title,
          snippet(pages_fts, 1, '<mark>', '</mark>', ' … ', 12) AS snippet
        FROM pages_fts
        JOIN pages p ON p.id = pages_fts.rowid
        WHERE pages_fts MATCH ?
        ORDER BY bm25(pages_fts) ASC
        LIMIT ? OFFSET ?;
        """,
        (query, limit, offset),
    ).fetchall()

    results = [
        SearchResult(
            id=int(r["id"]),
            url=r["url"],
            title=r["title"],
            snippet=r["snippet"] or "",
        )
        for r in rows
    ]
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
    mark { background: #ffef8a; }
    a { text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
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
<p class="muted">Tip: use FTS queries like <code>sqlite OR postgres</code>, <code>title:foo</code> (if you store it), phrases like <code>"exact phrase"</code>.</p>
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
  <p class="muted">{{ total }} result(s). Showing {{ results|length }}.</p>

  {% for r in results %}
    <div class="result">
      <div>
        <a href="{{ r.url }}"><strong>{{ r.title or ("Page #" ~ r.id) }}</strong></a>
      </div>
      {% if r.url %}
        <a href="{{ url_for('page', page_id=r.id) }}"><div class="muted">{{ r.title or ("Page #" ~ r.id) }} Cached </div></a>        
      {% endif %}
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
app.jinja_loader = DictLoader({
    "base.html": BASE_HTML,
    "home.html": HOME_HTML,
    "search.html": SEARCH_HTML,
    "page.html": PAGE_HTML,
})

# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    conn = get_db()
    if not fts5_available(conn):
        return "FTS5 is not available in this Python/SQLite build.", 500
    return render_template("home.html", title="Home")

@app.route("/search")
def search():
    conn = get_db()
    if not fts5_available(conn):
        return "FTS5 is not available in this Python/SQLite build.", 500

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
    row = conn.execute("SELECT id, url, title, html FROM pages WHERE id = ?;", (page_id,)).fetchone()
    if row is None:
        abort(404)
    back_q = (request.args.get("q") or "").strip()
    return render_template(
        "page.html",
        title=row["title"] or f"Page #{page_id}",
        page=row,
        back_q=back_q,
    )

if __name__ == "__main__":
    # Run: python app.py
    app.run(host="127.0.0.1", port=5000, debug=True)
