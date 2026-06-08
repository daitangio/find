# Agent Guide

## General rules

- Think before acting. Read existing files before writing code.
- Be concise in output but thorough in reasoning.
- Prefer editing over rewriting whole files.
- Do not re-read files you have already read.
- Test your code before declaring done.
- No sycophantic openers or closing fluff.
- Keep solutions simple and direct.

- At the end of this file, create a work in progress log, where you note what you already did, what is missing. Always update this log.
The overall project aims to be very compact (*less is more* mantra)

## Project Summary

Find is a compact Python search engine for small static sites and blogs. It has two main runtime surfaces:

- `crawl`: an asyncio crawler that fetches HTML, respects robots.txt, extracts text and links, and writes pages into SQLite.
- `findgui`: a Flask search UI backed by SQLite FTS5.

Maintenance commands:

- `reindex`: rebuilds and optimizes the FTS table from stored pages.
- `delete-pages`: deletes indexed pages whose URL matches a regular expression.

The package lives in `src/find`. Runtime entry points are declared in `pyproject.toml`:

```toml
crawl = "find.crawl:crawl_init"
findgui = "find.app:web_run"
reindex = "find.reindex:main"
delete-pages = "find.delete_pages:main"
```

The default database path is `~/.find.db`, overridden with `SEARCH_DB`.

## Current Source Layout

```text
src/find/
├── __init__.py
├── app.py                  # Flask app, search query parsing, result rendering
├── crawl.py                # Async crawler, URL normalization, HTML extraction
├── delete_pages.py         # Delete indexed pages by URL regexp
├── reindex.py              # FTS rebuild command
├── schema.sql              # SQLite schema, FTS5 table, triggers
├── utils.py                # Shared DB/bootstrap, version, robots.txt helpers
└── templates/
    ├── about.html
    ├── base.html
    ├── home.html
    ├── page.html
    ├── search.html
    └── search_navigation.html
```

Do not describe templates as inline; they are package templates under `src/find/templates`.

## Dependencies And Packaging

- Python: `>=3.12`
- Build backend: `flit_core<4`
- Runtime dependencies: `beautifulsoup4`, `aiohttp`, `aiosqlite`, `Flask`, `click`, `Flask-Limiter`, `gunicorn`
- Docker currently uses `python:3.14-slim-trixie`.

Local setup:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Database Model

`schema.sql` creates:

- `pages`: latest page per normalized URL, including title, raw HTML, extracted text, content hash, status, fetch time, and optional post date.
- `page_versions`: historical snapshots keyed by `(page_id, content_hash)`.
- `links`: link graph edges from a stored page to normalized outgoing URLs, with optional `to_page_id` backfilled once the target page is known.
  Tracks `first_seen_at` and `last_seen_at`.
- `pages_fts`: external-content FTS5 table over `title`, `text`, and `url`, using `porter unicode61`.

Triggers keep `pages_fts` synchronized with `pages` inserts, updates, and deletes. Database connections enable `foreign_keys` and WAL mode.
Indexes exist on `links.from_page_id`, `links.to_url`, `links.to_page_id`, and `page_versions.page_id`.

## Crawler Notes

Important functions in `crawl.py`:

- `normalize_url`: accepts only `http` and `https`, strips fragments, lowercases scheme/host, removes default ports, and collapses repeated path slashes.
- `html_to_text_and_links`: uses BeautifulSoup, removes `script`, `style`, and `noscript`, extracts title/text/links/post date, resolves relative links, and dedupes links in order.
- `remove_nav_content`: clears `<nav>` content before text extraction and link collection.
- `normalize_post_date`, `normalize_http_date`, and `extract_post_date`: prefer `.post_meta .post_date`, then fall back to HTTP date headers such as `Last-Modified`; store ISO timestamps when recognized.
- `auto_tune_concurrency`: derives worker count from politeness delay, with range `2..200`.
- `Crawler.db_writer`: the only database writer. It consumes `PageJob` items from a queue sized at `concurrency * 5`.

Crawler behavior:

- Defaults to same-host crawling.
- `--include-pattern` restricts the crawl frontier to URLs matching a Python regular expression.
- Uses a shared politeness delay across workers, default `0.190` seconds.
- Default request timeout is `5` seconds; default page size cap is `2_000_000` bytes.
- Fetches robots.txt per origin through `utils.get_robots_parser` and caches parsers.
- Robots policy handling is explicit: `401` and `403` mean disallow all, other `4xx` mean allow all, `5xx` mean disallow all, and fetch errors fall back to allow all.
- Uses `Find/{version} (+https://github.com/daitangio/find)` as User-Agent.
- Skips non-HTML and oversized pages.
- Indexes only `200` responses; non-`200` responses are skipped before indexing.
- Stores a new page version only when the HTML content hash changes.

Typical crawl command:

```sh
crawl --seed https://example.com --same-host --max-pages 4000 --delay 0.190
```

Use `--no-same-host` carefully; it permits the crawl frontier to leave seed hosts.

## Flask App Notes

`app.py` exposes:

- `GET /`: home/search form.
- `GET /about`: simple crawl stats page showing stored URL counts grouped by origin.
- `GET /search`: FTS search with query parsing, pagination, and sort by relevance or date.
- `GET /page/<id>`: cached HTML page view, registered only when `FIND_SHOW_CACHED_PAGE` is enabled.

Search protections:

- Flask-Limiter default: `400 per day`, `30 per hour`; `/search` has `20 per minute`.
- Query limits: `MAX_QUERY_LENGTH = 150`, `MAX_QUERY_TERMS = 12`.
- Pagination limits: `limit` defaults to `10`, accepts `1..50`; `offset` defaults to `0` and must be non-negative.
- Searches run in a `ThreadPoolExecutor(max_workers=4)` and time out after `1.1` seconds.

Search query parsing supports:

- FTS operators: `AND`, `OR`, `NOT`.
- Column queries: `title:...`, `text:...`, `url:...`.
- Google-like `site:example.com`, translated to `url:"example.com"`.
- Quoting of punctuation-only or unsafe bare tokens before sending to SQLite FTS5.

Search result behavior:

- Ranking combines weighted `bm25()` scoring with an inbound-link boost from the `links` table.
- Weights are configurable through `LINK_BOOST_WEIGHT` and `LINK_BOOST_CAP`.
- Snippets are rendered with `<mark>` highlights.

`FIND_SHOW_CACHED_PAGE` is parsed as an environment flag. Enabled values include `1`, `true`, `yes`, `on`, `enabled`, and `enable`; disabled values include `0`, `false`, `no`, `off`, `disabled`, and `disable`.
When cached page display is enabled, `page()` also detects HTML meta-refresh redirects and passes them to the template.

## Reindex Command

`reindex.py` rebuilds and optimizes the FTS table from `pages`.

```sh
reindex --db ~/.find.db
```

It requires an existing database. It will not create one.

Implementation details:

- Clears `pages_fts` with the FTS5 `delete-all` command.
- Runs `rebuild`, then `optimize`.
- Verifies the final FTS row count against `pages`.

## Delete Pages Command

`delete_pages.py` deletes indexed pages whose URL matches a regular expression.

```sh
delete-pages '/docs/' --db ~/.find.db
```

Notes:

- It requires an existing database.
- It deletes from `pages`; `page_versions`, `links`, and `pages_fts` are cleaned up through foreign keys and triggers.
- Invalid regular expressions fail fast with a Click error.

## Tests

The test suite is under `tests/`:

- `test_app.py`: search query parsing, date formatting, search ordering, cached page feature flag, template version rendering.
- `test_crawl.py`: URL normalization, HTML extraction, nav stripping, post date extraction, host restriction, fetch size handling, and concurrency auto-tuning.
- `test_delete_pages.py`: URL-regexp deletion behavior and CLI validation.
- `test_page_ranking.py`: inbound-link boost behavior, cap handling, and BM25 title weighting.
- `test_robots.py`: robots.txt fetching, parsing, cache behavior, and common status handling.

Run tests:

```sh
python3 -m unittest discover -s tests
```

The contributor/pre-commit path also runs pylint:

```sh
python3 -m unittest discover -s tests && pylint $(git ls-files '*.py')
```

`./etc/pre-commit` runs Black, unit tests, and pylint.

## Development Conventions

- Keep the project compact; README calls out a target below roughly 2000 lines of code.
- Use `from __future__ import annotations` in Python modules.
- Use parameterized SQL. The only intentional SQL interpolation in `app.py` is the validated `ORDER BY` clause.
- Prefer existing helpers in `utils.py` for DB creation and robots.txt handling.
- Keep crawler writes funneled through the mono-writer queue.
- If changing schema or FTS behavior, update `schema.sql`, relevant tests, and reindex guidance.
- If changing search query parsing, add or update tests in `tests/test_app.py`.
- If changing crawl policy or extraction, add or update tests in `tests/test_crawl.py` or `tests/test_robots.py`.
- If changing deletion behavior, add or update tests in `tests/test_delete_pages.py`.

## Deployment Notes

Docker:

- Image base: `python:3.14-slim-trixie`
- App runs as non-root `app`.
- `initAndFind.sh` is the container command.
- Compose maps host port `49152` to container port `7001`.
- Compose uses `SEARCH_DB=/opt/find/search.db`, `FIND_WEB_WORKERS=2`, `REINDEX_INTERVAL_HOURS=36`, and `FIND_SHOW_CACHED_PAGE=disabled`.
- Persistent data is mounted from `$FIND_HOME` to `/opt/find`.

For the web UI locally:

```sh
FLASK_DEBUG=true findgui
```

`web_run()` binds Flask to `0.0.0.0:7001`.

## Known Tradeoffs

- SQLite FTS5 with the Porter tokenizer is best suited to English text.
- The crawler indexes static HTML only; it does not execute JavaScript.
- Search ranking combines BM25 with an inbound-link boost and date sorting; it is not full PageRank.
- Cached page display intentionally renders stored HTML when enabled, so treat it as a feature with an explicit trust boundary.

## Work In Progress Log

- 2026-05-28: Read `AGENT.md` and verified constraints.
- 2026-05-28: Added this `Work In Progress Log` section at end of file, as required.
- 2026-05-28: Synced documentation to code: crawler response policy, Docker compose port mapping, tests list, and ranking notes.
- 2026-05-28: Corrected Docker compose host port mapping to `49152:7001` after re-validation.
- 2026-05-28: Missing: re-check this document whenever runtime behavior or configuration changes.
- 2026-05-31: Added crawler `--include-pattern` note; missing: re-check after future crawler option changes.
- 2026-06-04: Synced `AGENTS.md` to current `src/find` implementation: added `delete-pages`, `/about`, `about.html`, crawler date/nav details, search pagination/ranking notes, and `test_delete_pages.py`.
- 2026-06-04: Missing: re-check this document after future CLI additions or Flask route/template changes.
- 2026-06-05: Fixed `src/find/app.py` test regressions: preserve tiny BM25 deltas in `nice_score`, order `/about` origin logic changed and test updated by GG
- 2026-06-05: Missing: if date-format behavior changes again, re-check whether test-only frozen time still matches expected template output.
- 2026-06-08: Added workspace custom agent `.github/agents/python-code-review.agent.md` for Python code/test review with max-7 prioritized improvements and optional targeted test execution.

@RTK.md
