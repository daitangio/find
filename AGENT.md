## General rules

- Think before acting. Read existing files before writing code.
- Be concise in output but thorough in reasoning.
- Prefer editing over rewriting whole files.
- Do not re-read files you have already read.
- Test your code before declaring done.
- No sycophantic openers or closing fluff.
- Keep solutions simple and direct.

## Executive Summary

**Find** is a minimal, self-contained search engine built with Python and SQLite FTS5 (Full Text Search). It's designed to provide a simple, zero-configuration search solution for static websites and blogs. The project emphasizes compactness (targeting <2000 LOC), performance through async I/O, and respect for web etiquette (robots.txt compliance, politeness delays).

The system consists of two main components:
1. **Crawler** - An async web crawler that indexes web pages
2. **Web Interface** - A Flask-based search UI with caching and anti-DDoS protection

---

## Project Architecture

### Design Philosophy

The project follows several core principles:
- **Minimalism**: "Less is more" mantra with a target of <2000 lines of code
- **Zero Configuration**: Works out of the box with sensible defaults
- **Compact Solution**: Minimal dependencies with Flask templates kept in the package templates directory
- **Web Politeness**: Respects robots.txt and implements polite crawling delays
- **Performance**: Async I/O for crawling, optimized SQLite with FTS5

### Technology Stack

**Core Technologies:**
- Python 3.12+ (type hints, modern features)
- SQLite 3 with FTS5 extension (Full-Text Search)
- asyncio for concurrent I/O operations

**Key Dependencies:**
```python
beautifulsoup4>=4.14.3    # HTML parsing
aiohttp==3.13.4           # Async HTTP client
aiosqlite==0.22.1         # Async SQLite wrapper
Flask>=3.1.2              # Web framework
click>=8.3.1              # CLI interface
Flask-Limiter>=3.5.0      # Rate limiting & DDoS protection
gunicorn>=23.0.0          # Production WSGI server used by Docker
```

**Build System:**
- flit_core<4 (PEP 517 compliant)
- Simple packaging without complex setup.py

---

## Component Analysis

### 1. Database Schema (schema.sql)

The database uses a sophisticated design with versioning and link graph support:

#### Tables:

**`pages`** - Current/latest version of each indexed page
- Primary key: `id` (INTEGER)
- Unique constraint: `url` (normalized canonical URL)
- Fields: `title`, `html`, `text`, `content_hash` (SHA-256)
- Metadata: `status_code`, `fetched_at` (ISO timestamp), `post_date` (ISO timestamp)
- Purpose: Stores the most recent version of each page

**`page_versions`** - Historical snapshots
- Links to `pages(id)` via `page_id` (CASCADE DELETE)
- Unique constraint: `(page_id, content_hash)`
- Only stores versions when content changes (deduplication)
- Enables temporal queries and version comparison

**`links`** - Link graph for future PageRank
- `from_page_id` → `to_url` → `to_page_id` (when known)
- Tracks first/last seen timestamps
- Supports backfilling when target pages are discovered
- Indexes on `from_page_id`, `to_url`, and `to_page_id`

**`pages_fts`** - FTS5 Virtual Table
- External content table pointing to `pages`
- Indexed fields: `title`, `text`, `url`
- Tokenizer: `porter unicode61` (English stemming + Unicode support)
- Automatically synced via triggers (INSERT, UPDATE, DELETE)

#### Design Highlights:

1. **Content Deduplication**: Uses SHA-256 hash to detect unchanged content
2. **Foreign Key Constraints**: Ensures referential integrity
3. **WAL Mode**: Write-Ahead Logging for better concurrency
4. **Automatic FTS Sync**: Triggers keep FTS index current with pages table

---

### 2. Web Crawler (crawl.py)

The crawler is the most complex component (~802 lines), implementing async crawling with sophisticated performance tuning.

#### Core Features:

**Async I/O Architecture:**
- Uses `aiohttp.ClientSession` for concurrent HTTP requests
- `aiosqlite` for non-blocking database operations
- Worker pool pattern with configurable concurrency
- Queue-based task distribution

**Performance Optimizations:**

1. **Auto-tuned Concurrency:**
```python
def auto_tune_concurrency(delay_s: float) -> int:
    if delay_s <= 0:
        return 2
    return min(max(2, int(1 // delay_s) - 1), 200)
```
- Calculates optimal workers based on politeness delay
- Range: 2-200 workers
- Example: 0.190s delay → ~4 workers

2. **Mono-Writer Pattern:**
- Single dedicated database writer task (`db_writer()`)
- Prevents SQLite locking issues
- Queue size: 4× concurrency level
- Avoids data loss through proper backpressure

3. **Performance Logging:**
```python
PERF_THRESHOLD_MS = 2000.0
```
- Logs operations exceeding thresholds
- Helps identify slow sites and bottlenecks
- Tracks: fetch time, parse time, DB write time

**URL Normalization:**
```python
def normalize_url(url: str) -> Optional[str]:
    # - Strip fragments (#section)
    # - Lowercase scheme/host
    # - Remove default ports (:80, :443)
    # - Collapse multiple slashes
    # - Keep query strings as-is
```

**Politeness Features:**

1. **Shared Rate Limiting:**
- Cross-worker synchronization via `asyncio.Lock`
- Configurable delay (default: 0.190s)
- Prevents server overload

2. **robots.txt Compliance:**
- Fetches and caches robots.txt per origin
- RFC 9309 compliant behavior:
  - 401/403 on robots.txt → Disallow all
  - 404 → Allow all
  - 5xx → Disallow all (temporary)
  - Network error → Allow all (permissive)
- User-Agent: `Find/{version} (+https://github.com/daitangio/find)`

**Content Processing:**

1. **HTML Parsing:**
```python
def html_to_text_and_links(base_url, html, wid):
    # Uses BeautifulSoup
    # Removes: <script>, <style>, <noscript>
    # Extracts: title, text, links, post_date
    # Resolves relative URLs to absolute
    # Deduplicates links in order
```

2. **Post Date Extraction:**
- Heuristic-based extraction from `<div class="post_meta">`
- Supports multiple date formats (ISO, US, European)
- Normalizes to ISO 8601 with timezone

3. **Content Deduplication:**
- SHA-256 hash of HTML content
- Only creates new version if hash differs
- Saves database space and index time

**Link Graph Management:**

- **Forward Links**: Stores all outbound links from a page
- **Backfilling**: Updates `to_page_id` when target page is indexed
- **Temporal Tracking**: `first_seen_at`, `last_seen_at` timestamps
- Future use: PageRank calculation (infrastructure ready)

**Worker Lifecycle:**

1. **Initialization**: Seeds added to queue
2. **Worker Tasks**: 
   - Fetch URL from queue (5s timeout)
   - Check robots.txt
   - Apply politeness delay
   - Fetch HTML (with timeout, size limits)
   - Parse and extract data
   - Queue DB write job
   - Enqueue discovered links
3. **DB Writer**: Single task consumes jobs, writes to DB
4. **Logger**: Periodic status updates with performance metrics
5. **Shutdown**: All workers complete → DB queue drains → Writer stops

**Status Monitoring:**
```python
async def logger(self):
    # Adaptive sample time: 3-60 seconds based on concurrency/delay
    # Reports:
    #   - URLs queued
    #   - Pages fetched vs stored
    #   - Pages per second (PPS) vs expected
    #   - DB queue utilization
    #   - Saturation warnings
```

#### Command-Line Interface:

```bash
crawl --seed URL [--seed URL2 ...] \
      [--db ~/.find.db] \
      [--max-pages 4000] \
      [--delay 0.190] \
      [--concurrency -1] \
      [--timeout 5] \
      [--max-bytes 2000000] \
      [--same-host | --no-same-host]
```

**Parameters:**
- `--seed`: Required, repeatable for multiple seeds
- `--same-host`: Default True, restricts crawl to seed domains
- `--concurrency=-1`: Auto-tuned by default
- `--max-bytes=2MB`: Prevents memory issues

---

### 3. Web Interface (app.py)

Flask-based search UI with ~380 lines of code, including inline templates.

#### Security & DDoS Protection (3 layers):

**Layer 1: Rate Limiting**
```python
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["400 per day", "30 per hour"],
    storage_uri="memory://",
)

@app.route("/search")
@limiter.limit("20 per minute")  # Stricter for search
```

**Layer 2: Query Complexity Limits**
```python
MAX_QUERY_LENGTH = 150      # Characters
MAX_QUERY_TERMS = 12        # Words
```

**Layer 3: Search Timeout Protection**
```python
SEARCH_TIMEOUT_SECONDS = 1.1
_search_executor = ThreadPoolExecutor(max_workers=4)

# Each search runs in thread pool with timeout
future = _search_executor.submit(_search_pages_threaded, ...)
results, total = future.result(timeout=SEARCH_TIMEOUT_SECONDS)
```
- Prevents slow queries from blocking Flask
- Thread-safe: each thread gets own DB connection
- Aborts with 504 if timeout exceeded

#### Search Features:

**FTS5 Query Syntax Support:**
- Boolean: `sqlite OR postgres`
- Field search: `title:foo`, `url:bar`
- Phrases: `"exact phrase"`
- Google-like: `site:example.com` → `url:"example.com"`

**Ranking Algorithm:**
```sql
bm25(pages_fts) AS score
```
- BM25 (Best Match 25): probabilistic ranking function
- Lower score = better match (ascending order)
- Currently uses pure BM25 (inbound link boost commented out)

**Snippet Generation:**
```sql
snippet(pages_fts, 1, '<mark>', '</mark>', ' … ', 12)
```
- Highlights matching terms with `<mark>` tags
- Shows ~12 tokens of context
- Ellipsis separator for readability

**Special Handling:**

1. **404 Pages**: Prepends URL to title for clarity
2. **Meta Refresh Detection**: Extracts redirect URLs from HTML
3. **Post Dates**: Formats as "Jan 15, 2025 14:30"

#### Routes:

**`GET /`** - Home page with search form
- Tips for advanced queries
- Rate limit notice

**`GET /search?q=query&limit=10&offset=0`**
- Pagination support
- Total results count
- Prev/Next navigation
- Score display (rank = `floor(10 * -1 * bm25_score)`)

**`GET /page/<id>?q=back_query`**
- Cached page viewer
- Back link to search results
- Meta refresh warning if present
- Displays raw HTML or redirect message

#### UI Design:

**Package Templates** (`src/find/templates`, loaded by Flask):
- `base.html`: Common layout with VIC-20 logo
- `home.html`: Search form + tips
- `search.html`: Results list + sort controls
- `search_navigation.html`: Prev/Next pagination controls
- `page.html`: Cached page display

**Styling:**
- System fonts: `system-ui, -apple-system, Segoe UI, Roboto`
- Minimal CSS (~400 bytes)
- Responsive: `min(720px, 95vw)` for inputs
- Accessibility: `<mark>` for highlights

---

### 4. Reindex Utility (reindex.py)

Simple utility (~70 lines) to rebuild FTS index.

**Use Cases:**
- FTS index corruption
- Schema changes
- Performance optimization

**Process:**
```python
# 1. Clear all FTS entries
INSERT INTO pages_fts(pages_fts) VALUES('delete-all')

# 2. Rebuild from pages table
INSERT INTO pages_fts(pages_fts) VALUES('rebuild')

# 3. Optimize index
INSERT INTO pages_fts(pages_fts) VALUES('optimize')

# 4. Verify count matches
```

**Safety:**
- Requires existing database (won't create)
- Commits after each step
- Verification check at end

---

### 5. Utilities (utils.py)

Shared functions (~110 lines) used by crawler and reindex.

**Key Functions:**

**`ensure_database_present(db_file, create_if_missing=True)`**
- Creates database if missing
- Validates FTS5 availability
- Loads schema from `schema.sql` (via importlib.resources)
- Graceful error handling

**`get_robots_parser(session, url, cache, timeout)`**
- Async robots.txt fetcher
- Per-origin caching
- RFC 9309 compliant status handling
- Fallback to permissive on network errors

**`get_version()`**
- Uses `importlib.metadata.version("find")`
- Reads from installed package metadata

**`DATABASE_FILE`**
- Default: `~/.find.db`
- Overridable via `SEARCH_DB` env var (app.py uses this)

---

## Testing Strategy

The project includes 3 test modules with good coverage:

### test_app.py
- `extract_meta_refresh()` - redirect detection
- `parse_search_query()` - site: syntax transformation

### test_crawl.py
- URL normalization (fragments, ports, schemes)
- HTML parsing (title, text, links, deduplication)
- Post date extraction
- Host restriction policy
- Auto-concurrency calculation
- Edge cases: zero delay, bounds checking

### test_robots.py
- robots.txt fetching and caching
- Status code handling (200, 404, 403, 500)
- User-agent respect
- Cache hit avoidance

**Test Execution:**
```bash
python3 -m unittest discover -s tests
```

**Quality Checks:**
```bash
pylint $(git ls-files '*.py')
black $(git ls-files '*.py')  # Code formatting
```

---

## Deployment

### Docker Support

**Dockerfile Highlights:**
- Base: `python:3.14-slim-trixie` (Debian Trixie)
- Non-root user: `app:app` (UID/GID 1000)
- Installs the application plus Gunicorn
- Runs the Flask app through Gunicorn bound to `0.0.0.0:7001`
- Environment:
  - `FLASK_ENV=production`
  - `FIND_WEB_WORKERS=4` by default; override to tune Gunicorn worker processes

**docker-compose.yml:**
- Port mapping: `49152:7001` (external:internal)
- Environment: `SEARCH_DB=/opt/find/search.db`, `TZ=Europe/Rome`, `REINDEX_INTERVAL_HOURS=8`
- Volume: `/opt/find` (persistent storage for DB)
- Resource limit: 1.90 CPU
- Optional syslog logging (commented)

### Local Development

**Setup:**
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

**Entry Points:**
```bash
crawl --seed https://example.com --same-host
findgui  # Starts Flask on 127.0.0.1:5000
reindex  # Rebuilds FTS index
```

**Pre-commit Hook:**
```bash
./etc/pre-commit
# Runs: black → unittest → pylint
```

---

## Performance Characteristics

### Crawler Performance

**Benchmark Data (from code comments):**
- SQLite writer is "very fast" - hard to saturate even at high concurrency
- Default queue size (4× concurrency) prevents data loss
- Performance threshold: 2000ms for most operations
- Parsing threshold: ~66ms (PERF_THRESHOLD_MS / 30)

**Scaling Factors:**
1. **Network I/O**: Primary bottleneck (async helps)
2. **Politeness Delay**: Limits max throughput
   - 0.190s delay → ~5.26 pages/second theoretical max
   - Actual: ~80% of theoretical due to overheads
3. **HTML Parsing**: CPU-bound but fast (<100ms typical)
4. **DB Writes**: Minimal impact with mono-writer pattern

**Concurrency Sweet Spot:**
- Auto-tuned based on delay
- Range: 2-200 workers
- Higher concurrency useful only with low delays

### Search Performance

**Query Timeout:** 1.1 seconds (aborts slow queries)
**Database Mode:** WAL (Write-Ahead Logging)
- Allows concurrent reads during writes
- Better for web traffic patterns

**FTS5 Performance:**
- BM25 ranking computed at query time
- Porter stemming adds minimal overhead
- Index size: ~2-3× original text size

---

## Design Patterns & Best Practices

### 1. Mono-Writer Pattern
**Problem:** SQLite has limited write concurrency  
**Solution:** Single writer task with async queue  
**Benefits:**
- Eliminates lock contention
- Predictable performance
- No BUSY errors

### 2. Async Queue Sizing
**Formula:** `queue_size = concurrency * 4`
**Rationale:**
- Provides buffer during traffic spikes
- Prevents backpressure blocking workers
- 4× multiplier balances memory vs throughput

### 3. URL Normalization
**Importance:** Prevents duplicate indexing  
**Approach:**
- Case-insensitive hosts
- Fragment removal
- Default port stripping
- Path normalization

### 4. Content Hashing
**Purpose:** Detect unchanged pages efficiently  
**Method:** SHA-256 of HTML  
**Impact:** Reduces DB writes, saves space

### 5. Graceful Degradation
- robots.txt network errors → allow
- Missing post dates → continue
- Parse errors → skip page, not crash

### 6. Template Inlining
**Philosophy:** Keep project compact  
**Trade-off:** Less flexible, but self-contained  
**Example:** All 4 templates in app.py (~60 lines)

### 7. Thread Pool for Search
**Why:** Flask is synchronous, SQLite queries can block  
**How:** ThreadPoolExecutor with timeout  
**Result:** Prevents slow queries from DoS

---

## Strengths

1. **Simplicity**: Easy to understand, modify, deploy
2. **Performance**: Async I/O, optimized DB access
3. **Reliability**: Deduplication, versioning, graceful errors
4. **Web Citizenship**: robots.txt, delays, User-Agent
5. **Security**: Multi-layer DDoS protection
6. **Portability**: Single binary (via Docker), minimal deps
7. **Testability**: Good test coverage for core logic
8. **Maintainability**: Under 2000 LOC, well-commented

---

## Limitations & Trade-offs

### 1. Language Support
- **Current:** English-only stemming (Porter)
- **Impact:** Poor results for non-English content
- **Future:** Could support multiple tokenizers

### 2. Ranking Algorithm
- **Current:** Pure BM25, no link analysis
- **Trade-off:** Simple but less sophisticated than PageRank
- **Note:** Link graph infrastructure exists but unused

### 3. JavaScript Rendering
- **Current:** Static HTML only
- **Impact:** Can't index SPAs or JS-heavy sites
- **Alternative:** Would require headless browser (heavy)

### 4. Scalability Ceiling
- **SQLite Limit:** ~140 TB DB size, millions of pages feasible
- **Real Limit:** Single-machine I/O, no distributed crawling
- **Target:** Static blogs/docs (<100k pages)

### 5. Template Flexibility
- **Current:** Inline templates, hard to customize
- **Trade-off:** Compactness vs. extensibility
- **Mitigation:** Jinja2 allows template override

### 6. Crawl Scheduling
- **Current:** One-shot crawls, manual re-runs
- **Missing:** Incremental updates, scheduling
- **Workaround:** Cron + deduplication handles re-crawls

### 7. Search Features
- **Missing:** 
  - Fuzzy matching
  - Spelling correction
  - Related searches
  - Faceted search
- **Rationale:** Complexity vs. benefit for small sites

---

## Code Quality & Practices

### Static Analysis
- **Pylint:** No violations in CI
- **Type Hints:** Extensive use of `from __future__ import annotations`
- **Formatting:** Black (enforced in pre-commit)

### Documentation
- **Docstrings:** Present for modules and complex functions
- **Comments:** Explain design decisions
- **README:** Clear usage instructions
- **DEVELOPING.md:** Contributor guide

### Error Handling
- **Async Exceptions:** Caught and logged
- **Network Errors:** Graceful degradation
- **DB Errors:** Prevented via mono-writer pattern
- **Input Validation:** Query length/complexity limits

### Dependencies
- **Minimal:** 6 runtime dependencies
- **Pinned:** aiohttp and aiosqlite (stability)
- **Recent:** All packages from 2024-2025

---

## Security Considerations

### 1. DDoS Protection (3 layers)
- Rate limiting: 20 req/min for search
- Query complexity limits
- Timeout enforcement (1.1s)

### 2. Input Validation
- Query length: 150 chars max
- Query terms: 12 words max
- Page size: 2MB max
- URL schemes: http/https only

### 3. SQL Injection
- **Mitigation:** Parameterized queries throughout
- **FTS5:** Uses prepared statements
- **No string interpolation**

### 4. XSS Prevention
- **Cached Pages:** Displays raw HTML (intentional feature)
- **Search Results:** Snippets use `|safe` (FTS5 generates safe HTML)
- **Note:** Trust boundary at crawl time, not display

### 5. Information Disclosure
- **Cached Pages:** Public (by design)
- **Error Messages:** Minimal (404, 504 status codes)
- **Logs:** Performance metrics, no sensitive data

### 6. Container Security
- Non-root user (app:app)
- Minimal base image (slim-trixie)
- No shell access needed
- Health checks prevent zombie containers

---

## Future Roadmap (from README)

### Link-based Ranking
**Current Status:** Infrastructure ready, not implemented  
**SQL Example:**
```sql
SELECT p.url, COUNT(*) AS out_links
FROM links l JOIN pages p ON p.id = l.from_page_id
GROUP BY p.id
```
**Plan:** 
- Calculate inbound link counts
- Weight BM25 score by popularity
- Cap boost to prevent manipulation

### Potential Enhancements (not in docs)
1. **Incremental Crawling**: Track last-modified, ETags
2. **Sitemap Support**: Faster discovery
3. **Query Suggestions**: Based on indexed content
4. **Advanced Filters**: Date ranges, content type
5. **API Endpoint**: JSON search results
6. **Multi-language**: Parameterized tokenizers
7. **Highlighting**: Better snippet context

---

## Developer Notes

### Project Structure
```
find/
├── src/find/           # Main package
│   ├── app.py         # Flask web interface
│   ├── crawl.py       # Async crawler
│   ├── reindex.py     # FTS rebuild utility
│   ├── utils.py       # Shared code
│   └── schema.sql     # Database schema
├── tests/             # Unit tests
│   ├── test_app.py
│   ├── test_crawl.py
│   └── test_robots.py
├── etc/
│   └── pre-commit     # Git hook script
├── specification/
│   └── 01-CLAUDE-SPEC.md
├── pyproject.toml     # Package metadata
├── Dockerfile
├── docker-compose.yml
├── README.md
└── DEVELOPING.md
```

### Key Insights

1. **Async is Essential:** I/O-bound workload benefits massively
2. **SQLite is Fast:** Single writer handles high throughput
3. **FTS5 is Powerful:** Stemming, BM25, snippets out-of-box
4. **Complexity Grows Non-linearly:** Keeping under 2000 LOC requires discipline

### Development Workflow

1. **Edit Code**
2. **Run Tests:** `python3 -m unittest discover -s tests`
3. **Format:** `black src/ tests/`
4. **Lint:** `pylint src/ tests/`
5. **Pre-commit:** `./etc/pre-commit`
6. **Docker Build:** `docker build -t find .`

---

## Comparative Analysis

### vs. Elasticsearch
**Find Advantages:**
- Zero-config setup
- Single binary
- Minimal resources

**Elasticsearch Advantages:**
- Distributed
- Advanced analytics
- REST API

**Use Case:** Find for small sites, ES for enterprise

### vs. Algolia/Meilisearch
**Find Advantages:**
- Self-hosted
- No API limits
- Simple deployment

**Algolia/Meilisearch Advantages:**
- Better ranking
- Real-time updates
- Cloud-hosted option

**Use Case:** Find for self-hosting, Algolia for SaaS

### vs. Sphinx Search
**Find Advantages:**
- Pure Python
- Async crawler included
- Modern codebase

**Sphinx Advantages:**
- Mature
- Multi-language
- Proven at scale

**Use Case:** Find for new projects, Sphinx for legacy

---

## Notable Technical Decisions

### 1. Why asyncio?
**Decision:** Use async I/O for crawler  
**Rationale:** I/O-bound workload, 10-100× more efficient than threads  
**Trade-off:** Complexity vs. performance

### 2. Why SQLite?
**Decision:** Use SQLite instead of PostgreSQL  
**Rationale:**
- FTS5 built-in
- Zero-config
- Fast enough for target use case
- Portable (single file)

### 3. Why Package Templates?
**Decision:** Store HTML in `src/find/templates` and render it with Flask's standard template loader  
**Rationale:** Keep view code focused on request handling while using Flask's documented package template layout  
**Trade-off:** A few more files, but cleaner separation between Python logic and markup

### 4. Why Mono-writer Pattern?
**Decision:** Single DB writer instead of connection pool  
**Rationale:** SQLite WAL mode has write bottleneck anyway  
**Benefit:** Eliminates BUSY errors, simpler code

### 5. Why Content Hashing?
**Decision:** SHA-256 of HTML to detect changes  
**Rationale:** Efficient deduplication, versioning support  
**Cost:** Minimal (hashing is fast)

### 6. Why Thread Pool for Search?
**Decision:** ThreadPoolExecutor instead of sync queries  
**Rationale:** Prevent slow FTS queries from blocking Flask  
**Alternative:** Could use async Flask, but adds complexity

---

## Performance Tuning Guide

### Crawler Tuning

**For Fast Sites (< 500ms response):**
```bash
crawl --seed URL --delay 0.1 --concurrency 9
```

**For Slow Sites (> 2s response):**
```bash
crawl --seed URL --delay 0.5 --concurrency 2 --timeout 30
```

**For Large Sites (> 10k pages):**
```bash
crawl --seed URL --max-pages 50000 --delay 0.2
```

**For Unrestricted Crawl:**
```bash
crawl --seed URL --no-same-host  # WARNING: Can crawl entire web
```

### Database Tuning

**For Read-heavy Workloads:**
```sql
PRAGMA cache_size = -64000;  -- 64MB cache
PRAGMA mmap_size = 268435456;  -- 256MB mmap
```

**For Write-heavy Workloads:**
```sql
PRAGMA synchronous = NORMAL;  -- Trade safety for speed
PRAGMA journal_size_limit = 67108864;  -- 64MB journal
```

### Web Interface Tuning

**For High Traffic:**
- Use gunicorn: `FIND_WEB_WORKERS=4 gunicorn -w "$FIND_WEB_WORKERS" -b 0.0.0.0:7001 find.app:app`
- Increase rate limits
- Add reverse proxy (nginx) for caching

**For Low Latency:**
- Increase `SEARCH_TIMEOUT_SECONDS`
- Optimize FTS index: `INSERT INTO pages_fts(pages_fts) VALUES('optimize')`

---

## Maintenance Tasks

### Regular Updates
1. **Re-crawl periodically:** `crawl --seed URL` (deduplication handles unchanged pages)
2. **Optimize FTS:** `reindex` after major crawls
3. **Vacuum DB:** `sqlite3 ~/.find.db "VACUUM;"` to reclaim space

### Monitoring
- **DB Size:** `ls -lh ~/.find.db`
- **Page Count:** `sqlite3 ~/.find.db "SELECT COUNT(*) FROM pages;"`
- **FTS Count:** `sqlite3 ~/.find.db "SELECT COUNT(*) FROM pages_fts;"`
- **Version Count:** `sqlite3 ~/.find.db "SELECT COUNT(*) FROM page_versions;"`

### Troubleshooting

**Problem:** Slow searches  
**Solution:** Run `reindex`, check query complexity

**Problem:** Out of disk space  
**Solution:** VACUUM database, reduce max_pages

**Problem:** Crawl too slow  
**Solution:** Reduce delay, increase concurrency

**Problem:** Too many 403 errors  
**Solution:** Check User-Agent, respect robots.txt

---

## Conclusion

**Find** is a well-engineered, minimal search engine that successfully achieves its design goals. The codebase demonstrates:
- **Clean architecture** with clear separation of concerns
- **Performance-conscious design** using async I/O and optimized DB access
- **Web etiquette** through robots.txt compliance and rate limiting
- **Security mindfulness** with multi-layer DDoS protection
- **Maintainability** through simplicity and good test coverage

The project is ideal for:
- Static website search
- Personal blog indexing
- Documentation search
- Learning async Python and SQLite FTS

It's **not suitable** for:
- Large-scale web crawling (use Scrapy + Elasticsearch)
- Real-time search (use Meilisearch)
- Multi-language content (needs tokenizer work)
- JavaScript-heavy sites (needs headless browser)

**Overall Assessment:** Excellent implementation for its intended use case. The code quality is high, documentation is clear, and the architecture is sound. The project successfully balances simplicity with functionality.

---

## Technical Metrics

**Lines of Code:**
- `app.py`: ~380 lines
- `crawl.py`: ~802 lines
- `utils.py`: ~110 lines
- `reindex.py`: ~70 lines
- `schema.sql`: ~70 lines
- **Total:** ~1432 lines (well under 2000 LOC target)

**Test Coverage:**
- 3 test modules
- ~15 test cases
- Core functionality well-covered

**Dependencies:**
- Runtime: 7 packages
- Build: 1 package (flit_core)
- Total footprint: ~50MB (Docker image size would be ~150-200MB)

**Performance Benchmarks (Estimated):**
- Crawl speed: 3-5 pages/second (with 0.19s delay)
- Search latency: 10-50ms (typical FTS5 query)
- Index size: ~2-3× text content size
- Memory usage: ~100-200MB (crawler), ~50MB (web interface)

---

## References & Resources

**SQLite FTS5 Documentation:**
- https://sqlite.org/fts5.html

**Async Python:**
- https://docs.python.org/3/library/asyncio.html
- https://docs.aiohttp.org/

**BM25 Ranking:**
- https://en.wikipedia.org/wiki/Okapi_BM25

**robots.txt RFC:**
- RFC 9309: https://www.rfc-editor.org/rfc/rfc9309.html

**Flask Best Practices:**
- https://flask.palletsprojects.com/

**Project Repository:**
- https://github.com/daitangio/find

---

## Acknowledgments

This project was initially designed with ChatGPT 5.2 and then refined by Giovanni Giorgi. The implementation demonstrates effective use of AI-assisted development combined with expert refinement.

**Notable Design Inspirations:**
- Google's page caching (now deprecated)
- Minimal search engines (YaCy, Searx)
- SQLite FTS documentation examples
- Flask Mega-Tutorial patterns

---

**Report Prepared By:** AI Analysis (Claude Sonnet 4.5)  
**Analysis Depth:** Complete codebase review  
**Files Analyzed:** 13 source files, 3 test files, 5 configuration files  
**Documentation Quality:** High (clear README, inline comments, docstrings)
