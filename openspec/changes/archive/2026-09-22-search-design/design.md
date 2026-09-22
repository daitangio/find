## Context

The current search design is implemented as a Flask route backed by SQLite FTS5. The crawl stores canonical page metadata in `pages`, tracks link edges in `links`, and refreshes a virtual table `pages_fts` through triggers. Query execution occurs in `src/find/app.py`, where session-local work is isolated in a thread pool and the result list is shaped for the template layer.

## Goals / Non-Goals

**Goals:**
- Keep search fast and explainable for small static websites.
- Preserve predictable behavior across common inputs such as plain terms, `site:` filters, and quoted phrases.
- Protect the service with bounded query complexity and execution time.

**Non-Goals:**
- Building a distributed or multi-user search backend.
- Replacing FTS5 with a separate search engine.
- Introducing a complex ranking model beyond the existing BM25 + link boost approach.

## Decisions

### Decision: Use SQLite FTS5 as the single source of truth for matching
The project already stores canonical document text and metadata in SQLite, and FTS5 is the native mechanism for bounded full-text search. This keeps the search path compact, reduces operational surface area, and aligns with the repository’s “small and self-contained” architecture.

**Alternatives considered:**
- External search services: rejected because the project intentionally avoids extra infrastructure.
- Pure SQL LIKE queries: rejected because they do not provide ranking and tokenization semantics comparable to FTS5.

### Decision: Sanitize input at the query-parsing boundary
The search parser converts `site:` filters into URL-based constraints, escapes unsafe punctuation, and preserves boolean operators such as `AND`, `OR`, and `NOT`. This keeps malformed inputs from leaking into SQLite FTS syntax in ways that create ambiguous or unsafe behavior.

**Alternatives considered:**
- Passing raw user input directly to SQLite: rejected because it creates parsing edge cases and makes invalid queries difficult to control.
- Broad blocking of all punctuation: rejected because users expect quoted phrases and domain-scoped filtering to work.

### Decision: Rank with BM25 plus link popularity and title weight
The ranking function combines BM25 on title/text/url with a bounded link boost from inbound `links` edges and higher weight on title hits. This keeps scores stable and interpretable without requiring a large re-ranking pipeline.

**Alternatives considered:**
- Pure BM25 only: rejected because it underweights highly connected pages.
- Pure link-count ranking: rejected because it ignores the content signal and produces poor relevance on fresh or sparse pages.

### Decision: Keep safety limits at the HTTP layer and execution layer
The app enforces query length, maximum term count, page limit, offset validation, and a per-search timeout. This creates a defensive ring around the database and gives the UI predictable error responses when requests exceed operating limits.

**Alternatives considered:**
- Only database-side limits: rejected because they fail to produce graceful user-facing errors.
- No execution timeout: rejected because a pathological search could monopolize worker capacity.

## Risks / Trade-offs

- [Ranking quality can drift as the corpus grows] → Mitigate by keeping weights configurable via environment variables and validating them against the test suite.
- [Query edge cases may surprise users] → Mitigate by preserving common FTS operators while normalizing unsafe tokens and validating invalid expressions as HTTP 400 responses.
- [Search timeout pressure under heavy load] → Mitigate by using a bounded thread pool and strict query-complexity safeguards.

## Migration Plan

No data migration is required for the initial design. The existing `pages_fts` virtual table and trigger-backed sync model are sufficient to support this capability. If ranking parameters change in the future, they can be adjusted via environment variables without changing the schema contract.

## Open Questions

- None at this time. The current design is sufficient for the project’s compact search engine scope.
