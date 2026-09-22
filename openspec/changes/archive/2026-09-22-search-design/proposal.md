## Why

The existing project has a working search surface, but its behavior is spread across the Flask layer, SQLite FTS configuration, and rank heuristics. The search experience needs a clear contract: predictable query parsing, relevant ranking, and safe handling of malformed or oversized queries so the tool remains usable on small static sites without becoming brittle under abuse.

## What Changes

- Define the user-facing search capability for crawled pages as a first-class behavior contract.
- Document query parsing rules for plain terms, quoted phrases, `site:` filters, column filters, and FTS boolean operators.
- Define ranking behavior as a combination of SQLite FTS BM25 scoring, link-based popularity, and title weighting.
- Establish explicit limits for query size, number of terms, pagination, and timeout behavior to protect the service.
- Capture the search UX contract for results display, ordering, and cached-page handling without locking in a specific implementation.

## Capabilities

### New Capabilities
- `search`: Search across crawled pages with FTS-safe parsing, ranking, and bounded execution.

### Modified Capabilities
- None

## Impact

- Main behavior is centered in `src/find/app.py` and the SQLite FTS model in `src/find/schema.sql`.
- Planned changes affect search parsing semantics, ranking thresholds, and production safety checks.
- Existing tests in `tests/test_app.py` provide the primary validation surface for query parsing and result behavior.
