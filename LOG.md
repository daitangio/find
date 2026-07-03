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
- 2026-06-17: Closed test-side SQLite connections explicitly in `tests/test_app.py`, `tests/test_page_ranking.py`, and `tests/test_delete_pages.py` to remove Python 3.13 `ResourceWarning` noise about unclosed databases.
- 2026-06-17: Missing: re-check future tests for `sqlite3.connect(...)` usage since `with conn:` does not close the connection.
- 2026-06-17: Updated `AGENTS.md` to match current code in `src/find` and runtime files. Corrected database lifecycle notes, clarified crawler concurrency and metadata-refresh behavior, tightened search and delete-pages documentation, and aligned Docker notes with `Dockerfile`, `docker-compose.yml`, and `initAndFind.sh`.
- 2026-06-17: Missing after this pass: none for the requested task. Verification still needs to run.

