Read AGENT.md to understand the project.

Fix the search pagination parameter handling in app.py.

Problem:
- /search currently parses limit and offset with int() directly.
- Non-integer values such as limit=abc or offset=abc raise ValueError and become a 500 response.
- Negative limit values can bypass the intended maximum result cap in SQLite, because LIMIT -1 means no limit.
- Negative offset values are silently coerced to 0, which hides invalid user input.

Expected behavior:
- limit must be an integer from 1 through 50.
- offset must be an integer greater than or equal to 0.
- Missing limit defaults to 10.
- Missing offset defaults to 0.
- Invalid or out-of-range values must return HTTP 400 instead of reaching the search query.

Add focused unit tests for invalid pagination values.
