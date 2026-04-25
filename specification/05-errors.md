Read AGENT.md to understand the project.

Some characters provoke an internal server error:
- searching for / cause a "sqlite3.OperationalError: fts5: syntax error near "/"
- searching for * cause "sqlite3.OperationalError: unknown special query: "

Analyze the problem and produce a file called 05_BUG_REPORT.md
Fix the code to avoid this error, but retain the ability to narrow search with FTS5 idioms like "site:sitename".
