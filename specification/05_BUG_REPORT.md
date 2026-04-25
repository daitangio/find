# Bug Report: FTS5 Query Syntax Errors

## Summary

Searching for punctuation-only or punctuation-heavy input such as `/` or `*` can raise `sqlite3.OperationalError` from SQLite FTS5 and produce an HTTP 500 response.

## Reproduction

1. Start the Flask search UI against an indexed database.
2. Search for `/`.
3. Search for `*`.

Observed errors:

- `/` raises `sqlite3.OperationalError: fts5: syntax error near "/"`
- `*` raises `sqlite3.OperationalError: unknown special query:`

## Root Cause

The application passed user input directly into `pages_fts MATCH ?`. In FTS5, the `MATCH` argument is a query language, not a plain text string. Characters such as `/`, `.`, `-`, and bare `*` are parsed as query syntax, so ordinary user input can become an invalid FTS expression.

## Fix

`parse_search_query()` now converts ordinary unsafe terms into quoted FTS5 phrase values before executing `MATCH`. This makes punctuation inputs valid and returns zero results instead of an internal server error.

The parser still preserves supported search idioms:

- `site:example.com` is translated to `url:"example.com"`
- `url:`, `title:`, and `text:` column filters remain available
- boolean operators such as `AND`, `OR`, and `NOT` remain available

The `/search` route also catches any remaining `sqlite3.OperationalError` from FTS query parsing and returns HTTP 400 instead of HTTP 500.
