## Purpose

Define the behavior contract for searching the crawled site index with predictable relevance, bounded execution, and safe query handling.

## ADDED Requirements

### Requirement: Search accepts natural-language and fielded queries
The system SHALL accept free-text requests and support FTS-style operators alongside common query patterns such as quoted phrases, `site:` filters, and column-scoped terms.

#### Scenario: Plain text search
- **WHEN** a user submits a query such as `python sqlite`
- **THEN** the system matches pages using the FTS index and returns the most relevant results.

#### Scenario: Site-scoped search
- **WHEN** a user submits `site:example.com python`
- **THEN** the system interprets the filter as a URL constraint and searches only pages from that origin.

#### Scenario: Quoted phrase search
- **WHEN** a user submits a quoted phrase such as `"full text search"`
- **THEN** the system preserves the phrase semantics and searches for that exact multi-word expression.

### Requirement: Search ranking favors relevance and discoverability
The system SHALL rank results by a combination of FTS relevance, inbound link popularity, and stronger title matches while preserving a bounded score range suitable for display.

#### Scenario: More relevant title matches rank higher
- **WHEN** two pages match the same query and one has a title hit while the other only matches body text
- **THEN** the title-matched result SHALL be ranked ahead of the body-only result.

#### Scenario: Inbound links boost a page
- **WHEN** a page has repeated cross-links from other indexed pages
- **THEN** the search result SHALL receive a positive ranking adjustment without creating unbounded score inflation.

### Requirement: Search rejects malformed or abusive requests
The system SHALL reject requests with invalid parameters, excessive query length, too many terms, or invalid FTS expressions with explicit HTTP errors.

#### Scenario: Query too long
- **WHEN** a user submits a request longer than the configured maximum
- **THEN** the system SHALL return an HTTP 400 error explaining the query is too long.

#### Scenario: Too many terms
- **WHEN** a user submits more search terms than the configured maximum
- **THEN** the system SHALL reject the request before executing the database query.

#### Scenario: Search timeout
- **WHEN** a search requires longer than the allowed execution window
- **THEN** the system SHALL return an HTTP 504 error rather than hanging the request.
