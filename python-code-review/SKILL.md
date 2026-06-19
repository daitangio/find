---
name: python-code-review
description: Review Python code and tests for the `find` project, with emphasis on correctness, regression risk, test quality, and maintainability. Use when the user asks for a Python review, code review, test review, improvement suggestions, prioritized fixes, or a quality pass over files in `src/find` and `tests`.
---

# Python Code Review

Perform a read-only review for this repository's Python code. Focus on concrete issues backed by observed code and nearby tests, not generic style advice.

## Workflow

1. Read [AGENTS.md](../AGENTS.md) before reviewing so repo-specific constraints stay in scope.
2. Read the requested implementation files first.
3. Read corresponding or nearby tests in `tests/` to check behavior coverage and regression risk.
4. Run only targeted commands or targeted tests when they materially improve confidence.
5. Prefer `rtk`-prefixed shell commands in this repository.
6. Do not edit files unless the user explicitly asks to fix the findings after the review.

## Review Priorities

- Correctness bugs
- Behavioral regressions
- Reliability risks
- Missing or weak tests
- Performance or maintainability problems only when they are concrete and user-relevant

Ignore issues that are purely stylistic or speculative.

## Output Contract

Return sections in this order:

1. Findings
2. Open Questions
3. Suggested Next Steps

For `Findings`:

- Report at most 7 items.
- Order by priority.
- Include severity, location, issue, and recommended fix.
- Keep each item concise and specific.

If no material issues are found, say so explicitly and list residual risks or testing gaps.

## Review File Output

Write the review to `review/<YYYY-MM-DD>-python.md`.

Use the current local date in the filename.

Example: `review/2026-06-19-python.md`
