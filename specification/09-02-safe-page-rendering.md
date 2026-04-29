Read AGENT.md to understand the project.

Fix cached page rendering so crawled HTML is treated as untrusted content.

Problem:
- The cached page view renders stored crawled HTML with the Jinja safe filter.
- Stored HTML can contain scripts, inline event handlers, forms, hostile styles, or navigation behavior.
- Rendering that HTML directly inside the Find application document gives crawled pages the same DOM context as the app UI.
- FIND_SHOW_CACHED_PAGE is currently enabled by default in app.py, so the unsafe route is exposed unless explicitly disabled.

Expected behavior:
- Cached page viewing must be disabled by default.
- FIND_SHOW_CACHED_PAGE must still allow explicitly enabling the feature for trusted/local use.
- When enabled, cached HTML must not be injected directly into the parent application document.
- Render cached HTML inside a sandboxed iframe with no script/form/top-navigation permissions.
- Add focused unit tests for the default-disabled behavior and sandboxed rendering.
