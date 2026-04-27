Read AGENT.md to understand the project.

Add a new environment variable called FIND_SHOW_CACHED_PAGE to enable/disable cache usage on app.py

In particular if cache is disabled:
- disable the [Cached version] link in the serach.html page (it must not be present at all).
- do not expose the @app.route("/page/<int:page_id>") route. 

Disable it by default on docker-compose.yml

Add 2 relevant unit tests to test this functionality in both mode (FIND_SHOW_CACHED_PAGE=enabled and FIND_SHOW_CACHED_PAGE=disabled),

Always follow best flask practice.

Update AGENT.md accordingly.