Read AGENT.md to understand the project.

Inside app.py:
- Understand how date is printed in the search results: see SEARCH_HTML template line with  {% set formatted_post_date = r.post_date|format_post_date %}
- Implement a custom format_post_date called "find_format_date"
- The new implementation need to print the date relative to current timestamp (i.e. "2 days ago", "3 years ago" )
- Add unit tests
- Ensure old unit tests pass, and in case fix them