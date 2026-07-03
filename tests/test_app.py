import importlib
import os
import sqlite3
import tempfile
import unittest

from contextlib import closing
from datetime import datetime, timezone
from unittest.mock import patch


import find.app as find_app
from find.app import (
    count_crawled_urls_by_domain,
    extract_meta_refresh,
    find_format_date,
    nice_score,
    parse_search_query,
    search_pages,
)

ROOT = os.path.dirname(os.path.dirname(__file__))


def create_search_db(path: str) -> None:
    schema_path = os.path.join(ROOT, "src", "find", "schema.sql")
    with closing(sqlite3.connect(path)) as conn, conn:
        with open(schema_path, encoding="utf-8") as schema:
            conn.executescript(schema.read())
        conn.executemany(
            """
            INSERT INTO pages (
                id, url, title, html, text, content_hash, status_code, fetched_at, post_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    1,
                    "https://example.com/old",
                    "Old",
                    "<p>alpha old</p>",
                    "alpha old",
                    "old-hash",
                    200,
                    "2024-01-01T00:00:00+00:00",
                    "2024-01-01T00:00:00+00:00",
                ),
                (
                    2,
                    "https://example.com/new",
                    "New",
                    "<p>alpha new</p>",
                    "alpha new",
                    "new-hash",
                    200,
                    "2024-02-01T00:00:00+00:00",
                    "2024-02-01T00:00:00+00:00",
                ),
                (
                    3,
                    "https://example.com/undated",
                    "Undated",
                    "<p>alpha undated</p>",
                    "alpha undated",
                    "undated-hash",
                    200,
                    "2024-03-01T00:00:00+00:00",
                    None,
                ),
                (
                    4,
                    "https://example.com/pizza",
                    "Pizza Menu",
                    "<p>tomato basil</p>",
                    "tomato basil",
                    "pizza-hash",
                    200,
                    "2024-04-01T00:00:00+00:00",
                    "2024-04-01T00:00:00+00:00",
                ),
            ],
        )


class ExtractMetaRefreshTests(unittest.TestCase):
    def test_extract_meta_refresh_returns_url(self) -> None:
        html = (
            "<html><head>"
            '<meta http-equiv="refresh" content="0; URL=/c64ref/c64disasm/" />'
            "</head><body>Hi</body></html>"
        )
        self.assertEqual(extract_meta_refresh(html), "/c64ref/c64disasm/")

    def test_extract_meta_refresh_returns_none_without_tag(self) -> None:
        html = "<html><head></head><body>No redirect</body></html>"
        self.assertIsNone(extract_meta_refresh(html))


class ParseSearchQueryTests(unittest.TestCase):
    def test_parse_search_query_transforms_site_to_url(self) -> None:
        q = "site:example.com"
        self.assertEqual(parse_search_query(q), 'url:"example.com"')

    def test_parse_search_query_transforms_quoted_site(self) -> None:
        q = 'site:"example.com"'
        self.assertEqual(parse_search_query(q), 'url:"example.com"')

    def test_parse_search_query_preserves_other_terms(self) -> None:
        q = "python site:example.com testing"
        self.assertEqual(parse_search_query(q), 'python url:"example.com" testing')

    def test_parse_search_query_case_insensitive(self) -> None:
        q = "SITE:example.com"
        self.assertEqual(parse_search_query(q), 'url:"example.com"')

    def test_parse_search_query_no_site_term(self) -> None:
        q = "python testing"
        self.assertEqual(parse_search_query(q), "python testing")

    def test_parse_search_query_quotes_unsafe_punctuation(self) -> None:
        self.assertEqual(parse_search_query("/"), '"/"')
        self.assertEqual(parse_search_query("*"), '"*"')
        self.assertEqual(parse_search_query("example.com/path"), '"example.com/path"')

    def test_parse_search_query_preserves_fts_idioms(self) -> None:
        self.assertEqual(parse_search_query("sqlite OR postgres"), "sqlite OR postgres")
        self.assertEqual(parse_search_query("title:python"), "title:python")
        self.assertEqual(
            parse_search_query("url:example.com/path"), 'url:"example.com/path"'
        )


class FindFormatDateTests(unittest.TestCase):
    def test_find_format_date_returns_days_ago(self) -> None:
        now = datetime(2024, 2, 3, tzinfo=timezone.utc)

        self.assertEqual(
            find_format_date("2024-02-01T00:00:00+00:00", now=now),
            "2 days ago",
        )

    def test_find_format_date_returns_years_ago(self) -> None:
        now = datetime(2026, 2, 1, tzinfo=timezone.utc)

        self.assertEqual(
            find_format_date("2023-02-01T00:00:00+00:00", now=now),
            "3 years ago",
        )

    def test_find_format_date_handles_singular_units(self) -> None:
        now = datetime(2024, 2, 1, 1, 0, tzinfo=timezone.utc)

        self.assertEqual(
            find_format_date("2024-02-01T00:00:00+00:00", now=now),
            "1 hour ago",
        )

    def test_find_format_date_returns_empty_for_missing_or_invalid_dates(self) -> None:
        now = datetime(2024, 2, 1, tzinfo=timezone.utc)

        self.assertEqual(find_format_date(None, now=now), "")
        self.assertEqual(find_format_date("not a date", now=now), "")


class SearchPagesTests(unittest.TestCase):
    def test_count_crawled_urls_by_domain_groups_by_url_origin(self) -> None:
        with tempfile.NamedTemporaryFile() as db:
            schema_path = os.path.join(ROOT, "src", "find", "schema.sql")
            with closing(sqlite3.connect(db.name)) as conn, conn:
                with open(schema_path, encoding="utf-8") as schema:
                    conn.executescript(schema.read())
                conn.executemany(
                    """
                    INSERT INTO pages (
                        url, title, html, text, content_hash, status_code, fetched_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    [
                        (
                            "https://othersite.com/page",
                            "Other",
                            "<p>other</p>",
                            "other",
                            "other-hash",
                            200,
                            "2024-01-01T00:00:00+00:00",
                        ),
                        (
                            "https://gioorgi.com/one",
                            "One",
                            "<p>one</p>",
                            "one",
                            "one-hash",
                            200,
                            "2024-01-01T00:00:00+00:00",
                        ),
                        (
                            "https://gioorgi.com/two",
                            "Two",
                            "<p>two</p>",
                            "two",
                            "two-hash",
                            200,
                            "2024-01-01T00:00:00+00:00",
                        ),
                    ],
                )
                conn.row_factory = sqlite3.Row

                counts = count_crawled_urls_by_domain(conn)

        self.assertEqual(
            counts,
            [
                ("https://gioorgi.com", 2),
                ("https://othersite.com", 1),
            ],
        )

    def test_search_pages_weights_title_matches_above_text_matches(self) -> None:
        with tempfile.NamedTemporaryFile() as db:
            schema_path = os.path.join(ROOT, "src", "find", "schema.sql")
            with closing(sqlite3.connect(db.name)) as conn, conn:
                with open(schema_path, encoding="utf-8") as schema:
                    conn.executescript(schema.read())
                conn.executemany(
                    """
                    INSERT INTO pages (
                        id, url, title, html, text, content_hash, status_code, fetched_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    [
                        (
                            1,
                            "https://example.com/text",
                            "Reference",
                            "<p>sqlite docs</p>",
                            "sqlite docs",
                            "text-hash",
                            200,
                            "2024-01-01T00:00:00+00:00",
                        ),
                        (
                            2,
                            "https://example.com/title",
                            "SQLite",
                            "<p>reference docs</p>",
                            "reference docs",
                            "title-hash",
                            200,
                            "2024-01-01T00:00:00+00:00",
                        ),
                    ],
                )
                conn.row_factory = sqlite3.Row
                results, total = search_pages(conn, "sqlite")

        self.assertEqual(total, 2)
        # print(results)
        self.assertEqual(
            [result.title for result in results], ["SQLite", "Reference"], results
        )
        self.assertIn("<mark>SQLite</mark>", results[0].snippet)

    def test_nice_score_keeps_tiny_matches_nonzero(self) -> None:
        self.assertEqual(nice_score(-0.00000001), 0.0000001)
        # self.assertEqual(nice_score(-0.000000000001), 0.0)

    def test_search_pages_can_order_by_date(self) -> None:
        with tempfile.NamedTemporaryFile() as db:
            create_search_db(db.name)
            conn = sqlite3.connect(db.name)
            conn.row_factory = sqlite3.Row
            try:
                results, total = search_pages(conn, "alpha", order_by="date")
            finally:
                conn.close()

        self.assertEqual(total, 3)
        self.assertEqual(
            [result.title for result in results], ["New", "Old", "Undated"]
        )
        self.assertEqual(results[0].indexed_at, "2024-02-01T00:00:00+00:00")

    def test_search_template_links_to_date_sort(self) -> None:
        old_db_path = find_app.DB_PATH
        with tempfile.NamedTemporaryFile() as db:
            create_search_db(db.name)
            find_app.DB_PATH = db.name
            find_app.app.config["TESTING"] = True
            try:
                response = find_app.app.test_client().get("/search?q=alpha")
            finally:
                find_app.DB_PATH = old_db_path

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"orderBy=date", response.data)

    def test_search_template_shows_last_indexed_date_for_each_result(self) -> None:
        old_db_path = find_app.DB_PATH
        with tempfile.NamedTemporaryFile() as db:
            create_search_db(db.name)
            find_app.DB_PATH = db.name
            find_app.app.config["TESTING"] = True
            try:
                response = find_app.app.test_client().get("/search?q=alpha")
            finally:
                find_app.DB_PATH = old_db_path

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Indexed", response.data)
        self.assertIn(b"4 months ago", response.data)

    def test_search_template_preserves_date_sort_in_pagination(self) -> None:
        old_db_path = find_app.DB_PATH
        with tempfile.NamedTemporaryFile() as db:
            create_search_db(db.name)
            find_app.DB_PATH = db.name
            find_app.app.config["TESTING"] = True
            try:
                response = find_app.app.test_client().get(
                    "/search?q=alpha&limit=1&orderBy=date"
                )
            finally:
                find_app.DB_PATH = old_db_path

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"orderBy=date", response.data)
        self.assertIn(b"offset=1", response.data)

    def test_search_rejects_non_integer_pagination_values(self) -> None:
        old_db_path = find_app.DB_PATH
        with tempfile.NamedTemporaryFile() as db:
            create_search_db(db.name)
            find_app.DB_PATH = db.name
            find_app.app.config["TESTING"] = True
            try:
                limit_response = find_app.app.test_client().get(
                    "/search?q=alpha&limit=abc"
                )
                offset_response = find_app.app.test_client().get(
                    "/search?q=alpha&offset=abc"
                )
            finally:
                find_app.DB_PATH = old_db_path

        self.assertEqual(limit_response.status_code, 400)
        self.assertEqual(offset_response.status_code, 400)

    def test_search_rejects_out_of_range_pagination_values(self) -> None:
        old_db_path = find_app.DB_PATH
        with tempfile.NamedTemporaryFile() as db:
            create_search_db(db.name)
            find_app.DB_PATH = db.name
            find_app.app.config["TESTING"] = True
            try:
                negative_limit_response = find_app.app.test_client().get(
                    "/search?q=alpha&limit=-1"
                )
                zero_limit_response = find_app.app.test_client().get(
                    "/search?q=alpha&limit=0"
                )
                high_limit_response = find_app.app.test_client().get(
                    "/search?q=alpha&limit=51"
                )
                negative_offset_response = find_app.app.test_client().get(
                    "/search?q=alpha&offset=-1"
                )
            finally:
                find_app.DB_PATH = old_db_path

        self.assertEqual(negative_limit_response.status_code, 400)
        self.assertEqual(zero_limit_response.status_code, 400)
        self.assertEqual(high_limit_response.status_code, 400)
        self.assertEqual(negative_offset_response.status_code, 400)

    def test_search_handles_punctuation_only_queries(self) -> None:
        old_db_path = find_app.DB_PATH
        with tempfile.NamedTemporaryFile() as db:
            create_search_db(db.name)
            find_app.DB_PATH = db.name
            find_app.app.config["TESTING"] = True
            try:
                slash_response = find_app.app.test_client().get("/search?q=/")
                star_response = find_app.app.test_client().get("/search?q=*")
            finally:
                find_app.DB_PATH = old_db_path

        self.assertEqual(slash_response.status_code, 200)
        self.assertEqual(star_response.status_code, 200)
        self.assertIn(b"No results.", slash_response.data)
        self.assertIn(b"No results.", star_response.data)

    def test_search_marks_title_matches_for_title_queries(self) -> None:
        old_db_path = find_app.DB_PATH
        with tempfile.NamedTemporaryFile() as db:
            create_search_db(db.name)
            find_app.DB_PATH = db.name
            find_app.app.config["TESTING"] = True
            try:
                response = find_app.app.test_client().get("/search?q=title:pizza")
            finally:
                find_app.DB_PATH = old_db_path

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<mark>Pizza</mark> Menu", response.data)


class CachedPageConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        importlib.reload(find_app)

    def test_cached_page_enabled_exposes_link_and_route(self) -> None:
        old_db_path = find_app.DB_PATH
        with patch.dict(os.environ, {"FIND_SHOW_CACHED_PAGE": "enabled"}):
            reloaded_app = importlib.reload(find_app)
        with tempfile.NamedTemporaryFile() as db:
            create_search_db(db.name)
            reloaded_app.DB_PATH = db.name
            reloaded_app.app.config["TESTING"] = True
            try:
                with reloaded_app.app.test_client() as client:
                    search_response = client.get("/search?q=alpha")
                    page_response = client.get("/page/1")
            finally:
                reloaded_app.DB_PATH = old_db_path

        self.assertEqual(search_response.status_code, 200)
        self.assertIn(b"[Cached version]", search_response.data)
        self.assertEqual(page_response.status_code, 200)

    def test_cached_page_default_hides_link_and_route(self) -> None:
        old_db_path = find_app.DB_PATH
        clean_env = {"HOME": os.environ.get("HOME", "/tmp")}
        with patch.dict(os.environ, clean_env, clear=True):
            reloaded_app = importlib.reload(find_app)
        with tempfile.NamedTemporaryFile() as db:
            create_search_db(db.name)
            reloaded_app.DB_PATH = db.name
            reloaded_app.app.config["TESTING"] = True
            try:
                with reloaded_app.app.test_client() as client:
                    search_response = client.get("/search?q=alpha")
                    page_response = client.get("/page/1")
            finally:
                reloaded_app.DB_PATH = old_db_path

        self.assertEqual(search_response.status_code, 200)
        self.assertNotIn(b"[Cached version]", search_response.data)
        self.assertEqual(page_response.status_code, 404)

    def test_cached_page_disabled_hides_link_and_route(self) -> None:
        old_db_path = find_app.DB_PATH
        with patch.dict(os.environ, {"FIND_SHOW_CACHED_PAGE": "disabled"}):
            reloaded_app = importlib.reload(find_app)
        with tempfile.NamedTemporaryFile() as db:
            create_search_db(db.name)
            reloaded_app.DB_PATH = db.name
            reloaded_app.app.config["TESTING"] = True
            try:
                with reloaded_app.app.test_client() as client:
                    search_response = client.get("/search?q=alpha")
                    page_response = client.get("/page/1")
            finally:
                reloaded_app.DB_PATH = old_db_path

        self.assertEqual(search_response.status_code, 200)
        self.assertNotIn(b"[Cached version]", search_response.data)
        self.assertEqual(page_response.status_code, 404)

    def test_cached_page_renders_html_in_sandboxed_iframe(self) -> None:
        old_db_path = find_app.DB_PATH
        with patch.dict(os.environ, {"FIND_SHOW_CACHED_PAGE": "enabled"}):
            reloaded_app = importlib.reload(find_app)
        with tempfile.NamedTemporaryFile() as db:
            create_search_db(db.name)
            reloaded_app.DB_PATH = db.name
            reloaded_app.app.config["TESTING"] = True
            try:
                response = reloaded_app.app.test_client().get("/page/1")
            finally:
                reloaded_app.DB_PATH = old_db_path

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<iframe", response.data)
        self.assertIn(b"sandbox", response.data)
        self.assertIn(b'referrerpolicy="no-referrer"', response.data)
        self.assertIn(b'srcdoc="&lt;p&gt;alpha old&lt;/p&gt;"', response.data)
        self.assertNotIn(b"<div><p>alpha old</p></div>", response.data)


class AppTemplateTests(unittest.TestCase):
    def test_base_template_includes_find_version(self) -> None:
        find_app.app.config["TESTING"] = True
        with patch.object(find_app.utils, "get_version", return_value="1.2.3-test"):
            response = find_app.app.test_client().get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Find 1.2.3-test", response.data)


if __name__ == "__main__":
    unittest.main()
