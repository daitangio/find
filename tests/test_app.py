import os
import sqlite3
import sys
import tempfile
import unittest

from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import find.app as find_app
from find.app import extract_meta_refresh, find_format_date, parse_search_query, search_pages


def create_search_db(path: str) -> None:
    schema_path = os.path.join(ROOT, "src", "find", "schema.sql")
    with sqlite3.connect(path) as conn:
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
        self.assertEqual([result.title for result in results], ["New", "Old", "Undated"])

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


if __name__ == "__main__":
    unittest.main()
