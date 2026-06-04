import os
import sqlite3
import tempfile
import unittest

from click.testing import CliRunner

from find.delete_pages import delete_pages_by_pattern, main

ROOT = os.path.dirname(os.path.dirname(__file__))


def create_delete_db(path: str) -> None:
    schema_path = os.path.join(ROOT, "src", "find", "schema.sql")
    with sqlite3.connect(path) as conn:
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
                    "https://example.com/docs/one",
                    "Doc One",
                    "<p>doc one</p>",
                    "doc one",
                    "doc-one",
                    200,
                    "2024-01-01T00:00:00+00:00",
                ),
                (
                    2,
                    "https://example.com/docs/two",
                    "Doc Two",
                    "<p>doc two</p>",
                    "doc two",
                    "doc-two",
                    200,
                    "2024-01-02T00:00:00+00:00",
                ),
                (
                    3,
                    "https://example.com/blog/three",
                    "Blog Three",
                    "<p>blog three</p>",
                    "blog three",
                    "blog-three",
                    200,
                    "2024-01-03T00:00:00+00:00",
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO page_versions (
                page_id, content_hash, title, html, text, status_code, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    1,
                    "doc-one-v2",
                    "Doc One v2",
                    "<p>doc one v2</p>",
                    "doc one v2",
                    200,
                    "2024-02-01T00:00:00+00:00",
                ),
                (
                    2,
                    "doc-two-v2",
                    "Doc Two v2",
                    "<p>doc two v2</p>",
                    "doc two v2",
                    200,
                    "2024-02-02T00:00:00+00:00",
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO links (from_page_id, to_url, to_page_id, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            [
                (
                    1,
                    "https://example.com/docs/two",
                    2,
                    "2024-01-01T00:00:00+00:00",
                    "2024-01-01T00:00:00+00:00",
                ),
                (
                    3,
                    "https://example.com/docs/one",
                    1,
                    "2024-01-03T00:00:00+00:00",
                    "2024-01-03T00:00:00+00:00",
                ),
            ],
        )


class DeletePagesTests(unittest.IsolatedAsyncioTestCase):
    async def test_delete_pages_by_pattern_removes_matching_pages_and_versions(
        self,
    ) -> None:
        with tempfile.NamedTemporaryFile() as db:
            create_delete_db(db.name)

            deleted = await delete_pages_by_pattern(db.name, r"/docs/")

            self.assertEqual(deleted, 2)
            with sqlite3.connect(db.name) as conn:
                pages = conn.execute(
                    "SELECT id, url FROM pages ORDER BY id;"
                ).fetchall()
                versions = conn.execute(
                    "SELECT page_id FROM page_versions ORDER BY page_id;"
                ).fetchall()
                links = conn.execute(
                    "SELECT from_page_id, to_url, to_page_id FROM links ORDER BY id;"
                ).fetchall()
                fts_count = conn.execute(
                    "SELECT COUNT(*) FROM pages_fts;"
                ).fetchone()[0]

            self.assertEqual(pages, [(3, "https://example.com/blog/three")])
            self.assertEqual(versions, [])
            self.assertEqual(
                links,
                [(3, "https://example.com/docs/one", None)],
            )
            self.assertEqual(fts_count, 1)

    async def test_delete_pages_by_pattern_returns_zero_when_no_match(self) -> None:
        with tempfile.NamedTemporaryFile() as db:
            create_delete_db(db.name)

            deleted = await delete_pages_by_pattern(db.name, r"/missing/")

            self.assertEqual(deleted, 0)
            with sqlite3.connect(db.name) as conn:
                page_count = conn.execute("SELECT COUNT(*) FROM pages;").fetchone()[0]
            self.assertEqual(page_count, 3)


class DeletePagesCliTests(unittest.TestCase):
    def test_cli_rejects_invalid_regexp(self) -> None:
        with tempfile.NamedTemporaryFile() as db:
            create_delete_db(db.name)

            result = CliRunner().invoke(main, ["--db", db.name, "["])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid URL regexp", result.output)


if __name__ == "__main__":
    unittest.main()
