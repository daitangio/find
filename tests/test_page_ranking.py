import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

import find.app as find_app
from find.app import search_pages

ROOT = os.path.dirname(os.path.dirname(__file__))


def create_rank_db(
    path: str, pages: list[tuple], links: list[tuple] | None = None
) -> None:
    schema_path = os.path.join(ROOT, "src", "find", "schema.sql")
    with closing(sqlite3.connect(path)) as conn, conn:
        with open(schema_path, encoding="utf-8") as schema:
            conn.executescript(schema.read())
        conn.executemany(
            """
            INSERT INTO pages (
                id, url, title, html, text, content_hash, status_code, fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            pages,
        )
        if links:
            conn.executemany(
                """
                INSERT INTO links (from_page_id, to_url, to_page_id)
                VALUES (?, ?, ?);
                """,
                links,
            )


def rank_results(pages: list[tuple], links: list[tuple] | None = None):
    with tempfile.NamedTemporaryFile() as db:
        create_rank_db(db.name, pages, links)
        conn = sqlite3.connect(db.name)
        conn.row_factory = sqlite3.Row
        try:
            return search_pages(conn, "needle")
        finally:
            conn.close()


class PageRankTests(unittest.TestCase):
    def test_back_links_weight_above_pages_without_back_links(self) -> None:
        pages = [
            (
                1,
                "https://example.com/linked",
                "Needle",
                "<p>same body</p>",
                "same body",
                "linked-hash",
                200,
                "2024-01-01T00:00:00+00:00",
            ),
            (
                2,
                "https://example.com/unlinked",
                "Needle",
                "<p>same body</p>",
                "same body",
                "unlinked-hash",
                200,
                "2024-01-01T00:00:00+00:00",
            ),
            (
                3,
                "https://example.com/source",
                "Source",
                "<p>source body</p>",
                "source body",
                "source-hash",
                200,
                "2024-01-01T00:00:00+00:00",
            ),
        ]
        links = [(3, "https://example.com/linked", 1)]

        results, total = rank_results(pages, links)

        self.assertEqual(total, 2)
        self.assertEqual([result.id for result in results], [1, 2])
        self.assertGreater(results[0].rank, results[1].rank)

    def test_link_boost_weight_controls_inbound_link_boost(self) -> None:
        pages = [
            (
                1,
                "https://example.com/linked",
                "Needle",
                "<p>same body</p>",
                "same body",
                "linked-hash",
                200,
                "2024-01-01T00:00:00+00:00",
            ),
            (
                2,
                "https://example.com/unlinked",
                "Needle",
                "<p>same body</p>",
                "same body",
                "unlinked-hash",
                200,
                "2024-01-01T00:00:00+00:00",
            ),
            (
                3,
                "https://example.com/source",
                "Source",
                "<p>source body</p>",
                "source body",
                "source-hash",
                200,
                "2024-01-01T00:00:00+00:00",
            ),
        ]
        links = [(3, "https://example.com/linked", 1)]

        with patch.object(find_app, "LINK_BOOST_WEIGHT", 0.0):
            no_boost_results, _ = rank_results(pages, links)
        with patch.object(find_app, "LINK_BOOST_WEIGHT", 1.0):
            boosted_results, _ = rank_results(pages, links)

        no_boost_ranks = {result.id: result.rank for result in no_boost_results}
        boosted_ranks = {result.id: result.rank for result in boosted_results}
        self.assertEqual(no_boost_ranks[1], no_boost_ranks[2])
        self.assertGreater(boosted_ranks[1], boosted_ranks[2])

    def test_link_boost_cap_limits_inbound_link_boost(self) -> None:
        pages = [
            (
                1,
                "https://example.com/many-links",
                "Needle",
                "<p>same body</p>",
                "same body",
                "many-links-hash",
                200,
                "2024-01-01T00:00:00+00:00",
            ),
            (
                2,
                "https://example.com/capped-links",
                "Needle",
                "<p>same body</p>",
                "same body",
                "capped-links-hash",
                200,
                "2024-01-01T00:00:00+00:00",
            ),
        ]
        pages.extend(
            (
                source_id,
                f"https://example.com/source-{source_id}",
                f"Source {source_id}",
                "<p>source body</p>",
                "source body",
                f"source-{source_id}-hash",
                200,
                "2024-01-01T00:00:00+00:00",
            )
            for source_id in range(3, 8)
        )
        links = [
            (3, "https://example.com/many-links", 1),
            (4, "https://example.com/many-links", 1),
            (5, "https://example.com/many-links", 1),
            (6, "https://example.com/many-links", 1),
            (7, "https://example.com/many-links", 1),
            (3, "https://example.com/capped-links", 2),
            (4, "https://example.com/capped-links", 2),
        ]

        with (
            patch.object(find_app, "LINK_BOOST_WEIGHT", 1.0),
            patch.object(find_app, "LINK_BOOST_CAP", 2),
        ):
            results, total = rank_results(pages, links)

        ranks = {result.id: result.rank for result in results}
        self.assertEqual(total, 2)
        self.assertEqual(ranks[1], ranks[2])

    def test_bm25_title_weight_prioritizes_title_matches(self) -> None:
        pages = [
            (
                1,
                "https://example.com/text",
                "Reference",
                "<p>needle docs</p>",
                "needle docs",
                "text-hash",
                200,
                "2024-01-01T00:00:00+00:00",
            ),
            (
                2,
                "https://example.com/title",
                "Needle",
                "<p>reference docs</p>",
                "reference docs",
                "title-hash",
                200,
                "2024-01-01T00:00:00+00:00",
            ),
        ]

        with patch.object(find_app, "BM25_TITLE_WEIGHT", 5.0):
            results, total = rank_results(pages)

        self.assertEqual(total, 2)
        self.assertEqual([result.id for result in results], [2, 1])


if __name__ == "__main__":
    unittest.main()
