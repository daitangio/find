#!/usr/bin/env python3
"""
Docstring for find.reindex
Reindex the pages stored in the database.
Remove all the data from pages_fts and re-populate it from most recent pages, using the same logic inside crawl.py
"""
from __future__ import annotations

import asyncio

import aiosqlite
import click

from .utils import DATABASE_FILE, ensure_database_present


async def reindex_fts(db_path: str) -> None:
    """Reindex the FTS table by clearing and repopulating from pages table"""
    print(f"Starting reindex of {db_path}...")

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON;")
        await db.execute("PRAGMA journal_mode = WAL;")

        # Clear the FTS table: use the delete-all pseudo-command:
        # https://sqlite.org/fts5.html#the_delete_all_command
        print("Clearing pages_fts table...")
        await db.execute("INSERT INTO pages_fts(pages_fts) VALUES('delete-all')")

        # Get count of pages to reindex
        cursor = await db.execute("SELECT COUNT(*) as count FROM pages")
        row = await cursor.fetchone()
        total_pages = row["count"] if row else 0

        if total_pages == 0:
            print("No pages found in database. Nothing to reindex.")
            await db.commit()
            return

        print(f"✓ Reindexing {total_pages} pages...")

        await db.execute("INSERT INTO pages_fts(pages_fts) VALUES('rebuild')")
        await db.commit()

        print("✓ Optimizing index...")
        await db.execute("INSERT INTO pages_fts(pages_fts) VALUES('optimize')")
        await db.commit()

        # Verify the reindex worked
        cursor = await db.execute("SELECT COUNT(*) as count FROM pages_fts;")
        row = await cursor.fetchone()
        fts_count = row["count"] if row else 0

        if fts_count == total_pages:
            print(f"✓ Successfully reindexed {fts_count} entries in pages_fts")
        else:
            print(
                f"✗ Verification failed: {total_pages} pages but {fts_count} in pages_fts"
            )


@click.command(help="Reindex the FTS search table from the pages table")
@click.option("--db", default=DATABASE_FILE, help="Database file path")
def main(db: str) -> None:
    """Main entry point"""
    ensure_database_present(db, create_if_missing=False)
    asyncio.run(reindex_fts(db))
