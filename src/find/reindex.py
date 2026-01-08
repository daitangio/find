#!/usr/bin/env python3
"""
Docstring for find.reindex
Reindex the pages stored in the database.
Remove all the data from pages_fts and re-populate it from most recent pages, using the same logic inside crawl.py
"""
from __future__ import annotations

import argparse
import asyncio

import aiosqlite

from .db_utils import DATABASE_FILE, ensure_database_present


async def reindex_fts(db_path: str) -> None:
    """Reindex the FTS table by clearing and repopulating from pages table"""
    print(f"Starting reindex of {db_path}...")
    
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON;")
        await db.execute("PRAGMA journal_mode = WAL;")
        
        # Clear the FTS table
        print("Clearing pages_fts table...")
        await db.execute("DELETE FROM pages_fts;")
        
        # Get count of pages to reindex
        cursor = await db.execute("SELECT COUNT(*) as count FROM pages;")
        row = await cursor.fetchone()
        total_pages = row["count"] if row else 0
        
        if total_pages == 0:
            print("No pages found in database. Nothing to reindex.")
            await db.commit()
            return
        
        print(f"Reindexing {total_pages} pages...")
        
        # Repopulate FTS table from pages
        # Use the same logic as the triggers in schema.sql
        cursor = await db.execute("""
            SELECT id, title, text, url 
            FROM pages 
            ORDER BY id
        """)
        
        reindexed = 0
        async for row in cursor:
            await db.execute("""
                INSERT INTO pages_fts(rowid, title, text, url)
                VALUES (?, ?, ?, ?)
            """, (row["id"], row["title"], row["text"], row["url"]))
            
            reindexed += 1
            if reindexed % 100 == 0:
                print(f"Reindexed {reindexed}/{total_pages} pages...")
                await db.commit()  # Periodic commits
        
        await db.commit()
        print(f"Successfully reindexed {reindexed} pages.")
        
        # Verify the reindex worked
        cursor = await db.execute("SELECT COUNT(*) as count FROM pages_fts;")
        row = await cursor.fetchone()
        fts_count = row["count"] if row else 0
        
        if fts_count == total_pages:
            print(f"✓ Verification passed: {fts_count} entries in pages_fts")
        else:
            print(f"✗ Verification failed: {total_pages} pages but {fts_count} in pages_fts")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    p = argparse.ArgumentParser(
        description="Reindex the FTS search table from the pages table"
    )
    p.add_argument("--db", default=DATABASE_FILE, help="Database file path")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    """Main async function"""
    await reindex_fts(args.db)


def main():
    """Main entry point"""
    args = parse_args()
    ensure_database_present(args.db, create_if_missing=False)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()