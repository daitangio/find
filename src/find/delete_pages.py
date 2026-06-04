#!/usr/bin/env python3
"""
Delete indexed pages whose URL matches given regular expression.
"""

from __future__ import annotations

import asyncio
import re

import aiosqlite
import click

from .utils import DATABASE_FILE, ensure_database_present


async def delete_pages_by_pattern(db_path: str, url_regexp: str) -> int:
    """Delete pages matching URL regexp. Returns number of deleted pages."""
    try:
        pattern = re.compile(url_regexp)
    except re.error as exc:
        raise ValueError(f"Invalid URL regexp: {exc}") from exc

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys = ON;")
        await db.execute("PRAGMA journal_mode = WAL;")

        cursor = await db.execute("SELECT id, url FROM pages ORDER BY id;")
        rows = await cursor.fetchall()
        await cursor.close()

        matching_ids = [int(row["id"]) for row in rows if pattern.search(row["url"])]
        if not matching_ids:
            print(f"No pages matched regexp: {url_regexp}")
            return 0

        print(f"Deleting {len(matching_ids)} pages matching regexp: {url_regexp}")
        await db.executemany(
            "DELETE FROM pages WHERE id = ?;",
            ((page_id,) for page_id in matching_ids),
        )
        await db.commit()

    print(f"Deleted {len(matching_ids)} pages")
    return len(matching_ids)


@click.command(help="Delete indexed pages whose URL matches given regular expression")
@click.argument("url_regexp")
@click.option("--db", default=DATABASE_FILE, help="Database file path")
def main(url_regexp: str, db: str) -> None:
    """Main entry point."""
    ensure_database_present(db, create_if_missing=False)
    try:
        asyncio.run(delete_pages_by_pattern(db, url_regexp))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
