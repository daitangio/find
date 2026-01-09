#!/usr/bin/env python3
"""
Common database utilities shared between crawl.py and reindex.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
import urllib.robotparser

from urllib.parse import urljoin, urlparse
from importlib import resources
from importlib.metadata import version
import aiohttp

DATABASE_FILE = os.path.join(os.getenv("HOME"), ".find.db")


def fts5_available(conn: sqlite3.Connection) -> bool:
    """Check if FTS5 is available in SQLite"""
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5test USING fts5(x);")
        conn.execute("DROP TABLE __fts5test;")
        return True
    except sqlite3.OperationalError:
        return False


def ensure_database_present(db_file: str, create_if_missing: bool = True):
    """Ensure database exists and has proper schema

    Args:
        db_file: Path to database file
        create_if_missing: If True, create database if it doesn't exist.
                          If False, exit with error if database is missing.
    """
    if not os.path.exists(db_file):
        if create_if_missing:
            print(f"*** [INIT] Creating {db_file}")
            db = sqlite3.connect(db_file)
            if not fts5_available(db):
                raise SystemError("FTS5 needs to be available")
            schema_sql = (
                resources.files("find")
                .joinpath("schema.sql")
                .read_text(encoding="utf-8")
            )
            db.executescript(schema_sql)
            db.commit()
            db.close()
        else:
            print(
                f"*** [ERROR] Database {db_file} not found. Run crawl first to create it."
            )
            sys.exit(1)
    else:
        # Quick check that the database has the expected schema
        db = sqlite3.connect(db_file)
        try:
            if not fts5_available(db):
                raise SystemError("FTS5 needs to be available")
        finally:
            db.close()


def get_version():
    """Get version from pyproject.toml or fallback methods"""
    # First try importlib.metadata if package is installed
    return version("find")


async def get_robots_parser(
    session: aiohttp.ClientSession,
    url: str,
    robots_cache: dict[str, urllib.robotparser.RobotFileParser],
    timeout_s: int = 10,
) -> urllib.robotparser.RobotFileParser:
    parsed = urlparse(url)
    # Use scheme+netloc as key
    origin = f"{parsed.scheme}://{parsed.netloc}"

    if origin in robots_cache:
        return robots_cache[origin]

    robots_url = urljoin(origin, "/robots.txt")
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)

    # Default behaviour: if we can't fetch, we allow.
    # However, we should be polite.
    try:
        # Short timeout for robots.txt
        async with session.get(robots_url, timeout=timeout_s) as resp:
            if resp.status in (401, 403):
                # If strictly forbidden to access robots.txt, we assume 'Disallow: /'
                # (This is a common interpretation: if you hide robots.txt, you hide everything?)
                # Or it means "I don't want you to see my rules"?
                # RFC 9309 says: "If a robot can not obtain the robots.txt file ... due to
                # 401 or 403 ... the robot MUST assume valid file with Disallow: /"
                # But standard RobotFileParser doesn't have a simple flag for "Disallow all"
                # We can parse a disallow all string.
                rp.parse(["User-agent: *", "Disallow: /"])
            elif 400 <= resp.status < 500:
                # 4xx (except 401/403) => No robots.txt => Allow all
                print(f"Robots  Status:{resp.status} allow all")
                rp.parse([])
            elif resp.status >= 500:
                # Server error. RFC 9309: "MUST assume ... Disallow: /" (temporary)
                # For a simple crawler, we skip the site.
                rp.parse(["User-agent: *", "Disallow: /"])
            else:
                # 2xx
                content = await resp.text()
                rp.parse(content.splitlines())
    except Exception as _e:
        # Network errors etc. allow_all
        print(f"Robots fetch error for {origin}: {_e} allow all")
        rp.parse([])

    robots_cache[origin] = rp
    return rp
