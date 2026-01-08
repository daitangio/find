#!/usr/bin/env python3
"""
Common database utilities shared between crawl.py and reindex.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
from importlib import resources

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
