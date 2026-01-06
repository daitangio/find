#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.resources as resources
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse

import aiohttp
import aiosqlite
from bs4 import BeautifulSoup

import os

DATABASE_FILE = os.path.join(os.getenv("HOME"), ".find.db")

DEFAULT_UA = "Find/0.1 (+https://github.com/daitangio/find)"


def fts5_available(conn: sqlite3.Connection) -> bool:
    # Quick sanity check: will fail if SQLite not compiled with FTS5
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5test USING fts5(x);")
        conn.execute("DROP TABLE __fts5test;")
        return True
    except sqlite3.OperationalError:
        return False


def ensure_database_present(db_file: str):
    if not os.path.exists(db_file):
        import sqlite3

        print(f"*** [INIT] Creating {db_file}")
        db = sqlite3.connect(db_file)
        if fts5_available(db) == False:
            raise Exception("FT5 Need to be available")
        schema_sql = (
            resources.files("find").joinpath("schema.sql").read_text(encoding="utf-8")
        )
        db.executescript(schema_sql)
        db.commit()
        db.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_url(url: str) -> Optional[str]:
    """
    Basic URL normalization:
      - strip fragments
      - normalize scheme/host to lowercase
      - drop default ports (:80 http, :443 https)
      - collapse multiple slashes in path
    """
    if not url:
        return None
    url, _frag = urldefrag(url)
    url = url.strip()

    try:
        p = urlparse(url)
    except Exception:
        return None

    if p.scheme not in ("http", "https"):
        return None
    if not p.netloc:
        return None

    scheme = p.scheme.lower()
    netloc = p.netloc.lower()

    # Remove default ports
    if (scheme == "http" and netloc.endswith(":80")) or (
        scheme == "https" and netloc.endswith(":443")
    ):
        netloc = netloc.rsplit(":", 1)[0]

    # Normalize path
    path = p.path or "/"
    path = re.sub(r"/{2,}", "/", path)

    # Keep query as-is (real crawlers would canonicalize more carefully)
    normalized = urlunparse((scheme, netloc, path, "", p.query, ""))
    return normalized


def same_host(a: str, b: str) -> bool:
    return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()


def html_to_text_and_links(
    base_url: str, html: str
) -> tuple[str | None, str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else None
    text = soup.get_text(" ", strip=True)

    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        norm = normalize_url(abs_url)
        if norm:
            links.append(norm)

    # De-dup while preserving order
    seen = set()
    uniq_links = []
    for u in links:
        if u not in seen:
            seen.add(u)
            uniq_links.append(u)

    return title, text, uniq_links


def content_hash(html: str) -> str:
    return hashlib.sha256(html.encode("utf-8", errors="ignore")).hexdigest()


## Writer
@dataclass
class PageJob:
    url: str
    status_code: int | None
    html: str
    title: str | None
    text: str
    out_links: list[str]
    fetched_at: str


@dataclass
class FetchResult:
    url: str
    status: int | None
    content_type: str | None
    html: str | None
    error: str | None


async def fetch_html(
    session: aiohttp.ClientSession, url: str, timeout_s: int, max_bytes: int
) -> FetchResult:
    try:
        async with session.get(
            url, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as resp:
            ctype = resp.headers.get("content-type")
            status = resp.status

            # Only accept HTML-ish content types; still allow missing ctype
            if ctype and "html" not in ctype.lower():
                return FetchResult(
                    url=url,
                    status=status,
                    content_type=ctype,
                    html=None,
                    error="non-html",
                )

            # Read with size cap
            body = (
                await resp.content.readexactly(
                    min(max_bytes, resp.content_length or max_bytes)
                )
                if resp.content_length and resp.content_length > 0
                else await resp.content.read(max_bytes + 1)
            )

            if len(body) > max_bytes:
                return FetchResult(
                    url=url,
                    status=status,
                    content_type=ctype,
                    html=None,
                    error="too-large",
                )

            # Best-effort decode
            html = body.decode(resp.charset or "utf-8", errors="replace")
            return FetchResult(
                url=url, status=status, content_type=ctype, html=html, error=None
            )
    except asyncio.TimeoutError:
        return FetchResult(
            url=url, status=None, content_type=None, html=None, error="timeout"
        )
    except aiohttp.ClientError as e:
        return FetchResult(
            url=url,
            status=None,
            content_type=None,
            html=None,
            error=f"client-error:{e.__class__.__name__}",
        )
    except Exception as e:
        return FetchResult(
            url=url,
            status=None,
            content_type=None,
            html=None,
            error=f"error:{e.__class__.__name__}",
        )


async def fetchone(db: aiosqlite.Connection, sql: str, params=()):
    cur = await db.execute(sql, params)
    row = await cur.fetchone()
    await cur.close()
    return row


class Crawler:
    def __init__(
        self,
        db_path: str,
        seeds: list[str],
        max_pages: int,
        concurrency: int,
        timeout_s: int,
        max_bytes: int,
        restrict_same_host: bool,
        delay_s: float,
        user_agent: str,
    ):
        self.db_path = db_path
        self.seeds = [u for u in (normalize_url(s) for s in seeds) if u]
        if not self.seeds:
            raise ValueError("No valid seed URLs")

        self.max_pages = max_pages
        if concurrency == -1:
            suggested_concurrency= int(1 // delay_s)+1
            self.concurrency=suggested_concurrency
            print(f"*** [INIT] Auto tuned concurrency to {suggested_concurrency} for delay {delay_s}")
        else:
            self.concurrency = concurrency
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.restrict_same_host = restrict_same_host
        self.delay_s = delay_s
        self.user_agent = user_agent

        self.root_host = urlparse(self.seeds[0]).netloc.lower()
        self.q: asyncio.Queue[str] = asyncio.Queue()
        self.seen: set[str] = set()
        self.fetched_count = 0
        self._rate_lock = asyncio.Lock()
        self._last_fetch_ts = 0.0
        # Add a dedicated queue for the writer
        # GG: writer is very fast on SQLite
        self.dbq: asyncio.Queue[PageJob | None] = asyncio.Queue(maxsize=self.concurrency * 4)
        self.max_reached_size = -1

    def allowed(self, url: str) -> bool:
        if not url:
            return False
        if self.restrict_same_host:
            if urlparse(url).netloc.lower() != self.root_host:
                return False
        # DEBUG print(f"Checking {url}")
        # GG Skip special pages
        # GG check for exception
        if "/search/?query" in url:
            return False
        else:
            return True

    async def db_writer(self) -> None:
        """
        This is the only worker writing to database
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON;")
            await db.execute("PRAGMA journal_mode = WAL;")
            await db.execute("PRAGMA busy_timeout = 1000;")  # wait for locks (ms)
            writer_counter = 0
            while True:
                job = await self.dbq.get()
                try:
                    if job is None:
                        await db.commit()
                        return

                    page_id = await self.upsert_page_and_version(
                        db=db,
                        url=job.url,
                        title=job.title,
                        html=job.html,
                        text=job.text,
                        status_code=job.status_code,
                        fetched_at=job.fetched_at,
                    )
                    await self.backfill_inbound_links(db, job.url, page_id)
                    await self.upsert_links(
                        db,
                        from_page_id=page_id,
                        out_links=job.out_links,
                        ts=job.fetched_at,
                    )

                    # Commit per page (safe). You can batch later.
                    await db.commit()
                    writer_counter = writer_counter + 1
                    current_queue_size = self.dbq.qsize()
                    self.max_reached_size = max(
                        self.max_reached_size, current_queue_size
                    )
                finally:
                    self.dbq.task_done()

    async def enqueue(self, url: str) -> None:
        if url in self.seen:
            return
        if not self.allowed(url):
            # print(f"Not Allowed: {url}")
            return
        self.seen.add(url)
        await self.q.put(url)

    async def _polite_wait(self) -> None:
        if self.delay_s <= 0:
            return
        async with self._rate_lock:
            now = asyncio.get_event_loop().time()
            wait = (self._last_fetch_ts + self.delay_s) - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_fetch_ts = asyncio.get_event_loop().time()

    async def upsert_page_and_version(
        self,
        db: aiosqlite.Connection,
        url: str,
        title: str | None,
        html: str,
        text: str,
        status_code: int | None,
        fetched_at: str,
    ) -> int:

        if status_code == 404:
            print(f"Dead link {url}")
            text = text + " dead link"
        h = content_hash(html)
        row = await fetchone(
            db, "SELECT id, content_hash FROM pages WHERE url = ?;", (url,)
        )
        if row is None:
            # Insert new page (latest) + version
            cur = await db.execute(
                """
                INSERT INTO pages(url, title, html, text, content_hash, status_code, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (url, title, html, text, h, status_code, fetched_at),
            )
            page_id = cur.lastrowid

            await db.execute(
                """
                INSERT INTO page_versions(page_id, content_hash, title, html, text, status_code, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (page_id, h, title, html, text, status_code, fetched_at),
            )
            return int(page_id)

        page_id = int(row["id"])
        old_hash = row["content_hash"]

        if old_hash != h:
            # New version: store in versions and update latest in pages
            await db.execute(
                """
                INSERT OR IGNORE INTO page_versions(page_id, content_hash, title, html, text, status_code, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (page_id, h, title, html, text, status_code, fetched_at),
            )
            await db.execute(
                """
                UPDATE pages
                SET title=?, html=?, text=?, content_hash=?, status_code=?, fetched_at=?
                WHERE id=?;
                """,
                (title, html, text, h, status_code, fetched_at, page_id),
            )
        else:
            # Dedup: content unchanged; just refresh metadata
            await db.execute(
                """
                UPDATE pages
                SET status_code=?, fetched_at=?
                WHERE id=?;
                """,
                (status_code, fetched_at, page_id),
            )

        return page_id

    async def upsert_links(
        self,
        db: aiosqlite.Connection,
        from_page_id: int,
        out_links: list[str],
        ts: str,
    ) -> None:
        for to_url in out_links:
            # If we already know the target page_id, link it
            to_row = await fetchone(
                db, "SELECT id FROM pages WHERE url = ?;", (to_url,)
            )
            to_page_id = int(to_row["id"]) if to_row else None

            await db.execute(
                """
                INSERT INTO links(from_page_id, to_url, to_page_id, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(from_page_id, to_url) DO UPDATE SET
                  to_page_id = COALESCE(links.to_page_id, excluded.to_page_id),
                  last_seen_at = excluded.last_seen_at
                ;
                """,
                (from_page_id, to_url, to_page_id, ts, ts),
            )

    async def backfill_inbound_links(
        self, db: aiosqlite.Connection, url: str, page_id: int
    ) -> None:
        # When we discover/insert a page, update any existing edges pointing to its URL
        await db.execute(
            """
            UPDATE links
            SET to_page_id = ?
            WHERE to_url = ? AND (to_page_id IS NULL);
            """,
            (page_id, url),
        )

    async def worker(self, session: aiohttp.ClientSession, wid: int) -> None:
        while True:
            if self.fetched_count >= self.max_pages:
                print("Reached max pages")
                return

            try:
                url = await asyncio.wait_for(self.q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # queue empty-ish
                return

            try:
                # polite delay across workers
                await self._polite_wait()

                fr = await fetch_html(
                    session, url, timeout_s=self.timeout_s, max_bytes=self.max_bytes
                )
                if fr.html is None:
                    # still record “fetched attempt”? (kept simple: skip)
                    # print(f"[{wid}] skip {url} ({fr.error})")
                    continue

                title, text, links = html_to_text_and_links(url, fr.html)
                ts = now_iso()

                # enqueue a DB job (this may backpressure if DB is slower)
                await self.dbq.put(
                    PageJob(
                        url=url,
                        status_code=fr.status,
                        html=fr.html,
                        title=title,
                        text=text,
                        out_links=links,
                        fetched_at=ts,
                    )
                )

                self.fetched_count += 1
                # Enqueue discovered links
                for u in links:
                    await self.enqueue(u)
                queue_size = self.q.qsize()
                # if queue_size % 10 ==0:
                #     print(f"[{wid}] fetched {url} links={len(links)} QueueSize: {queue_size}")
            finally:
                self.q.task_done()

    async def logger(self) -> None:
        start_ts = asyncio.get_event_loop().time()
        start_iso = now_iso()
        sample_time = (self.concurrency * self.delay_s) / 2
        if sample_time < 2.0:
            sample_time = 2.0
        expected_page_for_seconds = 1 / self.delay_s
        print(
            f"*** CRAWL START {start_iso} Sample time {sample_time}s max_pps={expected_page_for_seconds}"
        )
        # Wait a bit to let queue fill-in
        await asyncio.sleep(sample_time)
        while True:
            try:
                url_queue_size = self.q.qsize()
                writer_current_queue_size = self.dbq.qsize()
                writer_max_size = self.max_reached_size
                elapsed_s = asyncio.get_event_loop().time() - start_ts
                pages_per_s = self.fetched_count / elapsed_s if elapsed_s > 0 else 0.0
                ratio = 100 * pages_per_s / expected_page_for_seconds
                print(
                    f"*** STATUS queued={url_queue_size} fetched={self.fetched_count} pps={pages_per_s:.2f}  {ratio:.2f}% DB QUEUE: {writer_current_queue_size} / {self.dbq.maxsize} max_reached_size={writer_max_size}"
                )
                if self.max_reached_size >= self.dbq.maxsize:
                    print(f"WARNING: DB Writer queue near saturation")
                if ratio < 90:
                    print(
                        f"WARNING: WE are too slow even respecting the delay. Delay limit is {expected_page_for_seconds} page per seconds"
                    )
                await asyncio.sleep(sample_time)
            # except asyncio.TimeoutError:
            except Exception as e:
                print("LOG FAILED:", e)
                raise

    async def run(self) -> None:
        # await self.init_db()

        for s in self.seeds:
            await self.enqueue(s)

        headers = {"User-Agent": self.user_agent}
        connector = aiohttp.TCPConnector(
            limit=self.concurrency, ssl=False
        )  # keep simple
        async with aiohttp.ClientSession(
            headers=headers, connector=connector
        ) as session:
            writer_task = asyncio.create_task(self.db_writer())
            logger_task = asyncio.create_task(self.logger())
            tasks = [
                asyncio.create_task(self.worker(session, i))
                for i in range(self.concurrency)
            ]

            await asyncio.gather(*tasks)

            # Wait for all pending DB jobs to be written
            await self.dbq.join()

            # Stop writer
            await self.dbq.put(None)
            await writer_task
            # logger_task.cancel


async def main_async(args: argparse.Namespace) -> None:
    crawler = Crawler(
        db_path=args.db,
        seeds=args.seed,
        max_pages=args.max_pages,
        concurrency=args.concurrency,
        timeout_s=args.timeout,
        max_bytes=args.max_bytes,
        restrict_same_host=args.same_host,
        delay_s=args.delay,
        user_agent=args.user_agent,
    )
    await crawler.run()
    print(f"Done. Seen={len(crawler.seen)} fetched={crawler.fetched_count}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simple asyncio crawler with SQLite versioning + link graph"
    )
    p.add_argument("--db", default=DATABASE_FILE)
    p.add_argument(
        "--seed", action="append", required=True, help="Seed URL (repeatable)"
    )    
    p.add_argument("--max-pages", type=int, default=40)
    p.add_argument(
        "--delay",
        type=float,
        default=0.150,
        help="Politeness delay (seconds) shared across workers",
    )    
    p.add_argument("--concurrency", type=int, default=-1, help="Normally auto-detected by delay")
    p.add_argument("--timeout", type=int, default=15)
    p.add_argument(
        "--max-bytes", type=int, default=2_000_000, help="Max bytes per page"
    )
    p.add_argument(
        "--same-host", action="store_true", help="Restrict crawl to the seed host"
    )

    p.add_argument("--user-agent", default=DEFAULT_UA)
    return p.parse_args()

def crawl_init():
    args = parse_args()
    ensure_database_present(args.db)
    asyncio.run(main_async(args))
