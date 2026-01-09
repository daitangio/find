#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional
from urllib.parse import urljoin, urldefrag, urlparse, urlunparse

import aiohttp, aiosqlite
import click
from bs4 import BeautifulSoup

from .db_utils import DATABASE_FILE, ensure_database_present

DEFAULT_UA = "Find/0.1 (+https://github.com/daitangio/find)"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def host_for_url(url: str) -> str:
    return urlparse(url).netloc.lower()


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


def normalize_seeds(seeds: Iterable[str]) -> list[str]:
    return [u for u in (normalize_url(s) for s in seeds) if u]


def auto_tune_concurrency(delay_s: float) -> int:
    """
    Compute the correct concurrency level.
    Given the polite delay, we can compute the optimal amount of workers to satisfay the delay.
    Because we have already a centralized database worker doing part of the processing, we reduce this value from one.
    We ensure a minimum of 2 workers.
    Last but noy least, we put also an upper limit (200).
    """
    if delay_s <= 0:
        return 2
    return min(max(2, int(1 // delay_s) - 1), 200)


def is_allowed_url(url: str, root_host: str, restrict_same_host: bool) -> bool:
    if not url:
        return False
    if restrict_same_host and host_for_url(url) != root_host:
        return False
    return True


def dedupe_in_order(items: Iterable[str]) -> list[str]:
    seen = set()
    uniq_items = []
    for item in items:
        if item not in seen:
            seen.add(item)
            uniq_items.append(item)
    return uniq_items


POST_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%b %d, %Y",
    "%B %d, %Y",
    "%b %d, %Y %H:%M",
    "%B %d, %Y %H:%M",
    "%b %d, %Y %I:%M %p",
    "%B %d, %Y %I:%M %p",
)


def normalize_post_date(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    cleaned = re.sub(
        r"^(posted on|published on|published|posted)\s*:?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"(\d)(st|nd|rd|th)", r"\1", cleaned, flags=re.IGNORECASE)
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"

    tz_hint = None
    if cleaned.endswith(" UTC"):
        cleaned = cleaned[:-4].rstrip()
        tz_hint = timezone.utc
    elif cleaned.endswith(" GMT"):
        cleaned = cleaned[:-4].rstrip()
        tz_hint = timezone.utc

    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        parsed = None

    if parsed is None:
        for fmt in POST_DATE_FORMATS:
            try:
                parsed = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                continue

    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz_hint or timezone.utc)
    return parsed.isoformat()


def extract_post_date(soup: BeautifulSoup) -> str | None:
    meta = soup.find("div", class_="post_meta")
    if not meta:
        return None
    post_date = meta.find(class_="post_date")
    if not post_date:
        return None
    raw = post_date.get("datetime") or post_date.get_text(" ", strip=True)
    return normalize_post_date(raw)


def html_to_text_and_links(
    base_url: str, html: str
) -> tuple[str | None, str, list[str], str | None]:

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else None
    text = soup.get_text(" ", strip=True)
    post_date = extract_post_date(soup)

    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        norm = normalize_url(abs_url)
        if norm:
            links.append(norm)

    return title, text, dedupe_in_order(links), post_date


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
    post_date: str | None


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
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        db_path: str,
        seeds: tuple[str, ...],
        max_pages: int,
        concurrency: int,
        timeout_s: int,
        max_bytes: int,
        restrict_same_host: bool,
        delay_s: float,
    ):
        self.db_path = db_path
        self.seeds = normalize_seeds(seeds)
        if not self.seeds:
            raise ValueError("No valid seed URLs")

        self.max_pages = max_pages
        if concurrency == -1:
            self.concurrency = auto_tune_concurrency(delay_s)
            print(
                f"*** [INIT] Auto tuned concurrency to {self.concurrency} for delay {delay_s}"
            )
        else:
            self.concurrency = concurrency
        if not restrict_same_host:
            print(
                "*** [INIT] WARNING: Crawl is not restricted to seed host(s) it can go anywhere!"
            )
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.restrict_same_host = restrict_same_host
        self.delay_s = delay_s

        self.root_host = host_for_url(self.seeds[0])
        self.q: asyncio.Queue[str] = asyncio.Queue()
        self.seen: set[str] = set()
        self.fetched_count = 0
        self._rate_lock = asyncio.Lock()
        self._last_fetch_ts = 0.0
        # Add a dedicated queue for the writer
        # GG: writer is very fast on SQLite
        self.dbq: asyncio.Queue[PageJob | None] = asyncio.Queue(
            maxsize=self.concurrency * 4
        )
        self.max_reached_size = -1

    def allowed(self, url: str) -> bool:
        return is_allowed_url(url, self.root_host, self.restrict_same_host)

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
                    #
                    page_id = await self.upsert_page_and_version(db, job)
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
        self, db: aiosqlite.Connection, page_job: PageJob
    ) -> int:
        url = page_job.url
        text = page_job.text
        if page_job.status_code == 404:
            print(f"Dead link {url}")
            text = text + " dead link"
        h = content_hash(page_job.html)
        row = await fetchone(
            db, "SELECT id, content_hash FROM pages WHERE url = ?;", (url,)
        )
        if row is None:
            # Insert new page (latest) + version
            cur = await db.execute(
                """
                INSERT INTO pages(url, title, html, text, content_hash, status_code, fetched_at, post_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    url,
                    page_job.title,
                    page_job.html,
                    text,
                    h,
                    page_job.status_code,
                    page_job.fetched_at,
                    page_job.post_date,
                ),
            )
            page_id = cur.lastrowid

            await db.execute(
                """
                INSERT INTO page_versions(page_id, content_hash, title, html, text, status_code, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    page_id,
                    h,
                    page_job.title,
                    page_job.html,
                    text,
                    page_job.status_code,
                    page_job.fetched_at,
                ),
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
                (
                    page_id,
                    h,
                    page_job.title,
                    page_job.html,
                    text,
                    page_job.status_code,
                    page_job.fetched_at,
                ),
            )
            await db.execute(
                """
                UPDATE pages
                SET title=?, html=?, text=?, content_hash=?, status_code=?, fetched_at=?, post_date=COALESCE(?, post_date)
                WHERE id=?;
                """,
                (
                    page_job.title,
                    page_job.html,
                    text,
                    h,
                    page_job.status_code,
                    page_job.fetched_at,
                    page_job.post_date,
                    page_id,
                ),
            )
        else:
            # Dedup: content unchanged; just refresh metadata
            await db.execute(
                """
                UPDATE pages
                SET status_code=?, fetched_at=?, post_date=COALESCE(?, post_date)
                WHERE id=?;
                """,
                (
                    page_job.status_code,
                    page_job.fetched_at,
                    page_job.post_date,
                    page_id,
                ),
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
                print(f"[{wid}] Reached max pages")
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
                    print(f"[{wid}] skip {url} ({fr.error})")
                    continue

                title, text, links, post_date = html_to_text_and_links(url, fr.html)
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
                        post_date=post_date,
                    )
                )

                self.fetched_count += 1
                # Enqueue discovered links
                for u in links:
                    await self.enqueue(u)
            finally:
                self.q.task_done()

    async def logger(self) -> None:
        start_ts = asyncio.get_event_loop().time()
        start_iso = now_iso()
        # Ensure Sample time is no little than 2 seconds
        sample_time = max((self.concurrency * self.delay_s) / 2, 2.0)
        expected_page_for_seconds = 1 / self.delay_s
        print(
            f"*** CRAWL Sample time {sample_time}s max_pps={expected_page_for_seconds}"
        )
        print(f"*** CRAWL Pages: {self.max_pages} started at {start_iso}")
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
                    f"*** STATUS queued={url_queue_size} fetched={self.fetched_count} pps={pages_per_s:.2f} "
                    + f"{ratio:.2f}% DB QUEUE: {writer_current_queue_size} / {self.dbq.maxsize} max_reached_size={writer_max_size}"
                )
                if self.fetched_count >= self.max_pages:
                    print("*** CRAWL Near completition: stopping logger")
                    return
                if self.max_reached_size >= self.dbq.maxsize:
                    print(
                        f"WARNING: DB Writer queue near saturation: {self.max_reached_size} / {self.dbq.maxsize}"
                    )
                if ratio < 80:
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

        headers = {"User-Agent": DEFAULT_UA}
        connector = aiohttp.TCPConnector(
            limit=self.concurrency, ssl=False
        )  # keep simple
        async with aiohttp.ClientSession(
            headers=headers, connector=connector
        ) as session:
            writer_task = asyncio.create_task(self.db_writer())
            _logger_task = asyncio.create_task(self.logger())
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
            # _logger_task.cancel


async def main_async(crawler: Crawler) -> None:
    await crawler.run()
    print(f"Done. Seen={len(crawler.seen)} fetched={crawler.fetched_count}")


# pylint: disable=too-many-arguments
@click.command(help="Simple asyncio crawler with SQLite versioning + link graph")
@click.option("--db", default=DATABASE_FILE, help="Database file path")
@click.option("--seed", multiple=True, required=True, help="Seed URL (repeatable)")
@click.option("--max-pages", type=int, default=40, help="Maximum pages to crawl")
@click.option(
    "--delay",
    type=float,
    default=0.190,
    help="Politeness delay (seconds) shared across workers",
)
@click.option(
    "--concurrency", type=int, default=-1, help="Normally auto-detected by delay"
)
@click.option("--timeout", type=int, default=5, help="Request timeout in seconds")
@click.option("--max-bytes", type=int, default=2_000_000, help="Max bytes per page")
@click.option(
    "--same-host/--no-same-host",
    is_flag=True,
    default=True,
    help="Restrict crawl to the seed host",
)
def crawl_init(
    db: str,
    seed: tuple[str, ...],
    max_pages: int,
    delay: float,
    concurrency: int,
    timeout: int,
    max_bytes: int,
    same_host: bool,
) -> None:
    ensure_database_present(db)
    crawler = Crawler(
        db_path=db,
        seeds=seed,
        max_pages=max_pages,
        concurrency=concurrency,
        timeout_s=timeout,
        max_bytes=max_bytes,
        restrict_same_host=same_host,
        delay_s=delay,
    )
    asyncio.run(main_async(crawler))
