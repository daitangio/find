import os
import re
import sys
import unittest

from find import crawl

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


class NormalizeUrlTests(unittest.TestCase):
    def test_normalize_url_strips_fragment_and_ports(self) -> None:
        url = "HTTP://Example.com:80/a//b?x=1#frag"
        self.assertEqual(
            crawl.normalize_url(url),
            "http://example.com/a/b?x=1",
        )

    def test_normalize_url_rejects_non_http(self) -> None:
        self.assertIsNone(crawl.normalize_url("ftp://example.com/file"))
        self.assertIsNone(crawl.normalize_url("example.com/no-scheme"))


class HtmlExtractionTests(unittest.TestCase):
    def test_html_to_text_and_links_dedupes_and_resolves(self) -> None:
        html = """
        <html>
          <head>
            <title>Example Title</title>
            <script>console.log('skip');</script>
          </head>
          <body>
            <p>First paragraph</p>
            <a href="/a">First Link</a>
            <a href="https://example.com/b">Second Link</a>
            <a href="/a">Duplicate Link</a>
            <a href="mailto:test@example.com">Email</a>
          </body>
        </html>
        """
        title, text, links, post_date = crawl.html_to_text_and_links(
            "https://example.com/base", html
        )
        self.assertEqual(title, "Example Title")
        self.assertIn("First paragraph", text)
        self.assertEqual(
            links,
            ["https://example.com/a", "https://example.com/b"],
        )
        self.assertIsNone(post_date)

    def test_html_to_text_and_links_extracts_post_date(self) -> None:
        html = """
        <html>
          <body>
            <div class="post_meta">
              <span class="post_date">2023-08-30</span>
            </div>
          </body>
        </html>
        """
        _title, _text, _links, post_date = crawl.html_to_text_and_links(
            "https://example.com/base", html
        )
        self.assertEqual(post_date, "2023-08-30T00:00:00+00:00")

    def test_html_to_text_and_links_uses_document_date_fallback(self) -> None:
        html = """
        <html>
          <body>
            <p>No visible publication date</p>
          </body>
        </html>
        """
        _title, _text, _links, post_date = crawl.html_to_text_and_links(
            "https://example.com/base",
            html,
            document_date="Wed, 30 Aug 2023 14:21:00 GMT",
        )
        self.assertEqual(post_date, "2023-08-30T14:21:00+00:00")

    def test_remove_nav_content_clears_nav_elements(self) -> None:
        html = """
        <html>
          <body>
            <nav><a href="/menu">Menu</a><span>Navigation</span></nav>
            <main><p>Article body</p></main>
          </body>
        </html>
        """
        cleaned = crawl.remove_nav_content(html)

        self.assertIn("<nav></nav>", cleaned)
        self.assertNotIn("Menu", cleaned)
        self.assertNotIn("Navigation", cleaned)
        self.assertIn("Article body", cleaned)

    def test_html_to_text_and_links_ignores_nav_content_and_links(self) -> None:
        html = """
        <html>
          <head><title>Example Title</title></head>
          <body>
            <nav>
              <a href="/nav">Nav Link</a>
              <span>Navigation copy</span>
            </nav>
            <main>
              <p>Article body</p>
              <a href="/article">Article Link</a>
            </main>
          </body>
        </html>
        """
        title, text, links, _post_date = crawl.html_to_text_and_links(
            "https://example.com/base", html
        )

        self.assertEqual(title, "Example Title")
        self.assertNotIn("Nav Link", text)
        self.assertNotIn("Navigation copy", text)
        self.assertIn("Article body", text)
        self.assertEqual(links, ["https://example.com/article"])


class FakeContent:
    def __init__(self, body: bytes):
        self.body = body

    async def readexactly(self, size: int) -> bytes:
        return self.body[:size]

    async def read(self, size: int) -> bytes:
        return self.body[:size]


class FakeResponse:
    def __init__(self, body: bytes, content_length: int):
        self.headers = {"content-type": "text/html"}
        self.status = 200
        self.content_length = content_length
        self.charset = "utf-8"
        self.content = FakeContent(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _tb):
        return False


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response

    def get(self, _url: str, timeout):
        return self.response


class FetchHtmlTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_html_rejects_declared_oversized_response(self) -> None:
        response = FakeResponse(b"<html>oversized</html>", content_length=21)
        result = await crawl.fetch_html(
            FakeSession(response),
            "https://example.com/oversized",
            timeout_s=5,
            max_bytes=10,
            wid=0,
        )

        self.assertEqual(result.error, "too-large")
        self.assertIsNone(result.html)


class CrawlPolicyTests(unittest.TestCase):
    def test_is_allowed_url_respects_host_restriction(self) -> None:
        root_host = "example.com"
        self.assertTrue(
            crawl.is_allowed_url("https://example.com/page", [root_host], True)
        )
        self.assertFalse(
            crawl.is_allowed_url("https://other.com/page", [root_host], True)
        )

    def test_is_allowed_url_respects_include_pattern(self) -> None:
        root_host = "example.com"
        include_pattern = re.compile(r"/blog/")

        self.assertTrue(
            crawl.is_allowed_url(
                "https://example.com/blog/post", [root_host], True, include_pattern
            )
        )
        self.assertFalse(
            crawl.is_allowed_url(
                "https://example.com/about", [root_host], True, include_pattern
            )
        )

    def test_crawler_rejects_invalid_include_pattern(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid include pattern"):
            crawl.Crawler(
                db_path=":memory:",
                seeds=("https://example.com/",),
                max_pages=1,
                concurrency=2,
                timeout_s=1,
                max_bytes=1000,
                restrict_same_host=True,
                delay_s=0,
                include_pattern="[",
            )


class ConcurrencyTests(unittest.TestCase):
    def test_auto_concurrency_uses_delay(self) -> None:
        """Must be rounded down."""
        self.assertEqual(crawl.auto_tune_concurrency(0.5), 2)

    def test_auto_concurrency_handles_zero_delay(self) -> None:
        self.assertEqual(crawl.auto_tune_concurrency(0), 2)

    def test_auto_concurrency_has_a_lower_limit(self) -> None:
        self.assertEqual(crawl.auto_tune_concurrency(5), 2)

    def test_auto_concurrency_has_a_upper_limit(self) -> None:
        self.assertEqual(crawl.auto_tune_concurrency(0.00001), 200)


if __name__ == "__main__":
    unittest.main()
