import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from find import crawl

try:
    import bs4
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


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


@unittest.skipUnless(BS4_AVAILABLE, "bs4 not installed")
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
        title, text, links = crawl.html_to_text_and_links(
            "https://example.com/base", html
        )
        self.assertEqual(title, "Example Title")
        self.assertIn("First paragraph", text)
        self.assertEqual(
            links,
            ["https://example.com/a", "https://example.com/b"],
        )


class CrawlPolicyTests(unittest.TestCase):
    def test_is_allowed_url_respects_host_restriction(self) -> None:
        root_host = "example.com"
        self.assertTrue(
            crawl.is_allowed_url("https://example.com/page", root_host, True)
        )
        self.assertFalse(
            crawl.is_allowed_url("https://other.com/page", root_host, True)
        )

    def test_is_allowed_url_skips_search_pages(self) -> None:
        root_host = "example.com"
        self.assertFalse(
            crawl.is_allowed_url(
                "https://example.com/search/?query=dead+link", root_host, False
            )
        )


class ConcurrencyTests(unittest.TestCase):
    def test_auto_concurrency_uses_delay(self) -> None:
        self.assertEqual(crawl.auto_concurrency(0.5), 3)

    def test_auto_concurrency_handles_zero_delay(self) -> None:
        self.assertEqual(crawl.auto_concurrency(0), 1)


if __name__ == "__main__":
    unittest.main()
