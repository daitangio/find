import os
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


class CrawlPolicyTests(unittest.TestCase):
    def test_is_allowed_url_respects_host_restriction(self) -> None:
        root_host = "example.com"
        self.assertTrue(
            crawl.is_allowed_url("https://example.com/page", root_host, True)
        )
        self.assertFalse(
            crawl.is_allowed_url("https://other.com/page", root_host, True)
        )


class ConcurrencyTests(unittest.TestCase):
    def test_auto_concurrency_uses_delay(self) -> None:
        """ Must be rounded down."""
        self.assertEqual(crawl.auto_tune_concurrency(0.5), 2)

    def test_auto_concurrency_handles_zero_delay(self) -> None:
        self.assertEqual(crawl.auto_tune_concurrency(0), 2)

    def test_auto_concurrency_has_a_lower_limit(self) -> None:
        self.assertEqual(crawl.auto_tune_concurrency(5), 2)
        
    def test_auto_concurrency_has_a_upper_limit(self) -> None:
        self.assertEqual(crawl.auto_tune_concurrency(0.00001), 200)



if __name__ == "__main__":
    unittest.main()
