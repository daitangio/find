import os
import sys
import unittest

from find.app import extract_meta_refresh, parse_search_query

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


class ExtractMetaRefreshTests(unittest.TestCase):
    def test_extract_meta_refresh_returns_url(self) -> None:
        html = (
            "<html><head>"
            '<meta http-equiv="refresh" content="0; URL=/c64ref/c64disasm/" />'
            "</head><body>Hi</body></html>"
        )
        self.assertEqual(extract_meta_refresh(html), "/c64ref/c64disasm/")

    def test_extract_meta_refresh_returns_none_without_tag(self) -> None:
        html = "<html><head></head><body>No redirect</body></html>"
        self.assertIsNone(extract_meta_refresh(html))


class ParseSearchQueryTests(unittest.TestCase):
    def test_parse_search_query_transforms_site_to_url(self) -> None:
        q = "site:example.com"
        self.assertEqual(parse_search_query(q), 'url:"example.com"')

    def test_parse_search_query_transforms_quoted_site(self) -> None:
        q = 'site:"example.com"'
        self.assertEqual(parse_search_query(q), 'url:"example.com"')

    def test_parse_search_query_preserves_other_terms(self) -> None:
        q = "python site:example.com testing"
        self.assertEqual(parse_search_query(q), 'python url:"example.com" testing')

    def test_parse_search_query_case_insensitive(self) -> None:
        q = "SITE:example.com"
        self.assertEqual(parse_search_query(q), 'url:"example.com"')

    def test_parse_search_query_no_site_term(self) -> None:
        q = "python testing"
        self.assertEqual(parse_search_query(q), "python testing")


if __name__ == "__main__":
    unittest.main()
