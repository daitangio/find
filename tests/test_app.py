import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from find.app import extract_meta_refresh


class ExtractMetaRefreshTests(unittest.TestCase):
    def test_extract_meta_refresh_returns_url(self) -> None:
        html = (
            '<html><head>'
            '<meta http-equiv="refresh" content="0; URL=/c64ref/c64disasm/" />'
            '</head><body>Hi</body></html>'
        )
        self.assertEqual(extract_meta_refresh(html), "/c64ref/c64disasm/")

    def test_extract_meta_refresh_returns_none_without_tag(self) -> None:
        html = "<html><head></head><body>No redirect</body></html>"
        self.assertIsNone(extract_meta_refresh(html))


if __name__ == "__main__":
    unittest.main()
