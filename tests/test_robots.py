import unittest
from unittest.mock import MagicMock, AsyncMock
import asyncio
import aiohttp
from find import utils


class RobotsTxtTests(unittest.TestCase):
    def setUp(self):
        self.robots_cache = {}
        self.timeout = 1

    def test_get_robots_parser_fetches_and_parses(self):
        async def run_test():
            mock_session = MagicMock(spec=aiohttp.ClientSession)
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.return_value = "User-agent: *\nDisallow: /private"

            # Setup the context manager for session.get
            mock_session.get.return_value.__aenter__.return_value = mock_response

            rp = await utils.get_robots_parser(
                mock_session,
                "https://example.com/some/path",
                self.robots_cache,
                self.timeout,
            )

            # Verify it called the correct URL
            mock_session.get.assert_called_with(
                "https://example.com/robots.txt", timeout=1
            )

            # Verify parser logic
            self.assertFalse(rp.can_fetch("AnyBot", "https://example.com/private"))
            self.assertTrue(rp.can_fetch("AnyBot", "https://example.com/public"))

            # Verify caching
            self.assertIn("https://example.com", self.robots_cache)
            self.assertIs(self.robots_cache["https://example.com"], rp)

            # Call again, should not fetch
            mock_session.get.reset_mock()
            rp2 = await utils.get_robots_parser(
                mock_session,
                "https://example.com/other",
                self.robots_cache,
                self.timeout,
            )
            self.assertIs(rp, rp2)
            mock_session.get.assert_not_called()

        asyncio.run(run_test())

    def test_get_robots_parser_404_allows_all(self):
        async def run_test():
            mock_session = MagicMock(spec=aiohttp.ClientSession)
            mock_response = AsyncMock()
            mock_response.status = 404

            mock_session.get.return_value.__aenter__.return_value = mock_response

            rp = await utils.get_robots_parser(
                mock_session,
                "https://example.com/some/path",
                self.robots_cache,
                self.timeout,
            )

            self.assertTrue(rp.can_fetch("AnyBot", "https://example.com/private"))

        asyncio.run(run_test())

    def test_get_robots_parser_403_disallows_all(self):
        async def run_test():
            mock_session = MagicMock(spec=aiohttp.ClientSession)
            mock_response = AsyncMock()
            mock_response.status = 403

            mock_session.get.return_value.__aenter__.return_value = mock_response

            rp = await utils.get_robots_parser(
                mock_session,
                "https://example.com/some/path",
                self.robots_cache,
                self.timeout,
            )

            self.assertFalse(rp.can_fetch("AnyBot", "https://example.com/public"))

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
