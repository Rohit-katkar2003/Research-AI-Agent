"""
tools/scraper.py
=================
Tavily-based web scraper with fallback and rate limiting.
Returns clean, structured source objects.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from core.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class Source:
    """A single scraped source."""
    url: str
    title: str
    content: str
    score: float = 0.0
    published_date: Optional[str] = None
    source_type: str = "web"
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def truncated_content(self, max_chars: int = 2000) -> str:
        return self.content[:max_chars] + ("..." if len(self.content) > max_chars else "")


class TavilyScraper:
    """
    Async Tavily scraper.
    Handles batched queries, deduplication, error recovery.
    """
    def __init__(self):
        self._client = None
        self._last_call = 0.0
        self._min_interval = 0.5    # rate limit: max 2 calls/sec

    def _get_client(self):
        if self._client is None:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            except ImportError:
                raise RuntimeError(
                    "tavily-python not installed. Run: pip install tavily-python"
                )
        return self._client

    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()

    def search(self, query: str, max_results: int = None) -> list[Source]:
        """Synchronous single search."""
        client = self._get_client()
        n = max_results or settings.TAVILY_MAX_RESULTS
        self._rate_limit()

        try:
            response = client.search(
                query        = query,
                max_results  = n,
                search_depth = settings.TAVILY_SEARCH_DEPTH,
                include_raw_content = True,
            )
            return [self._parse_result(r) for r in response.get("results", [])]
        except Exception as e:
            log.error(f"Tavily search failed for '{query}': {e}")
            return []

    async def async_search(self, query: str, max_results: int = None) -> list[Source]:
        """Async wrapper for non-blocking use in FastAPI."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.search, query, max_results)

    async def search_batch(self, queries: list[str]) -> list[Source]:
        """Search multiple queries concurrently, deduplicate results."""
        tasks = [self.async_search(q, settings.MAX_SOURCES_PER_QUERY) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_sources: list[Source] = []
        seen_urls: set[str] = set()

        for batch in results:
            if isinstance(batch, Exception):
                log.warning(f"Batch search error: {batch}")
                continue
            for source in batch:
                if source.url not in seen_urls:
                    seen_urls.add(source.url)
                    all_sources.append(source)

        # Sort by relevance score
        all_sources.sort(key=lambda s: s.score, reverse=True)
        log.info(f"Scraped {len(all_sources)} unique sources from {len(queries)} queries")
        return all_sources

    def _parse_result(self, result: dict) -> Source:
        content = result.get("raw_content") or result.get("content") or ""
        return Source(
            url            = result.get("url", ""),
            title          = result.get("title", "Untitled"),
            content        = content.strip(),
            score          = result.get("score", 0.0),
            published_date = result.get("published_date"),
            source_type    = "web",
        )


# Module-level singleton
scraper = TavilyScraper()