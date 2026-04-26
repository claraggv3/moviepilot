from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import httpx
import structlog
from cachetools import TTLCache
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from moviepilot.config import settings

logger = structlog.get_logger(__name__)

TMDB_BASE_URL = "https://api.themoviedb.org/3"


@dataclass(frozen=True)
class Movie:
    id: int
    title: str
    original_title: str
    overview: str
    release_date: str
    vote_average: float
    vote_count: int
    genre_ids: list[int]
    popularity: float
    original_language: str
    media_type: str
    adult: bool
    video: bool
    softcore: bool


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError)


class TMDBClient:
    def __init__(self) -> None:
        self._cache: TTLCache = TTLCache(
            maxsize=32,
            ttl=settings.tmdb_cache_ttl_seconds,
        )
        self._http = httpx.AsyncClient(
            base_url=TMDB_BASE_URL,
            params={"api_key": settings.tmdb_api_key},
            timeout=10.0,
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
    )
    async def get_trending(
        self,
        time_window: Literal["day", "week"] = "week",
        page: int = 1,
    ) -> list[Movie]:
        cache_key = (time_window, page)

        if cache_key in self._cache:
            logger.debug("tmdb_cache_hit", time_window=time_window, page=page)
            return self._cache[cache_key]

        logger.info("tmdb_fetch", time_window=time_window, page=page)
        response = await self._http.get(f"/trending/movie/{time_window}", params={"page": page})
        response.raise_for_status()

        movies = [
            Movie(
                id=item["id"],
                title=item["title"],
                original_title=item.get("original_title", item["title"]),
                overview=item.get("overview", ""),
                release_date=item.get("release_date", ""),
                vote_average=item.get("vote_average", 0.0),
                vote_count=item.get("vote_count", 0),
                genre_ids=item.get("genre_ids", []),
                popularity=item.get("popularity", 0.0),
                original_language=item.get("original_language", "en"),
                media_type=item.get("media_type", "movie"),
                adult=item.get("adult", False),
                video=item.get("video", False),
                softcore=item.get("softcore", False),
            )
            for item in response.json().get("results", [])
        ]

        self._cache[cache_key] = movies
        logger.info("tmdb_fetch_done", count=len(movies), time_window=time_window)
        return movies


# Module-level singleton — nodes import this directly
tmdb_client = TMDBClient()
