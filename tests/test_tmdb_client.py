import pytest
import respx
from httpx import Response

from moviepilot.tmdb.client import Movie, TMDBClient

FAKE_MOVIE = {
    "id": 101,
    "title": "Fake Movie",
    "original_title": "Fake Movie Original",
    "overview": "A movie used in tests.",
    "release_date": "2026-01-15",
    "vote_average": 7.8,
    "vote_count": 1200,
    "genre_ids": [28, 12],
    "popularity": 210.5,
    "original_language": "en",
    "media_type": "movie",
    "adult": False,
    "video": False,
    "softcore": False,
}

TMDB_URL = "https://api.themoviedb.org/3/trending/movie/week"


@pytest.fixture
def client():
    """Fresh client per test — no shared cache state."""
    return TMDBClient()


@respx.mock
@pytest.mark.asyncio
async def test_get_trending_returns_movies(client):
    respx.get(TMDB_URL).mock(return_value=Response(200, json={"results": [FAKE_MOVIE]}))

    movies = await client.get_trending()

    assert len(movies) == 1
    movie = movies[0]
    assert isinstance(movie, Movie)
    assert movie.id == 101
    assert movie.title == "Fake Movie"
    assert movie.original_title == "Fake Movie Original"
    assert movie.vote_average == 7.8
    assert movie.vote_count == 1200
    assert movie.original_language == "en"
    assert movie.adult is False


@respx.mock
@pytest.mark.asyncio
async def test_get_trending_cache_prevents_second_request(client):
    route = respx.get(TMDB_URL).mock(
        return_value=Response(200, json={"results": [FAKE_MOVIE]})
    )

    await client.get_trending()
    await client.get_trending()

    assert route.call_count == 1  # second call served from cache


@respx.mock
@pytest.mark.asyncio
async def test_get_trending_retries_on_429(client):
    respx.get(TMDB_URL).mock(
        side_effect=[
            Response(429),
            Response(200, json={"results": [FAKE_MOVIE]}),
        ]
    )

    movies = await client.get_trending()

    assert len(movies) == 1
    assert movies[0].title == "Fake Movie"


@respx.mock
@pytest.mark.asyncio
async def test_get_trending_missing_optional_fields(client):
    """Client should not crash when TMDB omits optional fields."""
    minimal = {"id": 202, "title": "Minimal Movie"}
    respx.get(TMDB_URL).mock(return_value=Response(200, json={"results": [minimal]}))

    movies = await client.get_trending()

    assert movies[0].title == "Minimal Movie"
    assert movies[0].overview == ""
    assert movies[0].vote_average == 0.0
    assert movies[0].adult is False
