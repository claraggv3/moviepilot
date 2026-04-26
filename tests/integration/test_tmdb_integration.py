"""
Integration tests — hit the real TMDB API.

Run with:
    poetry run pytest -m integration -v

Requires TMDB_API_KEY to be set in .env or environment.
Not included in the default test run (pytest -m "not integration").
"""

import pytest

from moviepilot.tmdb.client import Movie, TMDBClient


@pytest.fixture
async def client():
    return TMDBClient()


@pytest.mark.integration
async def test_real_trending_returns_movies(client):
    movies = await client.get_trending(time_window="week")

    assert len(movies) > 0, "Expected at least one trending movie"
    assert all(isinstance(m, Movie) for m in movies)


@pytest.mark.integration
async def test_real_trending_fields_are_populated(client):
    movies = await client.get_trending(time_window="week")
    first = movies[0]

    assert first.id > 0
    assert len(first.title) > 0
    assert len(first.overview) > 0
    assert 0.0 <= first.vote_average <= 10.0
    assert first.vote_count >= 0
    assert first.release_date != ""
    assert first.original_language != ""


@pytest.mark.integration
async def test_real_trending_day_and_week_differ(client):
    """Day and week windows should return different popularity snapshots."""
    day = await client.get_trending(time_window="day")
    week = await client.get_trending(time_window="week")

    day_ids = {m.id for m in day}
    week_ids = {m.id for m in week}

    # They may overlap but should not be identical
    assert day_ids != week_ids, "Day and week trending should differ"
