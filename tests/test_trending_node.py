from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from moviepilot.nodes.trending import _format_context, _pick_time_window, make_trending_node
from moviepilot.tmdb.client import Movie

FAKE_MOVIES = [
    Movie(
        id=1,
        title="Sinners",
        original_title="Sinners",
        overview="A plantation owner fights an unexpected supernatural evil.",
        release_date="2025-04-18",
        vote_average=7.4,
        vote_count=980,
        genre_ids=[27, 18],
        popularity=310.0,
        original_language="en",
        media_type="movie",
        adult=False,
        video=False,
        softcore=False,
    ),
    Movie(
        id=2,
        title="A Minecraft Movie",
        original_title="A Minecraft Movie",
        overview="Four misfits are transported to the Overworld.",
        release_date="2025-04-04",
        vote_average=6.1,
        vote_count=2400,
        genre_ids=[28, 35],
        popularity=290.0,
        original_language="en",
        media_type="movie",
        adult=False,
        video=False,
        softcore=False,
    ),
]


def _state(*human_messages: str) -> dict:
    return {"messages": [HumanMessage(content=m) for m in human_messages]}


def _make_node(reply: str = "Here are my recommendations."):
    chat_model = MagicMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content=reply))
    return make_trending_node(chat_model)


# --- _pick_time_window ---

def test_time_window_defaults_to_week():
    assert _pick_time_window("what's popular right now?") == "week"
    assert _pick_time_window("any good thrillers?") == "week"

def test_time_window_day_on_today():
    assert _pick_time_window("what's trending today?") == "day"

def test_time_window_day_on_tonight():
    assert _pick_time_window("anything good to watch tonight?") == "day"


# --- _format_context ---

def test_format_context_contains_title():
    text = _format_context(FAKE_MOVIES)
    assert "Sinners" in text
    assert "A Minecraft Movie" in text


def test_format_context_contains_score():
    text = _format_context(FAKE_MOVIES)
    assert "★7.4" in text


def test_format_context_contains_year():
    text = _format_context(FAKE_MOVIES)
    assert "2025" in text


def test_format_context_empty_list():
    assert _format_context([]) == ""


# --- make_trending_node ---

@pytest.mark.asyncio
async def test_trending_node_returns_ai_message():
    node = _make_node("Here are the top films this week.")
    with patch("moviepilot.nodes.trending.tmdb_client.get_trending", AsyncMock(return_value=FAKE_MOVIES)):
        result = await node(_state("what's popular right now?"))

    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "Here are the top films this week."


@pytest.mark.asyncio
async def test_trending_node_returns_retrieved_context():
    node = _make_node()
    with patch("moviepilot.nodes.trending.tmdb_client.get_trending", AsyncMock(return_value=FAKE_MOVIES)):
        result = await node(_state("anything new?"))

    assert "retrieved_context" in result
    assert "Sinners" in result["retrieved_context"]


@pytest.mark.asyncio
async def test_trending_node_passes_context_to_llm():
    chat_model = MagicMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="reply"))
    node = make_trending_node(chat_model)

    with patch("moviepilot.nodes.trending.tmdb_client.get_trending", AsyncMock(return_value=FAKE_MOVIES)):
        await node(_state("what's trending?"))

    call_args = chat_model.ainvoke.call_args[0][0]
    system_content = call_args[0].content
    assert "Sinners" in system_content
    assert "<context>" in system_content
    assert "what's trending?" in system_content


@pytest.mark.asyncio
async def test_trending_node_passes_conversation_to_llm():
    chat_model = MagicMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content="reply"))
    node = make_trending_node(chat_model)

    with patch("moviepilot.nodes.trending.tmdb_client.get_trending", AsyncMock(return_value=FAKE_MOVIES)):
        await node(_state("what's popular?"))

    call_args = chat_model.ainvoke.call_args[0][0]
    # SystemMessage + HumanMessage
    assert len(call_args) == 2
    assert call_args[1].content == "what's popular?"


@pytest.mark.asyncio
async def test_trending_node_empty_results():
    """Node must not crash when TMDB returns no movies."""
    node = _make_node("Nothing is trending right now.")
    with patch("moviepilot.nodes.trending.tmdb_client.get_trending", AsyncMock(return_value=[])):
        result = await node(_state("anything new?"))

    assert isinstance(result["messages"][0], AIMessage)
