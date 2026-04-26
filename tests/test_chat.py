from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk


def _make_event(node: str, token: str) -> dict:
    return {
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": node},
        "data": {"chunk": AIMessageChunk(content=token)},
    }


def _make_non_stream_event(event_type: str) -> dict:
    return {"event": event_type, "metadata": {}, "data": {}}


async def _stream(*events):
    for e in events:
        yield e


def _make_graph(events):
    graph = MagicMock()
    graph.astream_events = MagicMock(return_value=_stream(*events))
    return graph


# --- token filtering ---

@pytest.mark.asyncio
async def test_yields_netflix_tokens():
    from moviepilot.chat import chat
    graph = _make_graph([
        _make_event("netflix", "Here "),
        _make_event("netflix", "are "),
        _make_event("netflix", "some films."),
    ])
    tokens = [t async for t in chat("spy thriller", "thread-1", graph)]
    assert tokens == ["Here ", "are ", "some films."]


@pytest.mark.asyncio
async def test_yields_trending_tokens():
    from moviepilot.chat import chat
    graph = _make_graph([_make_event("trending", "Top films: ")])
    tokens = [t async for t in chat("what's popular?", "thread-1", graph)]
    assert tokens == ["Top films: "]


@pytest.mark.asyncio
async def test_yields_refusal_tokens():
    from moviepilot.chat import chat
    graph = _make_graph([_make_event("refusal", "I can only help with movies.")])
    tokens = [t async for t in chat("write a poem", "thread-1", graph)]
    assert tokens == ["I can only help with movies."]


@pytest.mark.asyncio
async def test_suppresses_router_tokens():
    from moviepilot.chat import chat
    graph = _make_graph([
        _make_event("router", "netflix"),   # router output — must be suppressed
        _make_event("netflix", "Great pick!"),
    ])
    tokens = [t async for t in chat("spy thriller", "thread-1", graph)]
    assert tokens == ["Great pick!"]


@pytest.mark.asyncio
async def test_suppresses_non_stream_events():
    from moviepilot.chat import chat
    graph = _make_graph([
        _make_non_stream_event("on_chain_start"),
        _make_event("netflix", "Here you go."),
        _make_non_stream_event("on_chain_end"),
    ])
    tokens = [t async for t in chat("drama", "thread-1", graph)]
    assert tokens == ["Here you go."]


@pytest.mark.asyncio
async def test_skips_empty_tokens():
    from moviepilot.chat import chat
    graph = _make_graph([
        _make_event("netflix", ""),      # empty — must be skipped
        _make_event("netflix", "Hello"),
    ])
    tokens = [t async for t in chat("something good", "thread-1", graph)]
    assert tokens == ["Hello"]


# --- error handling ---

@pytest.mark.asyncio
async def test_error_yields_friendly_message():
    from moviepilot.chat import chat, _ERROR_MESSAGE

    async def _failing_stream(*args, **kwargs):
        raise RuntimeError("something broke")
        yield  # make it a generator

    graph = MagicMock()
    graph.astream_events = _failing_stream

    tokens = [t async for t in chat("anything", "thread-1", graph)]
    assert tokens == [_ERROR_MESSAGE]
