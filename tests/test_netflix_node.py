from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from moviepilot.nodes.netflix import _format_context, make_netflix_node
from moviepilot.retrieval.chroma import ScoredDocument

FAKE_DOCS = [
    ScoredDocument(
        text="The Dark Knight (MOVIE, 2008) — Genres: action, crime. Countries: US.\n"
             "Director: Christopher Nolan. Starring: Christian Bale, Heath Ledger.\n"
             "When the menace known as the Joker wreaks havoc on Gotham...",
        metadata={"id": "tm1", "title": "The Dark Knight", "type": "MOVIE",
                  "release_year": 2008, "imdb_score": 9.0},
        score=0.95,
    ),
    ScoredDocument(
        text="Inception (MOVIE, 2010) — Genres: action, sci-fi. Countries: US.\n"
             "Director: Christopher Nolan. Starring: Leonardo DiCaprio.\n"
             "A thief who steals corporate secrets through dream-sharing technology...",
        metadata={"id": "tm2", "title": "Inception", "type": "MOVIE",
                  "release_year": 2010, "imdb_score": 8.8},
        score=0.91,
    ),
]


def _state(*messages: str) -> dict:
    return {"messages": [HumanMessage(content=m) for m in messages]}


def _make_node(reply: str = "Here are my picks."):
    chat_model = MagicMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content=reply))
    chat_model.with_structured_output = MagicMock()
    collection = MagicMock()
    return make_netflix_node(chat_model, collection), chat_model


# --- _format_context ---

def test_format_context_contains_title():
    text = _format_context(FAKE_DOCS)
    assert "The Dark Knight" in text
    assert "Inception" in text


def test_format_context_includes_imdb_score():
    text = _format_context(FAKE_DOCS)
    assert "IMDB: 9.0" in text


def test_format_context_numbered():
    text = _format_context(FAKE_DOCS)
    assert "[1]" in text
    assert "[2]" in text


def test_format_context_no_score_when_missing():
    doc = ScoredDocument(text="Some film.", metadata={"id": "x"}, score=0.5)
    text = _format_context([doc])
    assert "IMDB" not in text


def test_format_context_empty():
    assert _format_context([]) == ""


# --- make_netflix_node ---

@pytest.mark.asyncio
async def test_netflix_node_returns_ai_message():
    node, _ = _make_node("Here are some Nolan films.")
    with patch("moviepilot.nodes.netflix.hyde_search", return_value=FAKE_DOCS):
        result = await node(_state("Christopher Nolan films"))

    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "Here are some Nolan films."


@pytest.mark.asyncio
async def test_netflix_node_returns_retrieved_context():
    node, _ = _make_node()
    with patch("moviepilot.nodes.netflix.hyde_search", return_value=FAKE_DOCS):
        result = await node(_state("action movies"))

    assert "retrieved_context" in result
    assert "The Dark Knight" in result["retrieved_context"]


@pytest.mark.asyncio
async def test_netflix_node_passes_query_and_context_to_llm():
    node, chat_model = _make_node()
    with patch("moviepilot.nodes.netflix.hyde_search", return_value=FAKE_DOCS):
        await node(_state("show me Nolan films"))

    system_content = chat_model.ainvoke.call_args[0][0][0].content
    assert "show me Nolan films" in system_content
    assert "<context>" in system_content
    assert "The Dark Knight" in system_content


@pytest.mark.asyncio
async def test_netflix_node_no_results_does_not_crash():
    node, _ = _make_node("I couldn't find anything matching that.")
    with patch("moviepilot.nodes.netflix.hyde_search", return_value=[]):
        result = await node(_state("something obscure"))

    assert isinstance(result["messages"][0], AIMessage)
