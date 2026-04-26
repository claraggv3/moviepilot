from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from moviepilot.nodes.route import Route, make_route_node


def _make_node(route: str):
    """Return a route node backed by a mock that returns the given route."""
    model = MagicMock()
    model.with_structured_output.return_value.invoke.return_value = Route(route=route)
    return make_route_node(model)


def _state(*messages: str) -> dict:
    """Alternating human/ai messages, always ending with a human message."""
    msgs = []
    for i, text in enumerate(messages):
        msgs.append(HumanMessage(content=text) if i % 2 == 0 else AIMessage(content=text))
    return {"messages": msgs}


# --- routing decisions ---

def test_routes_trending():
    node = _make_node("trending")
    assert node(_state("what's popular this week?"))["route"] == "trending"


def test_routes_netflix_genre():
    node = _make_node("netflix")
    assert node(_state("I want a Korean crime thriller"))["route"] == "netflix"


def test_routes_netflix_director():
    node = _make_node("netflix")
    assert node(_state("show me Christopher Nolan films"))["route"] == "netflix"


def test_routes_refusal():
    node = _make_node("refusal")
    assert node(_state("what's the weather in Tokyo?"))["route"] == "refusal"


def test_routes_refusal_off_topic():
    node = _make_node("refusal")
    assert node(_state("write me a poem about the sea"))["route"] == "refusal"


# --- multi-turn: follow-up messages ---

def test_multiturn_followup_routed_correctly():
    """A vague follow-up should route using prior context, not just the latest message."""
    node = _make_node("netflix")
    state = _state(
        "recommend me a French romantic comedy",
        "Here are some great options: ...",
        "something shorter",
    )
    assert node(state)["route"] == "netflix"


def test_multiturn_trending_followup():
    node = _make_node("trending")
    state = _state(
        "what's trending right now?",
        "Here are the top films this week: ...",
        "any of those available in Spanish?",
    )
    assert node(state)["route"] == "trending"


# --- fallback ---

def test_fallback_to_netflix_on_error():
    model = MagicMock()
    model.with_structured_output.return_value.invoke.side_effect = RuntimeError("API error")
    node = make_route_node(model)
    assert node(_state("something good to watch"))["route"] == "netflix"
