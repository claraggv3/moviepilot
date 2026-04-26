from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from moviepilot.graph import ChatState, _pick_next


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_compiled_graph(route: str = "netflix"):
    """Build a minimal in-memory graph with stub nodes for structural/memory tests."""

    def stub_router(state):
        return {"route": route}

    def stub_netflix(state):
        return {"messages": [AIMessage(content="Here are some Netflix films.")], "retrieved_context": "ctx"}

    def stub_trending(state):
        return {"messages": [AIMessage(content="Here are trending films.")], "retrieved_context": "ctx"}

    def stub_refusal(state):
        return {"messages": [AIMessage(content="I can only help with movies.")], "retrieved_context": ""}

    g = StateGraph(ChatState)
    g.add_node("router", stub_router)
    g.add_node("netflix", stub_netflix)
    g.add_node("trending", stub_trending)
    g.add_node("refusal", stub_refusal)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        _pick_next,
        {"trending": "trending", "netflix": "netflix", "refusal": "refusal"},
    )
    g.add_edge("netflix", END)
    g.add_edge("trending", END)
    g.add_edge("refusal", END)

    return g.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# _pick_next — pure routing logic
# ---------------------------------------------------------------------------

def test_pick_next_trending():
    assert _pick_next({"route": "trending", "messages": [], "retrieved_context": ""}) == "trending"


def test_pick_next_netflix():
    assert _pick_next({"route": "netflix", "messages": [], "retrieved_context": ""}) == "netflix"


def test_pick_next_refusal():
    assert _pick_next({"route": "refusal", "messages": [], "retrieved_context": ""}) == "refusal"


def test_pick_next_missing_key_defaults_to_netflix():
    # route key absent from state — .get() fallback must kick in
    assert _pick_next({"messages": [], "retrieved_context": ""}) == "netflix"


# ---------------------------------------------------------------------------
# build_graph — compilation (no real API calls)
# ---------------------------------------------------------------------------

@patch("moviepilot.graph.load_collection")
@patch("moviepilot.graph.make_chat_model")
def test_build_graph_has_all_nodes(mock_make_model, mock_load_collection):
    mock_make_model.return_value = MagicMock()
    mock_load_collection.return_value = MagicMock()

    from moviepilot.graph import build_graph

    compiled = build_graph(MemorySaver())
    node_names = set(compiled.nodes)

    assert {"router", "trending", "netflix", "refusal"}.issubset(node_names)


# ---------------------------------------------------------------------------
# Memory across turns — add_messages accumulates correctly
# ---------------------------------------------------------------------------

def test_memory_accumulates_across_two_turns():
    graph = _make_compiled_graph(route="netflix")
    config = {"configurable": {"thread_id": "mem-test-thread"}}

    state1 = graph.invoke({"messages": [HumanMessage(content="spy thriller")]}, config)
    # After turn 1: 1 human + 1 AI = 2 messages
    assert len(state1["messages"]) == 2
    assert isinstance(state1["messages"][0], HumanMessage)
    assert isinstance(state1["messages"][1], AIMessage)

    state2 = graph.invoke({"messages": [HumanMessage(content="something shorter")]}, config)
    # After turn 2: previous 2 + 1 new human + 1 new AI = 4 messages
    assert len(state2["messages"]) == 4


def test_memory_isolated_across_threads():
    graph = _make_compiled_graph(route="netflix")

    config_a = {"configurable": {"thread_id": "thread-a"}}
    config_b = {"configurable": {"thread_id": "thread-b"}}

    graph.invoke({"messages": [HumanMessage(content="first message")]}, config_a)
    # Thread B starts fresh — must not see thread A's history
    state_b = graph.invoke({"messages": [HumanMessage(content="hello")]}, config_b)
    assert len(state_b["messages"]) == 2


def test_memory_trending_route():
    graph = _make_compiled_graph(route="trending")
    config = {"configurable": {"thread_id": "trending-thread"}}

    state = graph.invoke({"messages": [HumanMessage(content="what's popular?")]}, config)
    assert state["messages"][-1].content == "Here are trending films."
    assert len(state["messages"]) == 2
