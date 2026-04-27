from __future__ import annotations

from typing import Annotated, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from moviepilot.llm.client import make_chat_model
from moviepilot.nodes.netflix import make_netflix_node
from moviepilot.nodes.refusal import make_refusal_node
from moviepilot.nodes.route import make_route_node
from moviepilot.nodes.trending import make_trending_node
from moviepilot.retrieval.chroma import load_collection


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # append-only — LangGraph merges new messages in
    route: str
    retrieved_context: str


def _pick_next(state: ChatState) -> str:
    return state.get("route", "netflix")


def build_graph(checkpointer: BaseCheckpointSaver) -> Any:
    """
    Build and compile the MoviePilot state graph.

    Node layout:
        route → [trending | netflix | refusal] → END

    The route node runs first on every turn. Its output sets state["route"],
    which the conditional edge reads to pick the next node.
    """
    collection = load_collection()

    chat_model = make_chat_model(streaming=True)
    route_model = make_chat_model(streaming=False, temperature=0.0)

    route_node    = make_route_node(route_model)
    trending_node = make_trending_node(chat_model)
    netflix_node  = make_netflix_node(chat_model, collection)
    refusal_node  = make_refusal_node(chat_model)

    graph = StateGraph(ChatState)

    graph.add_node("router",   route_node)
    graph.add_node("trending", trending_node)
    graph.add_node("netflix",  netflix_node)
    graph.add_node("refusal",  refusal_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _pick_next,
        {"trending": "trending", "netflix": "netflix", "refusal": "refusal"},
    )

    graph.add_edge("trending", END)
    graph.add_edge("netflix",  END)
    graph.add_edge("refusal",  END)

    return graph.compile(checkpointer=checkpointer)
