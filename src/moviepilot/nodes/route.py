from __future__ import annotations

from typing import Any, Literal

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from moviepilot.config import settings

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a routing assistant for a movie and TV show recommendation chatbot.
Classify the conversation into exactly one of three categories:

trending  — the user wants to know what is popular, new, or recently released
netflix   — the user wants a recommendation by genre, mood, topic, actor, director, era, or similar
refusal   — the query is not about movies or TV shows at all

When the latest message is a follow-up (e.g. "something shorter", "more like that"), \
use the earlier messages to infer intent."""


class Route(BaseModel):
    route: Literal["trending", "netflix", "refusal"]


def _format_conversation(messages: list[BaseMessage]) -> str:
    lines = []
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def make_route_node(chat_model):
    """Return a LangGraph node that classifies the conversation into a route."""
    structured = chat_model.with_structured_output(Route)

    def _route(state: dict[str, Any]) -> dict[str, Any]:
        recent = state["messages"][-settings.router_context_messages:]
        conversation = _format_conversation(recent)

        try:
            result: Route = structured.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=conversation),
            ])
            route = result.route
        except Exception:
            logger.warning("router_fallback", reason="structured_output_failed")
            route = "netflix"

        logger.info("route_decision", route=route)
        return {"route": route}

    return _route
