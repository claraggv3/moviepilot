from __future__ import annotations

import traceback
from typing import AsyncIterator

import structlog
from langchain_core.messages import HumanMessage

logger = structlog.get_logger(__name__)

_AGENT_NODES = {"trending", "netflix", "refusal"}
_ERROR_MESSAGE = "Sorry, something went wrong. Please try again."


async def chat(
    message: str,
    thread_id: str,
    graph,
) -> AsyncIterator[str]:
    """
    Drive the graph for one user turn and yield response tokens as they arrive.

    Only tokens from agent nodes (trending, netflix, refusal) are surfaced —
    the router node's structured output is suppressed.

    Errors are logged in full internally and masked with a friendly message
    so the CLI never shows a raw traceback to the user.
    """
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=message)]}

    try:
        async for event in graph.astream_events(inputs, config, version="v2"):
            if event["event"] != "on_chat_model_stream":
                continue
            if event["metadata"].get("langgraph_node") not in _AGENT_NODES:
                continue
            token = event["data"]["chunk"].content
            if token:
                yield token

    except Exception:
        logger.error("chat_error", traceback=traceback.format_exc(), thread_id=thread_id)
        yield _ERROR_MESSAGE
