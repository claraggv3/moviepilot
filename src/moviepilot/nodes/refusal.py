from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage

_SYSTEM_PROMPT = """\
You are a movie and TV show recommendation assistant. \
The user asked: {query}

Politely let them know that you can only help with movie and TV show recommendations. \
If their query is adjacent (e.g. asking about a book that was adapted into a film), \
you may offer to help with the movie version instead. Keep it brief and friendly."""


def make_refusal_node(chat_model):
    """Return an async LangGraph node that politely declines off-topic queries."""

    async def _refusal(state: dict[str, Any]) -> dict[str, Any]:
        query = state["messages"][-1].content
        response = await chat_model.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT.format(query=query)),
            *state["messages"],
        ])
        return {"messages": [response]}

    return _refusal
