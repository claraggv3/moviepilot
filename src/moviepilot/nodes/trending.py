from __future__ import annotations

from typing import Any, Literal

import structlog
from langchain_core.messages import SystemMessage

from moviepilot.tmdb.client import Movie, tmdb_client

_DAY_SIGNALS = {"today", "tonight", "this morning", "this afternoon", "just came out"}


def _pick_time_window(query: str) -> Literal["day", "week"]:
    q = query.lower()
    return "day" if any(s in q for s in _DAY_SIGNALS) else "week"

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a movie recommendation assistant. The user is asking: {query}

The titles below are trending on TMDB this week.

<context>
{context}
</context>

Recommend from this list only, tailored to the user's request. \
If none of them fit what they're looking for, say so clearly — \
do not suggest titles that are not in the list above."""


def _format_context(movies: list[Movie]) -> str:
    lines = []
    for i, m in enumerate(movies, 1):
        score = f"★{m.vote_average:.1f}" if m.vote_average else ""
        parts = [f"{i}. **{m.title}**"]
        if m.release_date:
            parts.append(f"({m.release_date[:4]})")
        if score:
            parts.append(score)
        if m.overview:
            parts.append(f"— {m.overview}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def make_trending_node(chat_model):
    """Return an async LangGraph node that answers from TMDB trending data."""

    async def _trending(state: dict[str, Any]) -> dict[str, Any]:
        query = state["messages"][-1].content
        movies = await tmdb_client.get_trending(_pick_time_window(query))
        context = _format_context(movies)

        logger.info("trending_context_built", n_movies=len(movies))

        response = await chat_model.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT.format(query=query, context=context)),
            *state["messages"],
        ])

        return {
            "messages": [response],
            "retrieved_context": context,
        }

    return _trending
