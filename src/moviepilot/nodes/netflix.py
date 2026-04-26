from __future__ import annotations

import asyncio
from typing import Any

import structlog
from langchain_core.messages import SystemMessage

from moviepilot.retrieval.chroma import ScoredDocument, hyde_search

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a movie and TV show recommendation assistant. The user is asking: {query}

The titles below are from the Netflix catalog and best match their request.

<context>
{context}
</context>

Recommend from this list only, tailored to the user's request. \
Do not mention or invent titles that are not in the list above."""


def _format_context(docs: list[ScoredDocument]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        imdb = doc.metadata.get("imdb_score")
        score_str = f" (IMDB: {imdb})" if imdb else ""
        parts.append(f"[{i}]{score_str}\n{doc.text}")
    return "\n\n".join(parts)


def make_netflix_node(chat_model, collection):
    """Return an async LangGraph node that answers from the Netflix ChromaDB collection."""

    async def _netflix(state: dict[str, Any]) -> dict[str, Any]:
        query = state["messages"][-1].content

        docs = await asyncio.to_thread(hyde_search, collection, query, chat_model)
        context = _format_context(docs)

        logger.info("netflix_context_built", n_docs=len(docs))

        response = await chat_model.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT.format(query=query, context=context)),
            *state["messages"],
        ])

        return {
            "messages": [response],
            "retrieved_context": context,
        }

    return _netflix
