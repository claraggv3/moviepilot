from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from moviepilot.nodes.refusal import make_refusal_node


def _make_node(reply: str = "I can only help with movie recommendations."):
    chat_model = MagicMock()
    chat_model.ainvoke = AsyncMock(return_value=AIMessage(content=reply))
    return make_refusal_node(chat_model), chat_model


def _state(message: str) -> dict:
    return {"messages": [HumanMessage(content=message)]}


@pytest.mark.asyncio
async def test_refusal_returns_ai_message():
    node, _ = _make_node()
    result = await node(_state("what's the weather in Paris?"))
    assert isinstance(result["messages"][0], AIMessage)


@pytest.mark.asyncio
async def test_refusal_passes_query_in_system_prompt():
    node, chat_model = _make_node()
    await node(_state("write me a poem"))

    system_content = chat_model.ainvoke.call_args[0][0][0].content
    assert "write me a poem" in system_content


@pytest.mark.asyncio
async def test_refusal_passes_conversation_to_llm():
    node, chat_model = _make_node()
    await node(_state("write me a poem"))

    call_args = chat_model.ainvoke.call_args[0][0]
    # SystemMessage + HumanMessage
    assert len(call_args) == 2
    assert call_args[1].content == "write me a poem"


@pytest.mark.asyncio
async def test_refusal_has_no_retrieved_context():
    node, _ = _make_node()
    result = await node(_state("tell me a joke"))
    assert "retrieved_context" not in result
