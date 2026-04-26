from __future__ import annotations

import time
from typing import Any
from uuid import UUID

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from moviepilot.config import settings

logger = structlog.get_logger(__name__)


class _StructlogCallback(BaseCallbackHandler):
    """Emits a structured log line with token counts and latency on every LLM call."""

    def __init__(self) -> None:
        super().__init__()
        self._start: dict[UUID, float] = {}

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        elapsed_ms = round(
            (time.perf_counter() - self._start.pop(run_id, time.perf_counter())) * 1000
        )
        usage = (response.llm_output or {}).get("token_usage", {})
        logger.info(
            "llm_call_end",
            model=(response.llm_output or {}).get("model_name", settings.chat_model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=elapsed_ms,
        )


def make_chat_model(
    streaming: bool = True,
    temperature: float | None = None,
) -> ChatOpenAI:
    """The only place in the codebase that constructs a chat model.

    Pass temperature=0.0 explicitly for deterministic calls (e.g. the router).
    Omit it to use the project default from settings.

    Tests inject FakeListChatModel directly — they never call this function.
    """
    callbacks: list[BaseCallbackHandler] = [_StructlogCallback()]

    if settings.langfuse_enabled:
        from langfuse.callback import CallbackHandler  # optional dep, may not be installed
        callbacks.append(
            CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        )

    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        temperature=temperature if temperature is not None else settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.request_timeout,
        streaming=streaming,
        max_retries=3,
        callbacks=callbacks,
    )
