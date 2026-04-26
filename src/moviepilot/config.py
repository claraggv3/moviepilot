from __future__ import annotations

import logging
from functools import lru_cache

import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required
    openai_api_key: str
    tmdb_api_key: str

    # OpenAI — honor gateway base URL from the assessment
    openai_base_url: str = "https://api.openai.com/v1"

    # Models
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.7        # recommendation nodes; router always overrides to 0
    max_tokens: int = 512           # cap response length — recommendations should be concise
    request_timeout: int = 30       # seconds before an OpenAI call is abandoned

    # Retrieval
    top_k: int = 8
    chroma_path: str = "data/chroma"
    hyde_weight: float = 0.7      # weight for HyDE-expanded query vs raw query
    bm25_alpha: float = 0.5       # weight for BM25 score in hybrid retrieval
    router_context_messages: int = 3  # how many recent messages the router sees

    # TMDB
    tmdb_cache_ttl_seconds: int = 3600

    # Observability
    log_level: str = "INFO"

    # Langfuse (optional — all three must be set to enable)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def configure_logging() -> None:
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
