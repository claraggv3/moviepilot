# Phase 1 — External Data Clients

## What was implemented

### `llm/client.py` — LLM factory

A single function `make_chat_model()` that constructs and returns a configured `ChatOpenAI` model. It is the only place in the codebase that builds a chat model — all nodes receive one by injection and never construct their own.

**`_StructlogCallback`** hooks into LangChain's callback system:
- `on_llm_start` — timestamps the call
- `on_llm_end` — computes elapsed time, reads token usage from the response, emits a JSON log line: `{model, prompt_tokens, completion_tokens, latency_ms}`

**Langfuse** is attached only when both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present in settings. The import itself is inside the conditional block, so the optional dependency is never required.

**Parameters:**
- `streaming: bool = True` — set to True for user-facing nodes so `astream_events` can yield tokens
- `temperature: float | None = None` — defaults to `settings.temperature` (0.7); pass `0.0` explicitly for the router to get deterministic classification

**Settings fields added to `config.py`:**
- `temperature` (default 0.7) — controls randomness in recommendations
- `max_tokens` (default 512) — caps response length; recommendations should be concise
- `request_timeout` (default 30s) — abandons hung API calls

### `tmdb/client.py` — TMDB HTTP client

An async HTTP client wrapping `GET /trending/movie/{time_window}`.

**`Movie` dataclass** — typed, frozen, captures all fields returned by the API:
`id, title, original_title, overview, release_date, vote_average, vote_count, genre_ids, popularity, original_language, media_type, adult, video, softcore`

`vote_count` and `original_language` were added after inspecting the raw API response — `vote_count` is important context for the LLM (a 7.5 rating means very different things with 12 votes vs 12,000).

**`TMDBClient`:**
- `httpx.AsyncClient` with TMDB base URL and `api_key` query param (v3 auth)
- `TTLCache(maxsize=32, ttl=settings.tmdb_cache_ttl_seconds)` — results cached for 1 hour, keyed by `(time_window, page)`
- Tenacity retry on transient failures (see error handling section below)

**Module-level singleton `tmdb_client`** — nodes import this directly; one shared cache per process.

---

## Decisions

### v3 API key (query param) over v4 Bearer token
TMDB free accounts get a v3 API key by default. The v4 access token requires an extra generation step. Using `api_key` as a query param is simpler and works for all free accounts without extra setup.

### Module-level singleton for the client
A single `tmdb_client = TMDBClient()` at the bottom of the module means:
- One cache shared across the entire session — if the user asks two questions about trending movies, the second call is instant
- Nodes import the instance directly, no dependency injection needed at this layer

Tests create fresh `TMDBClient()` instances to avoid shared cache state between test cases.

### `temperature` as a call-site parameter, not only in config
Different nodes have different needs. The router needs `temperature=0.0` for deterministic classification (same query must always route the same way). Recommendation nodes want `temperature=0.7` for varied, natural-sounding responses. Exposing temperature as a parameter on `make_chat_model()` with the config default as fallback handles both cleanly without adding a `router_temperature` field to config.

---

## Error handling

### Retry logic — only transient failures

```python
def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError)
```

| Error | Retried? | Reason |
|---|---|---|
| `429 Too Many Requests` | Yes | Temporary rate limit — same request will succeed shortly |
| `500 / 502 / 503 / 504` | Yes | Server-side failures, usually momentary |
| `httpx.TransportError` | Yes | Network-level failures (DNS, connection reset) — transient |
| `401 Unauthorized` | No | Wrong API key — retrying won't fix it |
| `403 Forbidden` | No | Access denied — permanent |
| `404 Not Found` | No | Wrong endpoint — a code bug, not a transient condition |
| `400 Bad Request` | No | Malformed request — retrying is pointless |

Retry schedule: up to 3 attempts total, exponential backoff between 1s and 8s.

### What happens when errors are not retried (or retries are exhausted)

Two layers handle everything that escapes the retry:

**Layer 1 — exception propagates.** The calling node raises, LangGraph surfaces a graph execution error.

**Layer 2 — error-masking generator in `chat.py` (implemented in Phase 4).** This wraps the entire graph invocation. Any unhandled exception is caught here: structlog receives the full traceback, the user sees a friendly message. Nothing is silently swallowed — the error is always logged, just never shown raw to the user.

---

## Tests (`tests/test_tmdb_client.py`)

All offline — `respx` intercepts `httpx` at the transport layer, no real network traffic.

| Test | What it verifies |
|---|---|
| `test_get_trending_returns_movies` | Raw API JSON is correctly parsed into `Movie` dataclasses |
| `test_get_trending_cache_prevents_second_request` | Two calls produce only one HTTP request |
| `test_get_trending_retries_on_429` | Client retries after a 429 and ultimately succeeds |
| `test_get_trending_missing_optional_fields` | Sparse API responses don't crash the client |

Fresh `TMDBClient()` per test — no shared cache state between cases.
