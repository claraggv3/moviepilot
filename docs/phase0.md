# Phase 0 — Project Scaffold

## What was done

Set up the full project skeleton: dependency management, environment configuration, package structure, and the central settings module. No business logic yet — this phase exists so every subsequent phase has a clean, importable foundation to build on.

---

## Files created

```
moviepilot/
├── pyproject.toml                        # project definition + all dependencies
├── .gitignore                            # excludes data, secrets, caches
├── .env.example                          # documents every config variable
├── PLAN.md                               # living project plan (phases + NLP rationale)
├── docs/
│   └── phase0.md                         # this file
├── src/
│   └── moviepilot/
│       ├── __init__.py
│       ├── config.py                     # Settings + logging — the only place env vars are read
│       ├── llm/__init__.py
│       ├── retrieval/__init__.py
│       ├── tmdb/__init__.py
│       └── nodes/__init__.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py                       # stub — fixtures added in Phase 3
│   └── fixtures/
│       └── tiny_netflix.csv              # 10-row Netflix fixture for retrieval tests
├── scripts/                              # one-shot runners (build_collection, run_eval)
└── eval/                                 # golden cases + judge prompt
```

---

## Package skeleton — decisions

### `src/` layout

The source code lives under `src/moviepilot/` rather than directly at the repo root. This is a standard Python packaging convention that prevents accidental imports of the local directory instead of the installed package. When you run `poetry install`, the package is installed in editable mode and Python resolves imports from `src/` correctly. Without this layout, `import moviepilot` can silently import a different copy than the one being edited.

### Subpackage structure

| Subpackage | Responsibility |
|---|---|
| `llm/` | LLM client factory — the only place a chat model is constructed |
| `retrieval/` | Netflix data loading, ChromaDB collection, BM25, HyDE |
| `tmdb/` | TMDB HTTP client, caching, data model |
| `nodes/` | LangGraph node functions (route, trending, netflix, refusal) |

Each subpackage is a single responsibility boundary. Nodes depend on `llm/`, `retrieval/`, and `tmdb/` — but those three never depend on each other or on `nodes/`. This means any component can be tested in isolation without pulling in the full graph.

### Test fixture pre-created (`tests/fixtures/tiny_netflix.csv`)

A 10-row CSV with realistic columns (`id, title, type, release_year, genres, description, imdb_score`) is created now so Phase 2 retrieval tests have a ready fixture. Using a tiny deterministic dataset means retrieval tests never call the OpenAI embedding API and run in milliseconds.

---

## Dependency decisions (`pyproject.toml`)

### Poetry over uv

Poetry was chosen because it is already familiar. Both tools use `pyproject.toml` and produce a lockfile for exact reproducibility. The lockfile (`poetry.lock`, generated on first `poetry install`) pins every transitive dependency to an exact version — this is what guarantees the submission runs identically on the reviewer's machine.

### Version bounds, not exact pins in `pyproject.toml`

Dependencies use `>=min,<max` bounds rather than `==exact`. The lockfile handles exact pinning. Bounds in `pyproject.toml` communicate intent ("we need LangGraph 0.2+ but not 0.4 which may break the API") without making the file unreadable.

### LangGraph / LangChain bounded tightly

```toml
langgraph = ">=0.2.0,<0.4"
langchain-openai = ">=0.1.0,<0.3"
langchain-core = ">=0.2.0,<0.4"
```

The LangGraph ecosystem has a history of breaking changes between minor versions. Bounding the upper range prevents `poetry update` from pulling in a version that changes the `astream_events` API or the checkpointer interface.

### Langfuse as an optional group

```toml
[tool.poetry.group.langfuse]
optional = true

[tool.poetry.group.langfuse.dependencies]
langfuse = ">=2.0.0"
```

The assessment reviewer will not have Langfuse credentials. Making it an optional group means `poetry install --with dev` (the standard install) does not pull in Langfuse at all. Only `poetry install --with langfuse` does. The app checks at runtime whether the env vars are set before activating any tracing.

### NLP dependencies

```toml
spacy = ">=3.7.0"
rank-bm25 = ">=0.2.2"
```

These power two NLP features: `spacy` for named entity recognition in the groundedness evaluation (deterministic title hallucination detection), and `rank-bm25` for the lexical component of hybrid retrieval. Both are lightweight with no GPU requirements.

---

## `config.py` — line by line

### `Settings` class

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
```

`BaseSettings` from `pydantic-settings` reads values from environment variables and a `.env` file automatically. `case_sensitive=False` means `OPENAI_API_KEY` and `openai_api_key` are equivalent — avoids subtle bugs when mixing shell exports and `.env` entries.

### Required fields (no defaults)

```python
openai_api_key: str
tmdb_api_key: str
```

No default value means the app fails immediately at startup with a clear validation error if either key is missing. This is intentional — a missing key surfaces as a config error, not a cryptic API error 10 messages into a conversation.

### Gateway support

```python
openai_base_url: str = "https://api.openai.com/v1"
```

The assessment provides an API gateway. Honoring `OPENAI_BASE_URL` via this field means the same codebase works against the assessment gateway or the real OpenAI endpoint with no code changes.

### NLP tuning knobs

```python
hyde_weight: float = 0.7
bm25_alpha: float = 0.5
router_context_messages: int = 3
```

These control the NLP retrieval behaviour and are exposed as config rather than hardcoded constants. `hyde_weight` controls how much the HyDE-expanded query contributes vs the raw query (0.7 = 70% HyDE). `bm25_alpha` controls the BM25 vs semantic score balance in hybrid retrieval. `router_context_messages` sets how many recent messages the router sees for follow-up turn handling. All three can be tuned via `.env` without touching code.

### Langfuse toggle

```python
langfuse_public_key: str | None = None
langfuse_secret_key: str | None = None
langfuse_host: str = "https://cloud.langfuse.com"

@property
def langfuse_enabled(self) -> bool:
    return bool(self.langfuse_public_key and self.langfuse_secret_key)
```

Both keys must be present for Langfuse to activate. The `langfuse_enabled` property is used in `langfuse_setup.py` later — a single check, no scattered `if key is not None` blocks.

### Singleton pattern

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

`lru_cache` ensures `Settings()` is constructed exactly once per process. This matters because constructing it reads and validates all env vars — doing it on every import would be wasteful and could cause issues if env vars change mid-process (they shouldn't, but this makes the contract explicit). Tests can override by calling `get_settings.cache_clear()` before patching env vars.

### `configure_logging()`

```python
def configure_logging() -> None:
    logging.basicConfig(format="%(message)s", level=...)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        ...
    )
```

Structlog is configured to emit **JSON lines** — one JSON object per log event. This is the format that can be piped to a log aggregator (Datadog, CloudWatch, etc.) without any parsing. Each line carries `level`, `logger`, `timestamp`, and whatever fields the calling code adds (e.g., `latency_ms`, `tokens`, `node`). The `merge_contextvars` processor allows attaching a `thread_id` to every log line for the duration of a request without passing it explicitly to every log call.

This function is called once at process startup (in `cli.py` later). It is not called at import time to avoid side effects during testing.
