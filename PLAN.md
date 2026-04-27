# MoviePilot — Project Plan & Context

**Owner:** Clara  
**Submission type:** Senior MLE take-home (4-day budget)  
**Stack:** Python 3.11+, LangGraph, ChromaDB, OpenAI SDK, structlog, Langfuse (optional)  
**Last updated:** 2026-04-25

---

## What we're building — plain language

A chatbot that answers movie questions by looking up real data first, then talking about it. It never makes up movies. It knows two things: what's popular *right now* (TMDB), and what's on *Netflix* (Kaggle catalog, ~6k titles).

### The three agents

| Agent | Trigger | Data source | What it does |
|---|---|---|---|
| **Trending** | Recency/popularity signal ("what's hot", "out recently", "this week") | TMDB trending endpoint | Fetches this week's top films, LLM picks the most relevant for the query |
| **Netflix** | Genre, mood, topic — no recency signal | ~6k Netflix titles (Kaggle) | HyDE + hybrid BM25/semantic search, LLM answers from retrieved matches only |
| **Refusal** | Off-topic (weather, poems, etc.) | None | Short LLM call, polite decline |

### Turn-by-turn flow

```
User types message
    ↓
Router reads last 2–3 messages → decides: trending | netflix | refusal
    ↓
Agent fetches real data → LLM answers using only that data, streaming word-by-word
    ↓
Full conversation saved (SqliteSaver) → follow-ups work naturally
```

---

## NLP differentiators

These go beyond the baseline spec and showcase NLP expertise directly.

### 1. HyDE — Hypothetical Document Embedding
**Problem:** Query language ("spy thriller") ≠ document language ("A rogue CIA operative..."). Raw query embedding underperforms.  
**Fix:** Ask the LLM to write a short hypothetical Netflix-style description, embed *that*, then search. Bridges the query–document vocabulary gap.  
**Location:** `retrieval/chroma.py` → `hyde_expand()`

### 2. Hybrid retrieval — BM25 + semantic
**Problem:** Dense search is weak on specifics (actor names, partial titles, years).  
**Fix:** BM25 lexical first-pass (`rank_bm25`) + semantic re-ranking, weighted combination.  
**Location:** `retrieval/chroma.py` → `hybrid_search()`

### 3. NER-based groundedness check
**Problem:** Asking an LLM judge "are these titles real?" is expensive and non-deterministic.  
**Fix:** Extract title mentions from the response via spaCy/catalog pattern matching, check programmatically against `retrieved_context`.  
**Location:** `scripts/run_eval.py`

### 4. Retrieval evaluation: Recall@k + MRR
**Problem:** Baseline eval only measures final response quality, not whether the right movies were even retrieved.  
**Fix:** For Netflix cases in the golden set, label relevant titles. Measure Recall@k and MRR separately from generation quality.  
**Location:** `scripts/run_eval.py`, `eval/golden.yaml`

### 5. Router with conversational context
**Problem:** Router seeing only the latest message mis-routes follow-ups ("something more recent?").  
**Fix:** Pass the last 2–3 messages to the router.  
**Location:** `nodes/route.py`

---

## Architecture

```
src/moviepilot/
├── config.py                    # pydantic-settings + structlog init
├── llm/
│   └── client.py                # ChatOpenAI factory + logging callback
├── retrieval/
│   ├── netflix_loader.py        # CSV → cleaned records
│   └── chroma.py                # ChromaDB build/load + BM25 + HyDE search
├── tmdb/
│   └── client.py                # httpx, TTLCache, Movie dataclass
├── nodes/
│   ├── route.py                 # structured output classifier (last N messages)
│   ├── trending.py              # TMDB fetch + LLM
│   ├── netflix.py               # HyDE + hybrid retrieval + LLM
│   └── refusal.py               # short LLM call (uniform streaming contract)
├── graph.py                     # StateGraph + conditional edges + SqliteSaver
├── chat.py                      # astream_events wrapper + error masking
├── langfuse_setup.py            # optional, env-var-gated
└── cli.py                       # typer REPL
```

**Key choices vs baseline DESIGN.md:**
- ChromaDB replaces FAISS — metadata co-location, `EphemeralClient` for tests, no manual save/load
- `llm/streaming.py` eliminated — streaming logic lives in `chat.py`
- `observability/` sub-package flattened — structlog config in `config.py`
- Refusal node makes a short LLM call — keeps streaming contract uniform, no special case in `chat.py`
- Router sees last 2–3 messages — fixes multi-turn routing accuracy

---

## Phases & tickets

### Phase 0 — Project scaffold
**Goal:** Clean repo, deps locked, package importable.  
**Verify:** `uv sync` works; `python -c "from movie_chatbot.config import settings"` succeeds.

| ID | Task | Files |
|---|---|---|
| P0-1 | `pyproject.toml` — all deps pinned, entry point `chatbot = "movie_chatbot.cli:app"` | `pyproject.toml` |
| P0-2 | `.gitignore`, `.env.example` | `.gitignore`, `.env.example` |
| P0-3 | Package skeleton — all `__init__.py` files | `src/movie_chatbot/**/__init__.py` |
| P0-4 | `config.py` — `Settings` (pydantic-settings), `configure_logging()` (structlog JSON) | `src/movie_chatbot/config.py` |

---

### Phase 1 — External data clients
**Goal:** Reliable OpenAI + TMDB access with retries and caching.  
**Verify:** Unit tests pass with respx mocks; TTL cache skips HTTP on second call.

| ID | Task | Files |
|---|---|---|
| P1-1 | `llm/client.py` — `make_chat_model()` factory; honors `OPENAI_BASE_URL`; structlog callback for `{model, tokens, latency_ms}`; optional Langfuse handler | `src/movie_chatbot/llm/client.py` |
| P1-2 | `tmdb/client.py` — `Movie` dataclass; `TMDBClient`; httpx + tenacity retry on 429/5xx; `get_trending()` with TTLCache | `src/movie_chatbot/tmdb/client.py` |
| P1-3 | Tests: TMDB client with respx mock; assert cache hit; assert retry on 429 | `tests/test_tmdb_client.py` |

---

### Phase 2 — Netflix retrieval pipeline (NLP core)
**Goal:** Build, persist, and query the Netflix semantic + lexical index.  
**Verify:** `chatbot build-collection` completes; `hybrid_search("spy thriller")` returns ranked results; HyDE produces a non-empty expansion.

| ID | Task | Files |
|---|---|---|
| P2-1 | `netflix_loader.py` — CSV → cleaned records; `text` field: `{title} ({type}, {year}) — Genres: {genres}.\n{description}` | `src/movie_chatbot/retrieval/netflix_loader.py` |
| P2-2 | `chroma.py` — `build_collection()` with `PersistentClient` + `OpenAIEmbeddingFunction`; batched ingestion; idempotency check | `src/movie_chatbot/retrieval/chroma.py` |
| P2-3 | `chroma.py` — `load_collection()`; `semantic_search()` → `list[ScoredDocument]` | `src/movie_chatbot/retrieval/chroma.py` |
| P2-4 | `chroma.py` — `bm25_search()` via `rank_bm25`; `hybrid_search()` weighted combo (alpha configurable, default 0.5) | `src/movie_chatbot/retrieval/chroma.py` |
| P2-5 | `chroma.py` — `hyde_expand(query, chat_model)` — LLM generates hypothetical Netflix description; returns expanded query string | `src/movie_chatbot/retrieval/chroma.py` |
| P2-6 | `scripts/build_collection.py` — CLI entry; calls loader + builder; progress output | `scripts/build_collection.py` |
| P2-7 | Tests: `EphemeralClient` + 10-row fixture CSV; semantic search; hybrid scoring; HyDE expansion | `tests/test_retrieval.py`, `tests/fixtures/tiny_netflix.csv` |

---

### Phase 3 — LangGraph nodes
**Goal:** Four nodes individually testable with fake LLM + fake retrieval.  
**Verify:** Each node callable with a `ChatState`; correct keys returned; routing test covers follow-up turns.

| ID | Task | Files |
|---|---|---|
| P3-1 | `route.py` — reads last 2–3 messages; structured output `Route(route: Literal[...])`; fallback to `"netflix"` + warning | `src/movie_chatbot/nodes/route.py` |
| P3-2 | `trending.py` — `get_trending("week")` top 20; markdown context block; system prompt with `<context>` guard + relevance-filter instruction | `src/movie_chatbot/nodes/trending.py` |
| P3-3 | `netflix.py` — HyDE → hybrid search → context block; `asyncio.to_thread()` for ChromaDB call | `src/movie_chatbot/nodes/netflix.py` |
| P3-4 | `refusal.py` — short LLM call; uniform streaming contract | `src/movie_chatbot/nodes/refusal.py` |
| P3-5 | Node tests: 8 routing cases (incl. follow-up turns); trending + netflix node output structure | `tests/test_route_node.py`, `tests/test_trending_node.py`, `tests/test_netflix_node.py` |

---

### Phase 4 — Graph, streaming, CLI
**Goal:** Full working chatbot from terminal; multi-turn memory; streaming; error masking.  
**Verify:** `chatbot chat` runs; follow-ups work; Ctrl-C exits cleanly; thread_id printed.

| ID | Task | Files |
|---|---|---|
| P4-1 | `graph.py` — `ChatState` TypedDict; `build_graph(checkpointer)`; `SqliteSaver` at runtime; collection presence check at startup | `src/movie_chatbot/graph.py` |
| P4-2 | `chat.py` — `astream_events(version="v2")`; filter `on_chat_model_stream` by `langgraph_node` (exclude `"route"`); error-masking generator | `src/movie_chatbot/chat.py` |
| P4-3 | `cli.py` — `typer` app; `chat` command (UUID thread_id, `--thread-id` resume, "Thinking…" indicator); `build-collection` command; `eval` command | `src/movie_chatbot/cli.py` |
| P4-4 | Integration tests: streaming filter; checkpointer multi-turn; e2e with all mocks | `tests/test_graph_streaming.py`, `tests/test_checkpointer_memory.py`, `tests/test_chat_e2e.py` |

---

### Phase 5 — Evaluation harness (NLP-informed)
**Goal:** Measurable, reproducible scores across routing, retrieval, and generation.  
**Verify:** `chatbot eval` runs and prints score table; `eval/results.json` written.

| ID | Task | Files |
|---|---|---|
| P5-1 | `eval/golden.yaml` — 15 cases: 5 trending, 5 netflix, 3 genre-only (→ netflix), 2 off-topic; each with `relevant_titles` for retrieval eval | `eval/golden.yaml` |
| P5-2 | Retrieval eval — Recall@k and MRR for Netflix cases | `scripts/run_eval.py` |
| P5-3 | NER groundedness check — spaCy/pattern matching; deterministic title hallucination detection | `scripts/run_eval.py` |
| P5-4 | LLM-as-judge (`gpt-4o`) — `{relevance, helpfulness}`; 3 runs/case; mean + stdev | `scripts/run_eval.py`, `eval/judge_prompt.md` |
| P5-5 | Results output — `eval/results.json` + `eval/summary.md` (score table + case snippets) | `scripts/run_eval.py` |

---

### Phase 6 — Observability (stretch)
**Goal:** JSON structured logs on every run; optional Langfuse traces.  
**Verify:** `LOG_LEVEL=DEBUG chatbot chat` emits JSON; app runs without `LANGFUSE_*` vars.

| ID | Task | Files |
|---|---|---|
| P6-1 | structlog timing in all nodes — `{step, latency_ms, status, tokens}` | All node files |
| P6-2 | `langfuse_setup.py` — `make_langfuse_handler() -> CallbackHandler | None`; `None` when vars unset | `src/movie_chatbot/langfuse_setup.py` |

---

### Phase 7 — Report, README, polish
**Goal:** Reviewer runs the chatbot in under 5 minutes on a fresh machine.  
**Verify:** Fresh venv: `uv sync && chatbot build-collection && chatbot chat`.

| ID | Task | Files |
|---|---|---|
| P7-1 | `README.md` — copy-pasteable quickstart; how to run tests; how to run eval | `README.md` |
| P7-2 | `REPORT.md` — approach + diagram; design choices (LangGraph, ChromaDB, HyDE, hybrid retrieval, NER eval); results table; challenges; productization | `REPORT.md` |
| P7-3 | Lint + format — `ruff check --fix && ruff format`; fresh venv smoke test | — |

---

## Constraints (hard)

- TMDB: **trending endpoint only** — no search, no details
- Langfuse: **fully optional** — reviewer won't have credentials
- `SqliteSaver` at runtime, `MemorySaver` in tests — never write SQLite from test runs
- `OPENAI_BASE_URL` must be honored — assessment provides a gateway
- ChromaDB: `PersistentClient` at runtime, `EphemeralClient` in tests

## Open decisions

- HyDE weight: how much to weight expanded query vs raw query (default: 0.7 HyDE / 0.3 raw)
- BM25 alpha: hybrid weighting (default: 0.5/0.5, tunable via `config.py`)
- Router context: last 2 vs 3 messages (default: 3)
- FastAPI stretch: include only if ≥3 hours remain on Day 4
