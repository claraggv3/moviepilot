# Phase 3 — LangGraph nodes

Four nodes compose the graph: **router**, **trending**, **netflix**, **refusal**. Each is built with a factory function that captures its dependencies (model, collection) in a closure and returns a plain async callable — the signature that LangGraph expects.

---

## Factory pattern

Every node follows the same shape:

```python
def make_<name>_node(chat_model, ...):
    async def _<name>(state: dict) -> dict:
        ...
        return {"messages": [...], ...}
    return _<name>
```

This keeps instantiation (model wiring, collection loading) in `graph.py` and keeps the node functions themselves pure-ish and testable with mock models.

---

## Router node (`nodes/route.py`)

### Role
Reads the recent conversation and emits a routing decision: `"trending"`, `"netflix"`, or `"refusal"`.

### Structured output
Uses `chat_model.with_structured_output(Route)` — a Pydantic model with a single `Literal` field. LangChain enforces the JSON schema, so the router never returns a freeform string that has to be parsed.

```python
class Route(BaseModel):
    route: Literal["trending", "netflix", "refusal"]
```

### Context window
Passes the last `settings.router_context_messages` messages (default: 3) to the model — not just the latest one. This is critical for follow-up turns: "something shorter" on turn 2 has no standalone meaning; the router needs turns 1–2 to infer that the user is still asking about Netflix films.

```python
recent = state["messages"][-settings.router_context_messages:]
conversation = _format_conversation(recent)
```

### Fallback
If structured output fails (schema validation error, API timeout), the node logs a warning and defaults to `"netflix"` — the safest choice, since most queries are recommendation-seeking.

### Model
Uses a separate `route_model` with `temperature=0.0` and `streaming=False`. Temperature 0 makes routing deterministic; streaming is unnecessary because the router's output is not user-visible.

---

## Trending node (`nodes/trending.py`)

### Role
Fetches the top films from TMDB's trending endpoint, then asks the LLM to select and present those most relevant to the user's query.

### Time window heuristic
```python
_DAY_SIGNALS = {"today", "tonight", "this morning", "this afternoon", "just came out"}

def _pick_time_window(query: str) -> Literal["day", "week"]:
    q = query.lower()
    return "day" if any(s in q for s in _DAY_SIGNALS) else "week"
```

When the query contains day-level intent ("what came out today"), `get_trending("day")` is called instead of the default weekly window. The string set is deliberately small; false positives (routing weekly queries to daily) are lower-risk than missing the signal entirely.

### System prompt
The prompt injects both the user query and the retrieved context:

```
You are a movie recommendation assistant. The user is asking: {query}
The titles below are trending on TMDB this week.
<context>
{context}
</context>
Recommend from this list only... If none of them fit, say so clearly.
```

The `<context>` guard and the "recommend from this list only" instruction are the hallucination barrier. The LLM cannot cite a title it hasn't been shown.

### State keys returned
- `messages`: the LLM's response appended (LangGraph merges via `add_messages`)
- `retrieved_context`: formatted movie list — recorded for evaluation and Langfuse tracing

---

## Netflix node (`nodes/netflix.py`)

### Role
Retrieves the most relevant titles from the Netflix catalog via HyDE + hybrid search, then asks the LLM to answer from that context.

### Retrieval call
```python
docs = await asyncio.to_thread(hyde_search, collection, query, chat_model)
```

`hyde_search` is synchronous (ChromaDB's Python client is not async). `asyncio.to_thread` runs it in the executor thread pool so it doesn't block the event loop.

See [docs/phase2.md](phase2.md) for the full retrieval pipeline (HyDE expansion, constraint extraction, BM25 hybrid, dual-source merge).

### System prompt
Same structure as the trending prompt — query + `<context>` block + "recommend from this list only" instruction:

```
You are a movie and TV show recommendation assistant. The user is asking: {query}
The titles below are from the Netflix catalog and best match their request.
<context>
{context}
</context>
Recommend from this list only... Do not mention or invent titles not in the list above.
```

### Context formatting
Each retrieved document is shown with its IMDB score (when present) and the full text field (`{title} ({type}, {year}) — Genres: ... {description}`). The index `[1], [2], ...` makes it easy for the LLM to reference specific entries.

---

## Refusal node (`nodes/refusal.py`)

### Role
Politely declines queries that are off-topic. Unlike a static message, it makes a real (short) LLM call — which keeps the streaming contract uniform across all three agents and allows the model to reference the specific query.

### Why a live LLM call?
If refusal returned a hard-coded string, `chat.py` would need a special case: router-filtered events plus a static string branch. Instead, `refusal` is identical in shape to `trending` and `netflix` — it streams tokens from a chat model call, and `chat.py` treats all three identically.

### Adjacent query handling
The system prompt invites the LLM to offer a movie-version redirect if the query is adjacent:

```
If their query is adjacent (e.g. asking about a book that was adapted into a film),
you may offer to help with the movie version instead. Keep it brief and friendly.
```

This handles "can you recommend the book Gone Girl?" gracefully without fully refusing.

---

## Tests (`tests/test_route_node.py`, `test_trending_node.py`, `test_netflix_node.py`, `test_refusal_node.py`)

Each node is tested with a mock model that returns a controlled response. Real TMDB calls and ChromaDB queries are mocked via `respx` and `unittest.mock`.

| Test file | What it covers |
|---|---|
| `test_route_node.py` | 5 routing decisions (trending, netflix ×2, refusal ×2), 2 multi-turn follow-up cases, error fallback to netflix |
| `test_trending_node.py` | TMDB response formatted into context, LLM invoked with correct system prompt |
| `test_netflix_node.py` | hyde_search mocked, context formatted correctly, LLM invoked |
| `test_refusal_node.py` | LLM invoked, response returned as message |
