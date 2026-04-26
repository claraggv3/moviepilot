# Phase 4 — Graph, streaming, CLI

This phase wires the four nodes into a compiled LangGraph, adds the token-streaming layer, and exposes everything through a Typer CLI.

---

## Graph (`graph.py`)

### ChatState

```python
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    retrieved_context: str
```

- `messages` uses `add_messages` as its reducer — LangGraph appends new messages instead of replacing the whole list. This is what gives the chatbot multi-turn memory: each invocation adds the new human message and the AI response to the accumulated history.
- `route` is written by the router node and read by the conditional edge. It is not annotated with a reducer, so each turn overwrites the previous value (correct — routing is per-turn, not cumulative).
- `retrieved_context` is overwritten each turn by whichever agent node runs. It is available for evaluation and optional Langfuse tracing.

### Node layout

```
router → [conditional edge] → trending → END
                            → netflix  → END
                            → refusal  → END
```

The router runs first on every turn. The conditional edge calls `_pick_next(state)` which reads `state["route"]` (set by the router) and returns the name of the next node.

### Why "router" not "route"?
LangGraph prohibits a node from sharing its name with a state key. Since the state has a `route` field, the node is named `"router"` to avoid the collision.

### Two model instances

```python
chat_model  = make_chat_model(streaming=True)   # trending, netflix, refusal
route_model = make_chat_model(streaming=False, temperature=0.0)  # router only
```

`temperature=0.0` makes routing deterministic. `streaming=False` on the route model avoids emitting token events that would need to be filtered in `chat.py` — the router's output is structured JSON, never user-visible text.

### Checkpointer
`build_graph(checkpointer)` accepts any `BaseCheckpointSaver`. At runtime, `AsyncSqliteSaver` persists state to `data/checkpoints.sqlite`. In tests, `MemorySaver` (in-process, no disk) is used instead.

---

## Routing logic (`_pick_next`)

```python
def _pick_next(state: ChatState) -> str:
    return state.get("route", "netflix")
```

A pure function — no side effects, no IO. The `.get("route", "netflix")` default handles the edge case where the router node failed to write the key (defensive; should not happen in practice).

Tests in `tests/test_graph.py` cover all four paths (trending, netflix, refusal, missing-key fallback) as pure-function unit tests — no graph compilation needed.

---

## Streaming layer (`chat.py`)

```python
async def chat(message, thread_id, graph) -> AsyncIterator[str]:
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=message)]}
    try:
        async for event in graph.astream_events(inputs, config, version="v2"):
            if event["event"] != "on_chat_model_stream":
                continue
            if event["metadata"].get("langgraph_node") not in _AGENT_NODES:
                continue
            token = event["data"]["chunk"].content
            if token:
                yield token
    except Exception:
        logger.error("chat_error", traceback=traceback.format_exc())
        yield _ERROR_MESSAGE
```

### `astream_events` vs `astream`
`astream` yields full state diffs at each node boundary — useful for debugging but coarse. `astream_events(version="v2")` emits fine-grained events for every callback in the call stack, including `on_chat_model_stream` which fires once per token from any `ChatModel` call.

### Filtering by node name
Every `on_chat_model_stream` event carries `metadata["langgraph_node"]` — the name of the node that triggered it. This is the filter:

```python
_AGENT_NODES = {"trending", "netflix", "refusal"}
```

The router also calls a chat model (for structured output), which would otherwise emit token events. Excluding `"router"` from `_AGENT_NODES` suppresses those entirely — the user never sees the routing decision text.

### Error masking
Any exception during graph execution is caught, logged in full (with traceback) via structlog, and replaced with a user-friendly message. The CLI never shows a raw Python traceback.

### Tests (`tests/test_chat.py`)
7 tests covering: tokens from all three agent nodes are yielded; router tokens suppressed; non-stream events (chain_start, chain_end) suppressed; empty tokens skipped; exception during streaming yields `_ERROR_MESSAGE`.

---

## Multi-turn memory

Memory works automatically via two mechanisms:

1. **`add_messages` reducer** — each invocation appends (not replaces) messages. After turn 1: `[HumanMessage("spy thriller"), AIMessage("Here are…")]`. After turn 2 on the same thread: `[…turn 1…, HumanMessage("something shorter"), AIMessage("…")]`.

2. **Checkpointer persistence** — between invocations, the full `ChatState` is serialized by the checkpointer. The next invocation loads it back before running the router. This means state survives across process restarts (with `AsyncSqliteSaver`) or is scoped to the process lifetime (with `MemorySaver` in tests).

The thread_id in `config["configurable"]["thread_id"]` is the key. Two invocations with the same thread_id share state; different thread_ids are fully isolated (tested in `test_memory_isolated_across_threads`).

---

## CLI (`cli.py`)

Two commands exposed via Typer:

### `chat`
```
cinepilot chat [--thread-id TEXT]
```
- Generates a UUID thread_id if `--thread-id` is omitted (new session)
- `--thread-id` resumes a previous conversation from the SQLite checkpoint
- Streams tokens word-by-word using `astream_events`
- Prints the thread_id on exit so the user can resume later

### `build-collection`
```
cinepilot build-collection [--titles PATH] [--credits PATH] [--chroma-path TEXT] [--force]
```
Convenience wrapper around the retrieval pipeline — loads CSVs, builds the ChromaDB collection, reports count and timing.

### aiosqlite compatibility patch
`langgraph-checkpoint-sqlite` 2.0.x calls `conn.is_alive()` on the aiosqlite connection. This method was removed in aiosqlite ≥0.20.0. The CLI patches it back at import time:

```python
if not hasattr(aiosqlite.Connection, "is_alive"):
    aiosqlite.Connection.is_alive = lambda self: True
```

This is applied before `AsyncSqliteSaver` is imported so the attribute is present when the class inspects the connection. The patch is harmless — `is_alive` is only used as a liveness probe; returning `True` unconditionally is correct behavior for an active connection.

---

## Tests (`tests/test_graph.py`)

| Test | What it covers |
|---|---|
| `test_pick_next_*` (×4) | Pure routing logic — all three valid routes plus the missing-key fallback |
| `test_build_graph_has_all_nodes` | `build_graph()` compiles with mocked deps; all four node names present |
| `test_memory_accumulates_across_two_turns` | Same thread_id → messages list grows from 2 to 4 across turns |
| `test_memory_isolated_across_threads` | Different thread_ids → thread B starts with 2 messages, not 4 |
| `test_memory_trending_route` | Trending stub node wired correctly; response content correct |
