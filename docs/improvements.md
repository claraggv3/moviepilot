# Potential improvements

Design ideas deferred during implementation. Revisit once the core graph is working and there's evidence from real usage that they're needed.

---

## Trending node — agentic retry

**Current behaviour:** The node always fetches page 1 of weekly trending (20 films), regardless of whether those results match the user query. The LLM is instructed to say "nothing fits" if none are relevant.

**Improvement:** Give the LLM the ability to request more results — a second page or a switch to daily trending — when the first batch doesn't fit the query. This would require converting the trending node into a tool-calling agent (LangGraph `ToolNode` + conditional loop). `TMDBClient.get_trending` already supports `page` and `time_window` params, so the client side is ready.

**When to do it:** Only if real usage shows that page 1 weekly results are frequently insufficient for specific queries (e.g. a user asks for a trending Korean film and none happen to be in the top 20 that week).

---

## Trending node — time window

**Current behaviour:** `time_window` is hardcoded to `"week"`. Weekly trending is more stable and less noisy than daily.

**Improvement:** Expose `time_window` as a configurable setting, or let the router/LLM choose between `"day"` and `"week"` based on query intent ("what's hot today" vs "what's popular this week").

---

## Agent nodes — cap conversation history passed to LLM

**Current behaviour:** Agent nodes pass `*state["messages"]` — the full conversation history — to the LLM on every turn.

**Problem:** After 10+ turns, the accumulated messages cost tokens and may dilute the system prompt's constraint ("recommend only from this list"). A long prior conversation about Nolan films could subtly bias responses to a completely unrelated new query.

**Improvement:** Cap the messages passed to agent nodes at the last N turns (e.g. last 6 messages = 3 turns). The router already does this via `settings.router_context_messages`. Add a similar `agent_context_messages` setting and slice `state["messages"][-settings.agent_context_messages:]` before the `ainvoke` call in each node.

**When to do it:** Only if real usage shows degradation on long sessions or token costs become a concern.

---

## Retrieval — always dual-source (remove person gate)

**Current behaviour:** The dual-source merge in `hyde_search` (hyde_document embedding + raw query embedding) is gated on `if analysis.persons`. This is a proxy for "user's literal tokens appear verbatim in document text."

**Problem:** The same property holds for title-reference queries ("something like Parasite") and other named entities not caught by the person extractor.

**Improvement:** Remove the gate and always fetch both sources. For non-entity queries, raw candidates sit beyond the hyde top-k×5 in the merged list and never affect the top-k output — the result is identical, with one extra ChromaDB call (~15ms) as the only cost. If ranking quality still falls short after this, apply a cross-encoder reranker over the merged candidate pool.

**When to do it:** If retrieval quality on title-similarity or other named-entity queries proves insufficient in production eval. See `docs/phase2.md` for full reasoning.
