# Phase 2 — Netflix Retrieval Pipeline

## Overview

Phase 2 builds the NLP core of the system: the pipeline that takes a user query and finds the most relevant Netflix titles to pass to the LLM. It covers data loading, ChromaDB collection building, semantic search, and HyDE query expansion with LLM-extracted constraints. BM25 hybrid retrieval was implemented, evaluated, and removed (see below).

---

## Dataset

The Kaggle dataset (`victorsoeiro/netflix-tv-shows-and-movies`) has two files, both required:

**`data/kaggle/titles.csv`**

| Field | Type | Notes |
|---|---|---|
| `id` | string | e.g. `ts300399` |
| `title` | string | |
| `type` | string | `MOVIE` or `SHOW` |
| `description` | string | 18 rows missing — these are skipped |
| `release_year` | int | |
| `age_certification` | string | `R`, `TV-MA`, `PG`, etc. — 469 missing |
| `runtime` | int | minutes for movies, minutes/episode for shows |
| `genres` | string | stored as `['drama', 'crime']` — requires parsing |
| `production_countries` | string | stored as `['US', 'GB']` — requires parsing |
| `seasons` | float | null for movies |
| `imdb_id` | string | dropped — external reference only |
| `imdb_score` | float | 469 rows missing |
| `imdb_votes` | int | 469 rows missing |
| `tmdb_popularity` | float | |
| `tmdb_score` | float | dropped — redundant with imdb_score |

**`data/kaggle/credits.csv`**

| Field | Notes |
|---|---|
| `person_id` | dropped |
| `id` | join key → titles.csv |
| `name` | director or actor name |
| `character` | dropped — character name, not useful for search |
| `role` | `ACTOR` or `DIRECTOR` |

**Dataset stats after loading:**
- 5832 documents (18 skipped for missing description)
- 4037 with credits (~69%), 1795 without (~31%)
- 5363 with IMDB score, 469 without

---

## P2-1 — Data loader (`retrieval/netflix_loader.py`)

### What it does

Reads both CSVs, joins them, and produces a list of dicts — each with a `text` field (the string to embed) and a `metadata` dict (structured fields stored in ChromaDB alongside the vector). No embeddings happen here.

### Document text format (what gets embedded)

```
Taxi Driver (MOVIE, 1976) — Genres: drama, crime. Countries: US. Rated: R.
Director: Martin Scorsese. Starring: Robert De Niro, Jodie Foster, Albert Brooks.
A mentally unstable Vietnam War veteran works as a night-time taxi driver in New York City...
```

For titles with no credits:
```
Five Came Back: The Reference Films (SHOW, 1945) — Genres: documentation. Countries: US. Rated: TV-MA.
This collection includes 12 World War II-era propaganda films...
```

### Field-by-field decisions: document text

**Title, type, release year — always included.**
The most discriminative tokens appear first. Embedding models weight earlier tokens more heavily. Including `type` means "movie" vs "show" queries match on this signal directly.

**Genres — included.**
Genre is the most common search signal. Including the genre words in the embedded text means both BM25 (exact match on "drama") and semantic search (synonym matching — "emotional" → drama) benefit. Without this, genre matching relies entirely on the description's vocabulary.

**Production countries — included.**
Users ask for "Korean drama", "French film", "something Italian". Country codes appear in the metadata but the full text ("Countries: KR") in the document means semantic search can match "Korean" → "KR". BM25 also benefits from the country code as a lexical token.

**Age certification — included.**
This is categorical text with semantic meaning embedding models understand. "TV-MA" and "R" map to "mature content", "adult". "G" and "PG" map to "family-friendly", "for kids". A query "something I can watch with children" will semantically match documents containing "G" or "PG". Omitting it would make family-suitability queries rely purely on description content.

**Runtime — NOT included in document text.**
Numbers don't embed meaningfully. "114min" and "90min" look nearly identical to an embedding model — it has no concept of "long" vs "short". A query "short movie under 90 minutes" won't match documents containing "Runtime: 85min" any better than if the field were absent. Runtime lives in metadata only, where it can be used for exact filtering (`runtime < 90`).

**Seasons — NOT included in document text.**
Same reasoning as runtime. "Seasons: 1" and "Seasons: 8" are semantically opaque numbers. "Mini-series" vs "long-running show" requires numeric comparison, not embedding similarity. Metadata only.

**Director and top 5 actors — included when available.**
Actor and director names almost never appear in plot descriptions. Without the credits join, queries like "Christopher Nolan films" or "movies with Tom Hanks" would return poor results. With credits in the document text, the embedding captures cast as a feature, and the raw-query vector search in `hyde_search` scores directly against "Starring: Tom Hanks" text. Actors are capped at 5 (billing order in the CSV). All directors are included since films rarely have more than 2.

### Field-by-field decisions: metadata

Metadata is stored as structured key-value pairs in ChromaDB alongside each vector. It is never embedded, never used for similarity search. It has two jobs:
1. Exact filtering via ChromaDB `where` clause (e.g., `type == "MOVIE"`, `release_year >= 1990`)
2. Citation by the LLM in its response ("this is rated R, released in 1976, IMDB 8.2")

| Field | Stored | Rationale |
|---|---|---|
| `id` | yes | reference key |
| `title` | yes | LLM cites it |
| `type` | yes | filter MOVIE vs SHOW |
| `release_year` | yes | filter by era |
| `age_certification` | yes | filter family-friendly; also in text |
| `runtime` | yes | filter short/long |
| `genres` | yes | also in text; LLM cites it |
| `production_countries` | yes | also in text; filter by country |
| `seasons` | yes | filter mini-series vs long-running |
| `imdb_score` | yes (when present) | LLM cites quality |
| `imdb_votes` | yes (when present) | quality confidence signal |
| `tmdb_popularity` | yes | trending signal |
| `directors` | yes | LLM citation + exact `where` filtering |
| `actors` | yes (top 5) | LLM citation + exact `where` filtering |
| `imdb_id` | dropped | external reference only |
| `tmdb_score` | dropped | redundant |

### Why some fields appear in both document text and metadata

They serve completely different purposes:
- **In document text:** so the embedding captures the concept. "drama" in genres means "drama" queries land near drama films in vector space.
- **In metadata:** so the LLM can cite exact values ("Genres: drama, crime") and so exact filters can be applied (`genres contains "drama"`).

A field only in metadata can't be searched semantically. A field only in document text can't be filtered exactly or cited cleanly.

### Missing data handling

**Genres and countries** are stored as Python list strings (`['drama', 'crime']`) in the CSV. Parsed with `ast.literal_eval()` — handles malformed values gracefully by returning an empty list.

**Missing numeric fields (imdb_score, imdb_votes, runtime, seasons, release_year)** are omitted from metadata entirely rather than using 0 as a sentinel. A `0.0` imdb_score is indistinguishable from "rated 0/10" — misleading to the LLM. Absent keys are simply not cited. ChromaDB handles sparse metadata correctly.

**Missing string fields (age_certification)** default to `""` — ChromaDB requires consistent types for fields used in `where` filters.

**Titles with no credits (31%)** are fully searchable by plot, genre, mood, and topic. They simply don't surface for actor/director queries, which is correct — we have no data for those queries on these titles.

**Actors cap at 5, directors uncapped.** Top 5 actors by billing order covers leads and key supporting roles — enough to handle the vast majority of actor-based queries. Actors below position 5 (niche, minor roles) are a known limitation documented in REPORT.md. Directors are all included since most titles have 1–2. Adding actors to metadata (comma-separated string) enables exact `where` filtering and LLM citation of the full cast.

### Inspecting retrieved chunks

```python
results = collection.query(query_texts=["spy thriller"], n_results=5)
print(results['documents'])   # text chunks — readable, judgeable
print(results['metadatas'])   # structured metadata per result
print(results['distances'])   # cosine distances (lower = more similar)
```

---

## Multi-vector retrieval — considered and rejected

**The idea:** create separate embeddings for plot, cast, and genres. At query time, search all three and combine scores.

**Why we rejected it for this system:**

- HyDE with dual-source merge already handles actor/director queries — a separate cast embedding adds no meaningful gain.
- Genre text (`['drama', 'crime']`) is too short to produce a strong embedding — sparse input, weak vector.
- HyDE already closes the query-document vocabulary gap for plot queries, which is the main problem multi-vector addresses.
- Cost is 3× (3 embedding API calls per document at build time, 3 per query at runtime).
- At 6k rows the marginal recall improvement is invisible in practice.

**When it would make sense:** corpus > 100k titles, dedicated vector infrastructure (Weaviate, Vespa), actor/director queries as the dominant use case.

Documented here and in REPORT.md as a "next step at scale" — it signals awareness of the technique without over-engineering.

---

## Query testset (`eval/query_testset.md`)

64 realistic user queries across 15 categories, each testing a distinct NLP or routing problem:

| Category | Count | Key problem |
|---|---|---|
| Named director | 4 | LLM person extraction + dual-source merge + _promote |
| Named actor | 4 | LLM person extraction + dual-source merge + _promote |
| Pure genre | 4 | Routing decision (no temporal signal) |
| Mood / emotional state | 6 | Abstract semantic retrieval |
| Year / era | 4 | Temporal metadata without recency routing |
| Language / country | 4 | Metadata-driven retrieval |
| Topic / theme | 5 | Deep semantic retrieval |
| Recency / trending | 6 | Routing to trending agent |
| Similarity queries | 3 | Semantic search quality |
| Audience / context | 4 | Semantic inference from context |
| Format / length | 3 | Metadata filtering |
| Ambiguous routing | 4 | Router boundary cases |
| Multi-turn follow-ups | 4 | Router context window |
| Negative preference | 3 | Generation-time filtering |
| Out-of-scope | 6 | Refusal |

---

## BM25 hybrid retrieval — implemented, evaluated, and removed

BM25 was implemented as P2-4 and evaluated against pure semantic search across 41 queries. It was removed after evaluation showed it was net negative on this corpus.

### What was implemented

`BM25Index` wrapping `rank_bm25.BM25Okapi`. Tokenisation: whitespace split, lowercase. Scores normalised to [0, 1] by dividing by the top result's raw score. Combined with semantic via weighted sum: `score = alpha * semantic + (1 - alpha) * BM25`, default `alpha = 0.5`.

### Evaluation results

Metric comparison across 21 queries with objective relevance rules (eval/results/stats_summary.csv):

| Metric | Semantic | Hybrid (BM25 α=0.5) |
|---|---|---|
| Total Hits@5 / max | 65 / 86 | 44 / 86 |
| Total Hits@10 / max | 101 / 137 | 81 / 137 |
| Recall@5 | **75.6%** | 51.2% |
| Recall@10 | **73.7%** | 59.1% |
| Mean MRR | **0.917** | 0.562 |
| Mean 1st Rank | **1.2** | 3.0 |

Hybrid was strictly worse on every metric. Full per-query breakdown in `eval/results/hybrid.csv` and `eval/results/semantic.csv`.

### Why BM25 failed on this corpus

**Token collision on common first names.** BM25 tokenises on whitespace: "Tom Hanks" → `["tom", "hanks"]`. "tom" has moderate IDF weight and matches every Tom Segura, Tom Papa, Tom Jones in the catalog. All three `movies with Tom Hanks` queries got 0 hits@10 under hybrid — BM25 boosted irrelevant Toms to the top, displacing semantically-matched results. Same pattern for David (Batra, Letterman), Cate (Shortland as director).

**0.5 score artifact from one-sided matches.** Documents absent from semantic top-20 but present at the top of BM25 get `score = 0.5 × 0 + 0.5 × 1.0 = 0.5`. This pushes lexically-matched-but-semantically-irrelevant results to rank 5–8, displacing correct semantic results.

**High-IDF noise tokens.** "films" in "Bong Joon-ho films" boosted "Five Came Back: The Reference Films" because "films" had high IDF weight in the corpus. The query token "films" had nothing to do with the genre of the target document.

**The fundamental mismatch.** BM25 was designed for documents-vs-documents retrieval where terms have consistent meaning across the corpus. User queries like "movies with Tom Hanks" contain function words and common nouns alongside the entity of interest. Whitespace tokenisation destroys the identity of proper nouns.

### What replaces it: HyDE + LLM-extracted filters (P2-4, merged with P2-5)

See the full implementation section below.

---

## HyDE + LLM-extracted filters — final implementation

### Design

A single LLM call with structured output does two jobs simultaneously:

1. **HyDE (Hypothetical Document Embedding)**: generates a 2-sentence Netflix-style description of an ideal title that would answer the query. This hypothetical document is used as the embedding query instead of the raw user query. It bridges the vocabulary gap between user language ("spy thriller with action") and document text ("A rogue CIA operative goes off the grid after uncovering a conspiracy...").

2. **Constraint extraction**: pulls structured hard constraints out of the query — person names, country, media type (MOVIE/SHOW), decade. These are applied after retrieval as post-filters, not as ChromaDB pre-filters.

### QueryAnalysis schema

```python
class QueryAnalysis(BaseModel):
    hyde_document: str          # 2-sentence hypothetical Netflix description
    persons: list[str] = []     # full names of directors/actors mentioned
    country_code: str | None    # ISO 2-letter code (KR, FR, ES, IT, JP, IN, DE, GB)
    media_type: Literal["MOVIE", "SHOW"] | None
    decade_start: int | None    # e.g. "80s" → 1980
    decade_end: int | None      # e.g. "80s" → 1989
```

One `chat_model.with_structured_output(QueryAnalysis)` call at the start of every `hyde_search` invocation.

### Retrieval flow

```
User query
    ↓
analyze_query() — one LLM call → hyde_document + constraints
    ↓
_build_where() — type/decade constraints → ChromaDB where clause (or None)
    ↓
_query_collection(hyde_document, k×5, where)
    │   if where raises (filtered count < k×5):
    │       retry without where  ← fallback A
    ↓
[if persons present]
    │   semantic_search(raw_query, k×6)  ← second ChromaDB query
    │   merge deduped by title ID, hyde_candidates first
    ↓
_promote(persons)  — matched docs float to top, rest fill remaining slots
    ↓
[if country_code]
    _promote(country_code)
    ↓
candidates[:k]  → final results
```

### Why two candidate sources for person queries

Person queries need two embedding sources because neither alone is sufficient:

- **hyde_document** captures style/genre ("heartwarming everyman drama" → Tom Hanks films). Works for actors whose typical roles have a recognisable voice in the corpus. Fails for directors with only 1–2 sparse-style films (Fincher, Tarantino) where the style embedding is weak.
- **raw query** carries the name as a literal token ("Tom Hanks" scores against "Starring: Tom Hanks" in document text). Works for directors whose name is the only reliable signal. Fails for actors whose name isn't a strong semantic anchor (Cate Blanchett).

Neither alone is sufficient. Merging gives the union of both signals. `hyde_candidates` (k×5) come first in the merge to preserve their score ordering; `raw_candidates` (k×6) fill in items not already seen.

**Latency cost:** one additional ChromaDB query (~10–20 ms, local disk). Person queries are a minority of real traffic; the LLM analysis call preceding this already takes ~500 ms, so the overhead is <4%.

**Known limitation — gate is ad hoc.** The `if analysis.persons` condition is a proxy for "user's literal tokens appear verbatim in document text." The same property holds for title-reference queries ("something like Parasite") and other named entities. A more principled design would always fetch both sources — for non-entity queries the raw candidates sit beyond hyde's top-k×5 in the merged list and never affect the top-k output, so the result is identical with one extra ChromaDB call (~15ms) as the only cost. If retrieval quality proves insufficient on title-similarity or other entity queries in production, the fix is: (1) remove the `persons` gate and always merge, then (2) if that still leaves ranking quality gaps, apply a **cross-encoder reranker** (e.g. `sentence-transformers` `cross-encoder/ms-marco-MiniLM`) over the merged candidate pool to replace cosine-similarity ordering with a more precise query-document relevance score.

### Why post-filters instead of ChromaDB pre-filters

ChromaDB raises an exception when `n_results > count(filtered documents)`. A director with only 2 films in the catalog would cause every query for them to fail if we used a `where_document={"$contains": "Christopher Nolan"}` pre-filter with `n_results=50`. Post-filtering on a large candidate pool avoids this entirely.

**Fallback A** (in `_query_collection`) handles the one case where ChromaDB pre-filters are still used (type and decade via `where`): if the filtered count is smaller than the requested k×5, it retries without the filter and lets the embedding carry the signal.

### _promote pattern

```python
def _promote(docs, predicate):
    matched = [d for d in docs if predicate(d)]
    rest    = [d for d in docs if not predicate(d)]
    return matched + rest
```

Preserves relative score order within each group. A director with only 2 films always ranks 1–2, then 8 thematically similar films fill remaining slots — instead of returning 0 director results or raising.

---

## Evaluation — retrieval strategy comparison

21 queries with objective relevance labels. Denominator is `min(corpus_n, k)` per query (capped recall — measures recall against what's retrievable in k slots).

| Metric | Semantic | Hybrid BM25 | HyDE (final) |
|---|---|---|---|
| Total Hits@5 / max | 65 / 86 | 44 / 86 | **79 / 86** |
| Total Hits@10 / max | 101 / 137 | 81 / 137 | **126 / 137** |
| Recall@5 | 75.6% | 51.2% | **91.9%** |
| Recall@10 | 73.7% | 59.1% | **92.0%** |
| Mean MRR | 0.917 | 0.562 | **1.0** |
| Mean 1st rank | 1.2 | 3.0 | **1.0** |

Full per-query breakdown in `eval/results/stats.csv` and summary in `eval/results/stats_summary.csv`.

### Iteration history (how we got to final HyDE)

| Version | Change | Recall@10 | Hits@10 |
|---|---|---|---|
| Semantic baseline | Pure embedding search | 73.7% | 101/137 |
| hyde_old (first HyDE) | HyDE + where_document pre-filter for persons | 83.2% | 114/137 |
| hyde_old2 | Replaced where_document with _promote post-filter | 85.4% | 117/137 |
| hyde_old3 | Added dual-source merge (hyde + raw query) for persons | 90.5% | 124/137 |
| HyDE final | Increased raw_candidates from k×3 to k×6 | **92.0%** | **126/137** |

The primary gains came from: (1) switching from where_document pre-filter to _promote post-filter (avoids ChromaDB exception on sparse-catalog directors); (2) adding the raw-query candidate source for person queries (catches actors and directors that hyde_document embedding alone misses).

### Remaining gaps

| Query | Got / Max @10 | Root cause |
|---|---|---|
| Romantic comedy | 6 / 10 | LLM nondeterminism in hyde_document generation — not structural |
| Cate Blanchett | 2 / 5 | Embedding ceiling: 2 of her 5 films are not in the top-300 nearest neighbours for any query formulation we tried |
| Scorsese spy thriller | 4 / 6 | Peripheral matches: Scorsese doesn't make spy films; the 2 missing titles have weak embedding overlap |
| Pedro Pascal | 2 / 3 | Triple Frontier ranks ~271st in raw semantic search; below k×6 threshold |
| Tom Hanks drama | 6 / 7 | 1 film below the k×6 candidate pool |

The Cate Blanchett and Pedro Pascal ceilings are structural — the films exist in the catalog but their document text doesn't produce embedding vectors close enough to any formulation of the user query. Closing these gaps would require larger k or a different embedding model.

---

## P2-6 — CLI entry point (`scripts/build_collection.py`)

Standalone script that wires `load_titles` → `build_collection`. Run from the repo root:

```bash
poetry run python scripts/build_collection.py          # uses defaults from settings
poetry run python scripts/build_collection.py --force  # delete + rebuild even if count matches
```

Flags: `--titles PATH`, `--credits PATH`, `--chroma-path TEXT`, `--force`. Idempotency is handled by `build_collection` itself — the script just prints timing and document counts.

**ChromaDB v0.6.0 compatibility note:** `list_collections()` now returns collection names (strings) directly, not Collection objects. The `{c.name for c in ...}` set comprehension was updated to `set(client.list_collections())`.

---

## P2-7 — Tests (`tests/test_retrieval.py`)

17 tests, fully offline — no OpenAI API calls, no disk writes:

- **`FakeEmbeddingFn`**: hash-based deterministic embeddings (128-dim), replaces `_make_embedding_fn` via `unittest.mock.patch`.
- **`EphemeralClient`**: replaces `chromadb.PersistentClient` via patch — in-memory only.
- **`collection` fixture** (`scope="module"`): builds once for all tests; `build_collection` called with both patches active.

Coverage:
- `load_titles`: count, document structure, skip empty descriptions, credits join, no-credits case
- `build_collection`: count, idempotency (second call skips re-embedding)
- `load_collection`: raises `RuntimeError` with clear message if path missing
- `semantic_search`: returns results, `ScoredDocument` shape, score in [0,1], respects `n_results`
- `hyde_search`: returns k results, scores in range, person `_promote` puts match at rank 1, media-type filter + fallback A doesn't crash, no-persons path

Fixture files: `tests/fixtures/tiny_netflix.csv` (10 rows), `tests/fixtures/tiny_credits.csv` (credits for 4 titles).
