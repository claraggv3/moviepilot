from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import chromadb
import structlog
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from pydantic import BaseModel

from moviepilot.config import settings

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "netflix_titles"
EMBED_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_embedding_fn() -> OpenAIEmbeddingFunction:
    return OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        api_base=settings.openai_base_url,
        model_name=settings.embedding_model,
    )


@dataclass
class ScoredDocument:
    text: str
    metadata: dict[str, Any]
    score: float  # 0–1, higher is more similar


# ---------------------------------------------------------------------------
# P2-2: Build
# ---------------------------------------------------------------------------

def build_collection(
    documents: list[dict[str, Any]],
    chroma_path: str | None = None,
) -> chromadb.Collection:
    """
    Embed all documents and persist them to ChromaDB.

    Idempotent: if the collection already exists with the same number of
    documents, returns it without re-embedding. Rebuilds if count mismatches
    (e.g. partial build from a previous interrupted run).

    Args:
        documents: output of netflix_loader.load_titles()
        chroma_path: overrides settings.chroma_path (used in tests)
    """
    path = chroma_path or settings.chroma_path
    client = chromadb.PersistentClient(path=path)
    embedding_fn = _make_embedding_fn()

    existing = set(client.list_collections())
    if COLLECTION_NAME in existing:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)
        if collection.count() == len(documents):
            logger.info("collection_exists_skipping", count=collection.count(), path=path)
            return collection
        logger.info("collection_incomplete_rebuilding", path=path)
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(documents)
    logger.info("embedding_start", total=total, batch_size=EMBED_BATCH_SIZE)

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = documents[start : start + EMBED_BATCH_SIZE]
        collection.add(
            ids=[d["metadata"]["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        done = min(start + EMBED_BATCH_SIZE, total)
        logger.info("embedding_progress", done=done, total=total, pct=round(done / total * 100))

    logger.info("collection_built", count=collection.count(), path=path)
    return collection


# ---------------------------------------------------------------------------
# P2-3: Load + semantic search
# ---------------------------------------------------------------------------

def load_collection(chroma_path: str | None = None) -> chromadb.Collection:
    """
    Load an existing ChromaDB collection from disk.
    Raises RuntimeError with a clear message if the collection hasn't been built.
    """
    path = chroma_path or settings.chroma_path

    if not Path(path).exists():
        raise RuntimeError(
            f"ChromaDB collection not found at '{path}'. "
            "Run `chatbot build-collection` first."
        )

    client = chromadb.PersistentClient(path=path)
    try:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=_make_embedding_fn(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load collection '{COLLECTION_NAME}' from '{path}'. "
            "Run `chatbot build-collection` to rebuild it."
        ) from exc


def semantic_search(
    collection: chromadb.Collection,
    query: str,
    n_results: int | None = None,
) -> list[ScoredDocument]:
    """
    Embed the query and return the closest documents by cosine similarity.
    Distances are converted to scores (0–1, higher = more similar).
    """
    k = n_results or settings.top_k
    results = collection.query(query_texts=[query], n_results=k)

    docs = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance is in [0, 2]; convert to similarity [0, 1]
        docs.append(ScoredDocument(text=text, metadata=metadata, score=1 - distance / 2))

    return docs


# ---------------------------------------------------------------------------
# P2-4: HyDE + LLM-extracted filters
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPT = """\
You are helping a movie recommendation engine parse a user query.

Return a JSON object with these fields:
- hyde_document: A 2-sentence Netflix-style description of an ideal title that perfectly answers this query. Write it as a real plot description — no "The user wants" phrasing.
- persons: Full names of specific directors or actors mentioned in the query. Empty list if none.
- country_code: ISO 2-letter code if a specific country's cinema is requested (KR=Korean, FR=French, ES=Spanish, IT=Italian, JP=Japanese, IN=Indian, DE=German, GB=British). Null if not specified.
- media_type: "MOVIE" if the query clearly wants a film, "SHOW" if clearly a series/show/TV. Null if unspecified or ambiguous.
- decade_start: First year of the decade if a specific era is mentioned (e.g. "80s" → 1980, "90s" → 1990). Null if not specified.
- decade_end: Last year of the decade (e.g. "80s" → 1989, "90s" → 1999). Null if not specified.

Query: {query}"""


class QueryAnalysis(BaseModel):
    hyde_document: str
    persons: list[str] = []
    country_code: str | None = None
    media_type: Literal["MOVIE", "SHOW"] | None = None
    decade_start: int | None = None
    decade_end: int | None = None


def analyze_query(query: str, chat_model) -> QueryAnalysis:
    """Single LLM call: generate HyDE document + extract hard constraints."""
    structured = chat_model.with_structured_output(QueryAnalysis)
    return structured.invoke(_ANALYSIS_PROMPT.format(query=query))


def _build_where(analysis: QueryAnalysis) -> dict | None:
    """Build ChromaDB metadata `where` filter from type/decade constraints only."""
    clauses: list[dict] = []
    if analysis.media_type:
        clauses.append({"type": analysis.media_type})
    if analysis.decade_start is not None:
        clauses.append({"release_year": {"$gte": analysis.decade_start}})
    if analysis.decade_end is not None:
        clauses.append({"release_year": {"$lte": analysis.decade_end}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _query_collection(
    collection: chromadb.Collection,
    query_text: str,
    k: int,
    where: dict | None,
) -> list[ScoredDocument]:
    """
    Query ChromaDB. If the where filter leaves fewer documents than k,
    ChromaDB raises instead of returning a short list — in that case we
    drop the filter and let the embedding carry the signal (fallback A).
    """
    def _run(use_where: bool) -> list[ScoredDocument] | None:
        kwargs: dict[str, Any] = {"query_texts": [query_text], "n_results": k}
        if use_where and where:
            kwargs["where"] = where
        try:
            results = collection.query(**kwargs)
            return [
                ScoredDocument(text=t, metadata=m, score=1 - d / 2)
                for t, m, d in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ]
        except Exception:
            return None

    if where:
        docs = _run(use_where=True)
        if docs is None:
            logger.info("chroma_where_fallback", query=query_text[:60])
            docs = _run(use_where=False) or []
        return docs
    return _run(use_where=False) or []


def _promote(
    docs: list[ScoredDocument],
    predicate: Any,
) -> list[ScoredDocument]:
    """
    Partition docs into matching and non-matching, returning matched first.
    Preserves relative score order within each group.
    This lets a director with only 2 films in the catalog rank 1–2 rather
    than disappearing into a hard-filter fallback.
    """
    matched = [d for d in docs if predicate(d)]
    rest    = [d for d in docs if not predicate(d)]
    return matched + rest


def hyde_search(
    collection: chromadb.Collection,
    query: str,
    chat_model,
    n_results: int | None = None,
) -> list[ScoredDocument]:
    """
    HyDE retrieval with LLM-extracted constraints applied as post-filters.

    One LLM call produces two things:
      - hyde_document: a hypothetical Netflix description used as the embedding
        query (bridges the vocabulary gap between user language and catalog text).
      - Structured constraints: persons, country, media type, decade.

    All constraints are applied as post-filters on a large candidate pool
    (k * 5 results) rather than as ChromaDB pre-filters. This avoids the
    ChromaDB exception that fires when n_results > count(filtered documents) —
    which breaks director queries where only 1–2 films exist in the catalog.

    Persons and country use _promote(): matched results float to the top,
    unmatched fill remaining slots. A query for "Tarantino movies" always
    returns his 2 catalog films at ranks 1–2, then 8 thematically similar
    films — instead of silently returning 0 Tarantino results.
    """
    k = n_results or settings.top_k
    analysis = analyze_query(query, chat_model)

    logger.info(
        "hyde_analysis",
        persons=analysis.persons,
        country=analysis.country_code,
        media_type=analysis.media_type,
        decade=(analysis.decade_start, analysis.decade_end),
    )

    where = _build_where(analysis)

    # Primary candidate pool: hyde_document embedding with metadata filters.
    # _query_collection handles the case where the where filter is too narrow
    # for k*5 by retrying without it (fallback A).
    hyde_candidates = _query_collection(collection, analysis.hyde_document, k * 5, where)

    if analysis.persons:
        # Person queries need two candidate sources, not one:
        #
        # - hyde_document captures style/genre ("heartwarming everyman drama"
        #   → Tom Hanks films), which works for actors whose typical roles have
        #   a recognisable voice in the corpus.
        # - raw query carries the name as a literal token ("Tom Hanks" scores
        #   directly against "Starring: Tom Hanks" in document text), which
        #   works for directors whose name is the only reliable signal.
        #
        # Neither source alone is sufficient: hyde misses directors with only
        # 1–2 sparse-style films (Fincher, Tarantino); raw query undershoots
        # actors whose name isn't a strong semantic anchor (Cate Blanchett).
        # Merging gives the union of both signals.
        #
        # Latency cost: one additional ChromaDB query (~10–20 ms, local disk).
        # Person queries are a minority of real traffic; the LLM analysis call
        # preceding this already takes ~500 ms, so the overhead is < 4%.
        raw_candidates = semantic_search(collection, query, n_results=k * 6)

        seen_ids: set[str] = set()
        merged: list[ScoredDocument] = []
        for d in hyde_candidates + raw_candidates:
            doc_id = d.metadata.get("id", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                merged.append(d)

        persons = analysis.persons
        candidates = _promote(merged, lambda d: any(p in d.text for p in persons))
    else:
        candidates = hyde_candidates

    if analysis.country_code:
        code = analysis.country_code
        candidates = _promote(
            candidates,
            lambda d: code in d.metadata.get("production_countries", ""),
        )

    return candidates[:k]
