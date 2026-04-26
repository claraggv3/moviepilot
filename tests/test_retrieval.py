"""
P2-7: Retrieval tests — EphemeralClient + tiny fixture.

All tests are fully offline: ChromaDB PersistentClient is replaced with
EphemeralClient so no disk is written, and OpenAIEmbeddingFunction is
replaced with a deterministic fake so no API calls are made.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb
import pytest

from moviepilot.retrieval.chroma import (
    QueryAnalysis,
    ScoredDocument,
    build_collection,
    hyde_search,
    load_collection,
    semantic_search,
)
from moviepilot.retrieval.netflix_loader import load_titles

FIXTURES = Path(__file__).parent / "fixtures"
TINY_TITLES = FIXTURES / "tiny_netflix.csv"
TINY_CREDITS = FIXTURES / "tiny_credits.csv"


# ---------------------------------------------------------------------------
# Fake embedding function — deterministic, no API calls
# ---------------------------------------------------------------------------

class FakeEmbeddingFn:
    """Hash-based embeddings: same text → same vector, fast, no network."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        result = []
        for text in input:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(128)]
            norm = sum(x ** 2 for x in vec) ** 0.5
            result.append([x / norm for x in vec])
        return result


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def documents():
    return load_titles(titles_path=TINY_TITLES, credits_path=TINY_CREDITS)


@pytest.fixture(scope="module")
def collection(documents):
    """In-memory collection with fake embeddings — built once per module."""
    client = chromadb.EphemeralClient()
    embed_fn = FakeEmbeddingFn()
    with (
        patch("moviepilot.retrieval.chroma.chromadb.PersistentClient", return_value=client),
        patch("moviepilot.retrieval.chroma._make_embedding_fn", return_value=embed_fn),
    ):
        return build_collection(documents)


# ---------------------------------------------------------------------------
# load_titles
# ---------------------------------------------------------------------------

def test_load_titles_count(documents):
    assert len(documents) == 10


def test_load_titles_document_structure(documents):
    doc = documents[0]
    assert "text" in doc
    assert "metadata" in doc
    for field in ("id", "title", "type"):
        assert field in doc["metadata"]


def test_load_titles_skips_empty_description(tmp_path):
    csv = tmp_path / "titles.csv"
    csv.write_text("id,title,type,release_year,genres,description,imdb_score\n"
                   "ts1,Title,MOVIE,2020,action,,7.0\n")
    credits = tmp_path / "credits.csv"
    credits.write_text("person_id,id,name,character,role\n")
    assert load_titles(titles_path=csv, credits_path=credits) == []


def test_load_titles_text_contains_title(documents):
    doc = next(d for d in documents if d["metadata"]["id"] == "ts1")
    assert "The Spy Who Knew Too Much" in doc["text"]


def test_load_titles_credits_joined(documents):
    doc = next(d for d in documents if d["metadata"]["id"] == "ts1")
    assert "Martin Bourne" in doc["text"]
    assert "Martin Bourne" in doc["metadata"]["directors"]


def test_load_titles_no_credits_still_loads(documents):
    # ts2 has no credits in the fixture — should still be present
    doc = next(d for d in documents if d["metadata"]["id"] == "ts2")
    assert doc["metadata"]["directors"] == ""
    assert doc["metadata"]["actors"] == ""


# ---------------------------------------------------------------------------
# build_collection
# ---------------------------------------------------------------------------

def test_build_collection_count(collection):
    assert collection.count() == 10


def test_build_collection_idempotent(documents):
    client = chromadb.EphemeralClient()
    embed_fn = FakeEmbeddingFn()
    with (
        patch("moviepilot.retrieval.chroma.chromadb.PersistentClient", return_value=client),
        patch("moviepilot.retrieval.chroma._make_embedding_fn", return_value=embed_fn),
    ):
        col1 = build_collection(documents)
        col2 = build_collection(documents)  # second call — should skip re-embedding

    assert col1.count() == col2.count() == 10


# ---------------------------------------------------------------------------
# load_collection
# ---------------------------------------------------------------------------

def test_load_collection_raises_if_path_missing(tmp_path):
    missing = str(tmp_path / "nonexistent")
    with pytest.raises(RuntimeError, match="not found"):
        load_collection(chroma_path=missing)


# ---------------------------------------------------------------------------
# semantic_search
# ---------------------------------------------------------------------------

def test_semantic_search_returns_results(collection):
    results = semantic_search(collection, "spy thriller", n_results=3)
    assert len(results) == 3


def test_semantic_search_returns_scored_documents(collection):
    results = semantic_search(collection, "documentary about nature", n_results=5)
    for doc in results:
        assert isinstance(doc, ScoredDocument)
        assert doc.text
        assert "title" in doc.metadata
        assert 0.0 <= doc.score <= 1.0


def test_semantic_search_respects_n_results(collection):
    for n in (1, 3, 5):
        results = semantic_search(collection, "drama", n_results=n)
        assert len(results) == n


# ---------------------------------------------------------------------------
# hyde_search
# ---------------------------------------------------------------------------

def _fake_analysis(**overrides) -> QueryAnalysis:
    defaults = dict(
        hyde_document="A gripping spy thriller set across European capitals.",
        persons=[],
        country_code=None,
        media_type=None,
        decade_start=None,
        decade_end=None,
    )
    return QueryAnalysis(**{**defaults, **overrides})


def test_hyde_search_returns_k_results(collection):
    with patch("moviepilot.retrieval.chroma.analyze_query", return_value=_fake_analysis()):
        results = hyde_search(collection, "spy thriller", MagicMock(), n_results=5)
    assert len(results) == 5


def test_hyde_search_scores_in_range(collection):
    with patch("moviepilot.retrieval.chroma.analyze_query", return_value=_fake_analysis()):
        results = hyde_search(collection, "spy thriller", MagicMock(), n_results=5)
    for doc in results:
        assert 0.0 <= doc.score <= 1.0


def test_hyde_search_person_promotes_match(collection):
    """Director match must appear at rank 1 after _promote."""
    analysis = _fake_analysis(
        hyde_document="A spy thriller by a seasoned European director.",
        persons=["Martin Bourne"],
    )
    with patch("moviepilot.retrieval.chroma.analyze_query", return_value=analysis):
        results = hyde_search(collection, "Martin Bourne films", MagicMock(), n_results=5)
    assert results[0].metadata["id"] == "ts1"


def test_hyde_search_media_type_filter_no_crash(collection):
    """MOVIE filter with small corpus triggers fallback A — must not raise."""
    analysis = _fake_analysis(hyde_document="An action film.", media_type="MOVIE")
    with patch("moviepilot.retrieval.chroma.analyze_query", return_value=analysis):
        results = hyde_search(collection, "action movie", MagicMock(), n_results=5)
    assert len(results) > 0


def test_hyde_search_no_persons_uses_hyde_doc_only(collection):
    """Without persons, raw query path is skipped — result count still correct."""
    analysis = _fake_analysis(hyde_document="A romantic drama set in Tokyo.")
    with patch("moviepilot.retrieval.chroma.analyze_query", return_value=analysis):
        results = hyde_search(collection, "romance in Japan", MagicMock(), n_results=4)
    assert len(results) == 4
