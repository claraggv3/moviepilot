from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_TITLES_PATH = Path("data/kaggle/titles.csv")
DEFAULT_CREDITS_PATH = Path("data/kaggle/credits.csv")


def _parse_list_field(value: str) -> list[str]:
    """Parse Kaggle-style list strings like \"['drama', 'crime']\" into Python lists."""
    if not value or value.strip() in ("", "[]"):
        return []
    try:
        parsed = ast.literal_eval(value)
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _safe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_credits_index(credits_path: Path) -> dict[str, dict[str, list[str]]]:
    """
    Read credits.csv once into a lookup dict.
    Actors are capped at 3 — they appear in billing order in the CSV.
    Returns: {title_id: {"directors": [...], "actors": [...]}}
    """
    index: dict[str, dict[str, list[str]]] = {}

    with open(credits_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row["id"]
            if tid not in index:
                index[tid] = {"directors": [], "actors": []}

            if row["role"] == "DIRECTOR":
                index[tid]["directors"].append(row["name"])
            elif row["role"] == "ACTOR" and len(index[tid]["actors"]) < 5:
                index[tid]["actors"].append(row["name"])

    return index


def _build_document_text(
    title: str,
    media_type: str,
    release_year: str,
    genres: list[str],
    countries: list[str],
    age_certification: str,
    directors: list[str],
    actors: list[str],
    description: str,
) -> str:
    """
    Build the text string that will be embedded.

    Format:
        {title} ({type}, {year}) — Genres: {g}. Countries: {c}. Rated: {cert}.
        Director: {d}. Starring: {a}.
        {description}

    Fields are omitted gracefully when empty.
    Runtime and seasons are intentionally excluded — numbers don't embed
    meaningfully; they live in metadata for exact filtering.
    """
    # Line 1 — structured header
    details: list[str] = []
    if genres:
        details.append(f"Genres: {', '.join(genres)}")
    if countries:
        details.append(f"Countries: {', '.join(countries)}")
    if age_certification:
        details.append(f"Rated: {age_certification}")

    detail_str = ". ".join(details) + "." if details else ""
    header = f"{title} ({media_type}, {release_year})"
    line1 = f"{header} — {detail_str}" if detail_str else header

    # Line 2 — credits (omitted entirely if no credits available)
    credits_parts: list[str] = []
    if directors:
        credits_parts.append(f"Director: {', '.join(directors)}")
    if actors:
        credits_parts.append(f"Starring: {', '.join(actors)}")
    line2 = ". ".join(credits_parts) + "." if credits_parts else ""

    return "\n".join(line for line in [line1, line2, description] if line)


def load_titles(
    titles_path: Path = DEFAULT_TITLES_PATH,
    credits_path: Path = DEFAULT_CREDITS_PATH,
) -> list[dict[str, Any]]:
    """
    Load and join titles.csv + credits.csv into documents ready for embedding.

    Each document is a dict with:
      - "text":     the string to embed (document text)
      - "metadata": structured fields stored in ChromaDB alongside the vector

    Rows with no description are skipped — empty text can't be embedded.
    ChromaDB requires metadata values to be str/int/float — None is replaced
    with type-appropriate sentinels (0 / 0.0 / "").
    """
    logger.info("building_credits_index", path=str(credits_path))
    credits_index = _build_credits_index(credits_path)

    documents: list[dict[str, Any]] = []
    skipped = 0

    logger.info("loading_titles", path=str(titles_path))
    with open(titles_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            description = (row.get("description") or "").strip()
            if not description:
                skipped += 1
                continue

            genres = _parse_list_field(row.get("genres", ""))
            countries = _parse_list_field(row.get("production_countries", ""))
            title_credits = credits_index.get(row["id"], {"directors": [], "actors": []})

            text = _build_document_text(
                title=row["title"],
                media_type=row["type"],
                release_year=row.get("release_year", ""),
                genres=genres,
                countries=countries,
                age_certification=(row.get("age_certification") or "").strip(),
                directors=title_credits["directors"],
                actors=title_credits["actors"],
                description=description,
            )

            # ChromaDB metadata — never embedded, returned with results, cited by the LLM.
            # String fields default to "" (ChromaDB requires a value for consistent filtering).
            # Numeric fields are omitted entirely when missing — a 0 imdb_score is misleading
            # and indistinguishable from "no data". Sparse metadata is fine in ChromaDB.
            raw_metadata: dict[str, Any] = {
                "id": row["id"],
                "title": row["title"],
                "type": row["type"],
                "release_year": _safe_int(row.get("release_year")),
                "age_certification": (row.get("age_certification") or "").strip(),
                "runtime": _safe_int(row.get("runtime")),
                "genres": ", ".join(genres),
                "production_countries": ", ".join(countries),
                "seasons": _safe_int(row.get("seasons")),
                "imdb_score": _safe_float(row.get("imdb_score")),
                "imdb_votes": _safe_int(row.get("imdb_votes")),
                "tmdb_popularity": _safe_float(row.get("tmdb_popularity")),
                # Credits in metadata — LLM can cite cast; enables exact filtering
                "directors": ", ".join(title_credits["directors"]),
                "actors": ", ".join(title_credits["actors"]),
            }
            # Drop None numeric values — missing data should be absent, not zero
            metadata = {k: v for k, v in raw_metadata.items() if v is not None}

            documents.append({"text": text, "metadata": metadata})

    logger.info("titles_loaded", total=len(documents), skipped=skipped)
    return documents
