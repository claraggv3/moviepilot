"""
Evaluate retrieval quality across a fixed query set.

Usage:
    poetry run python scripts/eval_retrieval.py --label semantic
    poetry run python scripts/eval_retrieval.py --label hyde
    poetry run python scripts/eval_retrieval.py --label hyde --queries eval/queries_expanded.csv

Results are saved to:
    eval/results/{label}.md   — human-readable with full descriptions
    eval/results/{label}.csv  — one row per result, easy to diff across labels
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from moviepilot.config import configure_logging
from moviepilot.retrieval.chroma import load_collection, semantic_search, hyde_search

configure_logging()

DEFAULT_QUERIES_PATH = Path("eval/queries.csv")


def load_queries(path: Path) -> list[tuple[str, str]]:
    """Load (category, query) pairs from a CSV file."""
    with open(path, encoding="utf-8") as f:
        return [(row["category"], row["query"]) for row in csv.DictReader(f)]

N_RESULTS = 10


def _format_result(rank: int, doc) -> str:
    m = doc.metadata
    directors = m.get("directors", "") or "—"
    actors = m.get("actors", "") or "—"
    genres = m.get("genres", "") or "—"
    year = m.get("release_year", "?")
    imdb = f"IMDB {m['imdb_score']}" if "imdb_score" in m else "no IMDB"
    cert = m.get("age_certification", "") or "—"

    # Grab description from the last line(s) of the document text
    lines = doc.text.strip().splitlines()
    desc_lines = lines[2:] if len(lines) > 2 else lines[-1:]
    description = " ".join(desc_lines)[:200]

    return (
        f"**{rank}. {m['title']}** ({m['type']}, {year}) — score: `{doc.score:.3f}`\n"
        f"   Genres: {genres} | Rated: {cert} | {imdb}\n"
        f"   Director: {directors}\n"
        f"   Starring: {actors}\n"
        f"   _{description}_\n"
    )


def run_semantic(col, query: str) -> list:
    return semantic_search(col, query, n_results=N_RESULTS)


def run_hyde(col, chat_model, query: str):
    return hyde_search(col, query, chat_model, n_results=N_RESULTS, return_analysis=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        required=True,
        choices=["semantic", "hyde"],
        help="Label for the output file: eval/results/{label}.md",
    )
    parser.add_argument(
        "--queries",
        default=str(DEFAULT_QUERIES_PATH),
        help="Path to a queries CSV (category,query). Defaults to eval/queries.csv.",
    )
    args = parser.parse_args()

    queries = load_queries(Path(args.queries))

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.label}.md"
    csv_path = out_dir / f"{args.label}.csv"

    print(f"Loading collection...")
    col = load_collection()

    chat_model = None
    if args.label == "hyde":
        from moviepilot.llm.client import make_chat_model
        print("Building chat model for HyDE query analysis...")
        chat_model = make_chat_model(streaming=False, temperature=0)

    lines: list[str] = [
        f"# Retrieval eval — {args.label}",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Strategy: {args.label} | n_results: {N_RESULTS} | queries: {args.queries}",
        f"",
        "---",
        "",
    ]

    csv_rows: list[dict] = []
    analysis_rows: list[dict] = []

    current_category = None
    for category, query in queries:
        if category != current_category:
            current_category = category
            lines.append(f"## {category.replace('_', ' ').title()}")
            lines.append("")

        print(f"  {query}")

        analysis = None
        if args.label == "semantic":
            results = run_semantic(col, query)
        else:
            results, analysis = run_hyde(col, chat_model, query)

        lines.append(f"### \"{query}\"")
        lines.append("")

        if analysis is not None:
            lines.append(
                f"**Analysis** — "
                f"persons: `{analysis.persons or '—'}` | "
                f"country: `{analysis.country_code or '—'}` | "
                f"media_type: `{analysis.media_type or '—'}` | "
                f"decade: `{analysis.decade_start}–{analysis.decade_end}` | "
                f"runtime: `{analysis.runtime_min or '—'}–{analysis.runtime_max or '—'} min`"
            )
            lines.append("")
            lines.append(f"> *HyDE doc:* {analysis.hyde_document}")
            lines.append("")
            analysis_rows.append({
                "category": category,
                "query": query,
                "persons": "; ".join(analysis.persons),
                "country_code": analysis.country_code or "",
                "media_type": analysis.media_type or "",
                "decade_start": analysis.decade_start if analysis.decade_start is not None else "",
                "decade_end": analysis.decade_end if analysis.decade_end is not None else "",
                "runtime_min": analysis.runtime_min if analysis.runtime_min is not None else "",
                "runtime_max": analysis.runtime_max if analysis.runtime_max is not None else "",
                "hyde_document": analysis.hyde_document,
            })

        for i, doc in enumerate(results, 1):
            lines.append(_format_result(i, doc))
            m = doc.metadata
            csv_rows.append({
                "label": args.label,
                "category": category,
                "query": query,
                "rank": i,
                "score": round(doc.score, 4),
                "title": m.get("title", ""),
                "year": m.get("release_year", ""),
                "type": m.get("type", ""),
                "genres": m.get("genres", ""),
                "directors": m.get("directors", ""),
                "actors": m.get("actors", ""),
                "imdb_score": m.get("imdb_score", ""),
                "age_certification": m.get("age_certification", ""),
                "runtime": m.get("runtime", ""),
            })
        lines.append("---")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved → {out_path}")

    csv_fields = [
        "label", "category", "query", "rank", "score",
        "title", "year", "type", "genres", "directors", "actors",
        "imdb_score", "age_certification", "runtime",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved → {csv_path}")

    if analysis_rows:
        analysis_path = out_dir / f"{args.label}_analysis.csv"
        analysis_fields = [
            "category", "query", "persons", "country_code", "media_type",
            "decade_start", "decade_end", "runtime_min", "runtime_max", "hyde_document",
        ]
        with analysis_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=analysis_fields)
            writer.writeheader()
            writer.writerows(analysis_rows)
        print(f"Saved → {analysis_path}")


if __name__ == "__main__":
    main()
