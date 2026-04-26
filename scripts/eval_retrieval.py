"""
Evaluate retrieval quality across a fixed query set.

Usage:
    poetry run python scripts/eval_retrieval.py --label semantic
    poetry run python scripts/eval_retrieval.py --label hybrid   (after P2-4)
    poetry run python scripts/eval_retrieval.py --label hyde     (after P2-5)

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

QUERIES_PATH = Path("eval/queries.csv")


def load_queries() -> list[tuple[str, str]]:
    """Load (category, query) pairs from eval/queries.csv."""
    with open(QUERIES_PATH, encoding="utf-8") as f:
        return [(row["category"], row["query"]) for row in csv.DictReader(f)]


QUERIES = load_queries()

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


def run_hyde(col, chat_model, query: str) -> list:
    return hyde_search(col, query, chat_model, n_results=N_RESULTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        required=True,
        choices=["semantic", "hybrid", "hyde"],
        help="Label for the output file: eval/results/{label}.md",
    )
    args = parser.parse_args()

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
        f"Strategy: {args.label} | n_results: {N_RESULTS}",
        f"",
        "---",
        "",
    ]

    csv_rows: list[dict] = []

    current_category = None
    for category, query in QUERIES:
        if category != current_category:
            current_category = category
            lines.append(f"## {category.replace('_', ' ').title()}")
            lines.append("")

        print(f"  {query}")

        if args.label == "semantic":
            results = run_semantic(col, query)
        else:
            results = run_hyde(col, chat_model, query)

        lines.append(f"### \"{query}\"")
        lines.append("")
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
            })
        lines.append("---")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved → {out_path}")

    csv_fields = [
        "label", "category", "query", "rank", "score",
        "title", "year", "type", "genres", "directors", "actors",
        "imdb_score", "age_certification",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
