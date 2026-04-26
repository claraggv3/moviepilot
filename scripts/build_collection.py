"""
Build (or rebuild) the ChromaDB vector collection from the Netflix Kaggle dataset.

Usage:
    python scripts/build_collection.py
    python scripts/build_collection.py --titles data/kaggle/titles.csv --credits data/kaggle/credits.csv
    python scripts/build_collection.py --chroma-path data/chroma

The collection is idempotent: if it already exists with the correct document
count, this is a no-op. Run with --force to rebuild from scratch.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import typer

# Allow running as a standalone script from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moviepilot.config import configure_logging, settings
from moviepilot.retrieval.chroma import build_collection
from moviepilot.retrieval.netflix_loader import (
    DEFAULT_CREDITS_PATH,
    DEFAULT_TITLES_PATH,
    load_titles,
)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    titles: Path = typer.Option(DEFAULT_TITLES_PATH, help="Path to titles.csv"),
    credits: Path = typer.Option(DEFAULT_CREDITS_PATH, help="Path to credits.csv"),
    chroma_path: str = typer.Option(settings.chroma_path, help="ChromaDB persistence directory"),
    force: bool = typer.Option(False, "--force", help="Delete and rebuild even if collection exists"),
) -> None:
    configure_logging()

    for path, label in [(titles, "titles"), (credits, "credits")]:
        if not path.exists():
            typer.echo(f"Error: {label} file not found: {path}", err=True)
            raise typer.Exit(1)

    typer.echo(f"Loading titles from {titles} + {credits} ...")
    t0 = time.perf_counter()
    documents = load_titles(titles_path=titles, credits_path=credits)
    typer.echo(f"  Loaded {len(documents):,} documents ({time.perf_counter() - t0:.1f}s)")

    if force:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_path)
        existing = {c.name for c in client.list_collections()}
        if "netflix_titles" in existing:
            client.delete_collection("netflix_titles")
            typer.echo("  Deleted existing collection (--force)")

    typer.echo(f"Building ChromaDB collection at {chroma_path} ...")
    t1 = time.perf_counter()
    collection = build_collection(documents, chroma_path=chroma_path)
    elapsed = time.perf_counter() - t1

    typer.echo(f"  Done. {collection.count():,} documents in collection ({elapsed:.1f}s)")


if __name__ == "__main__":
    app()
