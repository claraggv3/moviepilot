from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import typer
from langchain_core.messages import HumanMessage

from moviepilot.config import configure_logging

app = typer.Typer(add_completion=False, help="MoviePilot — movie recommendation chatbot")


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@app.command()
def chat(
    thread_id: str = typer.Option(
        None,
        "--thread-id",
        help="Resume a previous conversation. Omit to start a new one.",
    ),
) -> None:
    """Start an interactive movie recommendation chat session."""
    import aiosqlite
    if not hasattr(aiosqlite.Connection, "is_alive"):
        aiosqlite.Connection.is_alive = lambda self: True  # removed in aiosqlite>=0.20

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from moviepilot.chat import chat as _chat
    from moviepilot.graph import build_graph

    configure_logging()

    tid = thread_id or str(uuid.uuid4())
    db_path = Path("data/checkpoints.sqlite")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async def _run_chat() -> None:
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            graph = build_graph(checkpointer)
            typer.echo(f"Session: {tid}  (Ctrl-C to exit)\n")

            while True:
                try:
                    message = typer.prompt("You")
                except (KeyboardInterrupt, EOFError):
                    break

                first_token = True
                async for token in _chat(message, tid, graph):
                    if first_token:
                        typer.echo("\nAssistant: ", nl=False)
                        first_token = False
                    typer.echo(token, nl=False)

                typer.echo("\n")

        typer.echo(f"\nThread ID: {tid}")

    _run(_run_chat())


@app.command()
def eval(
    judge: bool = typer.Option(False, "--judge", help="Run LLM-as-judge scoring (gpt-4o, ~$1)"),
    cases: int = typer.Option(None, "--cases", help="Limit to first N cases (smoke test)"),
) -> None:
    """Run the end-to-end evaluation harness."""
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent.parent / "scripts"))

    from run_eval import main as _eval_main

    configure_logging()
    _run(_eval_main(run_judge=judge, max_cases=cases))


@app.command("build-collection")
def build_collection_cmd(
    titles: Path = typer.Option(Path("data/kaggle/titles.csv"), help="Path to titles.csv"),
    credits: Path = typer.Option(Path("data/kaggle/credits.csv"), help="Path to credits.csv"),
    chroma_path: str = typer.Option(None, help="ChromaDB path (default: from settings)"),
    force: bool = typer.Option(False, "--force", help="Rebuild even if collection exists"),
) -> None:
    """Build the Netflix ChromaDB vector collection."""
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent.parent.parent / "scripts"))

    import time
    import chromadb
    from moviepilot.config import settings
    from moviepilot.retrieval.chroma import build_collection, COLLECTION_NAME
    from moviepilot.retrieval.netflix_loader import load_titles

    configure_logging()
    path = chroma_path or settings.chroma_path

    for p, label in [(titles, "titles"), (credits, "credits")]:
        if not p.exists():
            typer.echo(f"Error: {label} file not found: {p}", err=True)
            raise typer.Exit(1)

    typer.echo(f"Loading titles from {titles} + {credits} ...")
    t0 = time.perf_counter()
    documents = load_titles(titles_path=titles, credits_path=credits)
    typer.echo(f"  Loaded {len(documents):,} documents ({time.perf_counter() - t0:.1f}s)")

    if force:
        client = chromadb.PersistentClient(path=path)
        if COLLECTION_NAME in set(client.list_collections()):
            client.delete_collection(COLLECTION_NAME)
            typer.echo("  Deleted existing collection (--force)")

    typer.echo(f"Building collection at {path} ...")
    t1 = time.perf_counter()
    collection = build_collection(documents, chroma_path=path)
    typer.echo(f"  Done. {collection.count():,} documents ({time.perf_counter() - t1:.1f}s)")
