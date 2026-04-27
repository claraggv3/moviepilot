from __future__ import annotations

import asyncio
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="MoviePilot",
    page_icon="🎬",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading MoviePilot...")
def _get_graph():
    from langgraph.checkpoint.memory import MemorySaver

    from moviepilot.config import configure_logging
    from moviepilot.graph import build_graph

    configure_logging()
    return build_graph(MemorySaver())


def _stream_tokens(message: str, thread_id: str, graph):
    """Bridge the async chat generator to a sync generator for st.write_stream."""
    from moviepilot.chat import chat

    token_queue: queue.Queue[str | None] = queue.Queue()

    async def _producer() -> None:
        async for token in chat(message, thread_id, graph):
            token_queue.put(token)
        token_queue.put(None)

    threading.Thread(target=lambda: asyncio.run(_producer()), daemon=True).start()

    while True:
        token = token_queue.get()
        if token is None:
            break
        yield token


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Load graph (once per server process, shared across sessions)
# ---------------------------------------------------------------------------

try:
    graph = _get_graph()
except Exception:
    st.error(
        "MoviePilot failed to start. "
        "Make sure you have run `poetry run chatbot build-collection` first."
    )
    st.stop()

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎬 MoviePilot")
st.caption("Ask me about movies — trending picks, Netflix recommendations, or anything film-related.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("What would you like to watch?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(_stream_tokens(prompt, st.session_state.thread_id, graph))

    st.session_state.messages.append({"role": "assistant", "content": response})


# ---------------------------------------------------------------------------
# Entry point for `poetry run moviepilot-ui`
# ---------------------------------------------------------------------------

def run_ui() -> None:
    # Suppress Streamlit's first-run email prompt by ensuring credentials exist.
    credentials = Path.home() / ".streamlit" / "credentials.toml"
    if not credentials.exists():
        credentials.parent.mkdir(exist_ok=True)
        credentials.write_text('[general]\nemail = ""\n')

    sys.exit(
        subprocess.call([
            sys.executable, "-m", "streamlit", "run", str(Path(__file__)),
        ])
    )
