# CinePilot

An agentic movie recommendation chatbot. Ask it anything — trending films, Netflix picks by genre/mood/director, or follow-up questions like "something shorter" — and it answers from real data without making up titles.

---

## Prerequisites

- **Python 3.11 or newer** — check with `python3 --version`
- **Poetry** — install with:
  ```bash
  curl -sSL https://install.python-poetry.org | python3 -
  ```
  Then restart your terminal so `poetry` is on your PATH.

---

## Setup (one-time)

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd cinepilot
poetry install
```

### 2. Get API keys

You need two:

- **OpenAI** — [platform.openai.com/api-keys](https://platform.openai.com/api-keys) — used for embeddings and chat responses
- **TMDB** — [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api) — free account required, used for trending data

### 3. Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...
TMDB_API_KEY=...
```

Everything else is optional and has sensible defaults.

### 4. Download the Netflix dataset

Download the **Netflix TV Shows and Movies** dataset from Kaggle:
[kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies)

You need two files from it: `titles.csv` and `credits.csv`. Place them here:

```
data/
└── kaggle/
    ├── titles.csv
    └── credits.csv
```

Create the folder if it doesn't exist:

```bash
mkdir -p data/kaggle
```

### 5. Build the vector index

This embeds the Netflix catalog into a local ChromaDB collection. It makes ~60 OpenAI API calls (batched) and takes about a minute.

```bash
poetry run chatbot build-collection
```

You only need to do this once. The index is saved to `data/chroma/`.

---

## Usage

```bash
poetry run chatbot chat
```

Type your question and press Enter. To exit, press `Ctrl-C`. Your thread ID is printed on exit — pass it with `--thread-id` to resume the conversation later:

```bash
poetry run chatbot chat --thread-id <your-thread-id>
```

### Example queries

- "What's trending this week?"
- "Recommend a Korean thriller"
- "Something like Parasite but lighter"
- "Show me David Fincher films on Netflix"
- "I want a short comedy for tonight"

---

## Running tests

```bash
poetry run pytest
```

Tests use mocks for all external APIs — no credentials needed.
