"""
Compute retrieval quality metrics from eval CSV files.

Compares strategies across queries where relevance is objectively
verifiable from metadata — no manual labeling needed.

Metrics shown per query:
  Hits@5   — number of relevant results in top 5  (integer, max 5)
  Hits@10  — number of relevant results in top 10 (integer, max 10)
  R@10     — Hits@10 / total relevant in corpus   (fraction, e.g. 3/6)
  1st      — rank of the first relevant result     (1 = best, 10 = worst)
             "-" means no relevant result found in top 10

MRR (Mean Reciprocal Rank) explanation:
  MRR = 1 / rank_of_first_relevant_result
  rank 1  → MRR 1.000   rank 2 → MRR 0.500   rank 5 → MRR 0.200
  rank 8  → MRR 0.125   rank 10 → MRR 0.100  not found → MRR 0.000
  Averaged across all queries: higher = first relevant result appears earlier.
  Shown here as the raw "1st rank" column — same information, more readable.

Usage:
    poetry run python scripts/eval_stats.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

RESULTS_DIR = Path("eval/results")

# ---------------------------------------------------------------------------
# Relevance rules
# ---------------------------------------------------------------------------

def _has_director(name: str) -> Callable:
    return lambda r: name in r.get("directors", "")

def _has_actor(name: str) -> Callable:
    return lambda r: name in r.get("actors", "")

def _has_country(code: str) -> Callable:
    return lambda r: code in r.get("production_countries", "")

def _in_decade(start: int, end: int) -> Callable:
    def rule(r: dict) -> bool:
        y = r.get("year", "")
        return bool(y) and start <= int(y) <= end
    return rule

def _has_genre(genre: str) -> Callable:
    return lambda r: genre.lower() in r.get("genres", "").lower()

def _has_genres(*genres: str) -> Callable:
    return lambda r: all(g.lower() in r.get("genres", "").lower() for g in genres)

def _in_decade_and_genre(start: int, end: int, genre: str) -> Callable:
    decade_rule = _in_decade(start, end)
    genre_rule = _has_genre(genre)
    return lambda r: decade_rule(r) and genre_rule(r)


RELEVANCE_RULES: dict[str, Callable] = {
    "David Fincher movies":                         _has_director("David Fincher"),
    "Christopher Nolan films":                      _has_director("Christopher Nolan"),
    "Bong Joon-ho films":                          _has_director("Bong Joon-ho"),
    "Quentin Tarantino movies":                     _has_director("Quentin Tarantino"),
    "movies with Tom Hanks":                        _has_actor("Tom Hanks"),
    "something with Cate Blanchett":                _has_actor("Cate Blanchett"),
    "movies with Meryl Streep":                     _has_actor("Meryl Streep"),
    "show with Pedro Pascal":                       _has_actor("Pedro Pascal"),
    "Korean drama":                                 _has_country("KR"),
    "a good French film":                           _has_country("FR"),
    "spy thriller with action by Martin Scorsese":  _has_director("Martin Scorsese"),
    "Christopher Nolan mind-bending thriller":       _has_director("Christopher Nolan"),
    "Tom Hanks drama about friendship":              _has_actor("Tom Hanks"),
    "Korean crime thriller like Parasite":           _has_country("KR"),
    "French romantic comedy":                        _has_country("FR"),
    "documentary about nature and wildlife":         _has_genre("documentation"),
    "80s horror film":                               _in_decade_and_genre(1978, 1992, "horror"),
    "classic films from the 80s":                   _in_decade(1980, 1989),
    "good comedies from the 90s":                   _in_decade_and_genre(1990, 1999, "comedy"),
    "spy thriller":                                  _has_genre("thriller"),
    "romantic comedy":                               _has_genres("comedy", "romance"),
}

SUBJECTIVE_QUERIES = {
    "something light and funny to cheer me up",
    "I need a good cry, what should I watch?",
    "something intense that'll keep me on edge",
    "movies about artificial intelligence",
    "a film about grief and loss",
    "survival in the wild",
    "films about the financial crisis",
    "documentaries about climate change",
    "something comforting to watch when I'm sick",
    "I want something mind-bending",
    "something to watch with my 8-year-old",
    "a movie for date night",
    "something I can have on in the background",
    "a long series I can binge this weekend",
    "just a mini-series, nothing too long",
    "if I liked Breaking Bad, what should I watch?",
    "movies in the same vein as The Godfather",
    "a comedy that isn't crude or raunchy",
    "action movie but nothing too gory",
    "something like Inception but shorter",
}


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus() -> list[dict[str, Any]]:
    from moviepilot.retrieval.chroma import load_collection
    from moviepilot.config import configure_logging
    configure_logging()
    col = load_collection()
    return col.get(include=["metadatas"])["metadatas"]


def build_title_lookup(corpus: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for meta in corpus:
        key = f"{meta.get('title', '')}|{meta.get('release_year', '')}"
        lookup[key] = meta
    return lookup


def count_relevant_in_corpus(rule: Callable, corpus: list[dict]) -> int:
    count = 0
    for meta in corpus:
        row = {
            "directors": meta.get("directors", ""),
            "actors": meta.get("actors", ""),
            "genres": meta.get("genres", ""),
            "production_countries": meta.get("production_countries", ""),
            "year": str(meta.get("release_year", "")),
        }
        try:
            if rule(row):
                count += 1
        except Exception:
            pass
    return count


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_results(label: str) -> dict[str, list[dict]]:
    path = RESULTS_DIR / f"{label}.csv"
    if not path.exists():
        return {}
    results: dict[str, list[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results[row["query"]].append(row)
    for q in results:
        results[q].sort(key=lambda r: int(r["rank"]))
    return results


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def enrich_row(row: dict, lookup: dict) -> dict:
    key = f"{row['title']}|{row['year']}"
    prod = lookup.get(key, {}).get("production_countries", "")
    return {**row, "production_countries": prod}


def hits_at_k(rows: list[dict], rule: Callable, k: int) -> int:
    return sum(1 for r in rows[:k] if rule(r))


def first_relevant_rank(rows: list[dict], rule: Callable) -> int | None:
    for r in rows:
        if rule(r):
            return int(r["rank"])
    return None


def mrr(first_rank: int | None) -> float:
    return 1.0 / first_rank if first_rank else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading corpus from ChromaDB...")
    corpus = load_corpus()
    lookup = build_title_lookup(corpus)
    print(f"  {len(corpus)} documents\n")

    corpus_counts = {q: count_relevant_in_corpus(r, corpus) for q, r in RELEVANCE_RULES.items()}

    labels = sorted(p.stem for p in RESULTS_DIR.glob("*.csv") if not p.stem.startswith("stats"))
    all_results = {lab: load_results(lab) for lab in labels}

    if not all_results:
        print("No eval CSVs found. Run eval_retrieval.py first.")
        return

    # -----------------------------------------------------------------------
    # Build per-query stats
    # -----------------------------------------------------------------------
    stat_rows: list[dict] = []

    for query in sorted(RELEVANCE_RULES):
        rule = RELEVANCE_RULES[query]
        corpus_n = corpus_counts[query]
        max5  = min(corpus_n, 5)
        max10 = min(corpus_n, 10)
        row: dict[str, Any] = {"query": query, "corpus_n": corpus_n, "max5": max5, "max10": max10}

        for lab in labels:
            rows = all_results[lab].get(query, [])
            enriched = [enrich_row(r, lookup) for r in rows]

            h5  = hits_at_k(enriched, rule, 5)
            h10 = hits_at_k(enriched, rule, 10)
            fr  = first_relevant_rank(enriched, rule)

            max5  = min(corpus_n, 5)
            max10 = min(corpus_n, 10)

            row[f"{lab}_hits@5"]  = h5
            row[f"{lab}_hits@10"] = h10
            row[f"{lab}_R@10"]    = f"{h10}/{max10}"
            row[f"{lab}_R@10_val"]= round(h10 / max10, 3) if max10 else 0.0
            row[f"{lab}_R@5_val"] = round(h5  / max5,  3) if max5  else 0.0
            row[f"{lab}_1st"]     = fr if fr else "-"
            row[f"{lab}_MRR"]     = round(mrr(fr), 3)

        stat_rows.append(row)

    # -----------------------------------------------------------------------
    # Print table
    # -----------------------------------------------------------------------
    W = 46  # query column width
    col_w = 28  # per-label column group width

    # Header
    print(f"{'Query':<{W}} {'Corpus':>7}  " +
          "  ".join(f"{'── ' + lab.upper() + ' ──':^{col_w}}" for lab in labels))
    print(f"{'':^{W}} {'N':>7}  " +
          "  ".join(f"{'Hits@5':>6} {'Hits@10':>7} {'R@10':>8} {'1st rank':>8}" for _ in labels))
    print("-" * (W + 9 + len(labels) * (col_w + 2)))

    totals: dict[str, list] = defaultdict(list)

    for r in stat_rows:
        line = f"{r['query'][:W-1]:<{W}} {r['corpus_n']:>7}  "
        for lab in labels:
            h5  = r[f"{lab}_hits@5"]
            h10 = r[f"{lab}_hits@10"]
            rc  = r[f"{lab}_R@10"]
            fr  = r[f"{lab}_1st"]
            line += f"{h5:>6} {h10:>7} {rc:>8} {str(fr):>8}  "
            totals[f"{lab}_hits@5"].append(h5)
            totals[f"{lab}_hits@10"].append(h10)
            totals[f"{lab}_R@5"].append(r[f"{lab}_R@5_val"])
            totals[f"{lab}_R@10"].append(r[f"{lab}_R@10_val"])
            totals[f"{lab}_MRR"].append(r[f"{lab}_MRR"])
            if isinstance(fr, int):
                totals[f"{lab}_1st"].append(fr)
        print(line)

    print("-" * (W + 9 + len(labels) * (col_w + 2)))

    # Overall row
    n = len(stat_rows)
    total_line = f"{'TOTAL / MEAN (across ' + str(n) + ' queries)':<{W}} {'':>7}  "
    for lab in labels:
        sum_h5  = sum(totals[f"{lab}_hits@5"])
        sum_h10 = sum(totals[f"{lab}_hits@10"])
        avg_r   = round(sum(totals[f"{lab}_R@10"]) / n, 3)
        avg_1st_vals = totals[f"{lab}_1st"]
        avg_1st = round(sum(avg_1st_vals) / len(avg_1st_vals), 1) if avg_1st_vals else "-"
        total_line += f"{sum_h5:>6} {sum_h10:>7} {str(avg_r):>8} {str(avg_1st):>8}  "
    print(total_line)

    print(f"\n  Hits@5/10 = absolute count of relevant results in top 5 / top 10")
    print(f"  R@10      = Hits@10 / total relevant titles in full catalog")
    print(f"  1st rank  = position of first relevant result (lower = better, '-' = not found)")
    print(f"\n  {len(SUBJECTIVE_QUERIES)} queries excluded (mood/topic/format/similarity/negative) — need manual labeling")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * (W + 9 + len(labels) * (col_w + 2)))
    print("SUMMARY")
    print("=" * (W + 9 + len(labels) * (col_w + 2)))

    metric_w = 42
    val_w = 12

    # Header
    print(f"\n  {'Metric':<{metric_w}}" + "".join(f"{lab.upper():>{val_w}}" for lab in labels))
    print(f"  {'-' * metric_w}" + "".join(f"  {'─' * (val_w - 2)}" for _ in labels))

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    total_max5  = sum(r["max5"]  for r in stat_rows)
    total_max10 = sum(r["max10"] for r in stat_rows)

    summary_rows = []
    for lab in labels:
        vals_h5  = totals[f"{lab}_hits@5"]
        vals_h10 = totals[f"{lab}_hits@10"]
        vals_mrr = totals[f"{lab}_MRR"]
        vals_1st = totals[f"{lab}_1st"]
        total_h5  = sum(vals_h5)
        total_h10 = sum(vals_h10)
        summary_rows.append({
            "label":         lab,
            "total_hits5":   total_h5,
            "total_hits10":  total_h10,
            "total_max5":    total_max5,
            "total_max10":   total_max10,
            "recall5":       total_h5  / total_max5  if total_max5  else 0.0,
            "recall10":      total_h10 / total_max10 if total_max10 else 0.0,
            "mean_mrr":      sum(vals_mrr) / n,
            "mean_1st":      round(sum(vals_1st) / len(vals_1st), 1) if vals_1st else None,
        })

    rows_to_print = [
        ("Total Hits@5  / max possible",   lambda s: f"{s['total_hits5']} / {s['total_max5']}"),
        ("Total Hits@10 / max possible",   lambda s: f"{s['total_hits10']} / {s['total_max10']}"),
        ("Recall@5   (hits@5 / max@5)",    lambda s: _pct(s["recall5"])),
        ("Recall@10  (hits@10 / max@10)",  lambda s: _pct(s["recall10"])),
        ("Mean MRR   (avg 1 / 1st rank)",  lambda s: f"{s['mean_mrr']:.3f}"),
        ("Mean 1st Rank  (lower = better)",lambda s: str(s["mean_1st"]) if s["mean_1st"] else "-"),
    ]

    for label_text, fn in rows_to_print:
        print(f"  {label_text:<{metric_w}}" + "".join(f"{fn(s):>{val_w}}" for s in summary_rows))

    print()

    # -----------------------------------------------------------------------
    # Save summary CSV
    # -----------------------------------------------------------------------
    summary_path = RESULTS_DIR / "stats_summary.csv"
    summary_fields = ["metric"] + labels
    summary_csv_rows = [
        {"metric": "total_hits@5",   **{s["label"]: f"{s['total_hits5']}/{s['total_max5']}"   for s in summary_rows}},
        {"metric": "total_hits@10",  **{s["label"]: f"{s['total_hits10']}/{s['total_max10']}" for s in summary_rows}},
        {"metric": "recall@5_%",     **{s["label"]: round(s["recall5"]  * 100, 1)             for s in summary_rows}},
        {"metric": "recall@10_%",    **{s["label"]: round(s["recall10"] * 100, 1)             for s in summary_rows}},
        {"metric": "mean_mrr",       **{s["label"]: round(s["mean_mrr"], 3)                   for s in summary_rows}},
        {"metric": "mean_1st_rank",  **{s["label"]: s["mean_1st"] if s["mean_1st"] else "-"  for s in summary_rows}},
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_csv_rows)
    print(f"Saved → {summary_path}")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    out_path = RESULTS_DIR / "stats.csv"
    csv_fields = ["query", "corpus_n"] + [
        f"{lab}_{m}" for lab in labels for m in ["hits@5", "hits@10", "R@10", "1st", "MRR"]
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stat_rows)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
