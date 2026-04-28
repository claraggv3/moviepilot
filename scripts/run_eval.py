"""
End-to-end evaluation harness.

Runs each golden case through the full graph and measures:
  1. Routing accuracy     — did the router pick the right agent?
  2. Retrieval recall     — do labeled relevant_titles appear in retrieved_context?
  3. L1 groundedness      — are all titles in the response present in retrieved_context?
  4. LLM judge (--judge)  — relevance / groundedness / helpfulness scored by the chat model

Usage:
    poetry run python scripts/run_eval.py
    poetry run python scripts/run_eval.py --judge          # adds LLM-as-judge (costs ~$1)
    poetry run python scripts/run_eval.py --cases 5        # smoke test on first 5 cases
    poetry run python scripts/run_eval.py --judge-only     # judge pass on existing results.json
    poetry run python scripts/run_eval.py --judge-only --cases 5  # judge only first 5

Results:
    eval/results.json   — full per-case data
    eval/summary.md     — aggregate table + notable snippets
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from moviepilot.config import configure_logging
from moviepilot.graph import build_graph
from moviepilot.llm.client import make_chat_model
from moviepilot.retrieval.chroma import load_collection

configure_logging()

GOLDEN_PATH  = Path("eval/golden.yaml")
RESULTS_PATH = Path("eval/results.json")
SUMMARY_PATH = Path("eval/summary.md")


_JUDGE_PROMPT = """\
You are evaluating a movie recommendation chatbot.

User query:
{query}

Retrieved context:
{context}

Response:
{response}

Evaluate on:

1. Relevance (1–5)
5 = directly answers query with appropriate content
3 = partially relevant or incomplete
1 = irrelevant

2. Groundedness (1–5)
5 = all claims supported by context
3 = some unsupported claims
1 = mostly unsupported or incorrect

List unsupported claims explicitly.

3. Helpfulness (1–5)
5 = clear, useful, well-structured
3 = somewhat useful
1 = not useful

Return JSON:
{{
  "relevance": int,
  "groundedness": int,
  "helpfulness": int,
  "unsupported_claims": list[str],
  "notes": str
}}
"""



# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    id: str
    query: str
    expected_route: str
    actual_route: str
    routing_correct: bool
    response: str
    retrieved_context: str
    # retrieval
    retrieved_relevant: list[str]
    retrieval_recall: float | None      # None when no relevance_rule; Hits@k / min(catalog_n, k)
    # groundedness
    context_titles: list[str]
    hallucinated_titles: list[str]
    grounded: bool
    # fields with defaults — must follow all non-default fields
    catalog_relevant_count: int | None = None  # total catalog docs matching the rule
    # judge (populated only with --judge)
    judge_runs: list[dict] = field(default_factory=list)
    judge_mean: dict       = field(default_factory=dict)
    judge_stdev: dict      = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Title parsing
# ---------------------------------------------------------------------------

# Netflix context line: "Inception (MOVIE, 2010) — Genres: ..."
_NETFLIX_TITLE_RE = re.compile(r'^(.+?)\s+\((?:MOVIE|SHOW),\s*\d{4}\)', re.MULTILINE)
# Trending context line: "1. **Inception** (2010) ★8.8 — ..."
_TRENDING_TITLE_RE = re.compile(r'\*\*(.+?)\*\*')
# Bold text in LLM responses: "1. **Inception (2010)** — ..." or "**The Dark Knight Rises**"
_RESPONSE_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
# Year suffix to strip from bolded text: " (2010)" at end of a string
_YEAR_SUFFIX_RE   = re.compile(r'\s*\(\d{4}\)\s*$')


def _parse_context_titles(context: str, route: str) -> list[str]:
    if route == "trending":
        return _TRENDING_TITLE_RE.findall(context)
    return _NETFLIX_TITLE_RE.findall(context)


def _extract_response_titles(response: str) -> list[str]:
    """
    Extract titles the LLM explicitly bolded in its response.

    The LLM consistently formats movie/show recommendations as **Title (Year)**
    or **Title**. Extracting only bold text avoids the false-positive explosion
    that comes from scanning the whole response for catalog title substrings.
    Year suffixes ("(2010)") are stripped so titles match the context exactly.
    Empty strings and single-character artifacts are discarded.
    """
    titles = []
    for raw in _RESPONSE_BOLD_RE.findall(response):
        title = _YEAR_SUFFIX_RE.sub("", raw).strip()
        if len(title) > 1:
            titles.append(title)
    return titles


def _find_hallucinations(response: str, context_titles: list[str]) -> list[str]:
    """
    Return titles the LLM explicitly recommended (bolded) that are absent from
    the retrieved context it was given.
    """
    context_lower = {t.lower() for t in context_titles}
    return [
        t for t in _extract_response_titles(response)
        if t.lower() not in context_lower
    ]


def _build_predicate(rule: dict):
    """
    Compile a relevance_rule dict into a callable (metadata: dict) -> bool.

    Supported forms:
      {field: directors, value: "Christopher Nolan"}  — substring contains
      {field: release_year, gte: 1980, lte: 1989}     — numeric range
      {all: [{...}, {...}]}                            — all sub-rules must match
    """
    if "all" in rule:
        subs = [_build_predicate(r) for r in rule["all"]]
        return lambda m, subs=subs: all(p(m) for p in subs)

    field = rule["field"]

    if "value" in rule:
        v = str(rule["value"]).lower()
        return lambda m, f=field, v=v: v in str(m.get(f, "")).lower()

    lo, hi = rule.get("gte"), rule.get("lte")

    def _range(m, f=field, lo=lo, hi=hi):
        val = m.get(f)
        if val is None:
            return False
        try:
            val = int(val)
        except (TypeError, ValueError):
            return False
        return (lo is None or val >= lo) and (hi is None or val <= hi)

    return _range


# ---------------------------------------------------------------------------
# Graph invocation
# ---------------------------------------------------------------------------

async def _run_case(graph: Any, query: str) -> tuple[str, str, str]:
    """Return (actual_route, response_text, retrieved_context)."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = await graph.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )
    route    = state.get("route") or "unknown"
    context  = state.get("retrieved_context") or ""
    response = state["messages"][-1].content
    return route, response, context


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

class _JudgeScore(BaseModel):
    relevance:     int
    groundedness:  int
    helpfulness:   int
    unsupported_claims: list[str]
    notes:         str = ""

async def _judge_once(judge_model: Any, case: dict, result: CaseResult) -> dict:
    prompt = _JUDGE_PROMPT.format(
        query=case["query"],
        context=result.retrieved_context[:8000],
        response=result.response,
    )
    structured = judge_model.with_structured_output(_JudgeScore)
    score = await structured.ainvoke([HumanMessage(content=prompt)])
    return {
        "relevance":    score.relevance,
        "groundedness": score.groundedness,
        "helpfulness":  score.helpfulness,
        "unsupported_claims": score.unsupported_claims,
        "notes":        score.notes,
    }


def _aggregate_runs(runs: list[dict]) -> tuple[dict, dict]:
    keys  = ["relevance", "groundedness", "helpfulness"]
    mean  = {k: round(statistics.mean(r[k]  for r in runs), 2) for k in keys}
    stdev = {k: round(statistics.stdev(r[k] for r in runs), 2) if len(runs) > 1 else 0.0 for k in keys}
    return mean, stdev


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge",      action="store_true", help="Run LLM-as-judge (3 runs/case)")
    parser.add_argument("--judge-only", action="store_true", help="Judge pass on existing results.json; skips graph")
    parser.add_argument("--cases",      type=int, default=None, help="Limit to first N cases")
    args = parser.parse_args()
    return args


async def _run_full(args: argparse.Namespace) -> None:
    cases: list[dict] = yaml.safe_load(GOLDEN_PATH.read_text())
    if args.cases:
        cases = cases[:args.cases]

    print("Building graph...")
    graph = build_graph(MemorySaver())

    print("Loading catalog...")
    col = load_collection()
    all_metadata: list[dict] = col.get(include=["metadatas"])["metadatas"]
    title_to_meta: dict[str, dict] = {m["title"].lower(): m for m in all_metadata if m.get("title")}
    print(f"  {len(title_to_meta)} titles loaded")

    judge_model = make_chat_model(streaming=False, temperature=0) if args.judge else None
    results: list[CaseResult] = []

    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case['id']}: {case['query'][:60]}")

        actual_route, response, context = await _run_case(graph, case["query"])
        routing_correct = actual_route == case["expected_route"]
        route_status = "✓" if routing_correct else f"✗  expected {case['expected_route']}"
        print(f"  route:  {actual_route} ({route_status})")

        context_titles = _parse_context_titles(context, actual_route)

        rule = case.get("relevance_rule")
        if rule and actual_route != "refusal":
            predicate     = _build_predicate(rule)
            catalog_count = sum(1 for m in all_metadata if predicate(m))
            retrieved_r   = [t for t in context_titles if predicate(title_to_meta.get(t.lower(), {}))]
            k_cap         = min(catalog_count, len(context_titles) or 1)
            recall        = round(len(retrieved_r) / k_cap, 3) if k_cap > 0 else 0.0
            print(f"  recall: {len(retrieved_r)}/{k_cap} (catalog has {catalog_count})")
        else:
            retrieved_r, recall, catalog_count = [], None, None

        if actual_route == "refusal":
            hallucinated, grounded = [], True
            print(f"  ground: ✓ (refusal — no retrieval)")
        else:
            hallucinated = _find_hallucinations(response, context_titles)
            grounded     = not hallucinated
            if hallucinated:
                print(f"  ground: ✗  hallucinated — {hallucinated[:3]}")
            else:
                print(f"  ground: ✓")

        result = CaseResult(
            id=case["id"],
            query=case["query"],
            expected_route=case["expected_route"],
            actual_route=actual_route,
            routing_correct=routing_correct,
            response=response,
            retrieved_context=context,
            retrieved_relevant=retrieved_r,
            retrieval_recall=recall,
            catalog_relevant_count=catalog_count,
            context_titles=context_titles,
            hallucinated_titles=hallucinated,
            grounded=grounded,
        )

        if args.judge and actual_route != "refusal":
            print(f"  judge:  running × 3 ...")
            runs = [await _judge_once(judge_model, case, result) for _ in range(3)]
            result.judge_runs  = runs
            result.judge_mean, result.judge_stdev = _aggregate_runs(runs)
            m = result.judge_mean
            print(f"          rel={m['relevance']} grd={m['groundedness']} hlp={m['helpfulness']}")

        results.append(result)

    _write_json(results)
    _write_summary(results, with_judge=args.judge)
    print(f"\nDone.  {RESULTS_PATH}  {SUMMARY_PATH}")


async def _run_judge_only(args: argparse.Namespace) -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found — run without --judge-only first to generate results."
        )

    print(f"Loading existing results from {RESULTS_PATH}...")
    raw: list[dict] = json.loads(RESULTS_PATH.read_text())
    if args.cases:
        raw = raw[:args.cases]
    results = [CaseResult(**{k: v for k, v in r.items() if k != "relevant_titles"}) for r in raw]
    print(f"  {len(results)} cases loaded")

    judge_model = make_chat_model(streaming=False, temperature=0)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}/{len(results)}] {result.id}: {result.query[:60]}")
        if result.actual_route == "refusal":
            print(f"  judge:  skipped (refusal)")
            continue
        print(f"  judge:  running × 3 ...")
        runs = [await _judge_once(judge_model, {"query": result.query}, result) for _ in range(3)]
        result.judge_runs  = runs
        result.judge_mean, result.judge_stdev = _aggregate_runs(runs)
        m = result.judge_mean
        print(f"          rel={m['relevance']} grd={m['groundedness']} hlp={m['helpfulness']}")

    _write_json(results)
    _write_summary(results, with_judge=True)
    print(f"\nDone.  {RESULTS_PATH}  {SUMMARY_PATH}")


async def main() -> None:
    args = _parse_args()
    if args.judge_only:
        await _run_judge_only(args)
    else:
        await _run_full(args)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_json(results: list[CaseResult]) -> None:
    Path("eval").mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps([asdict(r) for r in results], indent=2))


def _write_summary(results: list[CaseResult], with_judge: bool) -> None:
    n              = len(results)
    n_correct      = sum(1 for r in results if r.routing_correct)
    n_grounded     = sum(1 for r in results if r.grounded)
    recall_cases   = [r for r in results if r.retrieval_recall is not None]
    avg_recall     = statistics.mean(r.retrieval_recall for r in recall_cases) if recall_cases else None
    judge_cases    = [r for r in results if r.judge_mean]

    lines = [
        "# Eval summary",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"Cases run: {n}  ",
        "",
        "## Aggregate scores",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Routing accuracy | {n_correct}/{n} ({n_correct/n*100:.0f}%) |",
        f"| L1 groundedness  | {n_grounded}/{n} ({n_grounded/n*100:.0f}%) |",
    ]

    if avg_recall is not None:
        lines.append(f"| Retrieval recall (predicate-based) | {avg_recall*100:.1f}% |")

    if with_judge and judge_cases:
        for axis in ("relevance", "groundedness", "helpfulness"):
            avg = statistics.mean(r.judge_mean[axis] for r in judge_cases)
            sd  = statistics.mean(r.judge_stdev[axis] for r in judge_cases)
            lines.append(f"| LLM judge — {axis:<12} | {avg:.2f} ± {sd:.2f} / 5 |")

    # per-case table
    judge_header = " R / G / H |" if with_judge else ""
    judge_sep    = " --- |"       if with_judge else ""
    lines += [
        "",
        "## Per-case results",
        "",
        f"| ID | Query | Expected | Actual | Grounded | Recall |{judge_header}",
        f"|---|---|---|---|---|---|{judge_sep}",
    ]

    for r in results:
        if r.retrieval_recall is not None and r.catalog_relevant_count is not None:
            k_cap = min(r.catalog_relevant_count, len(r.context_titles) or 1)
            hits  = round(r.retrieval_recall * k_cap)
            recall_str = f"{r.retrieval_recall*100:.0f}% ({hits}/{k_cap})"
        else:
            recall_str = "—"
        ground_str = "✓" if r.grounded else f"✗ ({len(r.hallucinated_titles)})"
        route_ok   = "✓" if r.routing_correct else "✗"
        row = (
            f"| {r.id} | {r.query[:38]} | {r.expected_route} "
            f"| {r.actual_route} {route_ok} | {ground_str} | {recall_str} |"
        )
        if with_judge:
            m = r.judge_mean
            row += f" {m.get('relevance','—')}/{m.get('groundedness','—')}/{m.get('helpfulness','—')} |" if m else " — |"
        lines.append(row)

    # notable cases
    failures      = [r for r in results if not r.routing_correct]
    hallucinations = [r for r in results if not r.grounded]

    if failures or hallucinations:
        lines += ["", "## Notable cases", ""]

    for r in failures[:2]:
        lines += [
            f"### Routing miss — `{r.id}`",
            f"> {r.query}",
            f"",
            f"Expected `{r.expected_route}`, got `{r.actual_route}`.",
            f"",
            f"**Response:** {r.response[:400]}",
            "",
        ]

    for r in hallucinations[:2]:
        lines += [
            f"### Hallucination — `{r.id}`",
            f"> {r.query}",
            f"",
            f"Titles in response not found in context: `{r.hallucinated_titles}`",
            f"",
            f"**Response:** {r.response[:400]}",
            "",
        ]

    SUMMARY_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
