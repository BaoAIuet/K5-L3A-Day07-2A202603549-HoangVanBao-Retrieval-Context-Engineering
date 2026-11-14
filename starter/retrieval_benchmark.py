"""Benchmark TF-IDF, Day 6 dense, or hybrid retrieval on one split."""
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from baseline import TfidfIndex, hit, pack_context, validate
from day6_retriever import Day6DenseRetriever
from retrievers import HybridRetriever


def evaluate_retriever(
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    top_k: int = 3,
    budget: int = 800,
    strategy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces = []
    for question in questions:
        if question.get("answerable", True) is False:
            continue
        started = time.perf_counter()
        results = retriever.search(
            question["question"],
            filters=question.get("filter"),
            top_k=top_k,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        marked = [{**result, "evidence_hit": hit(result, question["gold"])} for result in results]
        context, context_ids = pack_context(results, budget=budget)
        traces.append({
            "query_id": question["id"],
            "split": question["split"],
            "strategy": strategy,
            "question": question["question"],
            "filter": question.get("filter"),
            "top_k": top_k,
            "context_budget_chars": budget,
            "results": marked,
            "context": context,
            "context_ids": context_ids,
            "hit_at_1": bool(marked and marked[0]["evidence_hit"]),
            "evidence_hit_at_3": any(result["evidence_hit"] for result in marked[:3]),
            "context_evidence_hit": any(
                result["evidence_hit"] and result["id"] in context_ids for result in marked
            ),
            "retrieval_ms": elapsed_ms,
        })

    n = len(traces)
    if not n:
        raise ValueError("No answerable questions in the selected split")
    summary = {
        "strategy": strategy,
        "split": questions[0]["split"],
        "answerable_n": n,
        "unanswerable_n": sum(
            question.get("answerable", True) is False for question in questions
        ),
        "hit_at_1": sum(trace["hit_at_1"] for trace in traces),
        "evidence_hit_at_3": sum(trace["evidence_hit_at_3"] for trace in traces),
        "context_evidence_hit": sum(trace["context_evidence_hit"] for trace in traces),
        "median_retrieval_ms": statistics.median(
            trace["retrieval_ms"] for trace in traces
        ),
    }
    return traces, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--method", choices=("tfidf", "dense", "hybrid"), required=True)
    parser.add_argument("--day6-index", type=Path)
    parser.add_argument("--strategy", choices=("A", "B"), default="B")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1 or args.budget < 1 or args.candidate_k < 1 or args.rrf_k < 1:
        parser.error("--top-k, --budget, --candidate-k and --rrf-k must be positive")

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [question for question in questions if question.get("split") == args.split]
    validate(chunks, selected)

    lexical = TfidfIndex(chunks) if args.method in ("tfidf", "hybrid") else None
    dense = None
    if args.method in ("dense", "hybrid"):
        if args.day6_index is None:
            parser.error("--day6-index is required for dense and hybrid methods")
        dense = Day6DenseRetriever.from_persistent_index(
            args.day6_index, strategy=args.strategy
        )

    if args.method == "tfidf":
        retriever = lexical
    elif args.method == "dense":
        retriever = dense
    else:
        retriever = HybridRetriever(
            dense, lexical, candidate_k=args.candidate_k, rrf_k=args.rrf_k
        )

    traces, summary = evaluate_retriever(
        retriever,
        selected,
        top_k=args.top_k,
        budget=args.budget,
        strategy=args.method,
    )
    summary.update({
        "candidate_k": args.candidate_k if args.method == "hybrid" else args.top_k,
        "rrf_k": args.rrf_k if args.method == "hybrid" else None,
        "context_budget_chars": args.budget,
        "service_cost_usd": 0,
    })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
