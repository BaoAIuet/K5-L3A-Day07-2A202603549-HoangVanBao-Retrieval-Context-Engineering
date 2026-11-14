"""Run the real-model agent over one benchmark split and save auditable traces."""
import argparse
import json
import time
from pathlib import Path

from agentic import run_agent
from baseline import TfidfIndex
from day6_retriever import Day6DenseRetriever
from openai_controller import OpenAICompatibleController
from retrievers import HybridRetriever


_USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_prompt_tokens",
)


def build_retriever(
    *,
    method: str,
    index_path: Path,
    strategy: str,
    chunks_path: Path | None,
    candidate_k: int,
    rrf_k: int,
) -> Day6DenseRetriever | HybridRetriever:
    dense = Day6DenseRetriever.from_persistent_index(
        index_path, strategy=strategy
    )
    if method == "dense":
        return dense
    if chunks_path is None:
        raise ValueError("--chunks is required for hybrid retrieval")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return HybridRetriever(
        dense,
        TfidfIndex(chunks),
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )


def estimate_cost_usd(
    usage: dict[str, int],
    *,
    input_rate: float,
    output_rate: float,
    cached_input_rate: float,
) -> float:
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached_tokens = usage.get("cached_prompt_tokens", 0)
    if cached_tokens > prompt_tokens:
        raise ValueError("cached prompt tokens cannot exceed prompt tokens")
    input_cost = (
        (prompt_tokens - cached_tokens) * input_rate
        + cached_tokens * cached_input_rate
    )
    output_cost = usage.get("completion_tokens", 0) * output_rate
    return (input_cost + output_cost) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--day6-index", type=Path, required=True)
    parser.add_argument("--strategy", choices=("A", "B"), default="B")
    parser.add_argument("--method", choices=("dense", "hybrid"), default="dense")
    parser.add_argument("--chunks", type=Path, help="Day 7-schema chunk JSON; required for hybrid")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1600)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--cached-input-price-per-million", type=float)
    parser.add_argument("--limit", type=int, help="Run only the first N questions in this split")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.top_k, args.max_searches, args.budget, args.candidate_k, args.rrf_k) < 1:
        parser.error("--top-k, --max-searches, --budget, --candidate-k and --rrf-k must be positive")
    if args.method == "hybrid" and args.chunks is None:
        parser.error("--chunks is required when --method is hybrid")
    prices = (
        args.input_price_per_million,
        args.output_price_per_million,
        args.cached_input_price_per_million,
    )
    if any(price is not None for price in prices) and any(price is None for price in prices):
        parser.error("provide all three token prices together")
    if any(price is not None and price < 0 for price in prices):
        parser.error("token prices must be non-negative")
    if (
        args.cached_input_price_per_million is not None
        and args.cached_input_price_per_million > args.input_price_per_million
    ):
        parser.error("cached input price cannot exceed input price")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [question for question in questions if question["split"] == args.split]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError(f"No questions found for split {args.split}")
    retriever = build_retriever(
        method=args.method,
        index_path=args.day6_index,
        strategy=args.strategy,
        chunks_path=args.chunks,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
    )
    controller = OpenAICompatibleController()
    cases = []
    traces = []
    for question in selected:
        query_id = question["id"]
        cases.append({
            "id": query_id,
            "split": args.split,
            "answerable": question.get("answerable", True),
        })
        usage_before = dict(controller.total_usage)
        started = time.perf_counter()
        try:
            result = run_agent(
                question=question["question"],
                controller=controller,
                retriever=retriever,
                filters=question.get("filter"),
                max_searches=args.max_searches,
                top_k=args.top_k,
                budget=args.budget,
            )
            trace = {
                "query_id": query_id,
                "split": args.split,
                "question": question["question"],
                **result,
            }
        except (RuntimeError, ValueError, OSError) as exc:
            trace = {
                "query_id": query_id,
                "split": args.split,
                "question": question["question"],
                "status": "error",
                "error": str(exc),
                "events": [],
                "citations": [],
            }
        trace["elapsed_ms"] = (time.perf_counter() - started) * 1000
        trace["usage_tokens"] = {
            key: controller.total_usage.get(key, 0) - usage_before.get(key, 0)
            for key in _USAGE_KEYS
        }
        if (
            args.input_price_per_million is not None
            and args.output_price_per_million is not None
            and args.cached_input_price_per_million is not None
        ):
            trace["estimated_cost_usd"] = estimate_cost_usd(
                trace["usage_tokens"],
                input_rate=args.input_price_per_million,
                output_rate=args.output_price_per_million,
                cached_input_rate=args.cached_input_price_per_million,
            )
        traces.append(trace)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    estimated_cost = None
    if (
        args.input_price_per_million is not None
        and args.output_price_per_million is not None
        and args.cached_input_price_per_million is not None
    ):
        estimated_cost = estimate_cost_usd(
            controller.total_usage,
            input_rate=args.input_price_per_million,
            output_rate=args.output_price_per_million,
            cached_input_rate=args.cached_input_price_per_million,
        )
    summary = {
        "split": args.split,
        "model": controller.model,
        "method": args.method,
        "strategy": args.strategy if args.method in ("dense", "hybrid") else None,
        "candidate_k": args.candidate_k if args.method == "hybrid" else None,
        "rrf_k": args.rrf_k if args.method == "hybrid" else None,
        "top_k": args.top_k,
        "max_searches": args.max_searches,
        "context_budget_chars": args.budget,
        "cases": len(cases),
        "provider_errors": sum(trace["status"] == "error" for trace in traces),
        "usage_tokens": controller.total_usage,
        "estimated_cost_usd": estimated_cost,
        "price_per_million_tokens": (
            {
                "input": args.input_price_per_million,
                "output": args.output_price_per_million,
                "cached_input": args.cached_input_price_per_million,
            }
            if estimated_cost is not None
            else None
        ),
        "cost_note": (
            "Estimate based on supplied standard token rates; check provider billing for actual charges."
            if estimated_cost is not None
            else "Pass all three token price flags to calculate an estimate."
        ),
        "output": str(args.output),
        "elapsed_ms_total": sum(trace["elapsed_ms"] for trace in traces),
    }
    (args.output / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
