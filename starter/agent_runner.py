"""Run the real-model agent over one benchmark split and save auditable traces."""
import argparse
import json
import time
from pathlib import Path

from agentic import run_agent
from day6_retriever import Day6DenseRetriever
from openai_controller import OpenAICompatibleController


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--day6-index", type=Path, required=True)
    parser.add_argument("--strategy", choices=("A", "B"), default="B")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1600)
    parser.add_argument("--limit", type=int, help="Run only the first N questions in this split")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k < 1 or args.max_searches < 1 or args.budget < 1:
        parser.error("--top-k, --max-searches and --budget must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    selected = [question for question in questions if question["split"] == args.split]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError(f"No questions found for split {args.split}")
    retriever = Day6DenseRetriever.from_persistent_index(
        args.day6_index, strategy=args.strategy
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
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        traces.append(trace)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "split": args.split,
        "model": controller.model,
        "cases": len(cases),
        "provider_errors": sum(trace["status"] == "error" for trace in traces),
        "usage_tokens": controller.total_usage,
        "cost_usd": None,
        "cost_note": "Provider pricing is model-specific; calculate from token usage.",
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
