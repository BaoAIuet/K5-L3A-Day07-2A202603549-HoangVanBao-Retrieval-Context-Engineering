"""Aggregate human-verified, all-or-nothing advanced benchmark reviews."""
import argparse
import json
import statistics
from pathlib import Path


def _read_array(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    return value


def grade(
    questions_path: Path,
    traces_path: Path,
    reviews_path: Path,
    output_path: Path,
    *,
    split: str | None = None,
) -> dict:
    questions = _read_array(questions_path)
    if split is not None:
        if split not in ("dev", "test"):
            raise ValueError("split must be 'dev' or 'test'")
        questions = [item for item in questions if item.get("split") == split]
        if not questions:
            raise ValueError(f"No questions found for split {split}")
    traces = _read_array(traces_path)
    reviews = _read_array(reviews_path)
    question_by_id = {item["id"]: item for item in questions}
    trace_by_id = {item["query_id"]: item for item in traces}
    review_by_id = {item["query_id"]: item for item in reviews}
    expected_ids = set(question_by_id)
    if len(question_by_id) != len(questions):
        raise ValueError("benchmark question IDs must be unique")
    if len(trace_by_id) != len(traces):
        raise ValueError("trace query IDs must be unique")
    if len(review_by_id) != len(reviews):
        raise ValueError("review query IDs must be unique")
    if set(trace_by_id) != expected_ids:
        raise ValueError("every benchmark case must have exactly one trace")
    if set(review_by_id) != expected_ids:
        raise ValueError("every benchmark case, including errors, must have one manual review")

    for query_id, question in question_by_id.items():
        review = review_by_id[query_id]
        if not isinstance(review.get("passed"), bool) or not review.get("reason"):
            raise ValueError(f"{query_id}: review needs a boolean pass and a reason")
        trace = trace_by_id[query_id]
        if trace.get("status") == "error" and review["passed"]:
            raise ValueError(f"{query_id}: a system error cannot pass")
        if question["answerable"]:
            checks = review.get("evidence_verified")
            if not isinstance(checks, list) or len(checks) != len(question["gold"]):
                raise ValueError(f"{query_id}: verify each gold evidence item")
            if any(
                check.get("verified") is not True
                or check.get("pdf_page") != gold["page"]
                for check, gold in zip(checks, question["gold"])
            ):
                raise ValueError(f"{query_id}: evidence review does not match gold pages")
            should_pass = (
                review.get("answer_correct") is True
                and review.get("citation_supported") is True
                and all(check["verified"] for check in checks)
                and trace.get("status") == "answered"
            )
        else:
            if not isinstance(review.get("abstention_correct"), bool):
                raise ValueError(f"{query_id}: review abstention_correct")
            should_pass = (
                review["abstention_correct"]
                and trace.get("status") == "abstain"
            )
        if review["passed"] != should_pass:
            raise ValueError(f"{query_id}: pass/fail conflicts with review details")

    groups = {}
    for query_id, question in question_by_id.items():
        group = question["primary_group"]
        counts = groups.setdefault(group, {"passed": 0, "total": 0})
        counts["total"] += 1
        counts["passed"] += int(review_by_id[query_id]["passed"])
    for counts in groups.values():
        counts["rate"] = counts["passed"] / counts["total"]

    latencies = [
        trace["elapsed_ms"]
        for trace in traces
        if isinstance(trace.get("elapsed_ms"), (int, float))
    ]
    errors = sum(trace.get("status") == "error" for trace in traces)
    passed = sum(review["passed"] for review in reviews)
    n = len(questions)
    summary = {
        "split": split,
        "n": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": passed / n if n else 0,
        "pass_rate_percent": 100 * passed / n if n else 0,
        "groups": groups,
        "system_errors": errors,
        "latency_ms": {
            "median": statistics.median(latencies) if latencies else None,
            "total": sum(latencies),
            "queries_with_latency": len(latencies),
        },
        "estimated_cost_usd": sum(
            trace.get("estimated_cost_usd", 0) for trace in traces
        ),
        "review_method": "manual comparison with cited PDF pages/regions; all-or-nothing",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"))
    args = parser.parse_args()
    print(json.dumps(
        grade(
            args.questions,
            args.traces,
            args.reviews,
            args.output,
            split=args.split,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
