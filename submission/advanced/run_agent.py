"""Run the advanced OpenStax benchmark with a text-and-visual-caption TF-IDF index."""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))

from agentic import run_agent  # noqa: E402
from baseline import TfidfIndex  # noqa: E402
from openai_controller import OpenAICompatibleController  # noqa: E402


_REFERENCE_RE = re.compile(
    r"\b(?P<kind>tables?|figures?)\s+"
    r"(?P<labels>\d+(?:\.\d+)?(?:\s*(?:,|and|&)\s*\d+(?:\.\d+)?)*\b)",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(r"\d+(?:\.\d+)?")
_TABLE_LABEL_RE = re.compile(r"\bTable\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_KEY_TERMS_RE = re.compile(r"\bkey\s+terms\b", re.IGNORECASE)


USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_prompt_tokens",
)


class ReferenceAwareTfidfIndex(TfidfIndex):
    """Promote indexed table or figure chunks explicitly named by a search."""

    def search(self, question, top_k=3, filters=None):
        references = set()
        for match in _REFERENCE_RE.finditer(question):
            kind = match.group("kind").casefold().rstrip("s")
            references.update(
                (kind, label) for label in _LABEL_RE.findall(match.group("labels"))
            )
        key_terms_requested = bool(_KEY_TERMS_RE.search(question))
        if not references and not key_terms_requested:
            return super().search(question, top_k=top_k, filters=filters)

        candidates = super().search(
            question,
            top_k=len(self.chunks),
            filters=filters,
        )
        for candidate in candidates:
            metadata = candidate["metadata"]
            content_type = metadata.get("content_type")
            labels = set()
            if content_type == "table":
                labels.update(_TABLE_LABEL_RE.findall(candidate["text"]))
            elif content_type == "visual_caption":
                figure_label = metadata.get("figure_label")
                if isinstance(figure_label, str):
                    labels.add(figure_label)
            kind = "table" if content_type == "table" else "figure"
            if any((kind, label) in references for label in labels):
                candidate["score"] += 2.0
            if (
                key_terms_requested
                and content_type == "text"
                and _KEY_TERMS_RE.search(candidate["text"])
            ):
                candidate["score"] += 2.0
        return sorted(
            candidates,
            key=lambda candidate: (-candidate["score"], candidate["id"]),
        )[:top_k]


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
    return (
        (prompt_tokens - cached_tokens) * input_rate
        + cached_tokens * cached_input_rate
        + usage.get("completion_tokens", 0) * output_rate
    ) / 1_000_000


def run_split(
    questions_path: Path,
    chunks_path: Path,
    split: str,
    output: Path,
    *,
    top_k: int,
    max_searches: int,
    budget: int,
    input_rate: float,
    output_rate: float,
    cached_input_rate: float,
) -> dict:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    selected = [item for item in questions if item.get("split") == split]
    if not selected:
        raise ValueError(f"No questions found for split {split}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    retriever = ReferenceAwareTfidfIndex(chunks)
    controller = OpenAICompatibleController()
    cases = []
    traces = []

    for item in selected:
        query_id = item["id"]
        cases.append({
            "id": query_id,
            "split": split,
            "answerable": item["answerable"],
            "primary_group": item["primary_group"],
        })
        usage_before = dict(controller.total_usage)
        started = time.perf_counter()
        try:
            result = run_agent(
                question=item["question"],
                controller=controller,
                retriever=retriever,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget,
            )
            trace = {
                "query_id": query_id,
                "question": item["question"],
                **result,
            }
        except (RuntimeError, ValueError, OSError) as exc:
            trace = {
                "query_id": query_id,
                "question": item["question"],
                "status": "error",
                "error": str(exc),
                "events": [],
                "citations": [],
            }
        trace["elapsed_ms"] = (time.perf_counter() - started) * 1000
        trace["usage_tokens"] = {
            key: controller.total_usage.get(key, 0) - usage_before.get(key, 0)
            for key in USAGE_KEYS
        }
        trace["estimated_cost_usd"] = estimate_cost_usd(
            trace["usage_tokens"],
            input_rate=input_rate,
            output_rate=output_rate,
            cached_input_rate=cached_input_rate,
        )
        traces.append(trace)

    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "split": split,
        "model": controller.model,
        "method": (
            "TF-IDF over page text, structured tables, figure captions, and "
            "indexed visual descriptions, with exact named table/figure boosts"
        ),
        "top_k": top_k,
        "max_searches": max_searches,
        "context_budget_chars": budget,
        "cases": len(cases),
        "provider_errors": sum(trace["status"] == "error" for trace in traces),
        "usage_tokens": controller.total_usage,
        "estimated_cost_usd": estimate_cost_usd(
            controller.total_usage,
            input_rate=input_rate,
            output_rate=output_rate,
            cached_input_rate=cached_input_rate,
        ),
        "price_per_million_tokens": {
            "input": input_rate,
            "output": output_rate,
            "cached_input": cached_input_rate,
        },
        "elapsed_ms_total": sum(trace["elapsed_ms"] for trace in traces),
        "output": str(output),
    }
    (output / "run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--budget", type=int, default=7000)
    parser.add_argument("--input-price-per-million", type=float, default=0.15)
    parser.add_argument("--output-price-per-million", type=float, default=0.60)
    parser.add_argument("--cached-input-price-per-million", type=float, default=0.075)
    args = parser.parse_args()
    if min(args.top_k, args.max_searches, args.budget) < 1:
        parser.error("--top-k, --max-searches, and --budget must be positive")
    prices = (
        args.input_price_per_million,
        args.output_price_per_million,
        args.cached_input_price_per_million,
    )
    if any(price < 0 for price in prices):
        parser.error("token prices must be non-negative")
    if args.cached_input_price_per_million > args.input_price_per_million:
        parser.error("cached input price cannot exceed input price")

    summary = run_split(
        args.questions,
        args.chunks,
        args.split,
        args.output,
        top_k=args.top_k,
        max_searches=args.max_searches,
        budget=args.budget,
        input_rate=args.input_price_per_million,
        output_rate=args.output_price_per_million,
        cached_input_rate=args.cached_input_price_per_million,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
