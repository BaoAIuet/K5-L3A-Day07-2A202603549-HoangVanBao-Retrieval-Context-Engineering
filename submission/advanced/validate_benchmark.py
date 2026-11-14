"""Validate the advanced split sizes, category coverage, and gold evidence."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "starter"))

from baseline import validate  # noqa: E402


def _validate_multi_hop(item: dict) -> None:
    locations = set()
    for evidence in item["gold"]:
        reference = evidence.get("evidence_ref")
        if reference is not None and not isinstance(reference, str):
            raise ValueError(f"{item['id']}: evidence_ref must be a string")
        locations.add((evidence["page"], reference if reference else "page"))
    if len(locations) < 2:
        raise ValueError(
            f"{item['id']}: multi-hop gold must cite at least 2 pages, tables, or figures"
        )


def validate_benchmark(questions_path: Path, chunks_path: Path) -> dict:
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("benchmark must be a JSON array")
    for question in questions:
        if not isinstance(question.get("primary_group"), str):
            raise ValueError(f"{question.get('id')}: primary_group is required")
        if question.get("answerable") is True and question["primary_group"] == "unanswerable":
            raise ValueError(f"{question['id']}: answerable item cannot be in unanswerable group")
        if question.get("answerable") is False and question["primary_group"] != "unanswerable":
            raise ValueError(f"{question['id']}: unanswerable item must use unanswerable group")

    for split in ("dev", "test"):
        selected = [item for item in questions if item.get("split") == split]
        if not selected:
            raise ValueError(f"{split} split is empty")
        validate(chunks, selected)
        answerable = [item for item in selected if item["answerable"]]
        unanswerable = [item for item in selected if not item["answerable"]]
        if split == "dev":
            if len(answerable) < 15 or len(unanswerable) < 3:
                raise ValueError("dev requires at least 15 answerable and 3 unanswerable questions")
        else:
            if len(selected) < 30 or len(answerable) < 24 or len(unanswerable) < 6:
                raise ValueError("test requires 30 total, 24 answerable, and 6 unanswerable questions")
            counts = Counter(item["primary_group"] for item in answerable)
            minimums = {
                "table": 6,
                "image_diagram": 5,
                "chart": 5,
                "multi_hop": 4,
                "text": 4,
            }
            for group, minimum in minimums.items():
                if counts[group] < minimum:
                    raise ValueError(f"test needs at least {minimum} {group} questions")
        for item in answerable:
            if item["primary_group"] == "multi_hop":
                _validate_multi_hop(item)
        print({
            "split": split,
            "total": len(selected),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "answerable_primary_groups": dict(Counter(
                item["primary_group"] for item in answerable
            )),
        })
    return {"questions": len(questions), "chunks": len(chunks)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_benchmark(args.questions, args.chunks), indent=2))


if __name__ == "__main__":
    main()
