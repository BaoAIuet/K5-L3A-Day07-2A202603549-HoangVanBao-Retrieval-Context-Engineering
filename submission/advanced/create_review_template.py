"""Create a blank, one-review-per-query template for manual grading."""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = ROOT / "submission" / "advanced" / "questions.json"


def create_review_template(questions_path: Path, split: str) -> list[dict]:
    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test'")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    selected = [item for item in questions if item.get("split") == split]
    if not selected:
        raise ValueError(f"No questions found for split {split}")
    reviews = []
    for question in selected:
        review = {
            "query_id": question["id"],
            "passed": False,
            "reason": "Pending manual comparison with the cited source page(s).",
        }
        if question["answerable"]:
            review.update({
                "answer_correct": False,
                "citation_supported": False,
                "evidence_verified": [
                    {"pdf_page": evidence["page"], "verified": False}
                    for evidence in question["gold"]
                ],
            })
        else:
            review["abstention_correct"] = False
        reviews.append(review)
    return reviews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reviews = create_review_template(args.questions, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(reviews, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(reviews)} review entries to {args.output}")


if __name__ == "__main__":
    main()
