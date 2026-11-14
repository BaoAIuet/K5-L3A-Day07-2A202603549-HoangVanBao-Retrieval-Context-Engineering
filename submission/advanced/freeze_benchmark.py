"""Write a reproducible manifest for the current dev/test question splits."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = ROOT / "submission" / "advanced" / "questions.json"
DEFAULT_STATS = ROOT / "submission" / "advanced" / "data" / "source_stats.json"
DEFAULT_OUTPUT = ROOT / "submission" / "advanced" / "benchmark_manifest.json"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_manifest(questions_path: Path, stats_path: Path) -> dict:
    questions_bytes = questions_path.read_bytes()
    questions = json.loads(questions_bytes.decode("utf-8"))
    if not isinstance(questions, list):
        raise ValueError("questions must be a JSON array")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    splits = {
        split: [item for item in questions if item.get("split") == split]
        for split in ("dev", "test")
    }
    if any(not items for items in splits.values()):
        raise ValueError("both dev and test splits must be non-empty")
    return {
        "hash_algorithm": "SHA-256",
        "questions_file_sha256": _sha256(questions_bytes),
        "source_pdf_sha256": stats["pdf_sha256"],
        "splits": {
            split: {
                "count": len(items),
                "canonical_questions_sha256": _sha256(
                    json.dumps(
                        items,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
            }
            for split, items in splits.items()
        },
        "freeze_policy": (
            "Create after dev tuning and before the held-out test run. "
            "Do not tune against test results."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--source-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest(args.questions, args.source_stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
