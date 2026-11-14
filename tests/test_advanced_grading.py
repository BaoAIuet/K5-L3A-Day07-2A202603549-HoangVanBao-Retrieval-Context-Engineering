import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission" / "advanced"))

from grade_results import grade


class AdvancedGradingTests(unittest.TestCase):
    def test_grades_only_selected_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            questions_path = temp / "questions.json"
            traces_path = temp / "traces.json"
            reviews_path = temp / "reviews.json"
            output_path = temp / "summary.json"
            questions_path.write_text(json.dumps([
                {
                    "id": "dev-q",
                    "split": "dev",
                    "answerable": False,
                    "primary_group": "unanswerable",
                },
                {
                    "id": "test-q",
                    "split": "test",
                    "answerable": False,
                    "primary_group": "unanswerable",
                },
            ]), encoding="utf-8")
            traces_path.write_text(json.dumps([
                {"query_id": "test-q", "status": "abstain", "elapsed_ms": 1}
            ]), encoding="utf-8")
            reviews_path.write_text(json.dumps([
                {
                    "query_id": "test-q",
                    "passed": True,
                    "abstention_correct": True,
                    "reason": "The answer abstained without unsupported claims.",
                }
            ]), encoding="utf-8")

            summary = grade(
                questions_path,
                traces_path,
                reviews_path,
                output_path,
                split="test",
            )

        self.assertEqual(summary["split"], "test")
        self.assertEqual(summary["n"], 1)
        self.assertEqual(summary["passed"], 1)


if __name__ == "__main__":
    unittest.main()
