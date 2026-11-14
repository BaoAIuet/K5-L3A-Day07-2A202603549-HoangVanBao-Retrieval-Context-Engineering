import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
from prepare_day7_data import build_questions, validate_questions


class PrepareDay7DataTests(unittest.TestCase):
    def test_builds_minimum_splits_and_moves_seen_test_to_dev(self):
        raw = [
            {"id": "old-dev", "split": "dev", "question": "q1",
             "gold_source_id": "doc", "gold_page": 1, "evidence_quote": "evidence",
             "expected_answer": "answer"},
            {"id": "old-test", "split": "test", "question": "q2",
             "gold_source_id": "doc", "gold_page": 1, "evidence_quote": "evidence",
             "expected_answer": "answer"},
        ]
        questions = build_questions(raw)
        counts = {
            split: (
                sum(q.get("answerable", True) for q in questions if q["split"] == split),
                sum(q.get("answerable", True) is False for q in questions if q["split"] == split),
            )
            for split in ("dev", "test")
        }
        self.assertEqual(counts, {"dev": (2, 2), "test": (5, 2)})
        self.assertEqual(next(q for q in questions if q["id"] == "old-test")["split"], "dev")

    def test_validates_evidence_and_unanswerable_counts(self):
        questions = build_questions([])
        chunks = [{
            "id": "evidence-chunk", "text": "a decision needs to be made",
            "source_id": "principles-management", "source": "book.pdf", "page": 42,
            "metadata": {},
        }]
        with self.assertRaisesRegex(ValueError, "evidence not found"):
            validate_questions(chunks, questions)


if __name__ == "__main__":
    unittest.main()
