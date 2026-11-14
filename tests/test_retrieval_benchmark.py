import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from retrieval_benchmark import evaluate_retriever


class FakeRetriever:
    def search(self, query, *, filters, top_k):
        return [
            {"id": "noise", "text": "unrelated", "source_id": "doc", "source": "a.pdf",
             "page": 2, "metadata": {}, "score": 0.9},
            {"id": "evidence", "text": "the exact answer", "source_id": "doc",
             "source": "a.pdf", "page": 1, "metadata": {}, "score": 0.8},
        ][:top_k]


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_metrics_use_answerable_denominator_and_final_context(self):
        questions = [
            {
                "id": "q1", "split": "dev", "question": "What is the answer?",
                "answerable": True,
                "gold": [{"source_id": "doc", "page": 1, "evidence": "exact answer"}],
            },
            {
                "id": "q2", "split": "dev", "question": "What is absent?",
                "answerable": False,
                "expected_behavior": "search_and_abstain",
                "absence_reason": "Not present.",
            },
        ]
        traces, summary = evaluate_retriever(
            FakeRetriever(), questions, top_k=2, budget=1000, strategy="fake"
        )
        self.assertEqual(len(traces), 1)
        self.assertTrue(traces[0]["evidence_hit_at_3"])
        self.assertTrue(traces[0]["context_evidence_hit"])
        self.assertEqual(summary["answerable_n"], 1)
        self.assertEqual(summary["unanswerable_n"], 1)
        self.assertEqual(summary["evidence_hit_at_3"], 1)
        self.assertEqual(summary["context_evidence_hit"], 1)


if __name__ == "__main__":
    unittest.main()
