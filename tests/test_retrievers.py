import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from retrievers import HybridRetriever


class FakeRetriever:
    def __init__(self, rows, strategy):
        self.rows = rows
        self.strategy = strategy
        self.calls = []

    def search(self, query, *, filters, top_k):
        self.calls.append((query, filters, top_k))
        return [
            {**row, "strategy": self.strategy}
            for row in self.rows[:top_k]
        ]


class HybridRetrieverTests(unittest.TestCase):
    def test_rrf_promotes_chunks_found_by_both_retrievers(self):
        dense = FakeRetriever([
            {"id": "a", "text": "dense first"},
            {"id": "b", "text": "shared"},
        ], "dense")
        lexical = FakeRetriever([
            {"id": "b", "text": "shared"},
            {"id": "c", "text": "lexical second"},
        ], "tfidf")
        hybrid = HybridRetriever(dense, lexical, candidate_k=3, rrf_k=10)

        rows = hybrid.search("query", filters={"chapter": 2}, top_k=3)

        self.assertEqual([row["id"] for row in rows], ["b", "a", "c"])
        self.assertEqual(rows[0]["retrieval_sources"], ["dense", "tfidf"])
        self.assertEqual(dense.calls, [("query", {"chapter": 2}, 3)])
        self.assertEqual(lexical.calls, [("query", {"chapter": 2}, 3)])

    def test_rejects_non_positive_fusion_parameters(self):
        with self.assertRaises(ValueError):
            HybridRetriever(FakeRetriever([], "dense"), FakeRetriever([], "tfidf"), candidate_k=0)


if __name__ == "__main__":
    unittest.main()
