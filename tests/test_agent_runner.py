import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from agent_runner import build_retriever, estimate_cost_usd
from retrievers import HybridRetriever


class FakeDenseRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.requested_top_k = None

    def search(self, query, *, filters=None, top_k=3):
        self.requested_top_k = top_k
        return self.chunks[:top_k]


class AgentRunnerTests(unittest.TestCase):
    def test_builds_hybrid_retriever_from_dense_and_local_chunks(self):
        chunk = {
            "id": "chunk-1",
            "text": "Apples grow in an orchard.",
            "source_id": "book",
            "source": "https://example.test/book",
            "page": 1,
            "metadata": {},
        }
        dense = FakeDenseRetriever([chunk])
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks_path = Path(temp_dir) / "chunks.json"
            chunks_path.write_text(json.dumps([chunk]), encoding="utf-8")
            with patch(
                "agent_runner.Day6DenseRetriever.from_persistent_index",
                return_value=dense,
            ):
                retriever = build_retriever(
                    method="hybrid",
                    index_path=Path(temp_dir),
                    strategy="B",
                    chunks_path=chunks_path,
                    candidate_k=7,
                    rrf_k=60,
                )

        self.assertIsInstance(retriever, HybridRetriever)
        hits = retriever.search("apples", top_k=1)
        self.assertEqual([hit["id"] for hit in hits], ["chunk-1"])
        self.assertEqual(dense.requested_top_k, 7)
        self.assertEqual(hits[0]["strategy"], "hybrid_rrf")

    def test_estimates_standard_and_cached_input_cost(self):
        cost = estimate_cost_usd(
            {
                "prompt_tokens": 1000,
                "cached_prompt_tokens": 400,
                "completion_tokens": 100,
            },
            input_rate=0.15,
            output_rate=0.60,
            cached_input_rate=0.075,
        )
        self.assertAlmostEqual(cost, 0.00018)

    def test_rejects_cached_tokens_above_prompt_total(self):
        with self.assertRaisesRegex(ValueError, "cached prompt tokens"):
            estimate_cost_usd(
                {"prompt_tokens": 1, "cached_prompt_tokens": 2},
                input_rate=0.15,
                output_rate=0.60,
                cached_input_rate=0.075,
            )


if __name__ == "__main__":
    unittest.main()
