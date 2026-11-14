import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from day6_retriever import Day6DenseRetriever


class FakeEmbedder:
    def __init__(self, vector):
        self.vector = vector
        self.queries = []

    def query_embed(self, query):
        self.queries.append(query)
        return iter([self.vector])


class FakeCollection:
    def __init__(self):
        self.query_args = None

    def get(self, *, limit, include):
        return {"embeddings": [[0.1, 0.2]]}

    def query(self, **kwargs):
        self.query_args = kwargs
        return {
            "ids": [["chunk-1"]],
            "documents": [["evidence text"]],
            "metadatas": [[
                {"document_id": "doc-1", "source": "manual.pdf", "page": 7, "chapter": 2}
            ]],
            "distances": [[0.15]],
        }


class Day6RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.collection = FakeCollection()
        self.embedder = FakeEmbedder([0.3, 0.4])
        self.retriever = Day6DenseRetriever(self.collection, self.embedder)

    def test_maps_day6_result_and_filters_to_day7_contract(self):
        rows = self.retriever.search(
            "leadership", filters={"chapter": 2, "page": 7}, top_k=5
        )
        self.assertEqual(self.embedder.queries, ["leadership"])
        self.assertEqual(
            self.collection.query_args["where"],
            {"$and": [{"chapter": 2}, {"page": 7}]},
        )
        self.assertEqual(self.collection.query_args["n_results"], 5)
        self.assertEqual(
            rows,
            [{
                "id": "chunk-1",
                "text": "evidence text",
                "source_id": "doc-1",
                "source": "manual.pdf",
                "page": 7,
                "score": 0.85,
                "metadata": {
                    "document_id": "doc-1",
                    "source": "manual.pdf",
                    "page": 7,
                    "chapter": 2,
                },
            }],
        )

    def test_rejects_bad_filter_and_mismatched_embedding(self):
        with self.assertRaises(ValueError):
            self.retriever.search("leadership", filters={"unknown": 2})

        wrong_embedder = FakeEmbedder([0.1])
        retriever = Day6DenseRetriever(self.collection, wrong_embedder)
        with self.assertRaisesRegex(ValueError, "dimension"):
            retriever.search("leadership")

    def test_rejects_empty_index(self):
        class EmptyCollection(FakeCollection):
            def get(self, *, limit, include):
                return {"embeddings": []}

        with self.assertRaisesRegex(ValueError, "empty"):
            Day6DenseRetriever(EmptyCollection(), self.embedder)


if __name__ == "__main__":
    unittest.main()
