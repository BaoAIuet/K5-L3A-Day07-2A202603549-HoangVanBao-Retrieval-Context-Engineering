"""Additional retrieval methods for Day 7."""
from typing import Any


class HybridRetriever:
    """Reciprocal-rank fusion over dense and TF-IDF candidate lists."""

    def __init__(self, dense: Any, lexical: Any, candidate_k: int = 10, rrf_k: int = 60) -> None:
        if candidate_k < 1 or rrf_k < 1:
            raise ValueError("candidate_k and rrf_k must be positive")
        self._dense = dense
        self._lexical = lexical
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip() or top_k < 1:
            raise ValueError("query must be non-empty and top_k must be positive")
        dense_hits = self._dense.search(
            query, filters=filters, top_k=self._candidate_k
        )
        lexical_hits = self._lexical.search(
            query, filters=filters, top_k=self._candidate_k
        )
        fused: dict[str, dict[str, Any]] = {}
        for source_name, hits in (("dense", dense_hits), ("tfidf", lexical_hits)):
            for rank, hit in enumerate(hits, 1):
                chunk_id = hit["id"]
                entry = fused.setdefault(chunk_id, {**hit, "score": 0.0})
                entry["score"] += 1.0 / (self._rrf_k + rank)
                entry.setdefault("retrieval_sources", []).append(source_name)
        results = sorted(
            fused.values(), key=lambda hit: (-hit["score"], hit["id"])
        )[:top_k]
        for hit in results:
            hit["strategy"] = "hybrid_rrf"
        return results
