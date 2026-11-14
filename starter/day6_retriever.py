"""Dense retrieval adapter for the persistent Chroma indexes created in Day 6."""
from pathlib import Path
from typing import Any


_ALLOWED_FILTERS = {"chapter", "page", "document_id", "section"}


class Day6DenseRetriever:
    def __init__(self, collection: Any, embedder: Any) -> None:
        self._collection = collection
        self._embedder = embedder
        sample = collection.get(limit=1, include=["embeddings"])
        embeddings = sample.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Day 6 Chroma collection is empty or has no embeddings")
        self._dimensions = len(embeddings[0])

    @classmethod
    def from_persistent_index(
        cls,
        index_path: str | Path,
        *,
        strategy: str = "B",
        model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> "Day6DenseRetriever":
        if strategy not in {"A", "B"}:
            raise ValueError("strategy must be 'A' or 'B'")
        path = Path(index_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Day 6 Chroma index not found: {path}")

        import chromadb
        from fastembed import TextEmbedding

        client = chromadb.PersistentClient(path=str(path))
        collection = client.get_collection(name=f"chunks_{strategy}")
        return cls(collection, TextEmbedding(model_name=model_name))

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if filters is None:
            filters = {}
        elif not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary")
        if any(key not in _ALLOWED_FILTERS for key in filters):
            raise ValueError(f"unsupported Day 6 filter; allowed: {sorted(_ALLOWED_FILTERS)}")
        if any(
            value is None or not isinstance(value, (str, int, float, bool))
            for value in filters.values()
        ):
            raise ValueError("filter values must be non-null scalar values")

        vectors = iter(self._embedder.query_embed(query))
        try:
            query_vector = [float(value) for value in next(vectors)]
        except StopIteration as exc:
            raise RuntimeError("Day 6 embedder returned no query vector") from exc
        if len(query_vector) != self._dimensions:
            raise ValueError(
                f"query embedding dimension {len(query_vector)} does not match "
                f"Day 6 index dimension {self._dimensions}"
            )

        conditions = [{key: value} for key, value in filters.items()]
        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif conditions:
            where = {"$and": conditions}
        response = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = response["ids"][0]
        documents = response["documents"][0]
        metadatas = response["metadatas"][0]
        distances = response["distances"][0]
        results = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            if text is None or metadata is None or distance is None:
                raise RuntimeError("Day 6 Chroma returned incomplete chunk data")
            required = ("document_id", "source", "page")
            if any(key not in metadata or metadata[key] is None for key in required):
                raise RuntimeError(f"Day 6 chunk {chunk_id} is missing source metadata")
            results.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "source_id": metadata["document_id"],
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "score": 1.0 - float(distance),
                    "metadata": dict(metadata),
                }
            )
        return results
