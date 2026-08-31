"""Stage 3 — ExampleStore.

Retrieve the top-k most similar question -> Cypher pairs, gated by a similarity
threshold: below threshold, inject nothing (irrelevant examples hurt more than
they help). Seeded by hand and grown from logged successful runs behind a
feedback gate (:meth:`add`).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..clients.base import VectorPoint, VectorSearchClient
from ..contracts import ExamplePair
from ..embeddings.base import BaseEmbedder


class ExampleStore:
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorSearchClient,
        collection: str,
        *,
        top_k: int = 3,
        sim_threshold: float = 0.60,
        enabled: bool = True,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._collection = collection
        self._top_k = top_k
        self._threshold = sim_threshold
        self._enabled = enabled
        self._store.ensure_collection(collection, embedder.dim)
        self._next_id = 1

    def retrieve(self, question: str) -> list[ExamplePair]:
        if not self._enabled or self._top_k <= 0:
            return []
        vector = self._embedder.embed_one(question)
        hits = self._store.search(self._collection, vector, limit=self._top_k)
        pairs: list[ExamplePair] = []
        for h in hits:
            if h.score < self._threshold:
                continue
            q = h.payload.get("question")
            cy = h.payload.get("cypher")
            if not q or not cy:
                continue
            pairs.append(
                ExamplePair(question=q, cypher=cy, similarity=h.score, tags=list(h.payload.get("tags", []) or []))
            )
        return pairs

    def add(self, pairs: list[ExamplePair]) -> int:
        """Upsert example pairs (question embedded as the vector). Returns count."""
        points: list[VectorPoint] = []
        for pair in pairs:
            vector = self._embedder.embed_one(pair.question)
            points.append(
                VectorPoint(
                    id=self._next_id,
                    vector=vector,
                    payload={"question": pair.question, "cypher": pair.cypher, "tags": pair.tags},
                )
            )
            self._next_id += 1
        if points:
            self._store.upsert(self._collection, points)
        return len(points)

    def add_pair(self, question: str, cypher: str, tags: list[str] | None = None) -> None:
        self.add([ExamplePair(question=question, cypher=cypher, tags=tags or [])])

    def import_jsonl(self, path: str | Path) -> int:
        pairs = load_example_pairs(path)
        return self.add(pairs)

    def count(self) -> int:
        return self._store.count(self._collection)

    def list(self) -> list[ExamplePair]:
        # Not all backends can enumerate; a search with a zero vector returns
        # nothing meaningful, so this is a best-effort convenience over fakes.
        try:
            hits = self._store.search(self._collection, self._embedder.embed_one(""), limit=1000)
        except Exception:
            return []
        return [
            ExamplePair(question=h.payload.get("question", ""), cypher=h.payload.get("cypher", ""))
            for h in hits
            if h.payload.get("question")
        ]


def load_example_pairs(path: str | Path) -> list[ExamplePair]:
    pairs: list[ExamplePair] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            pairs.append(
                ExamplePair(
                    question=data["question"],
                    cypher=data["cypher"],
                    tags=list(data.get("tags", []) or []),
                )
            )
    return pairs


__all__ = ["ExampleStore", "load_example_pairs"]
