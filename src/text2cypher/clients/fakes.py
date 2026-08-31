"""In-memory fakes for every client protocol.

These are the reason the whole pipeline runs and tests without a live ArcadeDB,
Qdrant, or LLM. The vector store is a *real* cosine-similarity search over
in-memory points (so grounding and example retrieval genuinely work); the graph
and LLM are programmable so tests can script exact scenarios.
"""

from __future__ import annotations

from typing import Any, Callable

from ..embeddings.base import cosine
from .base import (
    ClientError,
    GraphError,
    VectorHit,
    VectorPoint,
)

# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------

GraphRule = Callable[[str, dict[str, Any]], "list[dict[str, Any]] | None"]


class FakeGraphClient:
    """Programmable, read-only-by-default graph client.

    Register rules with :meth:`when` (substring match) or :meth:`add_rule`
    (predicate). The first matching rule's rows (or raised error) is returned.
    Records every executed (cypher, params) in :attr:`executed`.
    """

    def __init__(self, default_rows: list[dict[str, Any]] | None = None, read_only: bool = True) -> None:
        self.read_only = read_only
        self._default_rows = default_rows if default_rows is not None else []
        self._rules: list[tuple[Callable[[str, dict[str, Any]], bool], GraphRule]] = []
        self.executed: list[tuple[str, dict[str, Any]]] = []

    # -- configuration -------------------------------------------------
    def add_rule(
        self,
        predicate: Callable[[str, dict[str, Any]], bool],
        responder: GraphRule,
    ) -> "FakeGraphClient":
        self._rules.append((predicate, responder))
        return self

    def when(self, substring: str, rows: list[dict[str, Any]]) -> "FakeGraphClient":
        return self.add_rule(lambda c, p, s=substring: s in c, lambda c, p, r=rows: r)

    def when_raises(self, substring: str, exc: Exception) -> "FakeGraphClient":
        def responder(c: str, p: dict[str, Any], e: Exception = exc):
            raise e
        return self.add_rule(lambda c, p, s=substring: s in c, responder)

    def set_default(self, rows: list[dict[str, Any]]) -> "FakeGraphClient":
        self._default_rows = rows
        return self

    # -- protocol ------------------------------------------------------
    def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params = params or {}
        self.executed.append((cypher, dict(params)))
        for predicate, responder in self._rules:
            if predicate(cypher, params):
                rows = responder(cypher, params)
                if rows is None:
                    continue
                if limit is not None:
                    return list(rows)[:limit]
                return list(rows)
        rows = list(self._default_rows)
        return rows[:limit] if limit is not None else rows

    def introspect(self) -> dict[str, Any]:
        return {}


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------


class FakeVectorStore:
    """Real cosine-similarity search over in-memory collections."""

    def __init__(self) -> None:
        self._collections: dict[str, list[VectorPoint]] = {}

    def ensure_collection(self, collection: str, dim: int) -> None:
        self._collections.setdefault(collection, [])

    def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        store = self._collections.setdefault(collection, [])
        by_id = {p.id: p for p in store}
        for p in points:
            by_id[p.id] = p
        self._collections[collection] = list(by_id.values())

    def count(self, collection: str) -> int:
        return len(self._collections.get(collection, []))

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 5,
        *,
        with_payload: bool = True,
        query_filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        points = self._collections.get(collection, [])
        scored = [
            VectorHit(id=p.id, score=cosine(vector, p.vector), payload=dict(p.payload) if with_payload else {})
            for p in points
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------


class FakeLLM:
    """Scripted chat client.

    Provide ``responses`` (consumed in FIFO order) and/or a ``handler`` callable
    that receives the messages and returns a string. The handler takes
    precedence when set. Records each call in :attr:`calls`.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        handler: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._handler = handler
        self.calls: list[list[dict[str, str]]] = []

    def push(self, response: str) -> "FakeLLM":
        self._responses.append(response)
        return self

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        self.calls.append(messages)
        if self._handler is not None:
            return self._handler(messages)
        if not self._responses:
            raise ClientError("FakeLLM ran out of scripted responses")
        return self._responses.pop(0)


# --------------------------------------------------------------------------
# Document store
# --------------------------------------------------------------------------


class FakeDocumentStore:
    enabled = True

    def __init__(self, docs: dict[str, dict[str, Any]] | None = None) -> None:
        self._docs = docs or {}

    def get_many(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        return {i: self._docs[i] for i in ids if i in self._docs}


__all__ = [
    "FakeGraphClient",
    "FakeVectorStore",
    "FakeLLM",
    "FakeDocumentStore",
    "GraphError",
]
