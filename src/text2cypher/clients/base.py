"""Client-layer interfaces, data types, and exceptions.

Every external dependency (graph DB, vector store, LLM, document store) is
reached through a small Protocol here. Real HTTP implementations live alongside;
in-memory fakes live in :mod:`text2cypher.clients.fakes`. The pipeline stages
depend only on these protocols, so the whole thing runs and tests offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------


class ClientError(Exception):
    """Base for all client-layer errors."""


class GraphError(ClientError):
    """A graph database error."""


class GraphQueryError(GraphError):
    """The query failed at the database (syntax/semantic/dialect/etc.)."""


class GraphTimeout(GraphError):
    """The query exceeded the configured timeout."""


class GraphAuthError(GraphError):
    """Authentication / authorization failure (e.g. write attempted as RO)."""


class ConfigurationError(ClientError):
    """A client was constructed with missing/invalid configuration."""


# --------------------------------------------------------------------------
# Vector store types
# --------------------------------------------------------------------------


@dataclass
class VectorPoint:
    id: str | int
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorHit:
    id: str | int
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Protocols
# --------------------------------------------------------------------------


@runtime_checkable
class GraphClient(Protocol):
    """Read-only access to the graph database."""

    read_only: bool

    def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run a read query and return rows. Raises GraphError subclasses on failure."""
        ...

    def introspect(self) -> dict[str, Any]:
        """Best-effort live schema introspection (may return {} if unsupported)."""


@runtime_checkable
class VectorSearchClient(Protocol):
    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 5,
        *,
        with_payload: bool = True,
        query_filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        ...

    def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        ...

    def ensure_collection(self, collection: str, dim: int) -> None:
        ...

    def count(self, collection: str) -> int:
        ...


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        ...


@runtime_checkable
class DocumentStore(Protocol):
    enabled: bool

    def get_many(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        ...


__all__ = [
    "ClientError",
    "GraphError",
    "GraphQueryError",
    "GraphTimeout",
    "GraphAuthError",
    "ConfigurationError",
    "VectorPoint",
    "VectorHit",
    "GraphClient",
    "VectorSearchClient",
    "LLMClient",
    "DocumentStore",
]
