"""Optional MongoDB document store for row hydration.

Hydration is off by default (it is not on the pipeline's critical path — see the
spec non-goals). When enabled, the executor can enrich returned rows with full
documents keyed by node id. ``pymongo`` is imported lazily so the dependency is
only needed when hydration is actually turned on.
"""

from __future__ import annotations

from typing import Any

from ..config import MongoConfig


class DisabledDocumentStore:
    """No-op document store used when hydration is disabled."""

    enabled = False

    def get_many(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        return {}


class MongoDocumentStore:
    enabled = True

    def __init__(self, url: str, database: str, collection: str, id_field: str = "_id") -> None:
        try:
            from pymongo import MongoClient  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "MongoDB hydration is enabled but pymongo is not installed. "
                "Install with `pip install pymongo` or disable mongodb in config."
            ) from exc
        self._client = MongoClient(url)
        self._collection = self._client[database][collection]
        self._id_field = id_field

    def get_many(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        docs = self._collection.find({self._id_field: {"$in": ids}})
        return {str(doc[self._id_field]): doc for doc in docs}


def build_document_store(config: MongoConfig):
    if not config.enabled or not config.url:
        return DisabledDocumentStore()
    return MongoDocumentStore(config.url, config.database, config.collection)  # pragma: no cover


__all__ = ["DisabledDocumentStore", "MongoDocumentStore", "build_document_store"]
