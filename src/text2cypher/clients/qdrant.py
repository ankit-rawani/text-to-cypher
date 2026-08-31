"""Qdrant HTTP client (vector search over node / example collections)."""

from __future__ import annotations

from typing import Any

import httpx

from ..config import QdrantConfig
from .base import ClientError, VectorHit, VectorPoint


class QdrantClient:
    def __init__(
        self,
        url: str,
        api_key: str = "",
        timeout_s: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_s
        self._client = client or httpx.Client(timeout=timeout_s)

    @classmethod
    def from_config(cls, config: QdrantConfig, client: httpx.Client | None = None) -> "QdrantClient":
        return cls(url=config.url, api_key=config.api_key, timeout_s=config.timeout_s, client=client)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["api-key"] = self._api_key
        return headers

    def search(
        self,
        collection: str,
        vector: list[float],
        limit: int = 5,
        *,
        with_payload: bool = True,
        query_filter: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        body: dict[str, Any] = {"vector": vector, "limit": limit, "with_payload": with_payload}
        if query_filter:
            body["filter"] = query_filter
        resp = self._client.post(
            f"{self._url}/collections/{collection}/points/search",
            json=body,
            headers=self._headers(),
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise ClientError(f"Qdrant search failed ({resp.status_code}): {resp.text[:300]}")
        data = resp.json()
        hits: list[VectorHit] = []
        for item in data.get("result", []):
            hits.append(
                VectorHit(id=item.get("id"), score=float(item.get("score", 0.0)), payload=item.get("payload") or {})
            )
        return hits

    def upsert(self, collection: str, points: list[VectorPoint]) -> None:
        body = {
            "points": [
                {"id": p.id, "vector": p.vector, "payload": p.payload} for p in points
            ]
        }
        resp = self._client.put(
            f"{self._url}/collections/{collection}/points",
            json=body,
            headers=self._headers(),
            params={"wait": "true"},
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise ClientError(f"Qdrant upsert failed ({resp.status_code}): {resp.text[:300]}")

    def ensure_collection(self, collection: str, dim: int) -> None:
        # Create the collection if it does not exist (cosine distance).
        get = self._client.get(
            f"{self._url}/collections/{collection}", headers=self._headers(), timeout=self._timeout
        )
        if get.status_code == 200:
            return
        body = {"vectors": {"size": dim, "distance": "Cosine"}}
        resp = self._client.put(
            f"{self._url}/collections/{collection}", json=body, headers=self._headers(), timeout=self._timeout
        )
        if resp.status_code >= 400:
            raise ClientError(f"Qdrant create collection failed ({resp.status_code}): {resp.text[:300]}")

    def count(self, collection: str) -> int:
        resp = self._client.post(
            f"{self._url}/collections/{collection}/points/count",
            json={"exact": True},
            headers=self._headers(),
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            return 0
        return int(resp.json().get("result", {}).get("count", 0))


__all__ = ["QdrantClient"]
