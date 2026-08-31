"""OpenAI-compatible embeddings client (``POST {endpoint}/embeddings``)."""

from __future__ import annotations

from typing import Any

import httpx

from .base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str = "",
        dim: int = 1536,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._dim = dim
        self._timeout = timeout_s
        self._client = client or httpx.Client(timeout=timeout_s)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: dict[str, Any] = {"model": self._model, "input": texts}
        resp = self._client.post(
            f"{self._endpoint}/embeddings", json=payload, headers=headers, timeout=self._timeout
        )
        resp.raise_for_status()
        data = resp.json()
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        vectors = [item["embedding"] for item in items]
        if vectors:
            self._dim = len(vectors[0])
        return vectors


__all__ = ["OpenAIEmbedder"]
