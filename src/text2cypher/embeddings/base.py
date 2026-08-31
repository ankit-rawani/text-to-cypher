"""Embedder interface + cosine helper."""

from __future__ import annotations

import abc
import math


class BaseEmbedder(abc.ABC):
    """Turns text into fixed-dimension vectors."""

    @property
    @abc.abstractmethod
    def dim(self) -> int:  # pragma: no cover - trivial
        ...

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity clamped to [-1, 1]. Zero vectors -> 0."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    val = dot / (math.sqrt(na) * math.sqrt(nb))
    if val > 1.0:
        return 1.0
    if val < -1.0:
        return -1.0
    return val


__all__ = ["BaseEmbedder", "cosine"]
