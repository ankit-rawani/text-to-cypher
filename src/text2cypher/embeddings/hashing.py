"""Deterministic, dependency-free hashing embedder.

No network, fully reproducible (uses ``hashlib`` rather than the salted builtin
``hash``). It combines word unigrams with character trigrams so that aliases and
lightly-misspelled surface forms land near their canonical names — good enough
to make entity grounding and example retrieval work offline and in tests.

This is NOT a substitute for a real embedding model in production; select the
``openai`` provider for that. But it keeps the whole pipeline runnable and
testable without any external service.
"""

from __future__ import annotations

import hashlib
import math
import re

from .base import BaseEmbedder

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    words = _WORD.findall(text.lower())
    feats: list[str] = []
    for w in words:
        feats.append(f"w:{w}")
        padded = f"#{w}#"
        for i in range(len(padded) - 2):
            feats.append(f"t:{padded[i:i + 3]}")
    return feats


class HashingEmbedder(BaseEmbedder):
    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for feat in _tokens(text):
            h = hashlib.md5(feat.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "big") % self._dim
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


__all__ = ["HashingEmbedder"]
