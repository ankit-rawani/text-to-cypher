"""Embedding providers + factory."""

from __future__ import annotations

from ..config import EmbeddingConfig
from .base import BaseEmbedder, cosine
from .hashing import HashingEmbedder


def build_embedder(config: EmbeddingConfig) -> BaseEmbedder:
    """Construct an embedder from config. Defaults to the offline hashing one."""
    provider = (config.provider or "hashing").lower()
    if provider in ("hashing", "local", "offline"):
        return HashingEmbedder(dim=config.dim)
    if provider in ("openai", "openai-compatible", "http"):
        from .openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder(
            endpoint=config.endpoint,
            model=config.model,
            api_key=config.api_key,
            dim=config.dim,
            timeout_s=config.timeout_s,
        )
    raise ValueError(f"Unknown embedding provider: {config.provider!r}")


__all__ = ["BaseEmbedder", "HashingEmbedder", "cosine", "build_embedder"]
