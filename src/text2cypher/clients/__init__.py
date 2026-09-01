"""Client protocols, real HTTP implementations, and in-memory fakes."""

from __future__ import annotations

from ..config import LLMConfig
from .anthropic import AnthropicClient
from .arcadedb import ArcadeDBClient
from .base import (
    ClientError,
    ConfigurationError,
    DocumentStore,
    GraphAuthError,
    GraphClient,
    GraphError,
    GraphQueryError,
    GraphTimeout,
    LLMClient,
    VectorHit,
    VectorPoint,
    VectorSearchClient,
)
from .fakes import FakeDocumentStore, FakeGraphClient, FakeLLM, FakeVectorStore
from .llm import OpenAIChatClient
from .mongodb import DisabledDocumentStore, build_document_store
from .qdrant import QdrantClient


def build_llm(config: LLMConfig, client=None) -> LLMClient:
    """Construct the LLM client for the configured provider."""
    provider = (config.provider or "openai").lower()
    if provider in ("anthropic", "claude"):
        return AnthropicClient.from_config(config, client=client)
    if provider in ("openai", "openai-compatible", "compat"):
        return OpenAIChatClient.from_config(config, client=client)
    raise ConfigurationError(f"Unknown LLM provider: {config.provider!r}")


__all__ = [
    "ArcadeDBClient",
    "QdrantClient",
    "OpenAIChatClient",
    "AnthropicClient",
    "build_llm",
    "build_document_store",
    "DisabledDocumentStore",
    "ClientError",
    "ConfigurationError",
    "GraphError",
    "GraphQueryError",
    "GraphTimeout",
    "GraphAuthError",
    "GraphClient",
    "VectorSearchClient",
    "LLMClient",
    "DocumentStore",
    "VectorHit",
    "VectorPoint",
    "FakeGraphClient",
    "FakeVectorStore",
    "FakeLLM",
    "FakeDocumentStore",
]
