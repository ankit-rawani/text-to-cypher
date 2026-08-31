"""Client protocols, real HTTP implementations, and in-memory fakes."""

from __future__ import annotations

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

__all__ = [
    "ArcadeDBClient",
    "QdrantClient",
    "OpenAIChatClient",
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
