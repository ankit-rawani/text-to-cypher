"""The 7-stage pipeline and its wiring factory."""

from __future__ import annotations

from ..config import AppConfig, load_config
from ..graph_config import load_graph_config
from .entity_grounder import EntityGrounder
from .example_store import ExampleStore
from .executor import Executor
from .generator import Generator
from .orchestrator import Pipeline
from .schema_provider import SchemaProvider
from .validator import Denylist, Validator, load_denylist


def build_pipeline(config: AppConfig | None = None) -> Pipeline:
    """Wire the pipeline against real clients configured from ``config`` / env.

    Raises :class:`~text2cypher.clients.base.ConfigurationError` if the read-only
    ArcadeDB credential (or other required setting) is missing.
    """
    from ..clients import build_llm
    from ..clients.arcadedb import ArcadeDBClient
    from ..clients.mongodb import build_document_store
    from ..clients.qdrant import QdrantClient
    from ..embeddings import build_embedder

    config = config or load_config()

    embedder = build_embedder(config.embeddings)
    graph_config = load_graph_config(config.resolve_path(config.graph_config_path))
    graph_client = ArcadeDBClient.from_config(config.arcadedb)
    vector_store = QdrantClient.from_config(config.qdrant)
    llm = build_llm(config.llm)
    doc_store = build_document_store(config.mongodb)

    schema_provider = SchemaProvider(graph_config, graph_client, ttl_s=config.schema_cache.ttl_s)
    grounder = EntityGrounder(
        embedder,
        vector_store,
        config.qdrant.node_collection,
        sim_threshold=config.grounding.sim_threshold,
        max_candidates=config.grounding.max_candidates,
        low_confidence_threshold=config.grounding.low_confidence_threshold,
        enabled=config.grounding.enabled,
    )
    example_store = ExampleStore(
        embedder,
        vector_store,
        config.qdrant.example_collection,
        top_k=config.examples.top_k,
        sim_threshold=config.examples.sim_threshold,
        enabled=config.examples.enabled,
    )
    generator = Generator(
        llm,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        timeout_s=config.llm.timeout_s,
    )
    validator = Validator(
        load_denylist(config.resolve_path(config.dialect.denylist_path)),
        row_cap=config.pipeline.row_cap,
    )
    executor = Executor(
        graph_client,
        row_cap=config.pipeline.row_cap,
        timeout_s=config.arcadedb.timeout_s,
        document_store=doc_store,
    )
    return Pipeline(
        schema_provider,
        grounder,
        example_store,
        generator,
        validator,
        executor,
        hard_max_attempts=config.pipeline.hard_max_attempts,
        allow_relaxation=config.pipeline.allow_relaxation,
        low_confidence_threshold=config.grounding.low_confidence_threshold,
    )


__all__ = [
    "Pipeline",
    "build_pipeline",
    "SchemaProvider",
    "EntityGrounder",
    "ExampleStore",
    "Generator",
    "Validator",
    "Denylist",
    "load_denylist",
    "Executor",
]
