"""Offline test/demo harness: wire the pipeline against in-memory fakes.

``build_fake_pipeline`` returns a :class:`FakePipeline` bundle exposing both the
:class:`~text2cypher.pipeline.orchestrator.Pipeline` and the underlying fakes so
tests can script LLM responses and assert on executed queries/params.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .clients.fakes import FakeGraphClient, FakeLLM, FakeVectorStore
from .config import load_config
from .embeddings.hashing import HashingEmbedder
from .graph_config import GraphConfig, default_graph_config
from .ingest import index_nodes
from .pipeline.entity_grounder import EntityGrounder
from .pipeline.example_store import ExampleStore
from .pipeline.executor import Executor
from .pipeline.generator import Generator
from .pipeline.orchestrator import Pipeline
from .pipeline.schema_provider import SchemaProvider
from .pipeline.validator import Validator, load_denylist


def gen_json(
    cypher: str,
    params: dict[str, Any] | None = None,
    reasoning: str = "generated",
    confidence: float | None = 0.9,
) -> str:
    """Build a scripted Generator JSON response string."""
    return json.dumps(
        {
            "cypher": cypher,
            "params_used": params or {},
            "reasoning": reasoning,
            "confidence": confidence,
        }
    )


@dataclass
class FakePipeline:
    pipeline: Pipeline
    llm: FakeLLM
    graph: FakeGraphClient
    vector_store: FakeVectorStore
    embedder: HashingEmbedder
    schema_provider: SchemaProvider
    grounder: EntityGrounder
    example_store: ExampleStore
    validator: Validator
    executor: Executor

    def answer(self, *args, **kwargs):
        return self.pipeline.answer(*args, **kwargs)


def default_denylist():
    cfg = load_config()
    return load_denylist(cfg.resolve_path(cfg.dialect.denylist_path))


def build_fake_pipeline(
    *,
    nodes: list[dict[str, Any]] | None = None,
    examples: list[dict[str, Any]] | None = None,
    llm_responses: list[str] | None = None,
    llm_handler: Callable[[list[dict[str, str]]], str] | None = None,
    graph: FakeGraphClient | None = None,
    graph_config: GraphConfig | None = None,
    grounding_threshold: float = 0.25,
    example_threshold: float = 0.25,
    max_candidates: int = 3,
    allow_relaxation: bool = True,
    node_collection: str = "concepts",
    example_collection: str = "t2c_examples",
    row_cap: int = 200,
    grounding_enabled: bool = True,
    examples_enabled: bool = True,
    dim: int = 256,
) -> FakePipeline:
    embedder = HashingEmbedder(dim=dim)
    vector_store = FakeVectorStore()
    graph = graph if graph is not None else FakeGraphClient()
    llm = FakeLLM(responses=llm_responses, handler=llm_handler)
    gcfg = graph_config or default_graph_config()

    if nodes:
        index_nodes(vector_store, embedder, node_collection, nodes)

    schema_provider = SchemaProvider(gcfg, graph_client=None)
    grounder = EntityGrounder(
        embedder,
        vector_store,
        node_collection,
        sim_threshold=grounding_threshold,
        max_candidates=max_candidates,
        enabled=grounding_enabled,
    )
    example_store = ExampleStore(
        embedder,
        vector_store,
        example_collection,
        top_k=3,
        sim_threshold=example_threshold,
        enabled=examples_enabled,
    )
    if examples:
        from .contracts import ExamplePair

        example_store.add(
            [
                ExamplePair(question=e["question"], cypher=e["cypher"], tags=e.get("tags", []))
                for e in examples
            ]
        )

    generator = Generator(llm)
    validator = Validator(default_denylist(), row_cap=row_cap)
    executor = Executor(graph, row_cap=row_cap)

    pipeline = Pipeline(
        schema_provider,
        grounder,
        example_store,
        generator,
        validator,
        executor,
        allow_relaxation=allow_relaxation,
    )
    return FakePipeline(
        pipeline=pipeline,
        llm=llm,
        graph=graph,
        vector_store=vector_store,
        embedder=embedder,
        schema_provider=schema_provider,
        grounder=grounder,
        example_store=example_store,
        validator=validator,
        executor=executor,
    )


__all__ = ["build_fake_pipeline", "FakePipeline", "gen_json"]
