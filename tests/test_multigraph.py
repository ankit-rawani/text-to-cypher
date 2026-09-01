from __future__ import annotations

import pytest

from text2cypher.config import load_config
from text2cypher.contracts import QueryRequest
from text2cypher.graph_config import GraphConfig, RelationshipDef
from text2cypher.pipeline import resolve_graph_config
from text2cypher.pipeline.orchestrator import MultiGraphPipeline
from text2cypher.testing import build_fake_pipeline, gen_json


# ---- resolve_graph_config: object / path / default ----------------------


def test_resolve_graph_config_object_passthrough():
    cfg = load_config()
    gc = GraphConfig(database="mine", node_labels=["Widget"])
    assert resolve_graph_config(cfg, gc) is gc


def test_resolve_graph_config_by_path():
    cfg = load_config()
    gc = resolve_graph_config(cfg, "config/graph_config.worms.yaml")
    assert gc.database == "worms"
    assert "Node" in gc.node_labels


def test_resolve_graph_config_default():
    cfg = load_config()
    gc = resolve_graph_config(cfg, None)
    assert gc.database == "concept_graph"  # packaged default


# ---- MultiGraphPipeline routing -----------------------------------------


def _pipe(database, label, rel, marker):
    gc = GraphConfig(
        database=database,
        node_labels=[label],
        node_properties={"name": "string"},
        relationships=[RelationshipDef(type=rel)],
    )
    fp = build_fake_pipeline(
        graph_config=gc,
        llm_responses=[gen_json(f"MATCH (n:{label}) RETURN n.name AS name")],
    )
    fp.graph.set_default([{"name": marker}])
    return fp.pipeline


def test_multigraph_routes_by_name():
    mgp = MultiGraphPipeline(
        {"concepts": _pipe("a", "Node", "causes", "from_concepts"),
         "taxonomy": _pipe("b", "Taxon", "parent_of", "from_taxonomy")},
        default="concepts",
    )
    assert mgp.graphs == ["concepts", "taxonomy"]
    assert mgp.default == "concepts"

    # default route
    assert mgp.answer(QueryRequest(question="x")).rows == [{"name": "from_concepts"}]
    # explicit route
    assert mgp.answer(QueryRequest(question="x"), graph="taxonomy").rows == [{"name": "from_taxonomy"}]


def test_multigraph_unknown_graph_raises():
    mgp = MultiGraphPipeline({"only": _pipe("a", "Node", "causes", "m")})
    with pytest.raises(KeyError):
        mgp.answer(QueryRequest(question="x"), graph="missing")


def test_multigraph_requires_pipeline():
    with pytest.raises(ValueError):
        MultiGraphPipeline({})
