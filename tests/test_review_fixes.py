"""Regression tests for defects found by the adversarial review."""

from __future__ import annotations

import httpx

from text2cypher.clients.arcadedb import ArcadeDBClient
from text2cypher.clients.fakes import FakeVectorStore
from text2cypher.contracts import (
    ExamplePair,
    GroundedEntity,
    GroundingResult,
    QueryRequest,
)
from text2cypher.cypher import analyze, ensure_limit
from text2cypher.embeddings.hashing import HashingEmbedder
from text2cypher.graph_config import default_graph_config
from text2cypher.ingest import index_nodes
from text2cypher.pipeline.entity_grounder import EntityGrounder
from text2cypher.pipeline.example_store import ExampleStore
from text2cypher.pipeline.schema_provider import SchemaProvider
from text2cypher.pipeline.validator import Validator, load_denylist
from text2cypher.config import load_config
from text2cypher.testing import build_fake_pipeline, gen_json


def _validator():
    cfg = load_config()
    return Validator(load_denylist(cfg.resolve_path(cfg.dialect.denylist_path)), row_cap=200)


def _schema():
    return SchemaProvider(default_graph_config()).get()


# 1. write keyword as a map/property key must NOT be a write op
def test_write_keyword_as_map_key_not_flagged():
    a = analyze("MATCH (n:Concept) RETURN {create: n.canonical_name, set: 1} AS m")
    assert a.write_ops == []
    r = _validator().validate("MATCH (n:Concept) RETURN {create: n.canonical_name} AS m", {}, _schema())
    assert r.passed


def test_write_keyword_as_label_or_alias_not_flagged():
    assert analyze("MATCH (n:Concept) RETURN n.canonical_name AS set").write_ops == []
    assert analyze("MATCH (n:Concept)-[:CREATE]->(m) RETURN n").write_ops == []  # rel type named CREATE


# 2. denylist must not be bypassable via back-quoting the call name
def test_backtick_denylist_bypass_blocked():
    v, s = _validator(), _schema()
    for q in (
        "MATCH (n:Concept) RETURN `apoc.text.join`([n.canonical_name])",
        "MATCH (n:Concept) RETURN `apoc`.text.join([n.canonical_name])",
    ):
        r = v.validate(q, {}, s)
        assert not r.passed and "DENYLISTED" in [i.code for i in r.issues], q


# 3. LIMIT enforced per UNION branch (clamp + inject)
def test_union_limit_per_branch():
    q = "MATCH (n:Concept) RETURN n.canonical_name AS x UNION MATCH (m:Concept) RETURN m.canonical_name AS x LIMIT 9999"
    out, action = ensure_limit(q, 200, analyze(q))
    assert action == "clamped"
    assert out.count("LIMIT 200") == 2 and "9999" not in out


# 4. injected LIMIT must not land inside a trailing line comment
def test_limit_injected_before_trailing_comment():
    q = "MATCH (n:Concept) RETURN n.canonical_name // top"
    out, action = ensure_limit(q, 200, analyze(q))
    assert action == "injected"
    assert out.index("LIMIT 200") < out.index("//")


# 7 (finding). direction autofix works on the default (node_type-discriminated) graph
def test_direction_autofix_on_default_graph():
    q = "MATCH (a:Concept {node_type:'Disease'})-[:TREATS]->(b:Concept {node_type:'Drug'}) RETURN a"
    r = _validator().validate(q, {}, _schema())
    assert r.passed and "DIRECTION_FIXED" in [i.code for i in r.issues]
    assert "<-[:TREATS]-" in r.final_cypher


# 12 (finding). param discipline is position-aware
def test_param_literal_position_aware():
    v, s = _validator(), _schema()
    g = GroundingResult(
        entities=[GroundedEntity(mention="drug", canonical_name="Drug", param_name="drug")],
        params={"drug": "Drug"},
    )
    # inlined against node_type enum -> allowed
    assert v.validate("MATCH (n:Concept) WHERE n.node_type = 'Drug' RETURN n.canonical_name", {}, s, g).passed
    # inlined against identity property -> rejected
    bad = v.validate("MATCH (n:Concept) WHERE n.canonical_name = 'Drug' RETURN n.canonical_name", {}, s, g)
    assert not bad.passed and "PARAM_LITERAL" in [i.code for i in bad.issues]


# grounding dedupe keeps distinct nodes with overlapping mentions
def test_grounding_keeps_distinct_overlapping_mentions():
    emb, vs = HashingEmbedder(), FakeVectorStore()
    index_nodes(vs, emb, "concepts", [
        {"node_id": "company", "canonical_name": "Tesla Motors", "node_type": "Company", "aliases": []},
        {"node_id": "person", "canonical_name": "Nikola Tesla", "node_type": "Person", "aliases": ["Tesla"]},
    ])
    g = EntityGrounder(emb, vs, "concepts", sim_threshold=0.2, max_candidates=3)
    res = g.ground("Did Tesla Motors use Tesla coils?")
    node_ids = {e.node_id for e in res.entities}
    assert "company" in node_ids and "person" in node_ids


# example store uses content-stable ids -> accumulate across instances, idempotent
def test_example_store_stable_ids():
    emb, shared = HashingEmbedder(), FakeVectorStore()
    ExampleStore(emb, shared, "ex", sim_threshold=0.1).add([ExamplePair(question="how many drugs?", cypher="A")])
    ExampleStore(emb, shared, "ex", sim_threshold=0.1).add([ExamplePair(question="list all genes", cypher="B")])
    s3 = ExampleStore(emb, shared, "ex", sim_threshold=0.1)
    assert s3.count() == 2
    s3.add([ExamplePair(question="how many drugs?", cypher="A2")])  # idempotent (same question)
    assert s3.count() == 2


# generator/LLM error surfaces as a traced 'failed' response, not a crash
def test_generator_error_becomes_failed_response():
    fp = build_fake_pipeline(nodes=[], llm_responses=[])  # FakeLLM raises when empty
    resp = fp.answer(QueryRequest(question="x", max_attempts=2))
    assert resp.status == "failed"
    assert resp.trace_id and resp.attempts
    assert "error" in resp.message.lower()
    assert fp.graph.executed == []


# relaxation respects the caller's effective max_attempts
def test_relaxation_respects_max_attempts_one():
    empty = gen_json("MATCH (c:Concept) WHERE c.canonical_name = $metformin RETURN c.canonical_name AS name")
    fp = build_fake_pipeline(
        nodes=[{"node_id": "n2", "canonical_name": "Metformin", "node_type": "Drug", "aliases": []}],
        llm_responses=[empty, empty], grounding_threshold=0.2,
    )
    fp.graph.set_default([])
    resp = fp.answer(QueryRequest(question="metformin", max_attempts=1))
    assert resp.status == "empty"
    assert [a.kind for a in resp.attempts].count("relax") == 0


# ArcadeDB introspection runs as SQL, not Cypher
def test_arcadedb_introspect_uses_sql():
    seen = {}

    def handler(req):
        import json
        seen["lang"] = json.loads(req.content)["language"]
        return httpx.Response(200, json={"result": [{"name": "Concept", "type": "vertex"}]})

    c = ArcadeDBClient("http://db", "g", "ro", "pw", client=httpx.Client(transport=httpx.MockTransport(handler)))
    c.introspect()
    assert seen["lang"] == "sql"
