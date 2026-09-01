from __future__ import annotations

from text2cypher.clients.base import GraphQueryError
from text2cypher.contracts import QueryRequest
from text2cypher.testing import build_fake_pipeline, gen_json

TREATS = "MATCH (drug:Concept)-[:TREATS]->(d:Concept) WHERE d.canonical_name = $type_2_diabetes_mellitus RETURN drug.canonical_name AS drug"


def test_happy_path(make_pipeline):
    fp = make_pipeline(llm_responses=[gen_json(TREATS)])
    fp.graph.when("TREATS", [{"drug": "Metformin"}])
    resp = fp.answer(QueryRequest(question="Which drug treats type 2 diabetes?"))
    assert resp.status == "ok"
    assert resp.rows == [{"drug": "Metformin"}]
    assert resp.final_cypher.strip().endswith("LIMIT 200")
    assert len(resp.attempts) == 1 and len(resp.validations) == 1 and len(resp.executions) == 1
    assert resp.trace_id


def test_repair_on_validation_reject(make_pipeline):
    bad = gen_json("MATCH (n:Concept) RETURN apoc.text.join([n.canonical_name], ',')")
    good = gen_json("MATCH (n:Concept) RETURN n.canonical_name AS name")
    fp = make_pipeline(llm_responses=[bad, good])
    fp.graph.set_default([{"name": "Metformin"}])
    resp = fp.answer(QueryRequest(question="list names"))
    assert resp.status == "ok"
    assert len(resp.attempts) == 2
    assert not resp.validations[0].passed and resp.validations[1].passed
    assert resp.attempts[1].kind == "repair"


def test_repair_on_execution_error(make_pipeline):
    first = gen_json("MATCH (n:Concept) RETURN n.canonical_name AS FIRST")
    second = gen_json("MATCH (n:Concept) RETURN n.canonical_name AS SECOND")
    fp = make_pipeline(llm_responses=[first, second])
    fp.graph.when_raises("FIRST", GraphQueryError("runtime boom"))
    fp.graph.when("SECOND", [{"SECOND": "Metformin"}])
    resp = fp.answer(QueryRequest(question="names"))
    assert resp.status == "ok"
    assert len(resp.executions) == 2
    assert resp.executions[0].status == "error"


def test_empty_then_relaxation_success(make_pipeline):
    empty = gen_json("MATCH (c:Concept) WHERE c.canonical_name = $metformin RETURN c.canonical_name AS name")
    relaxed = gen_json("MATCH (c:Concept) WHERE c.canonical_name CONTAINS $metformin RETURN c.canonical_name AS name")
    fp = make_pipeline(llm_responses=[empty, relaxed])
    fp.graph.add_rule(lambda c, p: "CONTAINS" in c, lambda c, p: [{"name": "Metformin"}])
    fp.graph.add_rule(lambda c, p: "CONTAINS" not in c, lambda c, p: [])
    resp = fp.answer(QueryRequest(question="metformin"))
    assert resp.status == "ok"
    assert "relaxation" in resp.message
    assert [a.kind for a in resp.attempts].count("relax") == 1


def test_empty_returns_empty_not_looped(make_pipeline):
    empty = gen_json("MATCH (c:Concept) WHERE c.canonical_name = $metformin RETURN c.canonical_name AS name")
    fp = make_pipeline(llm_responses=[empty, empty])  # relaxation also empty
    fp.graph.set_default([])
    resp = fp.answer(QueryRequest(question="metformin", max_attempts=3))
    assert resp.status == "empty"
    # exactly one relaxation was attempted, never loop-retried
    assert [a.kind for a in resp.attempts].count("relax") == 1
    assert "vector-retrieval fallback" in resp.message


def test_exhausted_attempts_failed(make_pipeline):
    apoc = gen_json("MATCH (n:Concept) RETURN apoc.x()")
    fp = make_pipeline(llm_responses=[apoc, apoc, apoc])
    resp = fp.answer(QueryRequest(question="x", max_attempts=3))
    assert resp.status == "failed"
    assert len(resp.attempts) == 3
    assert fp.graph.executed == []  # never executed a denylisted query


def test_empty_no_grounding_skips_relaxation():
    # Out-of-graph question grounds nothing -> must NOT relax into an arbitrary
    # full scan; return `empty` without a relaxation attempt.
    empty = gen_json("MATCH (n:Concept) WHERE n.canonical_name = 'X' RETURN n.canonical_name AS name")
    fp = build_fake_pipeline(nodes=[], llm_responses=[empty, empty])
    fp.graph.set_default([])
    resp = fp.answer(QueryRequest(question="what is the capital of france", max_attempts=3))
    assert resp.status == "empty"
    assert [a.kind for a in resp.attempts].count("relax") == 0
    assert len(resp.executions) == 1
    assert "grounded" in resp.message.lower()


def test_relaxation_dropping_anchor_rejected():
    # A relaxation that drops the grounded entity ($metformin) into a bare scan
    # must be rejected (not executed, not reported ok), even though it would
    # return rows.
    empty = gen_json("MATCH (c:Concept) WHERE c.canonical_name = $metformin RETURN c.canonical_name AS name")
    degenerate = gen_json("MATCH (c:Concept) RETURN c.canonical_name AS name")
    fp = build_fake_pipeline(
        nodes=[{"node_id": "n2", "canonical_name": "Metformin", "node_type": "Drug", "aliases": []}],
        llm_responses=[empty, degenerate], grounding_threshold=0.2,
    )
    fp.graph.add_rule(lambda c, p: "WHERE" in c, lambda c, p: [])
    fp.graph.add_rule(lambda c, p: "WHERE" not in c, lambda c, p: [{"name": "Metformin"}])
    resp = fp.answer(QueryRequest(question="show metformin", max_attempts=3))
    assert resp.status == "empty"
    assert [a.kind for a in resp.attempts].count("relax") == 1  # relaxation WAS generated
    assert len(fp.graph.executed) == 1  # ...but the anchorless scan never executed
    assert "WHERE" in fp.graph.executed[0][0]


def test_full_trace_always_present(make_pipeline):
    fp = make_pipeline(llm_responses=[gen_json(TREATS)])
    fp.graph.when("TREATS", [{"drug": "Metformin"}])
    resp = fp.answer(QueryRequest(question="Which drug treats type 2 diabetes?"))
    dumped = resp.model_dump()
    for key in ("attempts", "validations", "executions", "grounding", "trace_id", "final_cypher", "params"):
        assert key in dumped
