"""Acceptance criteria tests (spec section 9)."""

from __future__ import annotations

import pytest

from text2cypher.clients.base import ConfigurationError
from text2cypher.config import ArcadeConfig
from text2cypher.contracts import QueryRequest
from text2cypher.eval.harness import EvalReport, compare_reports
from text2cypher.testing import build_fake_pipeline, gen_json

MUTATIONS = [
    "MATCH (n:Concept) DELETE n",
    "CREATE (n:Concept {canonical_name: $x})",
    "MATCH (n:Concept) SET n.confidence = 1.0 RETURN n",
    "MATCH (n:Concept) REMOVE n.confidence RETURN n",
    "MERGE (n:Concept {canonical_name: 'x'}) RETURN n",
    "MATCH (n:Concept) DETACH DELETE n",
    "DROP INDEX foo",
    "FOREACH (x IN [1,2] | CREATE (:N))",
]


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_ac1_writes_never_reach_db(sample_nodes, mutation):
    fp = build_fake_pipeline(nodes=sample_nodes, llm_responses=[gen_json(mutation)])
    resp = fp.answer(QueryRequest(question="please mutate the graph"))
    assert resp.status == "failed"
    assert fp.graph.executed == []  # nothing hit the database
    # write was terminal — no repair attempts spent
    assert len(resp.attempts) == 1


def test_ac2_denylist_never_reaches_db(sample_nodes):
    fp = build_fake_pipeline(
        nodes=sample_nodes,
        llm_responses=[gen_json("MATCH (n:Concept) RETURN apoc.text.join([n.canonical_name], ',')")],
    )
    resp = fp.answer(QueryRequest(question="x", max_attempts=1))
    assert resp.status == "failed"
    assert fp.graph.executed == []


def test_ac3_grounded_values_are_bound_params(sample_nodes):
    cy = "MATCH (c:Concept)-[:TREATS]->(d:Concept) WHERE d.canonical_name = $type_2_diabetes_mellitus RETURN c.canonical_name AS drug"
    fp = build_fake_pipeline(nodes=sample_nodes, llm_responses=[gen_json(cy)], grounding_threshold=0.2)
    fp.graph.when("TREATS", [{"drug": "Metformin"}])
    resp = fp.answer(QueryRequest(question="What treats Type 2 Diabetes Mellitus?"))
    assert resp.status == "ok"
    executed_cypher, executed_params = fp.graph.executed[0]
    # The grounded value travels as a param, never inlined into the query text.
    assert "Type 2 Diabetes Mellitus" not in executed_cypher
    assert "Type 2 Diabetes Mellitus" in executed_params.values()


def test_ac4_full_trace(sample_nodes):
    fp = build_fake_pipeline(nodes=sample_nodes, llm_responses=[gen_json("MATCH (n:Concept) RETURN n.canonical_name AS name")])
    fp.graph.set_default([{"name": "Metformin"}])
    resp = fp.answer(QueryRequest(question="names"))
    assert resp.attempts and resp.validations and resp.executions
    assert resp.trace_id
    # round-trips through JSON (contract stability)
    from text2cypher.contracts import PipelineResponse

    PipelineResponse.model_validate_json(resp.model_dump_json())


def test_ac5_empty_not_retried_past_one_relaxation(sample_nodes):
    empty = gen_json("MATCH (c:Concept) WHERE c.canonical_name = $metformin RETURN c.canonical_name AS name")
    fp = build_fake_pipeline(nodes=sample_nodes, llm_responses=[empty, empty, empty, empty], grounding_threshold=0.2)
    fp.graph.set_default([])
    resp = fp.answer(QueryRequest(question="metformin", max_attempts=5))
    assert resp.status == "empty"
    assert [a.kind for a in resp.attempts].count("relax") <= 1
    # empty was reached on the first execution; only the single relaxation executed after
    assert len(resp.executions) <= 2


def test_ac6_eval_regression_gate():
    base = EvalReport(
        n=10, overall_accuracy=0.80, per_stratum={"1-hop": 0.9, "multi-hop": 0.7},
        per_stratum_counts={}, valid_query_rate=0.9, attempts_histogram={}, latency_p50_ms=1, latency_p95_ms=2,
    )
    good = base.model_copy(update={"overall_accuracy": 0.79, "per_stratum": {"1-hop": 0.9, "multi-hop": 0.68}})
    ok, regressions = compare_reports(good, base, threshold_pts=5.0)
    assert ok and not regressions

    bad = base.model_copy(update={"overall_accuracy": 0.60, "per_stratum": {"1-hop": 0.5, "multi-hop": 0.7}})
    ok2, regressions2 = compare_reports(bad, base, threshold_pts=5.0)
    assert not ok2 and any("overall" in r for r in regressions2)
    assert any("1-hop" in r for r in regressions2)


def test_ac7_readonly_credential_enforced():
    # No read-only user configured -> refuse to construct at the connection level.
    from text2cypher.clients.arcadedb import ArcadeDBClient

    cfg = ArcadeConfig(url="http://db:2480", database="g", user_readonly="", require_readonly_user=True)
    with pytest.raises(ConfigurationError):
        ArcadeDBClient.from_config(cfg)

    # With a read-only user, construction succeeds and read_only is asserted.
    cfg2 = ArcadeConfig(url="http://db:2480", database="g", user_readonly="ro", password_readonly="pw")
    client = ArcadeDBClient.from_config(cfg2)
    assert client.read_only is True
