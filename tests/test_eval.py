from __future__ import annotations

from text2cypher.eval import result_set_equal
from text2cypher.eval.harness import EvalHarness, GoldItem, load_gold
from text2cypher.testing import build_fake_pipeline, gen_json


def test_result_set_equal_order_insensitive():
    a = [{"x": 1}, {"x": 2}]
    b = [{"x": 2}, {"x": 1}]
    assert result_set_equal(a, b)


def test_result_set_equal_rounds_numerics():
    assert result_set_equal([{"v": 1.0000001}], [{"v": 1.0000002}])
    assert not result_set_equal([{"v": 1.0}], [{"v": 2.0}])


def test_result_set_equal_multiset():
    assert not result_set_equal([{"x": 1}, {"x": 1}], [{"x": 1}])


def test_empty_sets_equal():
    assert result_set_equal([], [])
    assert result_set_equal(None, [])


def test_load_gold(tmp_path):
    p = tmp_path / "gold.jsonl"
    p.write_text('{"question":"q","cypher":"MATCH (n) RETURN n","stratum":"1-hop"}\n', encoding="utf-8")
    items = load_gold(p)
    assert items[0].stratum == "1-hop"


def test_harness_scores_and_strata(sample_nodes):
    # Param-free on purpose: this test exercises the EVAL HARNESS (scoring /
    # strata), not entity grounding — so the scripted query must not depend on
    # the offline embedder producing any particular $param.
    treats = "MATCH (drug:Concept)-[:TREATS]->(d:Concept) RETURN drug.canonical_name AS drug"

    def handler(_msgs):
        return gen_json(treats)

    fp = build_fake_pipeline(nodes=sample_nodes, llm_handler=handler, grounding_threshold=0.2)
    fp.graph.when("TREATS", [{"drug": "Metformin"}])
    fp.graph.add_rule(lambda c, p: "TREATS" not in c, lambda c, p: [])

    gold = [
        GoldItem(id="ok", stratum="1-hop", question="drugs for T2DM",
                 cypher="MATCH (drug:Concept)-[:TREATS]->(d:Concept) WHERE d.canonical_name=$n RETURN drug.canonical_name AS drug",
                 params={"n": "Type 2 Diabetes Mellitus"}),
        GoldItem(id="oog", stratum="out-of-graph", question="unrelated",
                 cypher="MATCH (c:Concept) WHERE c.canonical_name=$n RETURN c.canonical_name AS name",
                 params={"n": "does-not-exist"}),
    ]
    report = EvalHarness(fp.pipeline, fp.graph).run(gold)
    assert report.n == 2
    assert "1-hop" in report.per_stratum and "out-of-graph" in report.per_stratum
    assert report.per_stratum["1-hop"] == 1.0
    # summary is renderable
    assert "per-stratum" in report.summary()
