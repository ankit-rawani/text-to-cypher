from __future__ import annotations

import pytest

from text2cypher.contracts import GroundedEntity, GroundingResult, SchemaContext
from text2cypher.graph_config import GraphConfig, RelationshipDef, default_graph_config
from text2cypher.pipeline.schema_provider import SchemaProvider
from text2cypher.pipeline.validator import Denylist, Validator, load_denylist
from text2cypher.config import load_config


@pytest.fixture
def schema() -> SchemaContext:
    return SchemaProvider(default_graph_config()).get()


@pytest.fixture
def validator() -> Validator:
    cfg = load_config()
    return Validator(load_denylist(cfg.resolve_path(cfg.dialect.denylist_path)), row_cap=200)


def codes(report):
    return [i.code for i in report.issues]


def test_parse_error_repairable(validator, schema):
    r = validator.validate("MATCH (a)-[r]-(b", {}, schema)
    assert not r.passed and not r.terminal
    assert "PARSE_ERROR" in codes(r)


def test_empty_output(validator, schema):
    r = validator.validate("   ", {}, schema)
    assert not r.passed and "PARSE_ERROR" in codes(r)


def test_denylist_apoc(validator, schema):
    r = validator.validate("MATCH (n:Concept) RETURN apoc.text.join([n.canonical_name], ',')", {}, schema)
    assert not r.passed and "DENYLISTED" in codes(r)
    assert not r.terminal  # repairable (read query)


def test_denylist_db_procedure(validator, schema):
    r = validator.validate("CALL db.labels() YIELD label RETURN label", {}, schema)
    assert not r.passed and "DENYLISTED" in codes(r)


def test_write_op_terminal(validator, schema):
    r = validator.validate("MATCH (n:Concept) SET n.x = 1 RETURN n", {}, schema)
    assert not r.passed and r.terminal
    assert "WRITE_OP" in codes(r)


def test_write_plus_apoc_is_terminal(validator, schema):
    # A mutation must be terminal even if it also trips the dialect lint.
    r = validator.validate("MATCH (n:Concept) CREATE (m) RETURN apoc.x()", {}, schema)
    assert not r.passed and r.terminal


def test_unknown_label(validator, schema):
    r = validator.validate("MATCH (n:Widget) RETURN n.canonical_name", {}, schema)
    assert not r.passed and "UNKNOWN_LABEL" in codes(r)


def test_unknown_rel_type(validator, schema):
    r = validator.validate("MATCH (a:Concept)-[:FROBS]->(b:Concept) RETURN a", {}, schema)
    assert not r.passed and "UNKNOWN_REL_TYPE" in codes(r)


def test_unknown_property(validator, schema):
    r = validator.validate("MATCH (n:Concept) RETURN n.bogus_prop", {}, schema)
    assert not r.passed and "UNKNOWN_PROPERTY" in codes(r)


def test_known_property_ok(validator, schema):
    r = validator.validate("MATCH (n:Concept) RETURN n.canonical_name, n.confidence", {}, schema)
    assert r.passed, codes(r)


def test_limit_injected_and_passes(validator, schema):
    r = validator.validate("MATCH (n:Concept) RETURN n.canonical_name", {}, schema)
    assert r.passed
    assert "LIMIT_INJECTED" in codes(r)
    assert r.final_cypher.strip().endswith("LIMIT 200")


def test_limit_clamped(validator, schema):
    r = validator.validate("MATCH (n:Concept) RETURN n.canonical_name LIMIT 5000", {}, schema)
    assert r.passed and "LIMIT_CLAMPED" in codes(r)
    assert r.final_cypher.strip().endswith("LIMIT 200")


def test_param_discipline_reject(validator, schema):
    grounding = GroundingResult(
        entities=[
            GroundedEntity(
                mention="metformin", node_id="n2", canonical_name="Metformin",
                similarity=0.9, param_name="metformin",
            )
        ],
        params={"metformin": "Metformin"},
    )
    r = validator.validate(
        "MATCH (n:Concept) WHERE n.canonical_name = 'Metformin' RETURN n.canonical_name",
        {}, schema, grounding,
    )
    assert not r.passed and "PARAM_LITERAL" in codes(r)


def test_param_binding_passes(validator, schema):
    grounding = GroundingResult(
        entities=[GroundedEntity(mention="metformin", canonical_name="Metformin", param_name="metformin")],
        params={"metformin": "Metformin"},
    )
    r = validator.validate(
        "MATCH (n:Concept) WHERE n.canonical_name = $metformin RETURN n.canonical_name",
        {"metformin": "Metformin"}, schema, grounding,
    )
    assert r.passed, codes(r)


def test_direction_autofix():
    # Schema where endpoints differ by LABEL so direction is determinable.
    gcfg = GraphConfig(
        node_labels=["Drug", "Disease"],
        relationships=[RelationshipDef(type="TREATS", start=["Drug"], end=["Disease"])],
    )
    schema = SchemaProvider(gcfg).get()
    v = Validator(Denylist(), row_cap=200)
    # written backwards: (Disease)-[:TREATS]->(Drug)
    r = v.validate("MATCH (a:Disease)-[:TREATS]->(b:Drug) RETURN a", {}, schema)
    assert r.passed
    assert "DIRECTION_FIXED" in codes(r)
    assert "<-[:TREATS]-" in r.final_cypher


def test_direction_ambiguous_not_touched():
    schema = SchemaProvider(default_graph_config()).get()
    v = Validator(Denylist(), row_cap=200)
    q = "MATCH (a:Concept)-[:RELATES_TO]->(b:Concept) RETURN a"
    r = v.validate(q, {}, schema)
    assert "DIRECTION_FIXED" not in codes(r)
