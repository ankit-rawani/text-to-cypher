from __future__ import annotations

import pytest

from text2cypher.cypher import analyze, ensure_limit, flip_direction


def test_parse_ok_simple():
    a = analyze("MATCH (n:Concept) RETURN n.canonical_name")
    assert a.ok and a.error is None
    assert a.clauses[0][0] == "MATCH"
    assert a.labels() == {"Concept"}


@pytest.mark.parametrize(
    "q,err_contains",
    [
        ("", "empty"),
        ("MATCH (a)-[r]-(b", "unclosed"),
        ("MATCH (a) RETURN ]", "unbalanced"),
        ("MATCH (a:Concept) WHERE n.x = 'oops", "unterminated"),
        ("RETURN (n) /* open", "unterminated"),
        ("(n) RETURN n", "reading clause"),
    ],
)
def test_parse_errors(q, err_contains):
    a = analyze(q)
    assert not a.ok
    assert err_contains in a.error


def test_labels_reltypes_direction():
    a = analyze("MATCH (c:Concept)-[r:TREATS]->(d:Disease) RETURN d")
    assert a.labels() == {"Concept", "Disease"}
    assert a.rel_types() == {"TREATS"}
    assert a.rel_patterns[0].direction == "ltr"

    a2 = analyze("MATCH (c:Concept)<-[:LINKS]-(d:Concept) RETURN c")
    assert a2.rel_patterns[0].direction == "rtl"

    a3 = analyze("MATCH (c:Concept)-[:LINKS]-(d:Concept) RETURN c")
    assert a3.rel_patterns[0].direction == "undirected"


def test_write_ops_detected():
    for kw in ["CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DROP", "FOREACH"]:
        a = analyze(f"MATCH (n) {kw} (m) RETURN n")
        assert any(w.keyword == kw for w in a.write_ops), kw


def test_write_op_not_flagged_as_property():
    a = analyze("MATCH (n:Concept) RETURN n.create AS x")
    assert a.write_ops == []


def test_calls_and_property_access():
    a = analyze("MATCH (n:Concept) RETURN apoc.text.join(collect(n.canonical_name), ',')")
    names = {c.name for c in a.calls}
    assert "apoc.text.join" in names and "collect" in names
    assert ("n", "canonical_name") in [(p.variable, p.key) for p in a.property_accesses]
    # apoc.text must NOT be a property access
    assert ("apoc", "text") not in [(p.variable, p.key) for p in a.property_accesses]


def test_procedure_call_after_CALL():
    a = analyze("CALL db.labels() YIELD label RETURN label")
    assert any(c.name == "db.labels" and c.is_procedure for c in a.calls)


def test_params_and_strings():
    a = analyze("MATCH (n:Concept) WHERE n.canonical_name = $x OR n.canonical_name = 'Foo' RETURN n")
    assert a.params == {"x"}
    assert [t.decoded for t in a.string_literals] == ["Foo"]


def test_limit_detection():
    assert analyze("MATCH (n) RETURN n LIMIT 50").limit.value == 50
    assert analyze("MATCH (n) RETURN n").limit.present is False
    # subquery LIMIT is not the final limit
    a = analyze("MATCH (n) CALL { MATCH (m) RETURN m LIMIT 5 } RETURN n")
    assert a.limit.present is False


def test_flip_direction_length_preserving():
    q = "MATCH (a:Concept)-[r:LINKS]->(b:Concept) RETURN a"
    a = analyze(q)
    flipped = flip_direction(q, a.rel_patterns[0])
    assert "<-[r:LINKS]-" in flipped
    assert len(flipped) == len(q)
    # flipping back returns original
    a2 = analyze(flipped)
    assert flip_direction(flipped, a2.rel_patterns[0]) == q


def test_flip_bracketless():
    q = "MATCH (a:Concept)-->(b:Concept) RETURN a"
    a = analyze(q)
    assert "<--" in flip_direction(q, a.rel_patterns[0])


def test_ensure_limit_inject_clamp_noop():
    a = analyze("MATCH (n:Concept) RETURN n")
    assert ensure_limit("MATCH (n:Concept) RETURN n", 200, a) == ("MATCH (n:Concept) RETURN n LIMIT 200", "injected")

    a2 = analyze("MATCH (n:Concept) RETURN n LIMIT 999")
    out, action = ensure_limit("MATCH (n:Concept) RETURN n LIMIT 999", 200, a2)
    assert action == "clamped" and out.endswith("LIMIT 200")

    a3 = analyze("MATCH (n:Concept) RETURN n LIMIT 10")
    assert ensure_limit("MATCH (n:Concept) RETURN n LIMIT 10", 200, a3) == ("MATCH (n:Concept) RETURN n LIMIT 10", None)


def test_ensure_limit_with_semicolon():
    q = "MATCH (n:Concept) RETURN n;"
    a = analyze(q)
    out, action = ensure_limit(q, 50, a)
    assert action == "injected" and out == "MATCH (n:Concept) RETURN n LIMIT 50;"


def test_grouping_paren_not_node_pattern():
    a = analyze("MATCH (n:Concept) WHERE (n.confidence + 1) > 2 RETURN n")
    # only the (n:Concept) node pattern, not the (n.confidence + 1) group
    assert len(a.node_patterns) == 1
    assert a.node_patterns[0].labels == ["Concept"]


def test_inline_property_keys():
    a = analyze("MATCH (n:Concept {canonical_name: $x, bogus: 1}) RETURN n")
    assert set(a.node_patterns[0].prop_keys) == {"canonical_name", "bogus"}
