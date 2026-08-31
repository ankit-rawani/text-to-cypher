from __future__ import annotations

import pytest

from text2cypher.clients.fakes import FakeGraphClient
from text2cypher.pipeline.executor import Executor
from text2cypher.prewritten import PrewrittenTools, query_shape


def _tools(graph):
    return PrewrittenTools(Executor(graph), node_label="Concept", id_property="canonical_name")


def test_candidate_edges_param_bound():
    g = FakeGraphClient(default_rows=[{"source": "A", "target": "B"}])
    _tools(g).candidate_edges(["A", "B"])
    cypher, params = g.executed[0]
    assert "IN $ids" in cypher and params == {"ids": ["A", "B"]}
    assert "CREATE" not in cypher and "DELETE" not in cypher


def test_neighborhood_hops_clamped():
    g = FakeGraphClient(default_rows=[])
    _tools(g).neighborhood("Metformin", hops=99)
    cypher, params = g.executed[0]
    assert "[*1..5]" in cypher  # clamped to 5
    assert params == {"node_id": "Metformin"}


def test_node_lookup_alias_clause():
    g = FakeGraphClient(default_rows=[])
    _tools(g).node_lookup("Glucophage")
    cypher, params = g.executed[0]
    assert "$name IN n.aliases" in cypher and params == {"name": "Glucophage"}


def test_unsafe_identifier_rejected():
    g = FakeGraphClient()
    with pytest.raises(ValueError):
        PrewrittenTools(Executor(g), node_label="Concept); DROP")
    with pytest.raises(ValueError):
        PrewrittenTools(Executor(g)).candidate_edges(["a"], id_property="x OR 1=1")


def test_query_shape_collapses_values():
    s1 = query_shape("MATCH (c:Concept) WHERE c.canonical_name = $a RETURN c.name LIMIT 200")
    s2 = query_shape("MATCH (c:Concept) WHERE c.canonical_name = $b RETURN c.name LIMIT 9")
    assert s1 == s2
