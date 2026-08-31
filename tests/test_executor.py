from __future__ import annotations

from text2cypher.clients.base import GraphQueryError, GraphTimeout
from text2cypher.clients.fakes import FakeDocumentStore, FakeGraphClient
from text2cypher.pipeline.executor import Executor


def test_ok_status():
    g = FakeGraphClient(default_rows=[{"a": 1}])
    res = Executor(g).execute("MATCH (n) RETURN n")
    assert res.status == "ok" and res.row_count == 1
    assert res.latency_ms >= 0


def test_empty_status():
    g = FakeGraphClient(default_rows=[])
    res = Executor(g).execute("MATCH (n) RETURN n")
    assert res.status == "empty" and res.row_count == 0


def test_error_status():
    g = FakeGraphClient().when_raises("boom", GraphQueryError("bad cypher"))
    res = Executor(g).execute("boom")
    assert res.status == "error" and "bad cypher" in res.error_message


def test_timeout_status():
    g = FakeGraphClient().when_raises("slow", GraphTimeout("deadline"))
    res = Executor(g).execute("slow query")
    assert res.status == "timeout"


def test_row_cap_hard_truncation():
    # A client that ignores the LIMIT arg -> the executor still caps rows.
    class NoLimitGraph:
        read_only = True

        def query(self, cypher, params=None, *, timeout_s=None, limit=None):
            return [{"i": i} for i in range(500)]

        def introspect(self):
            return {}

    res = Executor(NoLimitGraph(), row_cap=200).execute("MATCH (n) RETURN n", row_cap=200)
    assert res.row_count == 200 and res.truncated is True


def test_params_are_passed_not_interpolated():
    g = FakeGraphClient(default_rows=[{"x": 1}])
    Executor(g).execute("MATCH (n) WHERE n.canonical_name = $name RETURN n", {"name": "Metformin"})
    cypher, params = g.executed[0]
    assert "$name" in cypher and "Metformin" not in cypher  # value only in params
    assert params == {"name": "Metformin"}


def test_hydration_enrich():
    g = FakeGraphClient(default_rows=[{"node_id": "n1", "canonical_name": "X"}])
    docs = FakeDocumentStore({"n1": {"full": "document"}})
    res = Executor(g, document_store=docs).execute("MATCH (n) RETURN n")
    assert res.rows[0]["_document"] == {"full": "document"}
