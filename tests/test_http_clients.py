"""Real HTTP clients exercised against httpx.MockTransport (no network)."""

from __future__ import annotations

import base64

import httpx
import pytest

from text2cypher.clients.arcadedb import ArcadeDBClient
from text2cypher.clients.base import GraphAuthError, GraphQueryError, GraphTimeout, VectorPoint
from text2cypher.clients.llm import OpenAIChatClient
from text2cypher.clients.qdrant import QdrantClient


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---- ArcadeDB -----------------------------------------------------------


def test_arcadedb_query_binds_params_and_auth():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"result": [{"name": "Metformin"}]})

    client = ArcadeDBClient("http://db:2480", "g", "ro", "pw", client=mock_client(handler))
    rows = client.query("MATCH (n) WHERE n.canonical_name = $x RETURN n", {"x": "Metformin"}, limit=10)
    assert rows == [{"name": "Metformin"}]
    assert captured["body"]["params"] == {"x": "Metformin"}
    assert captured["body"]["language"] == "cypher"
    assert captured["body"]["limit"] == 10
    expected = "Basic " + base64.b64encode(b"ro:pw").decode()
    assert captured["auth"] == expected


def test_arcadedb_auth_error():
    client = ArcadeDBClient("http://db", "g", "ro", "pw", client=mock_client(lambda r: httpx.Response(403, json={"error": "no"})))
    with pytest.raises(GraphAuthError):
        client.query("MATCH (n) RETURN n")


def test_arcadedb_query_error():
    client = ArcadeDBClient("http://db", "g", "ro", "pw", client=mock_client(lambda r: httpx.Response(500, json={"error": "syntax"})))
    with pytest.raises(GraphQueryError):
        client.query("MATCH (n) RETURN n")


def test_arcadedb_timeout():
    def handler(request):
        raise httpx.TimeoutException("deadline", request=request)

    client = ArcadeDBClient("http://db", "g", "ro", "pw", client=mock_client(handler))
    with pytest.raises(GraphTimeout):
        client.query("MATCH (n) RETURN n")


# ---- Qdrant -------------------------------------------------------------


def test_qdrant_search_parses_hits():
    def handler(request):
        return httpx.Response(200, json={"result": [{"id": 1, "score": 0.9, "payload": {"canonical_name": "Metformin"}}]})

    client = QdrantClient("http://q:6333", client=mock_client(handler))
    hits = client.search("concepts", [0.1, 0.2], limit=1)
    assert hits[0].score == 0.9 and hits[0].payload["canonical_name"] == "Metformin"


def test_qdrant_upsert_ok():
    def handler(request):
        return httpx.Response(200, json={"result": {"status": "ok"}})

    client = QdrantClient("http://q:6333", client=mock_client(handler))
    client.upsert("concepts", [VectorPoint(id=1, vector=[0.1], payload={"a": 1})])  # no raise


# ---- LLM ----------------------------------------------------------------


def test_llm_complete_extracts_content():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"cypher":"RETURN 1"}'}}]})

    client = OpenAIChatClient("http://llm/v1", "m", client=mock_client(handler))
    out = client.complete([{"role": "user", "content": "hi"}])
    assert out == '{"cypher":"RETURN 1"}'


def test_llm_error_raises():
    from text2cypher.clients.base import ClientError

    client = OpenAIChatClient("http://llm/v1", "m", client=mock_client(lambda r: httpx.Response(500, text="boom")))
    with pytest.raises(ClientError):
        client.complete([{"role": "user", "content": "hi"}])
