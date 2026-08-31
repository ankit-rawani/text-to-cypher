from __future__ import annotations

import httpx

from text2cypher.clients.fakes import FakeVectorStore
from text2cypher.embeddings import build_embedder
from text2cypher.embeddings.hashing import HashingEmbedder
from text2cypher.embeddings.openai_embedder import OpenAIEmbedder
from text2cypher.config import EmbeddingConfig
from text2cypher.pipeline.example_store import ExampleStore


def test_build_embedder_factory():
    assert isinstance(build_embedder(EmbeddingConfig(provider="hashing")), HashingEmbedder)
    assert isinstance(build_embedder(EmbeddingConfig(provider="openai", endpoint="http://x", model="m")), OpenAIEmbedder)


def test_openai_embedder_over_mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        import json
        n = len(json.loads(request.content)["input"])
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(n)]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    emb = OpenAIEmbedder("http://llm/v1", "text-embed", client=client)
    vecs = emb.embed(["a", "b"])
    assert len(vecs) == 2 and vecs[0] == [0.1, 0.2, 0.3]
    assert emb.dim == 3  # dim inferred from response
    assert emb.embed([]) == []


def test_example_store_import_and_list(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text(
        '{"question": "how many drugs?", "cypher": "MATCH (n) RETURN count(n)"}\n'
        '{"question": "list genes", "cypher": "MATCH (g:Concept) RETURN g", "tags": ["1-hop"]}\n',
        encoding="utf-8",
    )
    store = ExampleStore(HashingEmbedder(), FakeVectorStore(), "ex", sim_threshold=0.1)
    n = store.import_jsonl(p)
    assert n == 2
    listed = store.list()
    assert {e.question for e in listed} == {"how many drugs?", "list genes"}
