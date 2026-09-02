from __future__ import annotations

from text2cypher.clients.fakes import FakeVectorStore
from text2cypher.contracts import ExamplePair
from text2cypher.embeddings.hashing import HashingEmbedder
from text2cypher.graph_config import default_graph_config
from text2cypher.ingest import index_nodes
from text2cypher.pipeline.entity_grounder import EntityGrounder, extract_mentions
from text2cypher.pipeline.example_store import ExampleStore
from text2cypher.pipeline.schema_provider import SchemaProvider


# ---- mention extraction -------------------------------------------------


def test_extract_mentions_quoted_and_capitalized():
    m = extract_mentions('Which drugs treat "Type 2 Diabetes" and Metformin?')
    joined = " | ".join(m).lower()
    assert "type 2 diabetes" in joined
    assert "metformin" in joined


def test_generic_single_words_not_extracted():
    # meta/schema words must not become standalone mentions (they ground to
    # nodes that merely contain them -> spurious params).
    m = [x.lower() for x in extract_mentions("How many relationships of each type, and which methods and processes?")]
    for w in ("relationships", "type", "methods", "processes"):
        assert w not in m, w
    # research/query meta words must not become standalone mentions either
    m2 = [x.lower() for x in extract_mentions("What contradictions and gaps exist in the current understanding?")]
    for w in ("contradictions", "gaps", "understanding"):
        assert w not in m2, w
    # real single-word entities still survive
    assert "metformin" in [x.lower() for x in extract_mentions("Tell me about Metformin")]


# ---- grounder -----------------------------------------------------------


def _grounder(nodes, threshold=0.2):
    emb = HashingEmbedder(dim=256)
    store = FakeVectorStore()
    index_nodes(store, emb, "concepts", nodes)
    return EntityGrounder(emb, store, "concepts", sim_threshold=threshold, max_candidates=3)


def test_grounding_binds_params(sample_nodes):
    g = _grounder(sample_nodes)
    result = g.ground("What treats Metformin?")
    names = {e.canonical_name for e in result.entities}
    assert "Metformin" in names
    # every resolved entity has a param bound to its canonical name
    for e in result.entities:
        if e.canonical_name:
            assert e.param_name in result.params
            assert result.params[e.param_name] == e.canonical_name


def test_grounding_alias_match(sample_nodes):
    g = _grounder(sample_nodes)
    result = g.ground("Tell me about Glucophage")  # alias of Metformin
    assert any(e.canonical_name == "Metformin" for e in result.entities)


def test_grounding_disabled():
    g = EntityGrounder(HashingEmbedder(), FakeVectorStore(), "concepts", enabled=False)
    assert g.ground("anything").entities == []


# ---- example store ------------------------------------------------------


def test_example_store_threshold_gating():
    emb = HashingEmbedder(dim=256)
    store = ExampleStore(emb, FakeVectorStore(), "ex", top_k=3, sim_threshold=0.9)
    store.add([ExamplePair(question="How many drugs are there?", cypher="MATCH (n) RETURN count(n)")])
    # a very different question below the high threshold -> nothing injected
    assert store.retrieve("completely unrelated astronomy question xyz") == []


def test_example_store_retrieves_similar():
    emb = HashingEmbedder(dim=256)
    store = ExampleStore(emb, FakeVectorStore(), "ex", top_k=3, sim_threshold=0.2)
    store.add([ExamplePair(question="Which drugs treat a disease?", cypher="MATCH (d)-[:TREATS]->(x) RETURN d")])
    hits = store.retrieve("which drugs treat diabetes")
    assert hits and hits[0].cypher.startswith("MATCH")


# ---- schema provider ----------------------------------------------------


def test_schema_context_shape():
    ctx = SchemaProvider(default_graph_config()).get()
    assert "Concept" in ctx.labels
    assert "TREATS" in ctx.rel_types
    assert "canonical_name" in ctx.properties["Concept"]
    assert ctx.version_hash
    assert ("Drug", "TREATS", "Disease") in ctx.edges


def test_schema_cache_ttl():
    calls = {"n": 0}
    cfg = default_graph_config()

    class Counter(SchemaProvider):
        def _build(self):
            calls["n"] += 1
            return super()._build()

    t = [0.0]
    sp = Counter(cfg, ttl_s=100, clock=lambda: t[0])
    sp.get()
    sp.get()
    assert calls["n"] == 1  # cached
    t[0] = 200
    sp.get()
    assert calls["n"] == 2  # expired -> rebuilt
