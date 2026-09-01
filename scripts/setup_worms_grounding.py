"""Index the disgust-mollusca concept nodes into Qdrant for grounding, and seed
a few schema-matched examples. Reads all config (incl. .env) via load_config.

Run once before `t2c ask`:  python scripts/setup_worms_grounding.py
"""

from __future__ import annotations

from text2cypher.clients.arcadedb import ArcadeDBClient
from text2cypher.clients.qdrant import QdrantClient
from text2cypher.config import load_config
from text2cypher.contracts import ExamplePair
from text2cypher.embeddings import build_embedder
from text2cypher.ingest import index_nodes
from text2cypher.pipeline.example_store import ExampleStore

GRAPH = "disgust-mollusca"

EXAMPLES = [
    ("What does climatic changes cause?",
     "MATCH (a:Node)-[:causes]->(b:Node) WHERE a.name = $x RETURN b.name AS effect LIMIT 200"),
    ("What causes gene frequencies?",
     "MATCH (a:Node)-[:causes]->(b:Node) WHERE b.name = $x RETURN a.name AS cause LIMIT 200"),
    ("What enables species classification?",
     "MATCH (a:Node)-[:enables]->(b:Node) WHERE b.name = $x RETURN a.name AS enabler LIMIT 200"),
    ("What is a component of Dreissena?",
     "MATCH (a:Node)-[:is_component_of]->(b:Node) WHERE b.name = $x RETURN a.name AS part LIMIT 200"),
    ("How many relationships of each type are in the concept graph?",
     "MATCH (a:Node)-[r]->(b:Node) WHERE a.graph = 'disgust-mollusca' RETURN type(r) AS relationship, count(*) AS n ORDER BY n DESC LIMIT 200"),
    ("Which established causal links exist?",
     "MATCH (a:Node)-[r:causes]->(b:Node) WHERE r.status = 'established' RETURN a.name AS cause, b.name AS effect LIMIT 200"),
    ("What methods measure climatic changes?",
     "MATCH (m:Node)-[:measures]->(b:Node) WHERE b.name = $x RETURN m.name AS method LIMIT 200"),
]


def main() -> None:
    cfg = load_config()
    emb = build_embedder(cfg.embeddings)
    arcade = ArcadeDBClient.from_config(cfg.arcadedb)
    qc = QdrantClient.from_config(cfg.qdrant)

    rows = arcade.query(
        "MATCH (n:Node) WHERE n.graph = $g "
        "RETURN n.id AS node_id, n.name AS canonical_name, n.kind AS node_type",
        {"g": GRAPH},
        limit=10000,
    )
    print(f"fetched {len(rows)} concept nodes from ArcadeDB")

    n = index_nodes(qc, emb, cfg.qdrant.node_collection, rows)
    print(f"indexed {n} nodes into Qdrant collection '{cfg.qdrant.node_collection}'")

    store = ExampleStore(
        emb, qc, cfg.qdrant.example_collection,
        top_k=cfg.examples.top_k, sim_threshold=cfg.examples.sim_threshold,
    )
    store.add([ExamplePair(question=q, cypher=c, tags=["concept"]) for q, c in EXAMPLES])
    print(f"seeded {len(EXAMPLES)} examples into '{cfg.qdrant.example_collection}'")


if __name__ == "__main__":
    main()
