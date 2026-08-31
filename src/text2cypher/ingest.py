"""Helpers to populate the vector store: node vectors and example pairs.

Used by the CLI (``t2c examples ...``) and by the offline/test harness. Node
text embeds canonical_name + aliases + node_type + definition so that
alias-phrased mentions land near their canonical concept.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .clients.base import VectorPoint, VectorSearchClient
from .embeddings.base import BaseEmbedder


def build_node_text(node: dict[str, Any]) -> str:
    parts: list[str] = []
    if node.get("canonical_name"):
        parts.append(str(node["canonical_name"]))
    for alias in node.get("aliases", []) or []:
        parts.append(str(alias))
    if node.get("node_type"):
        parts.append(str(node["node_type"]))
    if node.get("definition"):
        parts.append(str(node["definition"]))
    return " ".join(parts)


def _int_id(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:15], 16)


def index_nodes(
    vector_store: VectorSearchClient,
    embedder: BaseEmbedder,
    collection: str,
    nodes: list[dict[str, Any]],
    *,
    id_field: str = "node_id",
) -> int:
    vector_store.ensure_collection(collection, embedder.dim)
    points: list[VectorPoint] = []
    for i, node in enumerate(nodes):
        nid = str(node.get(id_field) or node.get("id") or f"n{i}")
        vector = embedder.embed_one(build_node_text(node))
        payload = {
            "node_id": nid,
            "canonical_name": node.get("canonical_name"),
            "node_type": node.get("node_type"),
            "aliases": list(node.get("aliases", []) or []),
        }
        points.append(VectorPoint(id=_int_id(nid), vector=vector, payload=payload))
    vector_store.upsert(collection, points)
    return len(points)


__all__ = ["index_nodes", "build_node_text"]
