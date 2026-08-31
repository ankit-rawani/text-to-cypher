"""GraphConfig — the *user-configured* graph schema.

The concept graph's shape is not hard-coded: node kinds, relationship kinds,
statuses, and the fixed node/edge attribute sets are declared here (or loaded
from ``config/graph_config.yaml``). :class:`~text2cypher.pipeline.schema_provider.SchemaProvider`
turns this (optionally merged with live introspection) into the prompt-facing
:class:`~text2cypher.contracts.SchemaContext`.

The default reflects the node/edge fields named in the spec:
node — ``canonical_name, aliases, node_type, definition, field_tags, confidence``;
edge — ``relationship, evidence_dois, status, directed, rationale``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RelationshipDef(BaseModel):
    type: str
    start: list[str] = Field(default_factory=list)  # allowed start node_type values
    end: list[str] = Field(default_factory=list)  # allowed end node_type values
    directed: bool = True
    description: str = ""


class GraphConfig(BaseModel):
    database: str = "concept_graph"
    node_labels: list[str] = Field(default_factory=lambda: ["Concept"])
    node_type_property: str = "node_type"
    node_type_values: list[str] = Field(
        default_factory=lambda: [
            "Disease", "Drug", "Gene", "Protein", "Phenotype", "Pathway",
            "Method", "Dataset", "Field", "Concept",
        ]
    )
    node_properties: dict[str, str] = Field(
        default_factory=lambda: {
            "canonical_name": "string",
            "aliases": "list<string>",
            "node_type": "string",
            "definition": "string",
            "field_tags": "list<string>",
            "confidence": "float",
        }
    )
    relationships: list[RelationshipDef] = Field(
        default_factory=lambda: [
            RelationshipDef(type="RELATES_TO", description="generic association between concepts"),
            RelationshipDef(type="CAUSES", start=["Disease", "Gene", "Drug"], end=["Disease", "Phenotype"], description="causal link"),
            RelationshipDef(type="TREATS", start=["Drug"], end=["Disease", "Phenotype"], description="therapeutic effect"),
            RelationshipDef(type="ASSOCIATED_WITH", description="statistical / observed association"),
            RelationshipDef(type="PART_OF", description="mereological / hierarchical containment"),
            RelationshipDef(type="SUBTYPE_OF", description="taxonomic specialization"),
            RelationshipDef(type="MEASURED_BY", start=["Phenotype", "Disease"], end=["Method", "Dataset"], description="measurement / assay"),
            RelationshipDef(type="INTERACTS_WITH", start=["Gene", "Protein", "Drug"], end=["Gene", "Protein", "Drug"], directed=False, description="interaction"),
        ]
    )
    edge_properties: dict[str, str] = Field(
        default_factory=lambda: {
            "relationship": "string",
            "evidence_dois": "list<string>",
            "status": "string",
            "directed": "boolean",
            "rationale": "string",
        }
    )
    status_property: str = "status"
    status_values: list[str] = Field(
        default_factory=lambda: ["proposed", "supported", "refuted", "deprecated"]
    )
    descriptions: dict[str, str] = Field(default_factory=dict)

    def relationship_types(self) -> list[str]:
        return [r.type for r in self.relationships]

    def edges(self) -> list[tuple[str, str, str]]:
        """Expand relationships into (start_label, type, end_label) triples.

        Endpoints are node_type values when declared; otherwise every node label
        pairs with itself. Used for the direction autofix — a triple present in
        exactly one orientation lets an unambiguously-backwards arrow be flipped.
        """
        triples: list[tuple[str, str, str]] = []
        for rel in self.relationships:
            starts = rel.start or list(self.node_labels)
            ends = rel.end or list(self.node_labels)
            for s in starts:
                for e in ends:
                    triples.append((s, rel.type, e))
                    if not rel.directed:
                        triples.append((e, rel.type, s))
        return triples

    def version_hash(self) -> str:
        payload = self.model_dump_json()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def default_graph_config() -> GraphConfig:
    return GraphConfig()


def load_graph_config(path: str | os.PathLike[str] | None) -> GraphConfig:
    """Load a GraphConfig from YAML, or the default when path is missing/absent."""
    if path is None:
        return default_graph_config()
    p = Path(path)
    if not p.exists():
        return default_graph_config()
    with open(p, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    return GraphConfig.model_validate(data)


__all__ = ["GraphConfig", "RelationshipDef", "default_graph_config", "load_graph_config"]
