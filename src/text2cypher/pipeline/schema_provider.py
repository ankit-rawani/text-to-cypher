"""Stage 1 — SchemaProvider.

Turns the user-configured :class:`GraphConfig` (optionally merged with live
ArcadeDB introspection) into a :class:`SchemaContext`: labels, relationship
types, per-label property types, enum values, connectivity edges, a stable
version hash, and a compact rendered block for the prompt. Cached with a TTL.
"""

from __future__ import annotations

import time

from ..clients.base import GraphClient
from ..contracts import SchemaContext
from ..graph_config import GraphConfig


class SchemaProvider:
    def __init__(
        self,
        graph_config: GraphConfig,
        graph_client: GraphClient | None = None,
        *,
        ttl_s: float = 300.0,
        clock=time.monotonic,
    ) -> None:
        self._config = graph_config
        self._client = graph_client
        self._ttl = ttl_s
        self._clock = clock
        self._cached: SchemaContext | None = None
        self._cached_at: float = 0.0

    def get(self, force_refresh: bool = False) -> SchemaContext:
        now = self._clock()
        if (
            not force_refresh
            and self._cached is not None
            and (now - self._cached_at) < self._ttl
        ):
            return self._cached
        ctx = self._build()
        self._cached = ctx
        self._cached_at = now
        return ctx

    def refresh(self) -> SchemaContext:
        return self.get(force_refresh=True)

    # ------------------------------------------------------------------
    def _build(self) -> SchemaContext:
        cfg = self._config

        labels = list(cfg.node_labels)
        rel_types = cfg.relationship_types()

        # Optional live introspection — augment discovered labels/rel types.
        introspected_hash = ""
        if self._client is not None:
            try:
                raw = self._client.introspect()
                introspected_hash = _hash_introspection(raw)
                for name in _discover_labels(raw):
                    if name not in labels:
                        labels.append(name)
            except Exception:
                introspected_hash = ""

        properties: dict[str, dict[str, str]] = {}
        for label in cfg.node_labels:
            properties[label] = dict(cfg.node_properties)
        for rel in rel_types:
            properties[rel] = dict(cfg.edge_properties)

        enums: dict[str, list[str]] = {}
        for label in cfg.node_labels:
            enums[f"{label}.{cfg.node_type_property}"] = list(cfg.node_type_values)
        if cfg.status_values:
            for rel in rel_types:
                enums[f"{rel}.{cfg.status_property}"] = list(cfg.status_values)

        edges = cfg.edges()
        rendered = self._render(cfg, labels, rel_types, edges)
        version_hash = cfg.version_hash()
        if introspected_hash:
            version_hash = f"{version_hash}.{introspected_hash}"

        return SchemaContext(
            rendered=rendered,
            labels=labels,
            rel_types=rel_types,
            properties=properties,
            version_hash=version_hash,
            enums=enums,
            edges=edges,
            node_type_property=cfg.node_type_property,
        )

    def _render(
        self,
        cfg: GraphConfig,
        labels: list[str],
        rel_types: list[str],
        edges: list[tuple[str, str, str]],
    ) -> str:
        lines: list[str] = [f"# Graph schema — database `{cfg.database}` (READ-ONLY)"]

        lines.append("Node labels:")
        for label in labels:
            desc = cfg.descriptions.get(label, "")
            suffix = f"  — {desc}" if desc else ""
            lines.append(f"  (:{label}){suffix}")
        lines.append("Node properties (on each node label):")
        for prop, ptype in cfg.node_properties.items():
            enum = cfg.node_type_values if prop == cfg.node_type_property else None
            enum_txt = f"  [one of: {', '.join(enum)}]" if enum else ""
            lines.append(f"    {prop}: {ptype}{enum_txt}")

        lines.append("Relationship types:")
        for rel in cfg.relationships:
            desc = f"  — {rel.description}" if rel.description else ""
            arrow = "->" if rel.directed else "-"
            start = "|".join(rel.start) if rel.start else "Concept"
            end = "|".join(rel.end) if rel.end else "Concept"
            lines.append(f"  ({start})-[:{rel.type}]{arrow}({end}){desc}")
        lines.append("Relationship properties (on each relationship):")
        for prop, ptype in cfg.edge_properties.items():
            enum = cfg.status_values if prop == cfg.status_property else None
            enum_txt = f"  [one of: {', '.join(enum)}]" if enum else ""
            lines.append(f"    {prop}: {ptype}{enum_txt}")

        lines.append("Guidance:")
        lines.append("  - Read-only queries only; bind entity values as $params.")
        lines.append("  - node_type is a PROPERTY of a node, not a label.")
        lines.append("  - Match entities on canonical_name or aliases.")
        return "\n".join(lines)


def _discover_labels(raw: dict) -> list[str]:
    out: list[str] = []
    for row in raw.get("types", []) or []:
        name = row.get("name") if isinstance(row, dict) else None
        typ = row.get("type") if isinstance(row, dict) else None
        if name and typ in (None, "vertex", "v", "document"):
            out.append(name)
    return out


def _hash_introspection(raw: dict) -> str:
    import hashlib
    import json

    try:
        payload = json.dumps(raw, sort_keys=True, default=str)
    except Exception:
        return ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


__all__ = ["SchemaProvider"]
