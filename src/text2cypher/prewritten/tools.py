"""Prewritten parameterized tools (spec section 5).

These bypass the LLM entirely: fixed query shapes with bound ``$params``. They
sit in the high-frequency / low-complexity (and known high-complexity) corner of
the complexity x frequency quadrant, where a hand-written parameterized tool
beats round-tripping through the generator. They are read-only and go through
the same guarded :class:`~text2cypher.pipeline.executor.Executor`.
"""

from __future__ import annotations

import re

from ..contracts import ExecutionResult
from ..cypher import analyze
from ..pipeline.executor import Executor

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_ident(name: str, kind: str) -> str:
    """Guard identifiers that are interpolated into query text (not params)."""
    if not _IDENT_RE.match(name):
        raise ValueError(f"Unsafe {kind} identifier: {name!r}")
    return name


class PrewrittenTools:
    def __init__(
        self,
        executor: Executor,
        *,
        node_label: str = "Concept",
        id_property: str = "canonical_name",
        row_cap: int = 200,
    ) -> None:
        self._executor = executor
        self._label = _safe_ident(node_label, "label")
        self._id = _safe_ident(id_property, "property")
        self._row_cap = row_cap

    def candidate_edges(
        self, ids: list[str], *, id_property: str | None = None, limit: int | None = None
    ) -> ExecutionResult:
        """Fetch edges among a candidate node set (both endpoints in ``$ids``)."""
        idp = _safe_ident(id_property, "property") if id_property else self._id
        cap = limit or self._row_cap
        cypher = (
            f"MATCH (a:{self._label})-[r]->(b:{self._label}) "
            f"WHERE a.{idp} IN $ids AND b.{idp} IN $ids "
            f"RETURN a.{idp} AS source, b.{idp} AS target, "
            f"type(r) AS type, r.relationship AS relationship, r.status AS status "
            f"LIMIT {int(cap)}"
        )
        return self._executor.execute(cypher, {"ids": list(ids)}, row_cap=cap)

    def neighborhood(
        self, node_id: str, *, hops: int = 1, id_property: str | None = None, limit: int | None = None
    ) -> ExecutionResult:
        """Expand the n-hop neighborhood from a node (hops clamped to 1..5)."""
        idp = _safe_ident(id_property, "property") if id_property else self._id
        hops = max(1, min(int(hops), 5))
        cap = limit or self._row_cap
        cypher = (
            f"MATCH p = (n:{self._label})-[*1..{hops}]-(m:{self._label}) "
            f"WHERE n.{idp} = $node_id "
            f"RETURN DISTINCT m.{idp} AS neighbor, m.node_type AS node_type, length(p) AS distance "
            f"LIMIT {int(cap)}"
        )
        return self._executor.execute(cypher, {"node_id": node_id}, row_cap=cap)

    def node_lookup(self, name: str, *, limit: int | None = None) -> ExecutionResult:
        """Look up a node by canonical_name or alias."""
        cap = limit or self._row_cap
        cypher = (
            f"MATCH (n:{self._label}) "
            f"WHERE n.{self._id} = $name OR $name IN n.aliases "
            f"RETURN n.{self._id} AS canonical_name, n.node_type AS node_type, "
            f"n.definition AS definition, n.confidence AS confidence "
            f"LIMIT {int(cap)}"
        )
        return self._executor.execute(cypher, {"name": name}, row_cap=cap)


def query_shape(cypher: str) -> str:
    """Normalize a query to a shape signature for promotion analysis.

    Params and string/number literals are abstracted away, so structurally
    identical queries (differing only in bound values) collapse to one shape.
    Recurring shapes in the query log are candidates to promote to a prewritten
    tool.
    """
    a = analyze(cypher)
    if not a.ok:
        return cypher.strip()
    out: list[str] = []
    for tok in a.sig:
        if tok.type in ("STRING", "NUMBER"):
            out.append("?")
        elif tok.type == "PARAM":
            out.append("$?")
        elif tok.type == "IDENT":
            out.append(tok.upper if tok.upper in _KEYWORDS else tok.value)
        else:
            out.append(tok.value)
    return " ".join(out)


_KEYWORDS = {
    "MATCH", "OPTIONAL", "WHERE", "RETURN", "WITH", "UNWIND", "CALL", "YIELD",
    "ORDER", "BY", "SKIP", "LIMIT", "AS", "AND", "OR", "NOT", "IN", "DISTINCT",
    "UNION", "ASC", "DESC", "IS", "NULL", "CONTAINS", "STARTS", "ENDS",
}


__all__ = ["PrewrittenTools", "query_shape"]
