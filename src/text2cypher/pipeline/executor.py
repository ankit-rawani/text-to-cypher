"""Stage 6 — Executor.

Runs the validated query against the graph through a **read-only** client, with
parameters bound (never interpolated), a timeout, a server-side ``LIMIT`` arg,
and a hard row cap applied to the returned rows (defense in depth over the
validator's LIMIT injection). Classifies the outcome as ok / empty / error /
timeout and measures latency. Optional MongoDB hydration enriches rows when
enabled.
"""

from __future__ import annotations

import time
from typing import Any

from ..clients.base import (
    DocumentStore,
    GraphAuthError,
    GraphClient,
    GraphError,
    GraphTimeout,
)
from ..clients.mongodb import DisabledDocumentStore
from ..contracts import ExecutionResult

_ID_FIELDS = ("node_id", "id", "@rid", "rid", "_id")


class Executor:
    def __init__(
        self,
        graph_client: GraphClient,
        *,
        row_cap: int = 200,
        timeout_s: float = 15.0,
        document_store: DocumentStore | None = None,
    ) -> None:
        self._client = graph_client
        self._row_cap = row_cap
        self._timeout = timeout_s
        self._docs = document_store or DisabledDocumentStore()

    def execute(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        row_cap: int | None = None,
        timeout_s: float | None = None,
    ) -> ExecutionResult:
        cap = row_cap if row_cap is not None else self._row_cap
        timeout = timeout_s if timeout_s is not None else self._timeout
        params = params or {}

        start = time.perf_counter()
        try:
            rows = self._client.query(cypher, params, timeout_s=timeout, limit=cap)
        except GraphTimeout as exc:
            return ExecutionResult(
                status="timeout",
                rows=None,
                row_count=0,
                error_message=f"Query timed out after {timeout}s: {exc}",
                latency_ms=_elapsed_ms(start),
            )
        except GraphAuthError as exc:
            return ExecutionResult(
                status="error",
                rows=None,
                row_count=0,
                error_message=f"Authorization error: {exc}",
                latency_ms=_elapsed_ms(start),
            )
        except GraphError as exc:
            return ExecutionResult(
                status="error",
                rows=None,
                row_count=0,
                error_message=str(exc),
                latency_ms=_elapsed_ms(start),
            )
        except Exception as exc:  # unexpected client failure
            return ExecutionResult(
                status="error",
                rows=None,
                row_count=0,
                error_message=f"{type(exc).__name__}: {exc}",
                latency_ms=_elapsed_ms(start),
            )

        rows = list(rows or [])
        truncated = len(rows) > cap
        if truncated:
            rows = rows[:cap]

        rows = self._maybe_hydrate(rows)

        latency = _elapsed_ms(start)
        if not rows:
            return ExecutionResult(
                status="empty", rows=[], row_count=0, error_message=None, latency_ms=latency
            )
        return ExecutionResult(
            status="ok",
            rows=rows,
            row_count=len(rows),
            error_message=None,
            latency_ms=latency,
            truncated=truncated,
        )

    def _maybe_hydrate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not getattr(self._docs, "enabled", False) or not rows:
            return rows
        ids: list[str] = []
        for row in rows:
            for field in _ID_FIELDS:
                if isinstance(row, dict) and field in row and row[field] is not None:
                    ids.append(str(row[field]))
                    break
        if not ids:
            return rows
        docs = self._docs.get_many(ids)
        if not docs:
            return rows
        enriched: list[dict[str, Any]] = []
        for row in rows:
            key = None
            for field in _ID_FIELDS:
                if isinstance(row, dict) and field in row and row[field] is not None:
                    key = str(row[field])
                    break
            if key and key in docs:
                row = {**row, "_document": docs[key]}
            enriched.append(row)
        return enriched


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


__all__ = ["Executor"]
