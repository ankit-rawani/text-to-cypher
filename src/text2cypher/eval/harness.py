"""Execution-accuracy eval harness (spec section 8).

Runs the pipeline over a gold set of question -> Cypher pairs *on our schema*,
executes each gold query against the same graph, and scores predicted vs gold by
result-set match. Reports per-stratum (never pooled only), plus valid-query
rate, attempts-to-success histogram, and latency p50/p95. Includes a CI
regression gate (fail on a drop greater than N points).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..clients.base import GraphClient
from ..contracts import QueryRequest
from ..pipeline.orchestrator import Pipeline
from .compare import result_set_equal

STRATA = ["1-hop", "multi-hop", "aggregation", "alias-phrased", "out-of-graph"]


class GoldItem(BaseModel):
    question: str
    cypher: str
    params: dict[str, Any] = Field(default_factory=dict)
    stratum: str = "1-hop"
    id: str | None = None


class EvalItemResult(BaseModel):
    id: str | None
    question: str
    stratum: str
    status: str
    correct: bool
    valid_first_try: bool
    attempts: int
    latency_ms: float
    predicted_rows: int
    gold_rows: int
    gold_error: str | None = None
    predicted_cypher: str | None = None


class StratumStats(BaseModel):
    n: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return (self.correct / self.n) if self.n else 0.0


class EvalReport(BaseModel):
    n: int
    overall_accuracy: float
    per_stratum: dict[str, float]
    per_stratum_counts: dict[str, dict[str, int]]
    valid_query_rate: float
    attempts_histogram: dict[str, int]
    latency_p50_ms: float
    latency_p95_ms: float
    items: list[EvalItemResult] = Field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"n={self.n}  overall_exec_accuracy={self.overall_accuracy:.1%}",
            f"valid_query_rate={self.valid_query_rate:.1%}  "
            f"latency p50={self.latency_p50_ms:.1f}ms p95={self.latency_p95_ms:.1f}ms",
            "per-stratum execution accuracy:",
        ]
        for stratum, acc in self.per_stratum.items():
            counts = self.per_stratum_counts.get(stratum, {})
            lines.append(f"  {stratum:<14} {acc:>6.1%}  ({counts.get('correct',0)}/{counts.get('n',0)})")
        hist = ", ".join(f"{k}:{v}" for k, v in sorted(self.attempts_histogram.items()))
        lines.append(f"attempts-to-success histogram: {hist or '(none)'}")
        return "\n".join(lines)


class EvalHarness:
    def __init__(
        self,
        pipeline: Pipeline,
        gold_graph_client: GraphClient,
        *,
        row_cap: int = 200,
        timeout_s: float = 15.0,
        max_attempts: int = 3,
    ) -> None:
        self._pipeline = pipeline
        self._gold = gold_graph_client
        self._row_cap = row_cap
        self._timeout = timeout_s
        self._max_attempts = max_attempts

    def run(self, gold: list[GoldItem], *, clock=time.perf_counter) -> EvalReport:
        items: list[EvalItemResult] = []
        latencies: list[float] = []
        for g in gold:
            start = clock()
            resp = self._pipeline.answer(
                QueryRequest(
                    question=g.question,
                    max_attempts=self._max_attempts,
                    row_cap=self._row_cap,
                    timeout_s=self._timeout,
                )
            )
            latency = (clock() - start) * 1000.0
            latencies.append(latency)

            gold_rows: list[dict] = []
            gold_error: str | None = None
            try:
                gold_rows = self._gold.query(g.cypher, g.params, timeout_s=self._timeout, limit=self._row_cap)
            except Exception as exc:
                gold_error = f"{type(exc).__name__}: {exc}"

            predicted_ok = resp.status in ("ok", "empty")
            correct = (
                gold_error is None
                and predicted_ok
                and result_set_equal(resp.rows or [], gold_rows)
            )
            items.append(
                EvalItemResult(
                    id=g.id,
                    question=g.question,
                    stratum=g.stratum,
                    status=resp.status,
                    correct=correct,
                    valid_first_try=bool(resp.validations and resp.validations[0].passed),
                    attempts=len(resp.attempts),
                    latency_ms=round(latency, 3),
                    predicted_rows=len(resp.rows or []),
                    gold_rows=len(gold_rows),
                    gold_error=gold_error,
                    predicted_cypher=resp.final_cypher,
                )
            )
        return self._aggregate(items, latencies)

    def _aggregate(self, items: list[EvalItemResult], latencies: list[float]) -> EvalReport:
        n = len(items)
        correct_total = sum(1 for it in items if it.correct)
        per_counts: dict[str, dict[str, int]] = {}
        for it in items:
            bucket = per_counts.setdefault(it.stratum, {"n": 0, "correct": 0})
            bucket["n"] += 1
            bucket["correct"] += 1 if it.correct else 0
        per_stratum = {s: (c["correct"] / c["n"] if c["n"] else 0.0) for s, c in per_counts.items()}

        valid = sum(1 for it in items if it.valid_first_try)
        hist: dict[str, int] = {}
        for it in items:
            if it.status in ("ok", "empty"):
                key = str(it.attempts)
                hist[key] = hist.get(key, 0) + 1

        return EvalReport(
            n=n,
            overall_accuracy=(correct_total / n) if n else 0.0,
            per_stratum=per_stratum,
            per_stratum_counts=per_counts,
            valid_query_rate=(valid / n) if n else 0.0,
            attempts_histogram=hist,
            latency_p50_ms=round(_percentile(latencies, 50), 3),
            latency_p95_ms=round(_percentile(latencies, 95), 3),
            items=items,
        )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def load_gold(path: str | Path) -> list[GoldItem]:
    items: list[GoldItem] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(GoldItem.model_validate_json(line))
    return items


def compare_reports(
    current: EvalReport, baseline: EvalReport, *, threshold_pts: float = 5.0
) -> tuple[bool, list[str]]:
    """CI gate: fail on any stratum/overall regression greater than threshold (points)."""
    regressions: list[str] = []
    delta = (baseline.overall_accuracy - current.overall_accuracy) * 100.0
    if delta > threshold_pts:
        regressions.append(f"overall: -{delta:.1f} pts ({baseline.overall_accuracy:.1%} -> {current.overall_accuracy:.1%})")
    for stratum, base_acc in baseline.per_stratum.items():
        cur_acc = current.per_stratum.get(stratum)
        if cur_acc is None:
            continue
        d = (base_acc - cur_acc) * 100.0
        if d > threshold_pts:
            regressions.append(f"{stratum}: -{d:.1f} pts ({base_acc:.1%} -> {cur_acc:.1%})")
    return (len(regressions) == 0, regressions)


__all__ = [
    "EvalHarness",
    "EvalReport",
    "EvalItemResult",
    "GoldItem",
    "STRATA",
    "load_gold",
    "compare_reports",
]
