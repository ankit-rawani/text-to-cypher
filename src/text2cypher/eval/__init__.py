"""Execution-accuracy evaluation (spec section 8)."""

from __future__ import annotations

from .compare import canonicalize, result_set_equal
from .harness import (
    STRATA,
    EvalHarness,
    EvalItemResult,
    EvalReport,
    GoldItem,
    compare_reports,
    load_gold,
)

__all__ = [
    "result_set_equal",
    "canonicalize",
    "EvalHarness",
    "EvalReport",
    "EvalItemResult",
    "GoldItem",
    "STRATA",
    "load_gold",
    "compare_reports",
]
