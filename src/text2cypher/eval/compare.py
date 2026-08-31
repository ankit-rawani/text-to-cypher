"""Result-set comparison for execution accuracy (spec section 8).

Execution accuracy is result-set match — order-insensitive across rows, with
numerics rounded — NOT string match, because many correct Cyphers exist per
question.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

DEFAULT_ROUND = 6


def canonicalize(value: Any, ndigits: int = DEFAULT_ROUND) -> Any:
    """Canonical, hashable form of a value for multiset comparison."""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("num", round(float(value), ndigits))
    if value is None:
        return ("none",)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (list, tuple)):
        return ("list", tuple(canonicalize(v, ndigits) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((str(k), canonicalize(v, ndigits)) for k, v in value.items())))
    return ("other", str(value))


def _canon_row(row: Any, ndigits: int) -> Any:
    if isinstance(row, dict):
        return ("dict", tuple(sorted((str(k), canonicalize(v, ndigits)) for k, v in row.items())))
    return canonicalize(row, ndigits)


def result_set_equal(
    a: list[dict] | None, b: list[dict] | None, *, ndigits: int = DEFAULT_ROUND
) -> bool:
    """Order-insensitive multiset equality of two result sets."""
    a = a or []
    b = b or []
    ca = Counter(_canon_row(r, ndigits) for r in a)
    cb = Counter(_canon_row(r, ndigits) for r in b)
    return ca == cb


__all__ = ["result_set_equal", "canonicalize", "DEFAULT_ROUND"]
