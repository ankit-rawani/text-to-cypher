"""Prewritten parameterized tools that bypass the LLM (spec section 5)."""

from __future__ import annotations

from .tools import PrewrittenTools, query_shape

__all__ = ["PrewrittenTools", "query_shape"]
