"""Text2Cypher: NL question -> grounded, validated, guarded Cypher.

A 7-stage pipeline (SchemaProvider, EntityGrounder, ExampleStore, Generator,
Validator, Executor, RepairLoop) that turns a natural-language question into a
read-only, schema-grounded Cypher query, executes it against ArcadeDB, and
returns structured rows plus a complete trace.

Accuracy comes from the pipeline, not the model: deterministic checks and
autofixes close the gap where they can, and the LLM is used only where it must.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .contracts import (  # noqa: F401
    ExecutionResult,
    GenerationAttempt,
    GroundedEntity,
    PipelineResponse,
    QueryRequest,
    SchemaContext,
    ValidationIssue,
    ValidationReport,
)

__all__ = [
    "__version__",
    "QueryRequest",
    "PipelineResponse",
    "GroundedEntity",
    "SchemaContext",
    "GenerationAttempt",
    "ValidationIssue",
    "ValidationReport",
    "ExecutionResult",
]
