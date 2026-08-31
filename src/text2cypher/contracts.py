"""Pydantic contracts for the Text2Cypher pipeline.

These models are the frozen interface between stages (see spec section 2).
Every ``PipelineResponse`` carries the complete trace: all attempts, all
validations, all executions, and the entity grounding. No silent retries,
no swallowed errors.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Public request/response contracts (spec section 2)
# --------------------------------------------------------------------------

Severity = Literal["reject", "autofixed", "warn"]
ExecStatus = Literal["ok", "empty", "error", "timeout"]
PipelineStatus = Literal["ok", "empty", "failed"]


class QueryRequest(BaseModel):
    """A single natural-language question to run through the pipeline."""

    question: str
    max_attempts: int = 3
    row_cap: int = 200
    timeout_s: float = 15.0
    # Optional per-request override of the config relaxation flag.
    allow_relaxation: bool | None = None


class GroundedEntity(BaseModel):
    """An entity mention from the question resolved against the node vectors."""

    mention: str  # surface form in question
    node_id: str | None = None  # resolved graph ID
    canonical_name: str | None = None
    similarity: float = 0.0
    ambiguous: bool = False  # >1 candidate above threshold
    # Name of the ``$param`` this entity is bound to (never inlined).
    param_name: str | None = None
    # All above-threshold candidates, best first (for ambiguity/relaxation).
    candidates: list["EntityCandidate"] = Field(default_factory=list)


class EntityCandidate(BaseModel):
    """A single Qdrant node hit considered for a mention."""

    node_id: str | None = None
    canonical_name: str | None = None
    node_type: str | None = None
    similarity: float = 0.0
    aliases: list[str] = Field(default_factory=list)


class SchemaContext(BaseModel):
    """Enhanced schema context rendered for the prompt (spec section 1, stage 1)."""

    rendered: str  # compact prompt block
    labels: list[str] = Field(default_factory=list)
    rel_types: list[str] = Field(default_factory=list)
    # label -> {prop: type}
    properties: dict[str, dict[str, str]] = Field(default_factory=dict)
    version_hash: str = ""
    # Optional enum values (label.prop -> allowed values) surfaced to the model.
    enums: dict[str, list[str]] = Field(default_factory=dict)
    # Optional connectivity triples (start_label, rel_type, end_label).
    edges: list[tuple[str, str, str]] = Field(default_factory=list)


class ExamplePair(BaseModel):
    """A retrieved (or seed) question -> Cypher example."""

    question: str
    cypher: str
    similarity: float = 0.0
    tags: list[str] = Field(default_factory=list)


class GenerationAttempt(BaseModel):
    """One LLM generation (or repair) attempt in the trace."""

    attempt: int
    cypher: str
    params: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    model_confidence: float | None = None
    # "generate" | "repair" | "relax" — how this attempt was produced.
    kind: str = "generate"


class ValidationIssue(BaseModel):
    """A single validator finding.

    ``code`` examples: PARSE_ERROR, DENYLISTED, WRITE_OP, UNKNOWN_LABEL,
    UNKNOWN_REL_TYPE, UNKNOWN_PROPERTY, DIRECTION_FIXED, PARAM_LITERAL,
    LIMIT_INJECTED, LIMIT_CLAMPED.
    """

    code: str
    severity: Severity
    detail: str


class ValidationReport(BaseModel):
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    final_cypher: str  # post-autofix
    # Whether the failure is terminal (write-op / denylist) and must NOT be
    # sent to the repair loop.
    terminal: bool = False


class ExecutionResult(BaseModel):
    status: ExecStatus
    rows: list[dict] | None = None
    row_count: int = 0
    error_message: str | None = None
    latency_ms: float = 0.0
    # True when the executor clamped the returned rows to ``row_cap``.
    truncated: bool = False


class PipelineResponse(BaseModel):
    status: PipelineStatus
    rows: list[dict] | None = None
    final_cypher: str | None = None
    params: dict[str, Any] | None = None
    attempts: list[GenerationAttempt] = Field(default_factory=list)
    validations: list[ValidationReport] = Field(default_factory=list)
    executions: list[ExecutionResult] = Field(default_factory=list)
    grounding: list[GroundedEntity] = Field(default_factory=list)
    trace_id: str = ""
    # Human-readable summary of why the pipeline ended in its final state.
    message: str = ""

    def executed_rows(self) -> int:
        return self.executions[-1].row_count if self.executions else 0


# --------------------------------------------------------------------------
# Internal contracts (not part of the external response, but stable)
# --------------------------------------------------------------------------


class GenerationOutput(BaseModel):
    """Structured output expected back from the Generator LLM call."""

    cypher: str
    params_used: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    confidence: float | None = None


class GroundingResult(BaseModel):
    """Output of the EntityGrounder: bound params + per-mention detail."""

    entities: list[GroundedEntity] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_ambiguous(self) -> bool:
        return any(e.ambiguous for e in self.entities)

    @property
    def has_low_confidence(self) -> bool:
        return any(
            e.node_id is not None and e.similarity < 0.65 for e in self.entities
        )


# Resolve forward references.
GroundedEntity.model_rebuild()

__all__ = [
    "Severity",
    "ExecStatus",
    "PipelineStatus",
    "QueryRequest",
    "GroundedEntity",
    "EntityCandidate",
    "SchemaContext",
    "ExamplePair",
    "GenerationAttempt",
    "ValidationIssue",
    "ValidationReport",
    "ExecutionResult",
    "PipelineResponse",
    "GenerationOutput",
    "GroundingResult",
]
