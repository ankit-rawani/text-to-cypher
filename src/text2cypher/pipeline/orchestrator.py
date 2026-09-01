"""Stage 7 + orchestration — the Pipeline.

Wires stages 1-6 and drives the reflection repair loop (capped) and the
empty-result branch (one relaxation, never loop-retried). Every response carries
the complete trace: all attempts, validations, executions, and grounding.
"""

from __future__ import annotations

import uuid
from typing import Any

from ..contracts import (
    GenerationAttempt,
    GroundingResult,
    PipelineResponse,
    QueryRequest,
    ValidationReport,
)
from ..cypher import analyze
from .entity_grounder import EntityGrounder
from .example_store import ExampleStore
from .executor import Executor
from .generator import Generator
from .schema_provider import SchemaProvider
from .validator import Validator


class Pipeline:
    def __init__(
        self,
        schema_provider: SchemaProvider,
        grounder: EntityGrounder,
        example_store: ExampleStore,
        generator: Generator,
        validator: Validator,
        executor: Executor,
        *,
        hard_max_attempts: int = 5,
        allow_relaxation: bool = True,
        low_confidence_threshold: float = 0.65,
    ) -> None:
        self._schema = schema_provider
        self._grounder = grounder
        self._examples = example_store
        self._generator = generator
        self._validator = validator
        self._executor = executor
        self._hard_max = hard_max_attempts
        self._allow_relax = allow_relaxation
        self._low_conf = low_confidence_threshold

    # ------------------------------------------------------------------
    def answer(self, request: QueryRequest) -> PipelineResponse:
        trace_id = uuid.uuid4().hex[:12]
        question = request.question
        schema = self._schema.get()
        # Grounding and example retrieval are optional enrichments — a transient
        # vector-store/embedder failure degrades gracefully rather than crashing.
        degraded: list[str] = []
        try:
            grounding = self._grounder.ground(question, schema)
        except Exception as exc:
            grounding = GroundingResult()
            degraded.append(f"grounding unavailable ({exc})")
        try:
            examples = self._examples.retrieve(question)
        except Exception as exc:
            examples = []
            degraded.append(f"examples unavailable ({exc})")

        attempts: list[GenerationAttempt] = []
        validations: list[ValidationReport] = []
        executions: list = []

        max_attempts = max(1, min(request.max_attempts, self._hard_max))
        allow_relax = (
            request.allow_relaxation if request.allow_relaxation is not None else self._allow_relax
        )
        row_cap = request.row_cap
        timeout = request.timeout_s

        failure_errors: list[str] = []
        failure_kind = "validation"
        prev_cypher = ""

        for attempt_no in range(1, max_attempts + 1):
            if attempt_no == 1:
                gen_attempt, gen_error = self._call_generator(
                    lambda: self._generator.generate(question, schema, grounding, examples, attempt_no),
                    attempt_no, "generate",
                )
            else:
                gen_attempt, gen_error = self._call_generator(
                    lambda: self._generator.repair(
                        question, schema, grounding, examples, prev_cypher, failure_errors, failure_kind, attempt_no
                    ),
                    attempt_no, "repair",
                )
            attempts.append(gen_attempt)
            prev_cypher = gen_attempt.cypher

            if gen_error is not None:
                # The LLM call itself failed (network/parse) — surface it in the
                # trace and reflect on it, rather than crashing.
                failure_kind = "generation"
                failure_errors = [gen_error]
                continue

            report = self._validator.validate(
                gen_attempt.cypher, gen_attempt.params, schema, grounding, row_cap=row_cap
            )
            validations.append(report)

            if not report.passed:
                if report.terminal:
                    return self._response(
                        "failed", None, report.final_cypher, gen_attempt.params,
                        attempts, validations, executions, grounding, trace_id,
                        message="Blocked by validator: " + _first_reject(report),
                    )
                failure_kind = "validation"
                failure_errors = [i.detail for i in report.issues if i.severity == "reject"]
                prev_cypher = report.final_cypher
                continue

            final_cypher = report.final_cypher
            execution = self._executor.execute(
                final_cypher, gen_attempt.params, row_cap=row_cap, timeout_s=timeout
            )
            executions.append(execution)

            if execution.status == "ok":
                return self._response(
                    "ok", execution.rows, final_cypher, gen_attempt.params,
                    attempts, validations, executions, grounding, trace_id,
                    message=f"{execution.row_count} row(s) returned.",
                )
            if execution.status == "empty":
                return self._handle_empty(
                    question, schema, grounding, examples, final_cypher, gen_attempt.params,
                    attempts, validations, executions, trace_id, allow_relax, row_cap, timeout,
                    attempt_no, max_attempts,
                )
            # error / timeout -> reflect and repair
            failure_kind = "execution"
            failure_errors = [execution.error_message or f"execution {execution.status}"]
            prev_cypher = final_cypher

        tail = f" Last error: {failure_errors[0]}" if failure_errors else ""
        return self._response(
            "failed", None, prev_cypher or None, attempts[-1].params if attempts else None,
            attempts, validations, executions, grounding, trace_id,
            message=f"Exhausted {max_attempts} attempt(s) without a successful query.{tail}",
        )

    def _call_generator(self, fn, attempt_no: int, kind: str):
        """Run a generator call, converting any exception into a traced failure."""
        try:
            attempt, _ = fn()
            return attempt, None
        except Exception as exc:
            attempt = GenerationAttempt(
                attempt=attempt_no, cypher="", params={},
                reasoning=f"GENERATION_ERROR: {type(exc).__name__}: {exc}", kind=kind,
            )
            return attempt, f"Generator/LLM error: {type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------
    def _handle_empty(
        self,
        question: str,
        schema,
        grounding: GroundingResult,
        examples,
        empty_cypher: str,
        empty_params: dict,
        attempts: list[GenerationAttempt],
        validations: list[ValidationReport],
        executions: list,
        trace_id: str,
        allow_relax: bool,
        row_cap: int,
        timeout: float,
        attempt_no: int,
        max_attempts: int,
    ) -> PipelineResponse:
        # The single relaxation counts as a generation and respects the caller's
        # effective attempt budget (and the hard cap).
        grounded_names = set(grounding.params.keys())
        # Guard 1: relaxation only makes sense when there is a grounded entity to
        # anchor it. With nothing grounded, an empty result usually means the
        # question is out-of-graph — relaxing would just widen into an arbitrary
        # full scan and report it as `ok`, so return `empty` instead.
        can_relax = (
            allow_relax
            and bool(grounded_names)
            and (attempt_no + 1) <= min(max_attempts, self._hard_max)
        )
        if can_relax:
            relax_attempt, relax_error = self._call_generator(
                lambda: self._generator.relax(question, schema, grounding, examples, empty_cypher, attempt_no + 1),
                attempt_no + 1, "relax",
            )
            attempts.append(relax_attempt)
            if relax_error is None:
                report = self._validator.validate(
                    relax_attempt.cypher, relax_attempt.params, schema, grounding, row_cap=row_cap
                )
                validations.append(report)
                # Guard 2: a legitimate relaxation loosens filters AROUND the
                # grounded entities — it must still bind at least one grounded
                # $param. One that drops the anchor (e.g. `MATCH (n) RETURN n
                # LIMIT k`) is a degenerate scan; reject it rather than surface
                # arbitrary rows as a successful answer.
                if report.passed and self._relaxation_anchored(report.final_cypher, grounded_names):
                    execution = self._executor.execute(
                        report.final_cypher, relax_attempt.params, row_cap=row_cap, timeout_s=timeout
                    )
                    executions.append(execution)
                    if execution.status == "ok":
                        return self._response(
                            "ok", execution.rows, report.final_cypher, relax_attempt.params,
                            attempts, validations, executions, grounding, trace_id,
                            message=f"{execution.row_count} row(s) after one relaxation.",
                        )

        return self._response(
            "empty", [], empty_cypher, empty_params,
            attempts, validations, executions, grounding, trace_id,
            message=self._empty_message(grounding),
        )

    @staticmethod
    def _relaxation_anchored(cypher: str, grounded_names: set[str]) -> bool:
        """True if a relaxed query still binds at least one grounded entity param."""
        if not grounded_names:
            return False
        try:
            referenced = set(analyze(cypher).params)
        except Exception:
            return False
        return bool(referenced & grounded_names)

    def _empty_message(self, grounding: GroundingResult) -> str:
        parts = ["Query executed successfully but returned 0 rows."]
        if not grounding.entities:
            parts.append("No entities from the question were confidently grounded in the graph.")
        ambiguous = [e.mention for e in grounding.entities if e.ambiguous]
        low = [
            e.mention
            for e in grounding.entities
            if e.node_id is not None and e.similarity < self._low_conf
        ]
        if ambiguous:
            parts.append(f"Ambiguous entities may need clarification: {', '.join(ambiguous)}.")
        if low:
            parts.append(f"Low-confidence groundings: {', '.join(low)}.")
        parts.append("Consider a vector-retrieval fallback for this question.")
        return " ".join(parts)

    def _response(
        self,
        status: str,
        rows: list[dict] | None,
        final_cypher: str | None,
        params: dict[str, Any] | None,
        attempts: list[GenerationAttempt],
        validations: list[ValidationReport],
        executions: list,
        grounding: GroundingResult,
        trace_id: str,
        *,
        message: str,
    ) -> PipelineResponse:
        return PipelineResponse(
            status=status,  # type: ignore[arg-type]
            rows=rows,
            final_cypher=final_cypher,
            params=params,
            attempts=attempts,
            validations=validations,
            executions=executions,
            grounding=grounding.entities,
            trace_id=trace_id,
            message=message,
        )


def _first_reject(report: ValidationReport) -> str:
    for issue in report.issues:
        if issue.severity == "reject":
            return f"[{issue.code}] {issue.detail}"
    return "validation failed"


__all__ = ["Pipeline"]
