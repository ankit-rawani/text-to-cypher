"""Stage 4 — Generator.

A single LLM call that maps (question + schema + grounded params + examples) to a
structured ``{cypher, params_used, reasoning, confidence}``. Grounded parameter
bindings are merged over whatever the model declares, so the authoritative entity
values always win and can never be silently altered by the model.

Also exposes ``repair`` (reflection retry with the failure context) and ``relax``
(one-shot widening for the empty-result branch).
"""

from __future__ import annotations

import json
import re

from ..clients.base import LLMClient
from ..contracts import (
    ExamplePair,
    GenerationAttempt,
    GenerationOutput,
    GroundingResult,
    SchemaContext,
)
from . import prompts

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_generation_output(text: str) -> GenerationOutput:
    """Robustly parse the model's JSON, tolerating fences and surrounding prose."""
    raw = (text or "").strip()
    candidate = raw
    fence = _FENCE_RE.search(raw)
    if fence:
        candidate = fence.group(1).strip()

    data = _try_json(candidate)
    if data is None:
        data = _try_json(_first_brace_object(raw))
    if data is None or not isinstance(data, dict):
        # Unparseable — surface as an empty cypher so the validator/repair loop
        # gives the model a precise, actionable error.
        return GenerationOutput(cypher="", params_used={}, reasoning=f"UNPARSEABLE_OUTPUT: {raw[:400]}", confidence=None)

    cypher = data.get("cypher") or data.get("query") or ""
    params = data.get("params_used") or data.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    reasoning = data.get("reasoning") or data.get("explanation") or ""
    confidence = data.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    return GenerationOutput(
        cypher=str(cypher).strip(),
        params_used=params,
        reasoning=str(reasoning),
        confidence=confidence,
    )


def _try_json(text: str | None):
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _first_brace_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


class Generator:
    def __init__(
        self,
        llm: LLMClient,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_s: float | None = None,
    ) -> None:
        self._llm = llm
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout_s

    def _call(
        self,
        messages: list[dict[str, str]],
        attempt_no: int,
        kind: str,
        grounded_params: dict,
    ) -> tuple[GenerationAttempt, GenerationOutput]:
        text = self._llm.complete(
            messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
            timeout_s=self._timeout,
        )
        out = parse_generation_output(text)
        # Grounded params are authoritative and win over model-declared values.
        merged = {**out.params_used, **grounded_params}
        attempt = GenerationAttempt(
            attempt=attempt_no,
            cypher=out.cypher,
            params=merged,
            reasoning=out.reasoning,
            model_confidence=out.confidence,
            kind=kind,
        )
        return attempt, out

    def generate(
        self,
        question: str,
        schema: SchemaContext,
        grounding: GroundingResult,
        examples: list[ExamplePair],
        attempt_no: int = 1,
    ) -> tuple[GenerationAttempt, GenerationOutput]:
        messages = prompts.build_generate_messages(question, schema, grounding, examples)
        return self._call(messages, attempt_no, "generate", grounding.params)

    def repair(
        self,
        question: str,
        schema: SchemaContext,
        grounding: GroundingResult,
        examples: list[ExamplePair],
        previous_cypher: str,
        errors: list[str],
        kind: str,
        attempt_no: int,
    ) -> tuple[GenerationAttempt, GenerationOutput]:
        messages = prompts.build_repair_messages(
            question, schema, grounding, examples, previous_cypher, errors, kind
        )
        return self._call(messages, attempt_no, "repair", grounding.params)

    def relax(
        self,
        question: str,
        schema: SchemaContext,
        grounding: GroundingResult,
        examples: list[ExamplePair],
        previous_cypher: str,
        attempt_no: int,
    ) -> tuple[GenerationAttempt, GenerationOutput]:
        messages = prompts.build_relax_messages(question, schema, grounding, examples, previous_cypher)
        return self._call(messages, attempt_no, "relax", grounding.params)


__all__ = ["Generator", "parse_generation_output"]
