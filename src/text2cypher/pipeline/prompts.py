"""Prompt construction for the Generator (and its repair / relax variants)."""

from __future__ import annotations

from ..contracts import ExamplePair, GroundingResult, SchemaContext

SYSTEM_PROMPT = """\
You are a precise text-to-Cypher generator for a READ-ONLY openCypher database \
(ArcadeDB dialect). You translate a natural-language question into a single, \
valid, read-only Cypher query.

Hard rules:
- READ-ONLY. Never use CREATE, MERGE, DELETE, SET, REMOVE, DROP, FOREACH, or LOAD CSV.
- No APOC, no db.*/dbms.* procedures, no Neo4j-only functions. Stay within the \
ArcadeDB openCypher subset.
- Use ONLY the labels, relationship types, and properties given in the schema.
- Bind every entity value as one of the provided $parameters. NEVER inline an \
entity name as a string literal.
- Respect relationship direction as defined by the schema.
- Project explicit properties (not whole nodes) and always include a LIMIT.

Respond with ONLY a JSON object, no prose and no code fences:
{"cypher": "<the query>", "params_used": {"<name>": <value>}, "reasoning": "<one sentence>", "confidence": <0.0-1.0>}
"""


def render_grounding(grounding: GroundingResult) -> str:
    if not grounding.entities:
        return "None resolved. If the question names entities, match on canonical_name/aliases."
    lines: list[str] = []
    for ent in grounding.entities:
        if ent.node_id is None and ent.param_name is None:
            lines.append(f'- "{ent.mention}": no confident match in the graph.')
            continue
        parts = [f'- "{ent.mention}" -> bind as ${ent.param_name}']
        if ent.canonical_name:
            parts.append(f'canonical_name="{ent.canonical_name}"')
        if ent.node_id:
            parts.append(f"id={ent.node_id}")
        parts.append(f"sim={ent.similarity:.2f}")
        if ent.ambiguous:
            alts = ", ".join(
                f'{c.canonical_name} ({c.similarity:.2f})' for c in ent.candidates[:3]
            )
            parts.append(f"AMBIGUOUS candidates: {alts}")
        lines.append(" ".join(parts))
    if grounding.params:
        binds = ", ".join(f"${k} = {v!r}" for k, v in grounding.params.items())
        lines.append(f"Parameter bindings available: {binds}")
    return "\n".join(lines)


def render_examples(examples: list[ExamplePair]) -> str:
    if not examples:
        return "None."
    blocks: list[str] = []
    for i, ex in enumerate(examples, 1):
        blocks.append(f"{i}. Q: {ex.question}\n   Cypher: {ex.cypher}")
    return "\n".join(blocks)


def build_generate_messages(
    question: str,
    schema: SchemaContext,
    grounding: GroundingResult,
    examples: list[ExamplePair],
) -> list[dict[str, str]]:
    user = (
        f"Question:\n{question}\n\n"
        f"{schema.rendered}\n\n"
        f"Grounded entities (bind these as $params):\n{render_grounding(grounding)}\n\n"
        f"Similar examples (for shape guidance only — adapt to THIS schema):\n{render_examples(examples)}\n\n"
        "Return the JSON object now."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_repair_messages(
    question: str,
    schema: SchemaContext,
    grounding: GroundingResult,
    examples: list[ExamplePair],
    previous_cypher: str,
    errors: list[str],
    kind: str,
) -> list[dict[str, str]]:
    err_block = "\n".join(f"- {e}" for e in errors) or "- (unspecified)"
    reason = "failed validation" if kind == "validation" else "failed at the database"
    user = (
        f"Question:\n{question}\n\n"
        f"{schema.rendered}\n\n"
        f"Grounded entities (bind these as $params):\n{render_grounding(grounding)}\n\n"
        f"Your previous query {reason}:\n{previous_cypher}\n\n"
        f"Problems to fix:\n{err_block}\n\n"
        "Produce a corrected, read-only Cypher query that resolves every problem "
        "above while still answering the question. Return the JSON object now."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_relax_messages(
    question: str,
    schema: SchemaContext,
    grounding: GroundingResult,
    examples: list[ExamplePair],
    previous_cypher: str,
) -> list[dict[str, str]]:
    user = (
        f"Question:\n{question}\n\n"
        f"{schema.rendered}\n\n"
        f"Grounded entities (bind these as $params):\n{render_grounding(grounding)}\n\n"
        f"This query executed successfully but returned ZERO rows:\n{previous_cypher}\n\n"
        "Relax it by exactly one step to surface near-matches: drop the single most "
        "restrictive filter, widen an exact match to a case-insensitive CONTAINS, or "
        "extend a fixed hop to a bounded variable-length path. Keep it read-only and "
        "keep the same entity $params. Return the JSON object now."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


__all__ = [
    "SYSTEM_PROMPT",
    "render_grounding",
    "render_examples",
    "build_generate_messages",
    "build_repair_messages",
    "build_relax_messages",
]
