"""Stage 5 — Validator.

Ordered, deterministic checks (spec section 3). Run in order; stop at the first
``reject`` unless the check is an autofix. Autofixes (relationship direction,
LIMIT) mutate the query and record an ``autofixed`` issue but do not stop.

    1. Grammar / structural parse   -> reject (repairable)
    2. Dialect lint (denylist)      -> reject (repairable; terminal if also a write)
    3. Write-op block               -> reject, NO repair (terminal)
    4. Schema consistency           -> reject (repairable) + fuzzy suggestions
    5. Relationship direction       -> autofix (DIRECTION_FIXED)
    6. Param discipline             -> reject (repairable)
    7. LIMIT enforcement            -> autofix (LIMIT_INJECTED / LIMIT_CLAMPED)

A mutation query is always ``terminal`` (never sent to the repair loop) — even if
it also trips the dialect lint — because repairing a write into a sneakier write
is worse than failing fast.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..contracts import GroundingResult, SchemaContext, ValidationIssue, ValidationReport
from ..cypher import analyze, ensure_limit, flip_direction
from ..cypher.analysis import RelPattern
from ..cypher.tokenizer import IDENT, OP, PUNCT, QUOTED_IDENT, Token

_COMPARISON_PUNCT = {"=", "<", ">"}
_COMPARISON_OP = {"<>", "<=", ">=", "=~", "!="}
_COMPARISON_KW = {"IN", "CONTAINS", "STARTS", "ENDS"}


def _is_comparison(tok: Token) -> bool:
    return (
        (tok.type == PUNCT and tok.value in _COMPARISON_PUNCT)
        or (tok.type == OP and tok.value in _COMPARISON_OP)
        or (tok.type == IDENT and tok.upper in _COMPARISON_KW)
    )


def _key_name(tok: Token) -> str:
    return tok.decoded if tok.type == QUOTED_IDENT and tok.decoded is not None else tok.value


def _literal_comparison_key(sig: list[Token], tok: Token) -> str | None:
    """The property key a string literal is compared/assigned against, or None.

    Recognizes ``var.key <op> 'lit'``, ``'lit' <op> var.key``, and the inline
    map form ``key: 'lit'``. Used to make param-discipline position-aware.
    """
    idx = next((i for i, t in enumerate(sig) if t is tok), None)
    if idx is None:
        return None
    n = len(sig)
    # inline map / property key:  key : 'lit'
    if idx >= 2 and sig[idx - 1].type == PUNCT and sig[idx - 1].value == ":" and sig[idx - 2].type in (IDENT, QUOTED_IDENT):
        return _key_name(sig[idx - 2])
    # var.key <op> 'lit'
    if idx >= 3 and _is_comparison(sig[idx - 1]) and sig[idx - 2].type == IDENT and sig[idx - 3].type == PUNCT and sig[idx - 3].value == ".":
        return sig[idx - 2].value
    # 'lit' <op> var.key
    if idx + 4 < n and _is_comparison(sig[idx + 1]) and sig[idx + 2].type == IDENT and sig[idx + 3].type == PUNCT and sig[idx + 3].value == "." and sig[idx + 4].type == IDENT:
        return sig[idx + 4].value
    return None


@dataclass
class Denylist:
    procedure_prefixes: list[str] = field(default_factory=list)
    functions: set[str] = field(default_factory=set)
    clauses: set[str] = field(default_factory=set)


def load_denylist(path: str | Path | None) -> Denylist:
    if path is None or not Path(path).exists():
        return Denylist()
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Denylist(
        procedure_prefixes=[str(p).lower() for p in (data.get("procedure_prefixes") or [])],
        functions={str(f).lower() for f in (data.get("functions") or [])},
        clauses={str(c).upper() for c in (data.get("clauses") or []) if c},
    )


class Validator:
    def __init__(self, denylist: Denylist | None = None, *, row_cap: int = 200) -> None:
        self._denylist = denylist or Denylist()
        self._row_cap = row_cap

    def validate(
        self,
        cypher: str,
        params: dict | None,
        schema: SchemaContext,
        grounding: GroundingResult | None = None,
        *,
        row_cap: int | None = None,
    ) -> ValidationReport:
        cap = row_cap if row_cap is not None else self._row_cap
        issues: list[ValidationIssue] = []

        # 0. Empty output
        if not cypher or not cypher.strip():
            issues.append(
                ValidationIssue(
                    code="PARSE_ERROR",
                    severity="reject",
                    detail="No Cypher produced. Return a JSON object with a non-empty 'cypher' field.",
                )
            )
            return ValidationReport(passed=False, issues=issues, final_cypher=cypher or "", terminal=False)

        # 1. Grammar / structural parse
        a = analyze(cypher)
        if not a.ok:
            pos = f" (at char {a.error_pos})" if a.error_pos is not None else ""
            issues.append(
                ValidationIssue(code="PARSE_ERROR", severity="reject", detail=f"{a.error}{pos}")
            )
            return ValidationReport(passed=False, issues=issues, final_cypher=cypher, terminal=False)

        is_write = bool(a.write_ops)

        # 2. Dialect lint (denylist)
        denied = self._denylisted(a)
        if denied:
            issues.append(
                ValidationIssue(
                    code="DENYLISTED",
                    severity="reject",
                    detail=(
                        "Uses constructs outside the ArcadeDB openCypher subset: "
                        + ", ".join(sorted(denied))
                        + ". Remove them and use only supported functions/procedures."
                    ),
                )
            )
            return ValidationReport(passed=False, issues=issues, final_cypher=cypher, terminal=is_write)

        # 3. Write-op block (terminal — never repaired)
        if is_write:
            kws = ", ".join(sorted({w.keyword for w in a.write_ops}))
            issues.append(
                ValidationIssue(
                    code="WRITE_OP",
                    severity="reject",
                    detail=f"Write operation(s) not allowed in a read-only pipeline: {kws}.",
                )
            )
            return ValidationReport(passed=False, issues=issues, final_cypher=cypher, terminal=True)

        # 4. Schema consistency
        schema_issues = self._schema_issues(a, schema)
        if schema_issues:
            issues.extend(schema_issues)
            return ValidationReport(passed=False, issues=issues, final_cypher=cypher, terminal=False)

        # 5. Relationship direction (autofix)
        current = cypher
        to_flip = self._direction_fixes(a, schema)
        for rp in to_flip:
            new = flip_direction(current, rp)
            if new != current:
                current = new
                issues.append(
                    ValidationIssue(
                        code="DIRECTION_FIXED",
                        severity="autofixed",
                        detail=(
                            f"Flipped relationship :{rp.types[0]} to match the schema direction."
                            if rp.types
                            else "Flipped relationship direction to match the schema."
                        ),
                    )
                )

        a2 = analyze(current)

        # 6. Param discipline
        param_issues = self._param_issues(a2, grounding)
        if param_issues:
            issues.extend(param_issues)
            return ValidationReport(passed=False, issues=issues, final_cypher=current, terminal=False)

        # 7. LIMIT enforcement (autofix)
        limited, action = ensure_limit(current, cap, a2)
        if action == "injected":
            current = limited
            issues.append(
                ValidationIssue(code="LIMIT_INJECTED", severity="autofixed", detail=f"Injected LIMIT {cap}.")
            )
        elif action == "clamped":
            current = limited
            issues.append(
                ValidationIssue(code="LIMIT_CLAMPED", severity="autofixed", detail=f"Clamped LIMIT to {cap}.")
            )

        return ValidationReport(passed=True, issues=issues, final_cypher=current, terminal=False)

    # ------------------------------------------------------------------
    def _denylisted(self, a) -> set[str]:
        offending: set[str] = set()
        for call in a.calls:
            name = call.name.lower()
            segments = name.split(".")
            if any(name.startswith(pref) for pref in self._denylist.procedure_prefixes):
                offending.add(call.name)
            elif name in self._denylist.functions:
                offending.add(call.name)
            elif segments and segments[-1] in self._denylist.functions:
                # e.g. a namespaced `foo.randomUUID(`
                offending.add(call.name)
        if self._denylist.clauses:
            for tok in a.sig:
                value = tok.decoded if tok.type == "QUOTED_IDENT" and tok.decoded else tok.value
                if tok.type in ("IDENT", "QUOTED_IDENT") and value.upper() in self._denylist.clauses:
                    offending.add(value)
        return offending

    def _schema_issues(self, a, schema: SchemaContext) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        known_labels = set(schema.labels)
        known_rels = set(schema.rel_types)

        for label in sorted(a.labels()):
            if label not in known_labels:
                sugg = difflib.get_close_matches(label, list(known_labels), n=1)
                hint = f" Did you mean :{sugg[0]}?" if sugg else ""
                issues.append(
                    ValidationIssue(
                        code="UNKNOWN_LABEL",
                        severity="reject",
                        detail=f"Unknown node label :{label}.{hint} Known labels: {', '.join(sorted(known_labels))}.",
                    )
                )
        for rel in sorted(a.rel_types()):
            if rel not in known_rels:
                sugg = difflib.get_close_matches(rel, list(known_rels), n=1)
                hint = f" Did you mean :{sugg[0]}?" if sugg else ""
                issues.append(
                    ValidationIssue(
                        code="UNKNOWN_REL_TYPE",
                        severity="reject",
                        detail=f"Unknown relationship type :{rel}.{hint} Known types: {', '.join(sorted(known_rels))}.",
                    )
                )

        issues.extend(self._property_issues(a, schema))
        return issues

    def _property_issues(self, a, schema: SchemaContext) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        def props_for(names: set[str]) -> set[str] | None:
            allowed: set[str] = set()
            known = False
            for name in names:
                if name in schema.properties:
                    known = True
                    allowed.update(schema.properties[name].keys())
            return allowed if known else None

        # property accesses  var.key
        for pa in a.property_accesses:
            scope = a.var_labels.get(pa.variable, set()) | a.var_types.get(pa.variable, set())
            allowed = props_for(scope)
            if allowed is None or not allowed:
                continue  # unknown scope — cannot verify
            if pa.key not in allowed:
                sugg = difflib.get_close_matches(pa.key, list(allowed), n=1)
                hint = f" Did you mean .{sugg[0]}?" if sugg else ""
                issues.append(
                    ValidationIssue(
                        code="UNKNOWN_PROPERTY",
                        severity="reject",
                        detail=f"Unknown property {pa.variable}.{pa.key}.{hint}",
                    )
                )

        # inline node maps
        for np in a.node_patterns:
            if not np.labels or not np.prop_keys:
                continue
            allowed = props_for(set(np.labels))
            if not allowed:
                continue
            for key in np.prop_keys:
                if key not in allowed:
                    sugg = difflib.get_close_matches(key, list(allowed), n=1)
                    hint = f" Did you mean {sugg[0]}?" if sugg else ""
                    issues.append(
                        ValidationIssue(
                            code="UNKNOWN_PROPERTY",
                            severity="reject",
                            detail=f"Unknown property '{key}' on :{'/'.join(np.labels)}.{hint}",
                        )
                    )

        # inline rel maps
        for rp in a.rel_patterns:
            if not rp.types or not rp.prop_keys:
                continue
            allowed = props_for(set(rp.types))
            if not allowed:
                continue
            for key in rp.prop_keys:
                if key not in allowed:
                    sugg = difflib.get_close_matches(key, list(allowed), n=1)
                    hint = f" Did you mean {sugg[0]}?" if sugg else ""
                    issues.append(
                        ValidationIssue(
                            code="UNKNOWN_PROPERTY",
                            severity="reject",
                            detail=f"Unknown property '{key}' on :{'/'.join(rp.types)}.{hint}",
                        )
                    )
        return issues

    def _direction_fixes(self, a, schema: SchemaContext) -> list[RelPattern]:
        edgeset = set(schema.edges)
        if not edgeset:
            return []
        ntp = schema.node_type_property or "node_type"

        def kinds(np) -> set[str]:
            # A node's "kind" is its label(s) and/or an inline node_type literal —
            # the default concept graph discriminates by the node_type PROPERTY,
            # not by label, so both must feed the direction check.
            k = set(np.labels)
            v = np.prop_values.get(ntp)
            if v:
                k.add(v)
            return k

        fixes: list[RelPattern] = []
        for rp in a.rel_patterns:
            if rp.direction not in ("ltr", "rtl"):
                continue
            if len(rp.types) != 1 or rp.left is None or rp.right is None:
                continue
            ls = kinds(rp.left)
            rs = kinds(rp.right)
            if not ls or not rs:
                continue
            t = rp.types[0]
            as_written = any((l, t, r) in edgeset for l in ls for r in rs)
            reversed_ok = any((r, t, l) in edgeset for l in ls for r in rs)
            if rp.direction == "ltr" and not as_written and reversed_ok:
                fixes.append(rp)
            elif rp.direction == "rtl" and not reversed_ok and as_written:
                fixes.append(rp)
        return fixes

    #: Node properties that identify an entity (a literal compared against one
    #: of these should be a bound param). Values compared against enum-like
    #: properties (node_type, status, ...) may legitimately be inlined.
    IDENTITY_PROPS = {"canonical_name", "aliases", "name"}

    def _param_issues(self, a, grounding: GroundingResult | None) -> list[ValidationIssue]:
        if grounding is None or not grounding.entities:
            return []
        # Map grounded surface value (lowercased) -> suggested param name.
        # Only the mention and the resolved canonical name — NOT fuzzy candidate
        # names, which could coincide with unrelated literals.
        grounded: dict[str, str] = {}
        for ent in grounding.entities:
            if ent.param_name is None:
                continue
            for val in (ent.mention, ent.canonical_name):
                if val:
                    grounded[val.lower()] = ent.param_name
        issues: list[ValidationIssue] = []
        for tok in a.string_literals:
            val = (tok.decoded or "").lower()
            if val not in grounded:
                continue
            # Position-aware: only reject when the literal is compared/assigned
            # against an IDENTITY property. A value inlined against node_type /
            # status / etc. (or in an indeterminate position) is not flagged.
            key = _literal_comparison_key(a.sig, tok)
            if key is None or key not in self.IDENTITY_PROPS:
                continue
            issues.append(
                ValidationIssue(
                    code="PARAM_LITERAL",
                    severity="reject",
                    detail=(
                        f"Entity value {tok.decoded!r} is inlined as a string literal; "
                        f"bind it as the parameter ${grounded[val]} instead."
                    ),
                )
            )
        return issues


__all__ = ["Validator", "Denylist", "load_denylist"]
