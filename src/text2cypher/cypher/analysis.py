"""Structural + semantic analysis of a Cypher query.

The analyzer produces a lightweight parse view rich enough for the validator's
deterministic checks (spec section 3) without pulling in a full openCypher
grammar. It is built to have a *low false-reject rate*: the structural "parse"
gate only fails on unambiguous syntax errors (unterminated literals, unbalanced
brackets, no leading reading clause, stray characters), leaving richer judgment
to the schema/dialect/write checks that run after it.

It also exposes two text transforms used by the validator's autofixes:

* :func:`flip_direction` — flip an unambiguously-backwards relationship arrow.
* :func:`ensure_limit` — inject or clamp the final ``LIMIT`` to a row cap.

Both operate on exact character spans recorded by the tokenizer, so they never
disturb the rest of the query.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tokenizer import (
    IDENT,
    NUMBER,
    PARAM,
    PUNCT,
    QUOTED_IDENT,
    STRING,
    Token,
    significant,
    tokenize,
)

# Clauses that may legally begin a query (read or write — write clauses are
# caught by the dedicated write-op check, not the structural parse).
LEADING_CLAUSES = {
    "MATCH", "OPTIONAL", "MERGE", "CREATE", "DELETE", "DETACH", "SET",
    "REMOVE", "DROP", "WITH", "UNWIND", "CALL", "RETURN", "FOREACH", "LOAD",
    "USE",
}

# Top-level clause keywords (for clause-sequence extraction / LIMIT location).
CLAUSE_KEYWORDS = LEADING_CLAUSES | {
    "WHERE", "ORDER", "SKIP", "LIMIT", "UNION", "YIELD", "AS", "BY", "ON",
}

# Keywords that mutate the graph — a read-only pipeline hard-blocks these.
WRITE_KEYWORDS = {
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "FOREACH",
    "LOAD",
}

# Reserved words that can be followed by '(' but are never function/procedure
# names (so they must not be recorded as calls for the denylist check).
RESERVED_NONCALL = CLAUSE_KEYWORDS | {
    "AND", "OR", "XOR", "NOT", "IN", "IS", "STARTS", "ENDS", "CONTAINS",
    "WHEN", "THEN", "ELSE", "CASE", "END", "DISTINCT", "ASC", "DESC",
    "TRUE", "FALSE", "NULL", "CONSTRAINT", "INDEX",
}

_OPEN = {"(", "[", "{"}
_CLOSE = {")": "(", "]": "[", "}": "{"}


@dataclass
class NodePattern:
    variable: str | None
    labels: list[str]
    prop_keys: list[str]
    start: int
    end: int
    sig_open: int
    sig_close: int


@dataclass
class RelPattern:
    variable: str | None
    types: list[str]
    prop_keys: list[str]
    direction: str  # "ltr" | "rtl" | "undirected"
    var_length: bool
    connector_start: int  # char offset just after the left node's ')'
    connector_end: int  # char offset of the right node's '('
    left: NodePattern | None
    right: NodePattern | None


@dataclass
class PropertyAccess:
    variable: str
    key: str
    start: int
    end: int


@dataclass
class CallRef:
    name: str
    start: int
    is_procedure: bool


@dataclass
class WriteOp:
    keyword: str
    start: int


@dataclass
class LimitInfo:
    present: bool = False
    is_literal: bool = False
    value: int | None = None
    num_start: int | None = None
    num_end: int | None = None
    return_exists: bool = False


@dataclass
class Analysis:
    source: str
    ok: bool
    error: str | None
    error_pos: int | None
    tokens: list[Token]
    sig: list[Token]
    clauses: list[tuple[str, int]] = field(default_factory=list)
    node_patterns: list[NodePattern] = field(default_factory=list)
    rel_patterns: list[RelPattern] = field(default_factory=list)
    property_accesses: list[PropertyAccess] = field(default_factory=list)
    string_literals: list[Token] = field(default_factory=list)
    params: set[str] = field(default_factory=set)
    write_ops: list[WriteOp] = field(default_factory=list)
    calls: list[CallRef] = field(default_factory=list)
    limit: LimitInfo = field(default_factory=LimitInfo)
    var_labels: dict[str, set[str]] = field(default_factory=dict)
    var_types: dict[str, set[str]] = field(default_factory=dict)

    def labels(self) -> set[str]:
        out: set[str] = set()
        for np in self.node_patterns:
            out.update(np.labels)
        return out

    def rel_types(self) -> set[str]:
        out: set[str] = set()
        for rp in self.rel_patterns:
            out.update(rp.types)
        return out


# --------------------------------------------------------------------------
# Bracket matching + depth
# --------------------------------------------------------------------------


def _match_brackets(sig: list[Token]) -> tuple[dict[int, int], str | None, int | None]:
    """Return (open_idx -> close_idx map, error, error_char_pos)."""
    stack: list[tuple[str, int]] = []
    pair: dict[int, int] = {}
    for idx, tok in enumerate(sig):
        if tok.type == PUNCT and tok.value in _OPEN:
            stack.append((tok.value, idx))
        elif tok.type == PUNCT and tok.value in _CLOSE:
            if not stack:
                return pair, f"unbalanced '{tok.value}'", tok.start
            open_char, open_idx = stack.pop()
            if open_char != _CLOSE[tok.value]:
                return pair, f"mismatched '{open_char}' and '{tok.value}'", tok.start
            pair[open_idx] = idx
    if stack:
        open_char, open_idx = stack[-1]
        return pair, f"unclosed '{open_char}'", sig[open_idx].start
    return pair, None, None


def _depths(sig: list[Token]) -> list[int]:
    """Nesting depth each token sits at (matching brackets share a depth)."""
    depths = [0] * len(sig)
    depth = 0
    for idx, tok in enumerate(sig):
        if tok.type == PUNCT and tok.value in _CLOSE:
            depth -= 1
            depths[idx] = depth
        else:
            depths[idx] = depth
            if tok.type == PUNCT and tok.value in _OPEN:
                depth += 1
    return depths


def _label_name(tok: Token) -> str:
    return tok.decoded if tok.type == QUOTED_IDENT and tok.decoded is not None else tok.value


# --------------------------------------------------------------------------
# Map / node / rel parsing
# --------------------------------------------------------------------------


def _map_keys(sig: list[Token], open_idx: int, bracket: dict[int, int]) -> list[str]:
    close = bracket.get(open_idx)
    if close is None:
        return []
    keys: list[str] = []
    depth = 0
    p = open_idx + 1
    while p < close:
        tok = sig[p]
        if depth == 0 and tok.type in (IDENT, QUOTED_IDENT, STRING):
            if p + 1 < close and sig[p + 1].type == PUNCT and sig[p + 1].value == ":":
                keys.append(tok.decoded if tok.decoded is not None else tok.value)
        if tok.type == PUNCT and tok.value in _OPEN:
            depth += 1
        elif tok.type == PUNCT and tok.value in _CLOSE:
            depth -= 1
        p += 1
    return keys


def _parse_node(sig: list[Token], open_idx: int, bracket: dict[int, int]) -> NodePattern | None:
    """Classify a ``(...)`` group as a node pattern; return it or None."""
    close = bracket.get(open_idx)
    if close is None:
        return None
    idx = open_idx + 1
    var: str | None = None
    labels: list[str] = []
    prop_keys: list[str] = []

    # optional variable
    if idx < close and sig[idx].type in (IDENT, QUOTED_IDENT):
        # A following '(' would make this a function call, not a node var.
        if not (idx + 1 < close and sig[idx + 1].type == PUNCT and sig[idx + 1].value == "("):
            var = _label_name(sig[idx])
            idx += 1

    # labels: (: | | | &) IDENT ...
    while idx < close and sig[idx].type == PUNCT and sig[idx].value in (":", "|", "&"):
        if idx + 1 < close and sig[idx + 1].type in (IDENT, QUOTED_IDENT):
            labels.append(_label_name(sig[idx + 1]))
            idx += 2
        else:
            break

    # optional property map
    if idx < close and sig[idx].type == PUNCT and sig[idx].value == "{":
        prop_keys = _map_keys(sig, idx, bracket)
        map_close = bracket.get(idx)
        if map_close is None:
            return None
        idx = map_close + 1

    # optional inline WHERE predicate (Cypher 5) — accept remainder.
    if idx < close and sig[idx].type == IDENT and sig[idx].upper == "WHERE":
        idx = close

    if idx != close:
        return None
    return NodePattern(
        variable=var,
        labels=labels,
        prop_keys=prop_keys,
        start=sig[open_idx].start,
        end=sig[close].end,
        sig_open=open_idx,
        sig_close=close,
    )


def _parse_rel_detail(
    sig: list[Token], open_idx: int, bracket: dict[int, int]
) -> tuple[str | None, list[str], list[str], bool]:
    """Parse a ``[...]`` relationship detail. Returns (var, types, keys, var_length)."""
    close = bracket.get(open_idx, open_idx)
    var: str | None = None
    types: list[str] = []
    prop_keys: list[str] = []
    var_length = False

    idx = open_idx + 1
    # optional variable (IDENT not immediately after ':'/'|' and not a func call)
    if idx < close and sig[idx].type in (IDENT, QUOTED_IDENT):
        if not (idx + 1 < close and sig[idx + 1].type == PUNCT and sig[idx + 1].value == "("):
            var = _label_name(sig[idx])
            idx += 1

    p = idx
    while p < close:
        tok = sig[p]
        if tok.type == PUNCT and tok.value in (":", "|"):
            if p + 1 < close and sig[p + 1].type in (IDENT, QUOTED_IDENT):
                types.append(_label_name(sig[p + 1]))
                p += 2
                continue
        if tok.type == PUNCT and tok.value == "*":
            var_length = True
        if tok.type == PUNCT and tok.value == "{":
            prop_keys = _map_keys(sig, p, bracket)
            mc = bracket.get(p)
            if mc is not None:
                p = mc + 1
                continue
        p += 1
    return var, types, prop_keys, var_length


# --------------------------------------------------------------------------
# Relationship connector detection
# --------------------------------------------------------------------------


def _connector_between(
    sig: list[Token], lo: int, hi: int, bracket: dict[int, int]
) -> tuple[bool, str, int | None]:
    """Is sig[lo:hi] a relationship connector?

    Returns (is_connector, direction, rel_bracket_open_idx).
    A connector is a run of ``-`` / ``<`` / ``>`` with at most one ``[...]``.
    """
    has_lt = False
    has_gt = False
    has_dash = False
    rel_bracket: int | None = None
    p = lo
    while p < hi:
        tok = sig[p]
        if tok.type == PUNCT and tok.value == "-":
            has_dash = True
        elif tok.type == PUNCT and tok.value == "<":
            has_lt = True
        elif tok.type == PUNCT and tok.value == ">":
            has_gt = True
        elif tok.type == PUNCT and tok.value == "[":
            # Must be a rel detail: preceded by a dash within the connector.
            if rel_bracket is not None:
                return False, "", None
            close = bracket.get(p)
            if close is None or close >= hi:
                return False, "", None
            rel_bracket = p
            p = close
        else:
            return False, "", None
        p += 1

    if not has_dash:
        return False, "", None
    if has_lt and has_gt:
        direction = "undirected"  # malformed <..> — treat as ambiguous
    elif has_gt:
        direction = "ltr"
    elif has_lt:
        direction = "rtl"
    else:
        direction = "undirected"
    return True, direction, rel_bracket


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def analyze(source: str) -> Analysis:
    tokens = tokenize(source)
    sig = significant(tokens)

    analysis = Analysis(
        source=source,
        ok=True,
        error=None,
        error_pos=None,
        tokens=tokens,
        sig=sig,
    )

    # --- Structural gate -------------------------------------------------
    if not sig:
        analysis.ok = False
        analysis.error = "empty query"
        return analysis

    for tok in tokens:
        if tok.type == "ERROR":
            analysis.ok = False
            analysis.error = f"unexpected character {tok.value!r}"
            analysis.error_pos = tok.start
            return analysis
        if tok.type in (STRING, QUOTED_IDENT) and not tok.terminated:
            analysis.ok = False
            analysis.error = "unterminated string literal" if tok.type == STRING else "unterminated quoted identifier"
            analysis.error_pos = tok.start
            return analysis
        if tok.type == "COMMENT" and not tok.terminated:
            analysis.ok = False
            analysis.error = "unterminated block comment"
            analysis.error_pos = tok.start
            return analysis

    bracket, berr, bpos = _match_brackets(sig)
    if berr:
        analysis.ok = False
        analysis.error = berr
        analysis.error_pos = bpos
        return analysis

    first = sig[0]
    if not (first.type == IDENT and first.upper in LEADING_CLAUSES):
        analysis.ok = False
        analysis.error = "query must begin with a reading clause (MATCH/WITH/UNWIND/CALL/RETURN/...)"
        analysis.error_pos = first.start
        return analysis

    # --- Semantic extraction (runs regardless; used by later checks) -----
    depths = _depths(sig)

    # Clauses at depth 0
    for idx, tok in enumerate(sig):
        if depths[idx] == 0 and tok.type == IDENT and tok.upper in CLAUSE_KEYWORDS:
            analysis.clauses.append((tok.upper, tok.start))

    # Write ops (any depth; not a property access, not back-quoted)
    for idx, tok in enumerate(sig):
        if tok.type == IDENT and tok.upper in WRITE_KEYWORDS:
            prev = sig[idx - 1] if idx > 0 else None
            if prev is not None and prev.type == PUNCT and prev.value == ".":
                continue  # property/namespace access
            analysis.write_ops.append(WriteOp(tok.upper, tok.start))

    # Params
    for tok in sig:
        if tok.type == PARAM and tok.decoded:
            analysis.params.add(tok.decoded)

    # String literals
    analysis.string_literals = [t for t in sig if t.type == STRING]

    # Calls: dotted-name run immediately followed by '('  -> function/proc
    n = len(sig)
    for idx, tok in enumerate(sig):
        if tok.type != IDENT:
            continue
        if tok.upper in RESERVED_NONCALL:
            continue  # clause/logical keywords are not call names
        # start of a dotted name only if not preceded by '.'
        prev = sig[idx - 1] if idx > 0 else None
        if prev is not None and prev.type == PUNCT and prev.value == ".":
            continue
        j = idx
        parts = [tok.value]
        while j + 2 < n and sig[j + 1].type == PUNCT and sig[j + 1].value == "." and sig[j + 2].type == IDENT:
            parts.append(sig[j + 2].value)
            j += 2
        if j + 1 < n and sig[j + 1].type == PUNCT and sig[j + 1].value == "(" and len(parts) >= 1:
            # a bare single-word call like count( is a function; keep dotted ones too
            analysis.calls.append(CallRef(".".join(parts), tok.start, is_procedure=False))

    # Procedure calls after CALL (dotted name, not a subquery '{')
    for idx, tok in enumerate(sig):
        if tok.type == IDENT and tok.upper == "CALL":
            if idx + 1 < n and not (sig[idx + 1].type == PUNCT and sig[idx + 1].value == "{"):
                j = idx + 1
                if sig[j].type == IDENT:
                    parts = [sig[j].value]
                    while j + 2 < n and sig[j + 1].type == PUNCT and sig[j + 1].value == "." and sig[j + 2].type == IDENT:
                        parts.append(sig[j + 2].value)
                        j += 2
                    analysis.calls.append(CallRef(".".join(parts), sig[idx + 1].start, is_procedure=True))

    # Property accesses: IDENT '.' IDENT  (not a call, not a namespace chain)
    for idx in range(1, n - 1):
        tok = sig[idx]
        if not (tok.type == PUNCT and tok.value == "."):
            continue
        left = sig[idx - 1]
        right = sig[idx + 1]
        if left.type != IDENT or right.type != IDENT:
            continue
        # skip namespace chains  a.b.c  (left preceded by '.', or right followed by '.')
        if idx - 2 >= 0 and sig[idx - 2].type == PUNCT and sig[idx - 2].value == ".":
            continue
        if idx + 2 < n and sig[idx + 2].type == PUNCT and sig[idx + 2].value == ".":
            continue
        # skip function call  ns.func(
        if idx + 2 < n and sig[idx + 2].type == PUNCT and sig[idx + 2].value == "(":
            continue
        analysis.property_accesses.append(
            PropertyAccess(variable=left.value, key=right.value, start=right.start, end=right.end)
        )

    # Node patterns
    for idx, tok in enumerate(sig):
        if tok.type == PUNCT and tok.value == "(":
            node = _parse_node(sig, idx, bracket)
            if node is not None:
                analysis.node_patterns.append(node)

    # Relationship connectors between consecutive node patterns
    nodes_sorted = sorted(analysis.node_patterns, key=lambda np: np.sig_open)
    for a, b in zip(nodes_sorted, nodes_sorted[1:]):
        lo, hi = a.sig_close + 1, b.sig_open
        if lo >= hi:
            continue
        is_conn, direction, rel_bracket = _connector_between(sig, lo, hi, bracket)
        if not is_conn:
            continue
        var = types = None
        prop_keys: list[str] = []
        var_length = False
        rtypes: list[str] = []
        if rel_bracket is not None:
            var, rtypes, prop_keys, var_length = _parse_rel_detail(sig, rel_bracket, bracket)
        analysis.rel_patterns.append(
            RelPattern(
                variable=var,
                types=rtypes,
                prop_keys=prop_keys,
                direction=direction,
                var_length=var_length,
                connector_start=sig[a.sig_close].end,
                connector_end=sig[b.sig_open].start,
                left=a,
                right=b,
            )
        )

    # var -> labels / types maps
    for np in analysis.node_patterns:
        if np.variable:
            analysis.var_labels.setdefault(np.variable, set()).update(np.labels)
    for rp in analysis.rel_patterns:
        if rp.variable:
            analysis.var_types.setdefault(rp.variable, set()).update(rp.types)

    # LIMIT info (after the last top-level RETURN)
    analysis.limit = _find_limit(sig, depths, analysis.clauses)

    return analysis


def _find_limit(sig: list[Token], depths: list[int], clauses: list[tuple[str, int]]) -> LimitInfo:
    info = LimitInfo()
    # Locate last top-level RETURN by char start; find its sig index.
    return_starts = [start for kw, start in clauses if kw == "RETURN"]
    if not return_starts:
        return info
    info.return_exists = True
    last_return_start = max(return_starts)
    last_return_sig = next((i for i, t in enumerate(sig) if t.start == last_return_start), None)
    if last_return_sig is None:
        return info
    for i in range(last_return_sig + 1, len(sig)):
        tok = sig[i]
        if depths[i] == 0 and tok.type == IDENT and tok.upper == "LIMIT":
            info.present = True
            # value follows
            if i + 1 < len(sig):
                nxt = sig[i + 1]
                if nxt.type == NUMBER:
                    try:
                        info.is_literal = True
                        info.value = int(nxt.value)
                        info.num_start = nxt.start
                        info.num_end = nxt.end
                    except ValueError:
                        info.is_literal = False
            break
    return info


# --------------------------------------------------------------------------
# Text transforms (autofixes)
# --------------------------------------------------------------------------


def flip_direction(source: str, rel: RelPattern) -> str:
    """Flip an ``ltr`` <-> ``rtl`` relationship arrow in-place by char span."""
    start, end = rel.connector_start, rel.connector_end
    seg = source[start:end]
    stripped = seg.strip()
    if not stripped:
        return source
    lead = seg[: len(seg) - len(seg.lstrip())]
    trail = seg[len(seg.rstrip()):]
    core = stripped
    if core.startswith("<"):
        new_core = core[1:]
        if not new_core.endswith(">"):
            new_core = new_core + ">"
    elif core.endswith(">"):
        new_core = core[:-1]
        if not new_core.startswith("<"):
            new_core = "<" + new_core
    else:
        return source  # undirected — nothing to flip
    return source[:start] + lead + new_core + trail + source[end:]


def ensure_limit(source: str, row_cap: int, analysis: Analysis) -> tuple[str, str | None]:
    """Inject or clamp a final ``LIMIT``.

    Returns (new_source, action) where action is ``"injected"``, ``"clamped"``,
    or ``None`` (already within cap / not applicable).
    """
    limit = analysis.limit
    if limit.present:
        if limit.is_literal and limit.value is not None and limit.value > row_cap:
            assert limit.num_start is not None and limit.num_end is not None
            new = source[: limit.num_start] + str(row_cap) + source[limit.num_end:]
            return new, "clamped"
        return source, None

    if not limit.return_exists:
        return source, None  # nothing sensible to limit

    core = source.rstrip()
    tail_ws = source[len(core):]
    if core.endswith(";"):
        body = core[:-1].rstrip()
        new = body + f" LIMIT {row_cap};" + tail_ws
    else:
        new = core + f" LIMIT {row_cap}" + tail_ws
    return new, "injected"


__all__ = [
    "Analysis",
    "NodePattern",
    "RelPattern",
    "PropertyAccess",
    "CallRef",
    "WriteOp",
    "LimitInfo",
    "analyze",
    "flip_direction",
    "ensure_limit",
    "LEADING_CLAUSES",
    "WRITE_KEYWORDS",
]
