"""A Cypher tokenizer that preserves exact source spans.

Source spans matter: the validator's autofixes (relationship-direction flip and
`LIMIT` injection/clamp) rewrite the query text by character offset, so every
token records where it came from. The tokenizer is deliberately lenient — it
never raises on malformed input; instead it flags unterminated strings /
comments / backticks so the analyzer can report a structural parse error.
"""

from __future__ import annotations

from dataclasses import dataclass

# Token type constants
WS = "WS"
COMMENT = "COMMENT"
STRING = "STRING"
NUMBER = "NUMBER"
IDENT = "IDENT"  # identifier or (unquoted) keyword
QUOTED_IDENT = "QUOTED_IDENT"  # `back-quoted`
PARAM = "PARAM"  # $name or $`name`
OP = "OP"  # multi-char operator (<=, >=, <>, =~, .., +=)
PUNCT = "PUNCT"  # single punctuation char
ERROR = "ERROR"  # unexpected character

# Multi-char operators, longest first so we match greedily.
_MULTI_OPS = ("<=", ">=", "<>", "=~", "..", "+=", "!=")

_IDENT_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_IDENT_CONT = _IDENT_START | set("0123456789")
_DIGITS = set("0123456789")
_SINGLE_PUNCT = set("()[]{}.,;:|+-*/%^=<>!~@?&")


@dataclass
class Token:
    type: str
    value: str  # raw source substring
    start: int  # inclusive char offset
    end: int  # exclusive char offset
    # For STRING / QUOTED_IDENT / PARAM: the decoded (unquoted) content.
    decoded: str | None = None
    # For STRING / QUOTED_IDENT / block COMMENT: whether the literal closed.
    terminated: bool = True

    @property
    def upper(self) -> str:
        return self.value.upper()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.type}, {self.value!r}, {self.start}:{self.end})"


def _read_quoted(text: str, i: int, quote: str) -> tuple[str, int, bool]:
    """Read a quoted run starting at ``text[i] == quote``.

    Returns (decoded, next_index, terminated). Supports backslash escapes and
    the doubled-quote escape (``''`` / ``""`` / ````` ``).
    """
    n = len(text)
    j = i + 1
    out: list[str] = []
    terminated = False
    while j < n:
        c = text[j]
        if c == "\\" and j + 1 < n:
            nxt = text[j + 1]
            mapping = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"', "`": "`", "/": "/"}
            out.append(mapping.get(nxt, nxt))
            j += 2
            continue
        if c == quote:
            # Doubled quote is an escaped quote.
            if j + 1 < n and text[j + 1] == quote:
                out.append(quote)
                j += 2
                continue
            terminated = True
            j += 1
            break
        out.append(c)
        j += 1
    return "".join(out), j, terminated


def tokenize(text: str) -> list[Token]:
    """Tokenize ``text`` into a flat list of :class:`Token` (no WS/COMMENT filtering)."""
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]

        # Whitespace
        if c.isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            tokens.append(Token(WS, text[i:j], i, j))
            i = j
            continue

        # Line comment //...
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = i + 2
            while j < n and text[j] != "\n":
                j += 1
            tokens.append(Token(COMMENT, text[i:j], i, j))
            i = j
            continue

        # Block comment /* ... */
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = i + 2
            terminated = False
            while j < n:
                if text[j] == "*" and j + 1 < n and text[j + 1] == "/":
                    j += 2
                    terminated = True
                    break
                j += 1
            tokens.append(Token(COMMENT, text[i:j], i, j, terminated=terminated))
            i = j
            continue

        # String literal ' ' or " "
        if c in ("'", '"'):
            decoded, j, terminated = _read_quoted(text, i, c)
            tokens.append(Token(STRING, text[i:j], i, j, decoded=decoded, terminated=terminated))
            i = j
            continue

        # Back-quoted identifier
        if c == "`":
            decoded, j, terminated = _read_quoted(text, i, c)
            tokens.append(Token(QUOTED_IDENT, text[i:j], i, j, decoded=decoded, terminated=terminated))
            i = j
            continue

        # Parameter $name or $`name` or $123 (positional)
        if c == "$":
            j = i + 1
            if j < n and text[j] == "`":
                decoded, k, terminated = _read_quoted(text, j, "`")
                tokens.append(Token(PARAM, text[i:k], i, k, decoded=decoded, terminated=terminated))
                i = k
                continue
            k = j
            while k < n and text[k] in _IDENT_CONT:
                k += 1
            if k > j:
                tokens.append(Token(PARAM, text[i:k], i, k, decoded=text[j:k]))
                i = k
                continue
            # Bare '$' — treat as punctuation-ish error marker.
            tokens.append(Token(ERROR, c, i, i + 1))
            i += 1
            continue

        # Number
        if c in _DIGITS or (c == "." and i + 1 < n and text[i + 1] in _DIGITS):
            j = i
            if text[j] == "0" and j + 1 < n and text[j + 1] in ("x", "X"):
                j += 2
                while j < n and text[j] in "0123456789abcdefABCDEF":
                    j += 1
            else:
                while j < n and text[j] in _DIGITS:
                    j += 1
                if j < n and text[j] == ".":
                    j += 1
                    while j < n and text[j] in _DIGITS:
                        j += 1
                if j < n and text[j] in ("e", "E"):
                    k = j + 1
                    if k < n and text[k] in ("+", "-"):
                        k += 1
                    if k < n and text[k] in _DIGITS:
                        j = k
                        while j < n and text[j] in _DIGITS:
                            j += 1
            tokens.append(Token(NUMBER, text[i:j], i, j))
            i = j
            continue

        # Identifier / keyword
        if c in _IDENT_START:
            j = i + 1
            while j < n and text[j] in _IDENT_CONT:
                j += 1
            tokens.append(Token(IDENT, text[i:j], i, j))
            i = j
            continue

        # Multi-char operators
        matched = False
        for op in _MULTI_OPS:
            if text.startswith(op, i):
                tokens.append(Token(OP, op, i, i + len(op)))
                i += len(op)
                matched = True
                break
        if matched:
            continue

        # Single punctuation
        if c in _SINGLE_PUNCT:
            tokens.append(Token(PUNCT, c, i, i + 1))
            i += 1
            continue

        # Anything else is an error character.
        tokens.append(Token(ERROR, c, i, i + 1))
        i += 1

    return tokens


def significant(tokens: list[Token]) -> list[Token]:
    """Drop whitespace and comments."""
    return [t for t in tokens if t.type not in (WS, COMMENT)]


__all__ = [
    "Token",
    "tokenize",
    "significant",
    "WS",
    "COMMENT",
    "STRING",
    "NUMBER",
    "IDENT",
    "QUOTED_IDENT",
    "PARAM",
    "OP",
    "PUNCT",
    "ERROR",
]
