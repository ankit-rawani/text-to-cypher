from __future__ import annotations

from text2cypher.cypher.tokenizer import (
    COMMENT,
    IDENT,
    NUMBER,
    PARAM,
    STRING,
    significant,
    tokenize,
)


def types(toks):
    return [t.type for t in toks]


def test_spans_are_exact():
    src = "MATCH (n)"
    toks = tokenize(src)
    for t in toks:
        assert src[t.start:t.end] == t.value


def test_string_decoding_and_escapes():
    toks = [t for t in tokenize("RETURN 'a\\'b'") if t.type == STRING]
    assert toks[0].decoded == "a'b"
    assert toks[0].terminated
    toks2 = [t for t in tokenize('RETURN "c"') if t.type == STRING]
    assert toks2[0].decoded == "c"


def test_unterminated_string_flagged():
    toks = [t for t in tokenize('RETURN "oops') if t.type == STRING]
    assert toks and not toks[0].terminated


def test_param_and_number():
    toks = significant(tokenize("WHERE n.x = $foo AND n.y = 12.5"))
    assert any(t.type == PARAM and t.decoded == "foo" for t in toks)
    assert any(t.type == NUMBER and t.value == "12.5" for t in toks)


def test_line_and_block_comments():
    toks = tokenize("MATCH (n) // hi\n/* block */ RETURN n")
    comments = [t for t in toks if t.type == COMMENT]
    assert len(comments) == 2
    assert all(c.terminated for c in comments)
    # comments dropped by significant()
    assert COMMENT not in [t.type for t in significant(toks)]


def test_unterminated_block_comment():
    toks = [t for t in tokenize("MATCH (n) /* unclosed") if t.type == COMMENT]
    assert toks and not toks[0].terminated


def test_backtick_identifier():
    toks = significant(tokenize("MATCH (`weird name`:Concept) RETURN 1"))
    assert any(t.value == "`weird name`" for t in toks)


def test_multichar_operators():
    toks = significant(tokenize("WHERE a <= 1 AND b >= 2 AND c <> 3 AND d =~ '.*'"))
    ops = [t.value for t in toks if t.type == "OP"]
    assert "<=" in ops and ">=" in ops and "<>" in ops and "=~" in ops
