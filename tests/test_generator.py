from __future__ import annotations

from text2cypher.clients.fakes import FakeLLM
from text2cypher.contracts import GroundingResult, SchemaContext
from text2cypher.pipeline.generator import Generator, parse_generation_output


def test_parse_plain_json():
    out = parse_generation_output('{"cypher":"MATCH (n) RETURN n","params_used":{"x":1},"reasoning":"r","confidence":0.8}')
    assert out.cypher == "MATCH (n) RETURN n"
    assert out.params_used == {"x": 1}
    assert out.confidence == 0.8


def test_parse_fenced_json():
    out = parse_generation_output('```json\n{"cypher":"MATCH (n) RETURN n"}\n```')
    assert out.cypher == "MATCH (n) RETURN n"


def test_parse_json_with_prose():
    out = parse_generation_output('Sure! Here is the query:\n{"cypher":"RETURN 1","params_used":{}}\nHope that helps.')
    assert out.cypher == "RETURN 1"


def test_parse_unparseable_yields_empty_cypher():
    out = parse_generation_output("I cannot do that.")
    assert out.cypher == ""
    assert "UNPARSEABLE" in out.reasoning


def test_grounded_params_win_over_model():
    llm = FakeLLM(responses=['{"cypher":"MATCH (n) WHERE n.canonical_name = $e RETURN n","params_used":{"e":"WRONG"}}'])
    gen = Generator(llm)
    grounding = GroundingResult(params={"e": "Correct"})
    schema = SchemaContext(rendered="schema")
    attempt, _ = gen.generate("q", schema, grounding, [])
    assert attempt.params["e"] == "Correct"  # grounded value wins


def test_response_format_requested():
    llm = FakeLLM(responses=['{"cypher":"RETURN 1"}'])
    gen = Generator(llm)
    gen.generate("q", SchemaContext(rendered="s"), GroundingResult(), [])
    # FakeLLM records the messages; ensure system prompt states read-only intent
    assert "READ-ONLY" in llm.calls[0][0]["content"]
