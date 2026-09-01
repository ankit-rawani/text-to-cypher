from __future__ import annotations

import json

import httpx
import pytest

from text2cypher.clients import AnthropicClient, OpenAIChatClient, build_llm
from text2cypher.clients.base import ClientError, ConfigurationError
from text2cypher.config import LLMConfig, load_config, load_dotenv


def _mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---- provider factory ---------------------------------------------------


def test_build_llm_selects_provider():
    a = build_llm(LLMConfig(provider="anthropic", model="claude-sonnet-5", endpoint="https://api.anthropic.com"))
    assert isinstance(a, AnthropicClient)
    o = build_llm(LLMConfig(provider="openai", model="gpt-4o", endpoint="http://x/v1"))
    assert isinstance(o, OpenAIChatClient)


def test_build_llm_unknown_provider():
    with pytest.raises(ConfigurationError):
        build_llm(LLMConfig(provider="mystery", model="m", endpoint="http://x"))


# ---- Anthropic client ---------------------------------------------------


def test_anthropic_request_shape():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["headers"] = dict(req.headers)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"stop_reason": "end_turn", "content": [{"type": "text", "text": "{\"cypher\":\"RETURN 1\"}"}]})

    c = AnthropicClient("https://api.anthropic.com", "claude-sonnet-5", api_key="sk-ant-x", client=_mock(handler))
    out = c.complete([{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}])
    assert out == '{"cypher":"RETURN 1"}'
    assert seen["url"].endswith("/v1/messages")
    assert seen["headers"]["x-api-key"] == "sk-ant-x"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["system"] == "SYS"
    assert [m["role"] for m in seen["body"]["messages"]] == ["user"]
    # temperature must NOT be sent by default (current Claude models 400 on it)
    assert "temperature" not in seen["body"]


def test_anthropic_send_temperature_opt_in():
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"stop_reason": "end_turn", "content": [{"type": "text", "text": "x"}]})

    c = AnthropicClient("https://api.anthropic.com", "claude-3-5-sonnet", send_temperature=True, client=_mock(handler))
    c.complete([{"role": "user", "content": "hi"}], temperature=0.0)
    assert seen["body"]["temperature"] == 0.0


def test_anthropic_refusal_raises():
    def handler(req):
        return httpx.Response(200, json={"stop_reason": "refusal", "stop_details": {"category": "cyber", "explanation": "no"}, "content": []})

    c = AnthropicClient("https://api.anthropic.com", "claude-opus-5", client=_mock(handler))
    with pytest.raises(ClientError, match="refused"):
        c.complete([{"role": "user", "content": "x"}])


def test_anthropic_error_status_raises():
    c = AnthropicClient("https://api.anthropic.com", "claude-opus-5", client=_mock(lambda r: httpx.Response(401, text="bad key")))
    with pytest.raises(ClientError):
        c.complete([{"role": "user", "content": "x"}])


def test_anthropic_empty_response_raises():
    def handler(req):
        return httpx.Response(200, json={"stop_reason": "max_tokens", "content": []})

    c = AnthropicClient("https://api.anthropic.com", "claude-opus-5", client=_mock(handler))
    with pytest.raises(ClientError):
        c.complete([{"role": "user", "content": "x"}])


# ---- config wiring ------------------------------------------------------


def test_config_provider_and_bool_coercion(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_SEND_TEMPERATURE", "false")
    monkeypatch.setenv("LLM_MAX_TOKENS", "8000")
    cfg = load_config()
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.send_temperature is False
    assert cfg.llm.max_tokens == 8000


# ---- dotenv -------------------------------------------------------------


def test_load_dotenv_sets_without_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('LLM_MODEL=claude-sonnet-5\nexport ARCADE_DB="graphdb"\n# comment\nEXISTING=fromfile\n', encoding="utf-8")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("EXISTING", "fromenv")
    load_dotenv(env)
    import os

    assert os.environ["LLM_MODEL"] == "claude-sonnet-5"
    assert os.environ["ARCADE_DB"] == "graphdb"  # quotes stripped, export prefix handled
    assert os.environ["EXISTING"] == "fromenv"  # real env not overridden
