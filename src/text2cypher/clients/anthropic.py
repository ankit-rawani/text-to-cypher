"""Anthropic Messages API client (native ``/v1/messages``).

A thin httpx client — same shape and testability (injectable client /
``MockTransport``) as the other clients in this package, no vendor SDK pulled in.
Lets you point the Generator at the real Claude API (or an Anthropic-compatible
proxy) with a custom base URL, model, and API key.

Notes for current Claude models (Opus 5 / 4.8 / 4.7, Sonnet 5, ...):
- ``temperature`` is rejected (HTTP 400) on these models, so it is NOT sent by
  default. Enable ``send_temperature`` only for older models that accept it.
- These models don't support assistant prefill or an OpenAI-style
  ``response_format``; the Generator's system prompt already asks for a bare
  JSON object and the parser tolerates prose/fences, so no JSON mode is needed.
- ``system`` messages are lifted into the top-level ``system`` field.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..config import LLMConfig
from .base import ClientError, ConfigurationError

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        *,
        max_tokens: int = 4096,
        timeout_s: float = 60.0,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        send_temperature: bool = False,
        client: httpx.Client | None = None,
    ) -> None:
        if not model:
            raise ConfigurationError("LLM model is required")
        self._base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._timeout = timeout_s
        self._version = anthropic_version
        self._send_temperature = send_temperature
        self._client = client or httpx.Client(timeout=timeout_s)

    @classmethod
    def from_config(cls, config: LLMConfig, client: httpx.Client | None = None) -> "AnthropicClient":
        return cls(
            base_url=config.endpoint or DEFAULT_BASE_URL,
            model=config.model,
            api_key=config.api_key,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            anthropic_version=config.anthropic_version,
            send_temperature=config.send_temperature,
            client=client,
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self._version,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,  # ignored (see module docstring)
        timeout_s: float | None = None,
    ) -> str:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        convo = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max(max_tokens, self._max_tokens),
            "messages": convo,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if self._send_temperature:
            payload["temperature"] = temperature

        resp = self._client.post(
            f"{self._base_url}/v1/messages",
            json=payload,
            headers=self._headers(),
            timeout=timeout_s or self._timeout,
        )
        if resp.status_code >= 400:
            raise ClientError(f"Anthropic call failed ({resp.status_code}): {resp.text[:500]}")
        data = resp.json()

        stop_reason = data.get("stop_reason")
        if stop_reason == "refusal":
            detail = data.get("stop_details") or {}
            raise ClientError(
                f"Claude refused the request (category={detail.get('category')}): "
                f"{detail.get('explanation') or ''}"
            )

        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text.strip():
            raise ClientError(f"Empty Claude response (stop_reason={stop_reason}).")
        return text


__all__ = ["AnthropicClient", "DEFAULT_BASE_URL", "DEFAULT_ANTHROPIC_VERSION"]
