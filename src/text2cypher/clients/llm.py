"""OpenAI-compatible chat completions client."""

from __future__ import annotations

from typing import Any

import httpx

from ..config import LLMConfig
from .base import ClientError, ConfigurationError


class OpenAIChatClient:
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str = "",
        timeout_s: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not endpoint:
            raise ConfigurationError("LLM endpoint is required")
        if not model:
            raise ConfigurationError("LLM model is required")
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_s
        self._client = client or httpx.Client(timeout=timeout_s)

    @classmethod
    def from_config(cls, config: LLMConfig, client: httpx.Client | None = None) -> "OpenAIChatClient":
        return cls(
            endpoint=config.endpoint,
            model=config.model,
            api_key=config.api_key,
            timeout_s=config.timeout_s,
            client=client,
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        resp = self._client.post(
            f"{self._endpoint}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout_s or self._timeout,
        )
        if resp.status_code >= 400:
            raise ClientError(f"LLM call failed ({resp.status_code}): {resp.text[:500]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ClientError(f"Unexpected LLM response shape: {data}") from exc


__all__ = ["OpenAIChatClient"]
