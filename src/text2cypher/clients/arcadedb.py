"""ArcadeDB HTTP client (openCypher over the /command endpoint).

Read-only by construction: the client is built from a *read-only* database
credential (spec acceptance criterion 7 — enforced at the connection level, not
just the validator). If a write ever slips past the validator, the database
rejects it because the credential lacks write grants.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from ..config import ArcadeConfig
from .base import (
    ConfigurationError,
    GraphAuthError,
    GraphQueryError,
    GraphTimeout,
)


class ArcadeDBClient:
    """Minimal ArcadeDB HTTP client for read-only Cypher queries."""

    read_only = True

    def __init__(
        self,
        url: str,
        database: str,
        user: str,
        password: str,
        *,
        timeout_s: float = 15.0,
        client: httpx.Client | None = None,
        language: str = "cypher",
    ) -> None:
        if not url:
            raise ConfigurationError("ArcadeDB url is required")
        if not database:
            raise ConfigurationError("ArcadeDB database is required")
        self._url = url.rstrip("/")
        self._database = database
        self._user = user
        self._password = password
        self._timeout = timeout_s
        self._language = language
        self._client = client or httpx.Client(timeout=timeout_s)

    @classmethod
    def from_config(cls, config: ArcadeConfig, client: httpx.Client | None = None) -> "ArcadeDBClient":
        # Defense in depth: refuse to construct without a read-only user.
        if config.require_readonly_user and not config.user_readonly:
            raise ConfigurationError(
                "A read-only ArcadeDB user (arcadedb.user_readonly / ARCADE_RO_USER) is "
                "required. The pipeline is read-only and must connect with a credential "
                "that has no write grants (acceptance criterion 7)."
            )
        return cls(
            url=config.url,
            database=config.database,
            user=config.user_readonly,
            password=config.password_readonly,
            timeout_s=config.timeout_s,
            client=client,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._user:
            token = base64.b64encode(f"{self._user}:{self._password}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        return headers

    def query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
        limit: int | None = None,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"language": language or self._language, "command": cypher}
        if params:
            payload["params"] = params
        if limit is not None:
            payload["limit"] = limit
        endpoint = f"{self._url}/api/v1/command/{self._database}"
        try:
            resp = self._client.post(
                endpoint,
                json=payload,
                headers=self._headers(),
                timeout=timeout_s or self._timeout,
            )
        except httpx.TimeoutException as exc:  # pragma: no cover - network
            raise GraphTimeout(str(exc)) from exc
        except httpx.HTTPError as exc:  # pragma: no cover - network
            raise GraphQueryError(str(exc)) from exc

        if resp.status_code in (401, 403):
            raise GraphAuthError(
                f"ArcadeDB rejected the credential ({resp.status_code}); "
                "the read-only user may lack access or a write was attempted."
            )
        if resp.status_code >= 400:
            raise GraphQueryError(_extract_error(resp))

        data = resp.json()
        result = data.get("result", data)
        if isinstance(result, dict):
            result = [result]
        return list(result or [])

    def introspect(self) -> dict[str, Any]:
        """Best-effort live schema introspection via ArcadeDB's schema view."""
        try:
            rows = self.query("SELECT FROM schema:types", limit=10_000, language="sql")
        except Exception:  # pragma: no cover - depends on server
            return {}
        return {"types": rows}


def _extract_error(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        return str(data.get("error") or data.get("detail") or data)
    except Exception:  # pragma: no cover - non-JSON body
        return f"HTTP {resp.status_code}: {resp.text[:500]}"


__all__ = ["ArcadeDBClient"]
