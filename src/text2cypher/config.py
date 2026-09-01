"""Configuration models and loader (spec section 6).

Config is layered: a packaged ``config/default.yaml`` provides defaults, an
optional user file overrides it, and ``${VAR}`` / ``${VAR:-default}`` tokens
are interpolated from the environment. The result is a validated
:class:`AppConfig`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

# Repo root: .../src/text2cypher/config.py -> parents[2]
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parents[1]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"


# --------------------------------------------------------------------------
# Config models
# --------------------------------------------------------------------------


class LLMConfig(BaseModel):
    # "openai" (OpenAI-compatible /chat/completions) or "anthropic" (Claude Messages API)
    provider: str = "openai"
    endpoint: str = ""  # base URL; for anthropic this is e.g. https://api.anthropic.com
    model: str = ""
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 60.0
    # Anthropic-specific:
    anthropic_version: str = "2023-06-01"
    # Current Claude models reject `temperature` (HTTP 400); keep this False for them.
    send_temperature: bool = False


class ArcadeConfig(BaseModel):
    url: str = ""
    database: str = "concept_graph"
    # Read-only credential is REQUIRED for real use (acceptance criterion 7).
    user_readonly: str = ""
    password_readonly: str = ""
    timeout_s: float = 15.0
    # Extra guard: refuse to construct an executor unless a RO user is set.
    require_readonly_user: bool = True


class QdrantConfig(BaseModel):
    url: str = ""
    api_key: str = ""
    node_collection: str = "concepts"
    example_collection: str = "t2c_examples"
    timeout_s: float = 15.0


class MongoConfig(BaseModel):
    url: str = ""
    database: str = ""
    collection: str = ""
    enabled: bool = False


class EmbeddingConfig(BaseModel):
    # "openai" (OpenAI-compatible endpoint) or "hashing" (offline, deterministic)
    provider: str = "hashing"
    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    dim: int = 256
    timeout_s: float = 30.0


class GroundingConfig(BaseModel):
    sim_threshold: float = 0.55
    max_candidates: int = 3
    # Below this similarity a resolved entity is flagged low-confidence.
    low_confidence_threshold: float = 0.65
    enabled: bool = True


class ExamplesConfig(BaseModel):
    top_k: int = 3
    sim_threshold: float = 0.60
    enabled: bool = True
    # Local seed file (JSONL) used to warm the store / offline fake.
    seed_path: str = "examples/seed_examples.jsonl"


class SchemaCacheConfig(BaseModel):
    ttl_s: float = 300.0


class DialectConfig(BaseModel):
    denylist_path: str = "config/arcade_denylist.yaml"


class PipelineConfig(BaseModel):
    max_attempts: int = 3
    hard_max_attempts: int = 5
    row_cap: int = 200
    allow_relaxation: bool = True


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    arcadedb: ArcadeConfig = Field(default_factory=ArcadeConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    mongodb: MongoConfig = Field(default_factory=MongoConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    examples: ExamplesConfig = Field(default_factory=ExamplesConfig)
    schema_cache: SchemaCacheConfig = Field(default_factory=SchemaCacheConfig)
    dialect: DialectConfig = Field(default_factory=DialectConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    # Path to the user-configured GraphConfig (node/edge kinds, statuses, attrs).
    graph_config_path: str = "config/graph_config.yaml"
    # Absolute repo root, used to resolve relative paths (denylist, seeds).
    root: str = str(_REPO_ROOT)

    def resolve_path(self, path: str | os.PathLike[str]) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return Path(self.root) / p


# --------------------------------------------------------------------------
# Loading + env interpolation
# --------------------------------------------------------------------------


def _interpolate(value: Any) -> Any:
    """Recursively substitute ``${VAR}`` / ``${VAR:-default}`` in strings."""
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            env = os.environ.get(name)
            if env is not None:
                return env
            return default if default is not None else ""

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_dotenv(path: str | os.PathLike[str]) -> None:
    """Load ``KEY=VALUE`` lines from a .env file into os.environ.

    Existing environment variables are never overridden (real env wins over the
    file), matching standard dotenv semantics. Silently ignores a missing file.
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _autoload_dotenv() -> None:
    # Repo-root .env first, then a cwd .env (cwd cannot override already-set keys).
    load_dotenv(_REPO_ROOT / ".env")
    cwd_env = Path.cwd() / ".env"
    if cwd_env.resolve() != (_REPO_ROOT / ".env").resolve():
        load_dotenv(cwd_env)


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    interpolate_env: bool = True,
    use_dotenv: bool = True,
) -> AppConfig:
    """Load configuration.

    Parameters
    ----------
    path:
        Optional user config file. Merged on top of the packaged defaults.
    overrides:
        In-memory dict merged last (useful for tests / programmatic use).
    interpolate_env:
        Whether to substitute ``${VAR}`` tokens from the environment.
    """
    if use_dotenv and interpolate_env:
        _autoload_dotenv()
    data: dict[str, Any] = {}
    if DEFAULT_CONFIG_PATH.exists():
        data = load_yaml(DEFAULT_CONFIG_PATH)
    if path is not None:
        data = _deep_merge(data, load_yaml(path))
    if overrides:
        data = _deep_merge(data, overrides)
    if interpolate_env:
        data = _interpolate(data)
    return AppConfig.model_validate(data)


__all__ = [
    "AppConfig",
    "LLMConfig",
    "ArcadeConfig",
    "QdrantConfig",
    "MongoConfig",
    "EmbeddingConfig",
    "GroundingConfig",
    "ExamplesConfig",
    "SchemaCacheConfig",
    "DialectConfig",
    "PipelineConfig",
    "load_config",
    "load_yaml",
    "load_dotenv",
    "DEFAULT_CONFIG_PATH",
]
