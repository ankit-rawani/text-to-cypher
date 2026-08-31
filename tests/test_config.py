from __future__ import annotations

from text2cypher.config import AppConfig, load_config
from text2cypher.config import _interpolate  # type: ignore


def test_defaults_load():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.pipeline.row_cap == 200
    assert cfg.grounding.sim_threshold == 0.55
    assert cfg.examples.top_k == 3
    assert cfg.arcadedb.require_readonly_user is True


def test_env_interpolation(monkeypatch):
    monkeypatch.setenv("ARCADE_URL", "http://db:2480")
    cfg = load_config()
    assert cfg.arcadedb.url == "http://db:2480"


def test_env_default_when_unset(monkeypatch):
    monkeypatch.delenv("ARCADE_DB", raising=False)
    cfg = load_config()
    assert cfg.arcadedb.database == "concept_graph"


def test_interpolate_default_syntax(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    assert _interpolate("${NOPE:-fallback}") == "fallback"
    monkeypatch.setenv("NOPE", "real")
    assert _interpolate("${NOPE:-fallback}") == "real"


def test_overrides_merge():
    cfg = load_config(overrides={"pipeline": {"row_cap": 42}})
    assert cfg.pipeline.row_cap == 42
    # untouched keys retain defaults
    assert cfg.pipeline.max_attempts == 3


def test_resolve_path_relative():
    cfg = load_config()
    p = cfg.resolve_path("config/arcade_denylist.yaml")
    assert p.is_absolute()
    assert p.exists()
