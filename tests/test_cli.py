"""CLI tests for the offline-capable subcommands."""

from __future__ import annotations

from typer.testing import CliRunner

from text2cypher.cli import app

runner = CliRunner()


def test_schema_show():
    result = runner.invoke(app, ["schema", "show"])
    assert result.exit_code == 0
    assert "Graph schema" in result.stdout
    assert "TREATS" in result.stdout
    assert "version_hash" in result.stdout


def test_validate_good_query(tmp_path):
    f = tmp_path / "q.cypher"
    f.write_text("MATCH (c:Concept) WHERE c.canonical_name = $x RETURN c.canonical_name AS name", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(f)])
    assert result.exit_code == 0
    assert "passed: True" in result.stdout
    assert "LIMIT 200" in result.stdout


def test_validate_write_query(tmp_path):
    f = tmp_path / "q.cypher"
    f.write_text("MATCH (n:Concept) DELETE n", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(f)])
    assert result.exit_code == 1
    assert "WRITE_OP" in result.stdout


def test_validate_denylist(tmp_path):
    f = tmp_path / "q.cypher"
    f.write_text("MATCH (n:Concept) RETURN apoc.text.join([n.canonical_name], ',')", encoding="utf-8")
    result = runner.invoke(app, ["validate", str(f)])
    assert result.exit_code == 1
    assert "DENYLISTED" in result.stdout


def test_validate_missing_file():
    result = runner.invoke(app, ["validate", "/nonexistent/file.cypher"])
    assert result.exit_code == 1


def test_ask_without_config_fails_cleanly(monkeypatch):
    # No ARCADE_RO_USER etc. -> pipeline build fails cleanly (exit 2).
    monkeypatch.delenv("ARCADE_RO_USER", raising=False)
    result = runner.invoke(app, ["ask", "hello"])
    assert result.exit_code == 2
