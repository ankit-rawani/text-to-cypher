"""``t2c`` command-line surface (spec section 7).

    t2c ask "which concepts link X to Y?"     full pipeline, prints rows + trace summary
    t2c ask ... --trace                        full trace JSON
    t2c validate query.cypher                  validator only (offline)
    t2c schema show | refresh                  inspect / refresh cached schema
    t2c examples add | list | import pairs.jsonl
    t2c eval --gold gold.jsonl                 execution-accuracy eval, per stratum
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

from .config import AppConfig, load_config
from .contracts import QueryRequest
from .graph_config import load_graph_config

app = typer.Typer(add_completion=False, help="Text2Cypher pipeline CLI.")
schema_app = typer.Typer(help="Inspect / refresh the schema context.")
examples_app = typer.Typer(help="Manage the example store.")
app.add_typer(schema_app, name="schema")
app.add_typer(examples_app, name="examples")


def _cfg(config_path: Optional[Path]) -> AppConfig:
    return load_config(config_path)


def _err(msg: str, code: int = 1) -> None:
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


# --------------------------------------------------------------------------
# ask
# --------------------------------------------------------------------------


@app.command()
def ask(
    question: str = typer.Argument(..., help="The natural-language question."),
    trace: bool = typer.Option(False, "--trace", help="Print the full trace JSON."),
    as_json: bool = typer.Option(False, "--json", help="Print the whole response as JSON."),
    max_attempts: int = typer.Option(3, help="Max generation attempts."),
    row_cap: int = typer.Option(200, help="Row cap."),
    timeout_s: float = typer.Option(15.0, help="Query timeout (seconds)."),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Config file."),
) -> None:
    from .pipeline import build_pipeline

    cfg = _cfg(config_path)
    try:
        pipeline = build_pipeline(cfg)
    except Exception as exc:  # ConfigurationError etc.
        _err(f"Cannot build pipeline: {exc}\nSet ARCADE_URL / ARCADE_RO_USER / LLM_ENDPOINT / QDRANT_URL.", 2)

    resp = pipeline.answer(
        QueryRequest(question=question, max_attempts=max_attempts, row_cap=row_cap, timeout_s=timeout_s)
    )

    if as_json:
        typer.echo(resp.model_dump_json(indent=2))
        raise typer.Exit(0 if resp.status in ("ok", "empty") else 1)

    typer.secho(f"status: {resp.status}   trace_id={resp.trace_id}", bold=True)
    typer.echo(f"message: {resp.message}")
    if resp.final_cypher:
        typer.echo(f"\ncypher:\n  {resp.final_cypher}")
        typer.echo(f"params: {json.dumps(resp.params or {})}")
    if resp.rows:
        typer.echo(f"\nrows ({len(resp.rows)}):")
        for row in resp.rows[:50]:
            typer.echo(f"  {json.dumps(row, default=str)}")
    if resp.grounding:
        typer.echo("\ngrounding:")
        for g in resp.grounding:
            typer.echo(f"  {g.mention!r} -> {g.canonical_name} (${g.param_name}, sim={g.similarity:.2f})")
    typer.echo(
        f"\ntrace: {len(resp.attempts)} attempt(s), "
        f"{len(resp.validations)} validation(s), {len(resp.executions)} execution(s)"
    )
    if trace:
        typer.echo("\nfull trace:")
        typer.echo(resp.model_dump_json(indent=2))
    raise typer.Exit(0 if resp.status in ("ok", "empty") else 1)


# --------------------------------------------------------------------------
# validate (offline)
# --------------------------------------------------------------------------


@app.command()
def validate(
    query_file: Path = typer.Argument(..., help="Path to a .cypher file to validate."),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Config file."),
) -> None:
    from .pipeline.schema_provider import SchemaProvider
    from .pipeline.validator import Validator, load_denylist

    cfg = _cfg(config_path)
    if not query_file.exists():
        _err(f"No such file: {query_file}")
    cypher = query_file.read_text(encoding="utf-8")

    graph_config = load_graph_config(cfg.resolve_path(cfg.graph_config_path))
    schema = SchemaProvider(graph_config).get()
    validator = Validator(load_denylist(cfg.resolve_path(cfg.dialect.denylist_path)), row_cap=cfg.pipeline.row_cap)
    report = validator.validate(cypher, {}, schema)

    color = typer.colors.GREEN if report.passed else typer.colors.RED
    typer.secho(f"passed: {report.passed}", fg=color, bold=True)
    for issue in report.issues:
        typer.echo(f"  [{issue.severity}] {issue.code}: {issue.detail}")
    if report.final_cypher.strip() != cypher.strip():
        typer.echo(f"\nfinal (post-autofix):\n  {report.final_cypher}")
    raise typer.Exit(0 if report.passed else 1)


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


@schema_app.command("show")
def schema_show(config_path: Optional[Path] = typer.Option(None, "--config")) -> None:
    from .pipeline.schema_provider import SchemaProvider

    cfg = _cfg(config_path)
    graph_config = load_graph_config(cfg.resolve_path(cfg.graph_config_path))
    ctx = SchemaProvider(graph_config).get()
    typer.echo(ctx.rendered)
    typer.echo(f"\nversion_hash: {ctx.version_hash}")
    typer.echo(f"labels: {', '.join(ctx.labels)}")
    typer.echo(f"rel_types: {', '.join(ctx.rel_types)}")


@schema_app.command("refresh")
def schema_refresh(config_path: Optional[Path] = typer.Option(None, "--config")) -> None:
    from .pipeline.schema_provider import SchemaProvider

    cfg = _cfg(config_path)
    graph_config = load_graph_config(cfg.resolve_path(cfg.graph_config_path))
    client = None
    try:
        from .clients.arcadedb import ArcadeDBClient

        client = ArcadeDBClient.from_config(cfg.arcadedb)
    except Exception as exc:
        typer.secho(f"(no live introspection: {exc})", fg=typer.colors.YELLOW, err=True)
    ctx = SchemaProvider(graph_config, client).refresh()
    typer.echo(ctx.rendered)
    typer.echo(f"\nversion_hash: {ctx.version_hash}")


# --------------------------------------------------------------------------
# examples
# --------------------------------------------------------------------------


def _example_store(cfg: AppConfig):
    from .clients.qdrant import QdrantClient
    from .embeddings import build_embedder
    from .pipeline.example_store import ExampleStore

    embedder = build_embedder(cfg.embeddings)
    store = QdrantClient.from_config(cfg.qdrant)
    return ExampleStore(
        embedder,
        store,
        cfg.qdrant.example_collection,
        top_k=cfg.examples.top_k,
        sim_threshold=cfg.examples.sim_threshold,
    )


@examples_app.command("import")
def examples_import(
    pairs_file: Path = typer.Argument(..., help="JSONL of {question, cypher, tags?}."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    cfg = _cfg(config_path)
    try:
        store = _example_store(cfg)
        count = store.import_jsonl(pairs_file)
    except Exception as exc:
        _err(f"Import failed: {exc}", 2)
    typer.secho(f"Imported {count} example pair(s).", fg=typer.colors.GREEN)


@examples_app.command("add")
def examples_add(
    question: str = typer.Option(..., "--question", "-q"),
    cypher: str = typer.Option(..., "--cypher", "-c"),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    cfg = _cfg(config_path)
    try:
        store = _example_store(cfg)
        store.add_pair(question, cypher)
    except Exception as exc:
        _err(f"Add failed: {exc}", 2)
    typer.secho("Added 1 example pair.", fg=typer.colors.GREEN)


@examples_app.command("list")
def examples_list(config_path: Optional[Path] = typer.Option(None, "--config")) -> None:
    cfg = _cfg(config_path)
    try:
        store = _example_store(cfg)
        pairs = store.list()
    except Exception as exc:
        _err(f"List failed: {exc}", 2)
    for i, p in enumerate(pairs, 1):
        typer.echo(f"{i}. Q: {p.question}\n   Cypher: {p.cypher}")
    typer.echo(f"\n{len(pairs)} example(s).")


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------


@app.command()
def eval(
    gold: Path = typer.Option(..., "--gold", help="Gold JSONL (question, cypher, stratum)."),
    baseline: Optional[Path] = typer.Option(None, "--baseline", help="Baseline report JSON to compare."),
    fail_under: Optional[float] = typer.Option(None, "--fail-under", help="Fail if overall accuracy < this fraction."),
    threshold_pts: float = typer.Option(5.0, "--threshold-pts", help="Max allowed regression vs baseline (points)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write the report JSON here."),
    config_path: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    from .clients.arcadedb import ArcadeDBClient
    from .eval import EvalHarness, compare_reports, load_gold
    from .eval.harness import EvalReport
    from .pipeline import build_pipeline

    cfg = _cfg(config_path)
    try:
        pipeline = build_pipeline(cfg)
        gold_client = ArcadeDBClient.from_config(cfg.arcadedb)
    except Exception as exc:
        _err(f"Cannot build pipeline for eval: {exc}", 2)

    items = load_gold(gold)
    harness = EvalHarness(
        pipeline, gold_client, row_cap=cfg.pipeline.row_cap, timeout_s=cfg.arcadedb.timeout_s
    )
    report = harness.run(items)
    typer.echo(report.summary())

    if out:
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"\nWrote {out}")

    exit_code = 0
    if fail_under is not None and report.overall_accuracy < fail_under:
        typer.secho(
            f"\nFAIL: overall accuracy {report.overall_accuracy:.1%} < {fail_under:.1%}",
            fg=typer.colors.RED,
        )
        exit_code = 1
    if baseline is not None and baseline.exists():
        base = EvalReport.model_validate_json(baseline.read_text(encoding="utf-8"))
        ok, regressions = compare_reports(report, base, threshold_pts=threshold_pts)
        if not ok:
            typer.secho("\nRegressions vs baseline:", fg=typer.colors.RED)
            for r in regressions:
                typer.echo(f"  {r}")
            exit_code = 1
        else:
            typer.secho("\nNo regression beyond threshold vs baseline.", fg=typer.colors.GREEN)
    raise typer.Exit(exit_code)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
