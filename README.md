# Text2Cypher Pipeline

Natural-language question → grounded, validated, guarded Cypher → executed
against an ArcadeDB concept graph → structured rows + full trace.

> **Design stance** — accuracy comes from the *pipeline*, not the model.
> Zero-shot frontier LLMs sit near ~60% execution accuracy on public
> text-to-Cypher benchmarks. Every stage below exists to close that gap
> deterministically where possible, with the LLM only where it must be.

The pipeline is **read-only** by construction: write operations are blocked at
the validator *and* enforced at the database connection (read-only credential).
Grounded entity values always travel as bound `$params` — never string
interpolation of user input. Every response carries the complete trace.

## Architecture — 7 stages

```
question
  → [1] SchemaProvider   introspect ArcadeDB + GraphConfig → enhanced schema (cached, TTL)
  → [2] EntityGrounder   mentions → embed → Qdrant node search → $param bindings
  → [3] ExampleStore     top-k similar Q→Cypher pairs (threshold-gated)
  → [4] Generator        single LLM call → {cypher, params_used, reasoning, confidence}
  → [5] Validator        ordered deterministic checks + autofix (§ Validator)
  → [6] Executor         guarded ArcadeDB call (read-only creds, timeout, LIMIT, row cap)
  → [7] RepairLoop       reflection retry on failure (capped); empty-result branch
  → PipelineResponse
```

### Validator — ordered checks (stop at first hard reject)

| Order | Check | Action |
|---|---|---|
| 1 | Grammar / structural parse | reject → repair |
| 2 | Dialect lint (denylist: `apoc.*`, `db.*`, …) | reject → repair |
| 3 | Write-op block (`CREATE/MERGE/DELETE/SET/REMOVE/DROP/LOAD CSV`, `FOREACH`) | **reject, no repair** |
| 4 | Schema consistency (labels / rel types / properties exist) | reject → repair (with fuzzy suggestions) |
| 5 | Relationship direction | **autofix** (`DIRECTION_FIXED`) |
| 6 | Param discipline (grounded values must be `$params`) | reject → repair |
| 7 | `LIMIT` enforcement | **autofix** (`LIMIT_INJECTED` / `LIMIT_CLAMPED`) |

Write-op detection never routes to the repair loop — repairing a write into a
sneakier write is worse. It is a hard stop.

### Empty-result branch

`status: empty` when a query executes fine but returns 0 rows: check grounding
(ambiguous / low-sim entities are surfaced to the caller), allow **one**
relaxation attempt (config flag), then return `empty` with a suggestion to fall
back to vector retrieval. Empty is a data answer, never loop-retried.

## Install

```bash
pip install -e ".[dev]"
```

Core deps: `pydantic>=2`, `httpx`, `PyYAML`, `typer`. No live services are
required to run the test suite — every client has an in-memory fake, and an
offline deterministic embedder (`provider: hashing`) lets grounding and the
example store work without a network.

## CLI

```
t2c ask "which concepts link X to Y?"     # full pipeline, prints rows + trace summary
t2c ask ... --trace                        # full trace JSON
t2c validate query.cypher                  # validator only
t2c schema show | refresh                  # inspect / refresh cached schema
t2c examples add | list | import pairs.jsonl
t2c eval --gold gold.jsonl                 # execution-accuracy eval, per stratum
```

Point at real services with environment variables. Copy the sample env file and
fill it in — the CLI/pipeline auto-loads `.env` from the repo root (real
environment variables still win over the file):

```bash
cp .env.example .env    # then edit, then:
t2c ask "which drugs treat type 2 diabetes?"
```

### LLM provider

The Generator supports two providers, selected by `LLM_PROVIDER`:

- **`anthropic`** — native Claude Messages API. Set a **custom base URL, model,
  and API key**:
  ```bash
  LLM_PROVIDER=anthropic
  LLM_ENDPOINT=https://api.anthropic.com     # or any Anthropic-compatible proxy
  LLM_MODEL=claude-sonnet-5                  # any Claude model id
  LLM_API_KEY=sk-ant-...
  ```
  Current Claude models reject `temperature` (HTTP 400), so it is not sent;
  determinism comes from the pipeline's validation + repair, not sampling.
- **`openai`** — any OpenAI-compatible `/chat/completions` endpoint
  (`LLM_ENDPOINT=https://api.openai.com/v1`, `LLM_MODEL=gpt-4o`, `LLM_API_KEY=…`).

Other services: `ARCADE_URL`, `ARCADE_RO_USER`, `ARCADE_RO_PASSWORD`,
`QDRANT_URL`, `EMBED_PROVIDER`, … — see `config/default.yaml` and `.env.example`.

## Configuration

`config/default.yaml` is the single source of defaults; a user file and
`${ENV}` interpolation layer on top. The dialect denylist
(`config/arcade_denylist.yaml`) is a separate config file that grows from
production errors.

## Eval

`t2c eval --gold gold.jsonl` reports **execution accuracy** (result-set match,
order-insensitive, rounded numerics — not string match) per stratum:
`1-hop | multi-hop | aggregation | alias-phrased | out-of-graph`. Secondary
metrics: valid-query rate, attempts-to-success histogram, latency p50/p95.
CI fails on a regression greater than 5 points (`--baseline`, `--fail-under`).

## Programmatic use

```python
from text2cypher import QueryRequest
from text2cypher.pipeline import build_pipeline
from text2cypher.config import load_config

pipeline = build_pipeline(load_config())          # real clients from config/env
resp = pipeline.answer(QueryRequest(question="which concepts link X to Y?"))
print(resp.status, resp.final_cypher, resp.rows)
```

For tests / offline demos, `text2cypher.testing.build_fake_pipeline(...)` wires
the pipeline against in-memory fakes with a scripted LLM.

## Layout

```
src/text2cypher/
  contracts.py         # pydantic contracts (§2) — the frozen interface
  config.py            # YAML + ${ENV} config loader (§6)
  cypher/              # tokenizer + analyzer (parse tree + transforms)
  clients/             # ArcadeDB / Qdrant / LLM / Mongo (httpx) + in-memory fakes
  embeddings/          # OpenAI-compatible + offline hashing embedder
  pipeline/            # the 7 stages + orchestrator + prompts
  prewritten/          # parameterized tools (§5) — bypass the LLM
  eval/                # execution-accuracy harness + result comparison (§8)
  cli.py               # `t2c` command surface (§7)
```
