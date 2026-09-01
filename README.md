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

## Accuracy — a real research session

Beyond the unit tests, the pipeline was run against a **real ArcadeDB concept
graph** — the [open-nb-worms](https://en.wikipedia.org/wiki/World_Register_of_Marine_Species)
`disgust-mollusca` graph: 240 nodes across 6 kinds
(`Entity/Property/Process/Phenomenon/Method/Model`) and 271 edges across 10
relationship types (`causes/enables/inhibits/increases/reduces/requires/`
`is_component_of/measures/contradicts/analogous_to`) — driven by
`claude-sonnet-5` through an Anthropic-compatible gateway, with offline
`hashing` embeddings for grounding.

On a 15-question session simulating how a researcher would actually explore the
graph, the pipeline produced **correct, executed answers for all 14 graded
questions** (graded by result-set match against a hand-written reference query,
projection-tolerant), answered the 1 open-ended question sensibly, and returned
`empty` for the out-of-graph question. Reproduce with
`python scripts/research_session.py`.

| Stratum | Score |
|---|---|
| 1-hop | 3/3 |
| alias-phrased | 1/1 |
| multi-hop | 3/3 |
| filtered (by edge attribute) | 2/2 |
| relation | 1/1 |
| aggregation | 3/3 |
| out-of-graph (→ `empty`) | 1/1 |
| **graded total** | **14/14** |

14 of 15 resolved in a **single generation attempt** (one needed the empty-result
relaxation); latency p50 ≈ 4 s (LLM-gateway-bound).

**Representative questions → the Cypher the model generated (unprompted):**

```cypher
-- "What reduces the shell colour phenotype?"  (grounded → $param, reverse edge)
MATCH (a:Node)-[r:reduces]->(b:Node {name: $shell_colour_phenotype})
RETURN a.name AS reducer, r.confidence AS confidence, r.status AS status
--> white lip gene, hyalozonate gene

-- "Through what intermediate concept do predators causally affect morph frequencies?"  (multi-hop)
MATCH (a:Node {name: $predators})-[:causes]->(mid:Node)-[:causes]->(b:Node {name: $morph_frequencies})
RETURN DISTINCT mid.name AS intermediate
--> visual selection

-- "What inhibits or reduces morph frequencies?"  (edge-type alternation; correctly empty)
MATCH (a:Node)-[r:inhibits|reduces]->(b:Node {name: $morph_frequencies}) RETURN a.name
--> (0 rows) status: empty

-- "Which three concepts have the most outgoing relationships?"  (aggregation)
MATCH (c:Node)-[r]->() WHERE c.kind <> 'Taxon'
RETURN c.name AS name, count(r) AS outgoing ORDER BY outgoing DESC LIMIT 3
--> species description (11), alien species (7), invasive species (6)

-- "How is visual selection connected to morph frequencies?"  (variable-length path)
MATCH (a:Node {name: $visual_selection})-[r*1..2]-(b:Node {name: $morph_frequencies})
RETURN [x IN r | type(x)] AS rel_types
--> ['causes'] ; ['causes','measures']
```

Every generated query travels grounded entity values as `$params` (never
inlined), gets a `LIMIT` injected, and executes under a **read-only** credential.
The deterministic guards hold regardless of the model:

```console
$ t2c validate delete.cypher     # MATCH (n:Node) DETACH DELETE n
passed: False
  [reject] WRITE_OP: Write operation(s) not allowed in a read-only pipeline: DELETE, DETACH.

$ t2c validate apoc.cypher       # RETURN apoc.coll.sum([1,2,3])
passed: False
  [reject] DENYLISTED: Uses constructs outside the ArcadeDB openCypher subset: apoc.coll.sum.
```

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

## Integrating with your own graph

text2cypher is graph-agnostic: you describe your schema, point it at your
ArcadeDB + Qdrant + LLM, index your nodes for grounding, then ask. The
`worms` setup in this repo is a complete worked example — copy these three
files and adapt:

- `config/graph_config.worms.yaml` — the schema description
- `scripts/setup_worms_grounding.py` — one-time node indexing + example seeding
- `scripts/research_session.py` — a graded question harness

### 1. Install it as a dependency

```bash
pip install -e /path/to/text-to-cypher      # or add the path/VCS ref to your requirements
```

### 2. Describe your schema (a GraphConfig YAML)

Mirror **what is actually queryable in your ArcadeDB** (if your graph is a
projection, it may expose fewer properties than the source of truth — list only
the ones present). A single vertex label with a `kind` discriminator is the
common shape:

```yaml
# my_graph.yaml
database: my_db
node_labels: [Node]              # your ArcadeDB vertex type(s)
node_type_property: kind         # property that discriminates node kind
node_type_values: [Gene, Disease, Drug, ...]
node_properties:                 # name -> type; ONLY props present in ArcadeDB
  id: string
  name: string                   # the property entities are matched on
  kind: string
relationships:                   # one entry per edge type
  - {type: causes,  description: A causes B}
  - {type: treats,  start: [Drug], end: [Disease], description: therapeutic}
  - {type: parent_of, directed: true}
edge_properties: {confidence: float, status: string}
status_property: status
status_values: [hypothesized, established]
```

The `start`/`end` node kinds (when set) power the direction autofix; `name` is
what grounding binds to `$params` and what the model filters on.

### 3. Configure `.env`

```bash
cp .env.example .env
# LLM: LLM_PROVIDER=anthropic|openai, LLM_ENDPOINT, LLM_MODEL, LLM_API_KEY
# Graph: ARCADE_URL, ARCADE_DB, ARCADE_RO_USER, ARCADE_RO_PASSWORD
# Point at your schema + a dedicated grounding collection:
GRAPH_CONFIG_PATH=my_graph.yaml
QDRANT_URL=http://localhost:6333
QDRANT_NODE_COLLECTION=my_nodes          # avoid colliding with existing collections
EMBED_PROVIDER=hashing                    # offline; or `openai` for a real model
```

### 4. Use a read-only database credential

Create a DB user with no write grants and put it in `ARCADE_RO_USER` /
`ARCADE_RO_PASSWORD`. The pipeline refuses to start without one
(`require_readonly_user`) — read-only enforced at the connection, on top of the
validator's write-block.

### 5. Index your nodes for grounding (one time)

Grounding resolves question mentions to your nodes via vector search, so index
them once (re-run when the graph changes):

```python
from text2cypher.config import load_config
from text2cypher.clients.arcadedb import ArcadeDBClient
from text2cypher.clients.qdrant import QdrantClient
from text2cypher.embeddings import build_embedder
from text2cypher.ingest import index_nodes

cfg = load_config()
emb = build_embedder(cfg.embeddings)
rows = ArcadeDBClient.from_config(cfg.arcadedb).query(
    "MATCH (n:Node) RETURN n.id AS node_id, n.name AS canonical_name, n.kind AS node_type",
    limit=100000,
)
index_nodes(QdrantClient.from_config(cfg.qdrant), emb, cfg.qdrant.node_collection, rows)
```

Optionally seed a handful of hand-written question→Cypher examples for dynamic
few-shot: `t2c examples import pairs.jsonl` (or `ExampleStore.add(...)`).

### 6. Ask — CLI or programmatic

```bash
t2c schema show                      # sanity-check the rendered schema
t2c ask "which genes cause X?"       # full pipeline; add --trace for JSON
t2c eval --gold gold.jsonl           # per-stratum execution accuracy (CI-gate ready)
```

```python
from text2cypher import QueryRequest
from text2cypher.pipeline import build_pipeline
from text2cypher.config import load_config

pipeline = build_pipeline(load_config())     # build once, reuse across requests
resp = pipeline.answer(QueryRequest(question="which genes cause X?"))

if resp.status == "ok":
    for row in resp.rows:
        ...
elif resp.status == "empty":
    ...      # nothing matched — resp.message suggests a vector-retrieval fallback
else:        # "failed" — blocked write / denylist / exhausted repairs
    ...
# resp also carries: final_cypher, params, grounding[], attempts[], validations[],
# executions[], trace_id — the complete, inspectable trace for every request.
```

`build_pipeline` returns a reusable, synchronous object (schema is cached with a
TTL); construct it once per process and call `answer()` per request.

### Switching graphs dynamically

The schema is not hardcoded — there are three levels of control:

1. **By env / file** — `GRAPH_CONFIG_PATH` selects the default schema.
2. **Per pipeline (in-memory or by path)** — pass `graph_config` to
   `build_pipeline` to target any schema without editing env/files:
   ```python
   from text2cypher.graph_config import GraphConfig
   p1 = build_pipeline(load_config(), graph_config="config/other.yaml")
   p2 = build_pipeline(load_config(), graph_config=GraphConfig(node_labels=["Gene"], ...))
   ```
3. **Per request (several graphs in one process)** — route with
   `MultiGraphPipeline`:
   ```python
   from text2cypher.pipeline import build_pipeline, MultiGraphPipeline

   mgp = MultiGraphPipeline({
       "concepts": build_pipeline(load_config()),                        # default schema
       "taxonomy": build_pipeline(tax_cfg, graph_config=tax_graph_cfg),  # different schema
   }, default="concepts")

   mgp.answer(QueryRequest(question="how many taxa?"), graph="taxonomy")
   ```
   Each named pipeline can target a different GraphConfig, database, and grounding
   collection — and two schema *views* over the **same** database work too (e.g. a
   concept-graph view and a taxonomy view of one ArcadeDB).

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
config/
  default.yaml         # defaults + ${ENV} interpolation
  graph_config.yaml    # example GraphConfig; graph_config.worms.yaml = worked example
  arcade_denylist.yaml # dialect denylist (grows from prod errors)
scripts/
  setup_worms_grounding.py  # index nodes + seed examples (worked example)
  research_session.py       # graded question harness (worked example)
```
