# NL→Cypher: notebook method vs. text2cypher pipeline

Head-to-head on the **same LLM gateway, same graph, same questions** — the only
variable is the *approach*.

- **Approach A — "notebook"**: a faithful port of
  `open-nb-worms/notebooks/nl_to_cypher_experiments.ipynb` — a schema-grounded
  system prompt, model-driven `toLower(name) CONTAINS` matching, a regex
  read-only check, and one self-repair pass. Its OpenRouter call is replaced by
  the same gateway model used by the pipeline (fair, model-controlled).
- **Approach B — "pipeline"**: the text2cypher pipeline — vector grounding →
  bound `$params`, an AST validator (write-block, denylist, schema check,
  unbound-param check, direction & LIMIT autofix), a capped repair loop, and the
  empty-result branch.

**Setup**: gateway `https://api.servicesessentials.ibm.com`; ArcadeDB `worms`
graph, `disgust-mollusca` concept subgraph (240 nodes / 271 edges, 6 node kinds,
10 edge types); offline `hashing` embedder for grounding; 13 graded questions
across strata (1-hop, alias-phrased, multi-hop, filtered, relation, aggregation,
out-of-graph). Reproduce: `python scripts/compare_notebook_vs_pipeline.py`
(override the model with `LLM_PROVIDER=openai LLM_MODEL=<id>`).

**Grading** is value-based and projection-tolerant against a hand-written
reference query: `EXACT` = matches the reference set; `SUPERSET` = contains all
reference rows plus extras (e.g. `CONTAINS` over-matching); `WRONG` = missing
rows / error / rows for an out-of-graph question. "Correct" = EXACT + SUPERSET.

## Results (correct / 13)

| Model | Tier | Notebook | Pipeline |
|---|---|---:|---:|
| `claude-sonnet-5` | frontier | 13 | **13** |
| `gpt-5.6-luna` | frontier (reasoning) | 12\* | 12\* |
| `gemini-3.7-flash` | small/fast | 12 | **13** |
| `gemma-4-26b-a4b-it` | small (MoE) | 7 | **13** |
| `claude-haiku-4-5` | small | 11 | **13** |
| `meta-llama/llama-4-maverick-17b` | small (MoE) | 7 | **11** |
| `ibm/granite-4-h-small` | very small | 5 | **7** |

\* On `gpt-5.6-luna` the single non-match for both is Q10 ("top-3 by out-degree"):
a genuine 3-way tie at degree 6, where each side returns a valid — but
different — third node. A grading artifact, not a miss.

## Findings

1. **The pipeline meets or beats the notebook method on every model**, and the
   margin widens as the model weakens — evidence for the design thesis that
   accuracy should come from the pipeline, not just the model.
2. **On strong models it's a near-tie on accuracy**; the pipeline's edge is
   *precision* — exact `$param` matching returns exactly the right set, whereas
   the notebook's `CONTAINS` occasionally over-matches (SUPERSET).
3. **On weaker models the pipeline's structure prevents failures** the notebook
   hits: `CONTAINS` matching the wrong surface form (e.g. "morph frequency" vs
   "morph frequenc*ies*"), and putting an edge attribute (`status`) on a node.
   Vector grounding + AST schema-validation catch these.
4. **Two pipeline bugs were found via the weak-model runs and fixed** (on `main`):
   - grounding over-grounded generic words ("relationships", "type") into
     spurious `$params` that weak models then filtered on — now dropped;
   - a query referencing an unbound `$param` failed opaquely at the DB — the
     validator now rejects it with a clear, repairable message.
   Verified on `granite-4-h-small`: the two questions it previously got wrong for
   these reasons now return correct results on the first attempt.

## Caveats (eval methodology, not pipeline behavior)

- **Boundary ties**: `top-N` over tied values makes exact-set matching brittle
  (Q10). Both approaches return valid-but-different members of the tie.
- **Map projections**: a model that returns `RETURN {a:…, b:…}` (one map column)
  instead of flat columns is semantically correct but the value-grader can't
  unwrap it — counts as WRONG on the weakest models.
- **Offline embedder**: the `hashing` embedder dilutes short aliases against long
  node text; a real embedding model grounds those better. (The `worms` concept
  nodes have short names, so grounding is strong there.)

## Latency (p50, gateway-bound)

Frontier: `sonnet-5` ~4 s, `gpt-5.6-luna` ~5 s (reasoning). Small: `haiku-4-5`
~2 s, `gemini-3.7-flash` ~2.5 s, `granite`/`llama` ~1–4 s; `gemma-4-26b` is a
reasoning model at ~25–30 s.
