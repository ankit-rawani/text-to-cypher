# NL→Cypher benchmark — gpt-5.6-luna (strict grading)

Scope: **`gpt-5.6-luna` only** (via the gateway's OpenAI-compatible `/v1` path),
against the ArcadeDB `worms` graph, `disgust-mollusca` concept subgraph (240
nodes / 271 edges; 6 node kinds, 10 edge types), offline `hashing` embedder for
grounding.

> **Grading is STRICT.** Only **EXACT** — the predicted result set equals the
> hand-written reference (extra *projected columns* tolerated, same rows) —
> counts as correct. **SUPERSET** (all reference rows **plus extras**, e.g.
> `CONTAINS` over-matching or a broader-but-defensible reading) and **WRONG**
> (missing rows / error / rows for an out-of-graph question) are reported but
> **not** counted. Open-ended discovery prompts with no single right answer are
> **INSPECT** (query + rows shown, judged by hand).
>
> An earlier multi-model table here counted `EXACT+SUPERSET` as "correct", which
> over-credited over-matching queries; it has been removed. These numbers use the
> strict grader.

Reproduce:
```bash
LLM_PROVIDER=openai LLM_ENDPOINT=https://api.servicesessentials.ibm.com/v1 LLM_MODEL=gpt-5.6-luna \
  python scripts/compare_notebook_vs_pipeline.py     # notebook method vs pipeline
LLM_PROVIDER=openai LLM_ENDPOINT=https://api.servicesessentials.ibm.com/v1 LLM_MODEL=gpt-5.6-luna \
  python scripts/researcher_gap_session.py           # researcher gap-discovery scenarios
```

## 1. Notebook method vs. text2cypher pipeline (13 questions)

| Approach | EXACT (correct) | SUPERSET (not counted) | WRONG |
|---|---:|---:|---:|
| notebook (ported to the gateway) | 12/13 | 1 | 0 |
| text2cypher pipeline | 12/13 | 0 | 1 |

Both land at 12/13 EXACT. The pipeline's one non-EXACT is Q "top-3 by
out-degree": a genuine **3-way tie at degree 6** where each side returns a valid
but different third node — a boundary-tie artifact, not a wrong answer. The
notebook's 1 SUPERSET is `CONTAINS` over-matching, now (correctly) not counted.
Net: on a strong model the two are even on accuracy; the pipeline's edge is
precision (exact `$param` matching vs fuzzy `CONTAINS`) and its guarantees
(read-only validator, bound params, full trace).

## 2. Researcher gap-discovery scenarios (23 prompts: 14 graded, 9 open)

Strict: **EXACT 10/14**, SUPERSET 2 (not counted), WRONG 2, INSPECT 9.

**What the pipeline expressed correctly** (EXACT / sensible INSPECT):
- Structural gaps: isolated concepts (`NOT (n)--()`), unexplained phenomena
  (rewrote the unsupported anonymous-start negation as an `OPTIONAL MATCH …
  count(r)=0` anti-join), idle methods, least-represented kind, most-contested
  concept, hubs, candidate missing links (`NOT (a)--(b)`), sink entities.
- Aggregation: avg confidence by status, kind participating in most relationships.
- Connectivity: Cepaea↔climate (correctly empty), Alzheimer's out-of-graph
  (correctly empty) — no hallucinated rows.

**Not counted, with honest cause (none are pipeline bugs):**
- SUPERSET Q1 "hypothesized causal claims" and Q9 "processes with no downstream
  effects": the model broadened "causal"/"downstream" to all effect-type edges
  (`causes|enables|inhibits|increases|reduces`) vs the reference's narrower
  definition. Defensible readings, but supersets → not counted.
- WRONG Q2 "what findings have been refuted?": the graph has **both** a
  `contradicts` edge type **and** a `refuted` status; the model read "refuted" as
  `contradicts`, the reference as `status='refuted'`. Genuine NL ambiguity.
- WRONG Q3 "established effects of alien species": the model kept
  `status='established'` but restricted to causal edge types (1 row); the
  reference counted every established edge incl. `is_component_of` (6). Partly an
  over-broad reference.

**One real bug this run surfaced — and fixed (on `main`):**
- Q21 "what contradictions exist in the current understanding?" first came back
  WRONG: grounding matched the meta word **"contradictions" → an unrelated node**
  ("monitoring and conservation actions") at a borderline **0.55**, which luna
  then filtered on. Fix: drop research/query meta words from grounding
  (commit `5d335e4`). After the fix, Q21 is **EXACT** (9 contradictions), taking
  gap discovery from 9 → **10/14 EXACT**.

## Caveats (eval methodology / data, not pipeline behavior)

- **Offline embedder**: the `hashing` embedder produces borderline (~0.55)
  lexical false-matches for short/meta words. A real embedding model, or a higher
  `grounding.sim_threshold`, is the general mitigation; the meta-word filter
  handles the common query words.
- **Boundary ties**: `top-N` over tied values makes exact-set matching brittle.
- **Map projections**: a `RETURN {a:…,b:…}` (one map column) is semantically
  correct but the value-grader can't unwrap it → shows as not-EXACT.
- **"refuted" / "effects"** are genuinely ambiguous against a graph that encodes
  both edge types and edge statuses.
