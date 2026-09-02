"""Head-to-head: the open-nb-worms notebook's inline NL->Cypher method vs. the
text2cypher pipeline, BOTH driven by the same Anthropic gateway (claude-sonnet-5)
and the same `worms` ArcadeDB, so the comparison isolates the *approach*.

- Approach A ("notebook"): faithful port of notebooks/nl_to_cypher_experiments.ipynb
  — schema-grounded system prompt, model-driven `CONTAINS` matching, regex
  read-only check, one self-repair pass. Its OpenRouter call is replaced by the
  Anthropic gateway (via text2cypher's AnthropicClient).
- Approach B ("pipeline"): text2cypher (vector grounding -> $params, AST
  validator, LIMIT autofix, repair loop, empty-branch).

Grading is STRICT: only EXACT (same row set as the reference, tolerating extra
projected columns) counts as correct. SUPERSET (all reference rows plus extras —
e.g. CONTAINS over-matching) and WRONG (missing rows / error / rows for
out-of-graph) are reported but NOT counted as correct.

    python scripts/compare_notebook_vs_pipeline.py
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter

from text2cypher.clients import build_llm
from text2cypher.clients.arcadedb import ArcadeDBClient
from text2cypher.config import load_config
from text2cypher.contracts import QueryRequest
from text2cypher.eval.compare import canonicalize
from text2cypher.pipeline import build_pipeline

GRAPH = "disgust-mollusca"

# ---- graded questions (question, reference cypher, params); "__EMPTY__" = expect empty ----
GRADED = [
    ("What does climatic changes cause?",
     "MATCH (a:Node)-[:causes]->(b:Node) WHERE a.name=$n RETURN b.name AS x", {"n": "climatic changes"}),
    ("What is a component of Dreissena?",
     "MATCH (a:Node)-[:is_component_of]->(b:Node) WHERE b.name=$n RETURN a.name AS x", {"n": "Dreissena"}),
    ("What reduces the shell colour phenotype?",
     "MATCH (a:Node)-[:reduces]->(b:Node) WHERE b.name=$n RETURN a.name AS x", {"n": "shell colour phenotype"}),
    ("Which techniques are used to measure climatic changes?",
     "MATCH (m:Node)-[:measures]->(b:Node) WHERE b.name=$n RETURN m.name AS x", {"n": "climatic changes"}),
    ("Through what intermediate concept do predators causally affect morph frequencies?",
     "MATCH (a:Node)-[:causes]->(b:Node)-[:causes]->(c:Node) WHERE a.name=$x AND c.name=$y RETURN DISTINCT b.name AS n",
     {"x": "predators", "y": "morph frequencies"}),
    ("List the established causal relationships.",
     "MATCH (a:Node)-[r:causes]->(b:Node) WHERE r.status='established' RETURN a.name AS a, b.name AS b", {}),
    ("Which pairs of concepts contradict each other?",
     "MATCH (a:Node)-[:contradicts]->(b:Node) RETURN a.name AS a, b.name AS b", {}),
    ("Count the relationships of each type in the concept graph.",
     "MATCH (a:Node)-[r]->(b:Node) WHERE a.graph='disgust-mollusca' RETURN type(r) AS rel, count(*) AS n", {}),
    ("How many concepts are there of each kind?",
     "MATCH (n:Node) WHERE n.graph='disgust-mollusca' RETURN n.kind AS kind, count(*) AS n", {}),
    ("Which three concepts have the most outgoing relationships?",
     "MATCH (a:Node)-[r]->() WHERE a.graph='disgust-mollusca' RETURN a.name AS name, count(r) AS d ORDER BY d DESC LIMIT 3", {}),
    ("What enables the shell colour locus?",
     "MATCH (a:Node)-[:enables]->(b:Node) WHERE b.name=$n RETURN a.name AS x", {"n": "shell colour locus"}),
    ("List properties that increase with temperature.",
     "MATCH (t:Node)-[:increases]->(p:Node) WHERE t.name=$n AND p.kind='Property' RETURN p.name AS x", {"n": "temperature"}),
    ("What is the GDP of Japan?", "__EMPTY__", {}),
]

# ---------------------------------------------------------------------------
# Approach A: faithful port of the notebook method (LLM -> Anthropic gateway)
# ---------------------------------------------------------------------------

_WRITE = re.compile(r"\b(create|merge|delete|set|remove|drop)\b", re.I)
_SYSTEM = """You translate a question into ONE read-only openCypher query for an ArcadeDB knowledge graph.

GRAPH SCHEMA
- Every node has label Node: (:Node {{id, kind, name, graph, phylum}}).
  `kind` is the node type (Taxon, Entity, Process, Property, Method, Model,
  Phenomenon). `graph` names a concept dataset; `phylum` names a taxonomy.
- Relationships are typed. The ONLY valid relationship types are: {rels}.
  Every relationship also carries {{id, kind, confidence, status, graph}}.

RULES
- Exactly ONE query. Read-only: use only MATCH / WHERE / WITH / RETURN /
  ORDER BY / LIMIT / UNWIND. NEVER CREATE / MERGE / SET / DELETE / REMOVE.
- Always RETURN each node's `id` plus its `name`.
- Match names case-insensitively by containment:
  WHERE toLower(n.name) CONTAINS toLower('term').
- Use relationship types verbatim, e.g. (a)-[:causes]->(b).
- Always add a LIMIT (<= 50).{scope}

Return ONLY a JSON object of the form: {{"cypher": "<the query>"}}"""


def _strip_fences(s: str) -> str:
    s = s.strip()
    return s.removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _parse_cypher(raw: str) -> str:
    raw = _strip_fences(raw)
    try:
        return json.loads(raw)["cypher"].strip()
    except Exception:
        m = re.search(r'\{.*"cypher".*\}', raw, re.S)
        if m:
            return json.loads(m.group(0))["cypher"].strip()
        raise


class NotebookMethod:
    def __init__(self, llm, arcade, rel_types):
        self._llm = llm
        self._arcade = arcade
        self._system = _SYSTEM.format(
            rels=", ".join(rel_types),
            scope=f"\n- SCOPE: restrict to the concept graph '{GRAPH}' — add WHERE n.graph = '{GRAPH}'.",
        )

    def _gen(self, user):
        raw = self._llm.complete(
            [{"role": "system", "content": self._system}, {"role": "user", "content": user}],
            max_tokens=1024,
        )
        return _parse_cypher(raw)

    def answer(self, question):
        try:
            cy = self._gen(question)
        except Exception as e:
            return {"cypher": None, "rows": None, "error": f"gen/parse: {e}"}
        try:
            if _WRITE.search(cy):
                raise ValueError("write clause blocked")
            return {"cypher": cy, "rows": self._arcade.query(cy, limit=200)}
        except Exception as e:
            repair = (f"{question}\n\nYour previous query was:\n{cy}\n\nIt failed with: {e}\n"
                      'Return a corrected JSON object {"cypher": "<read-only openCypher>"}.')
            try:
                fixed = self._gen(repair)
                if _WRITE.search(fixed):
                    raise ValueError("write clause blocked")
                return {"cypher": fixed, "rows": self._arcade.query(fixed, limit=200), "repaired": True}
            except Exception as e2:
                return {"cypher": cy, "rows": None, "error": str(e2)}


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def _vals(row):
    items = [v for k, v in row.items() if not str(k).startswith("@")] if isinstance(row, dict) else [row]
    return Counter(str(canonicalize(v)) for v in items)


def verdict(pred_rows, gold_cypher, gold_params, arcade, timeout):
    if gold_cypher == "__EMPTY__":
        return "EXACT" if not pred_rows else "WRONG", 0
    gold = arcade.query(gold_cypher, gold_params, timeout_s=timeout, limit=200)
    if pred_rows is None:
        return "WRONG", len(gold)
    if not gold:
        return ("EXACT" if not pred_rows else "WRONG"), 0
    pred = [_vals(r) for r in pred_rows]
    used = [False] * len(pred)
    for g in [_vals(r) for r in gold]:
        hit = next((i for i, p in enumerate(pred) if not used[i] and g <= p), None)
        if hit is None:
            return "WRONG", len(gold)
        used[hit] = True
    return ("EXACT" if len(pred) == len(gold) else "SUPERSET"), len(gold)


def main():
    cfg = load_config()
    arcade = ArcadeDBClient.from_config(cfg.arcadedb)
    timeout = cfg.arcadedb.timeout_s
    llm = build_llm(cfg.llm)
    pipeline = build_pipeline(cfg)
    rel_types = [r["t"] for r in arcade.query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t ORDER BY t")]
    notebook = NotebookMethod(llm, arcade, rel_types)

    print(f"model: {cfg.llm.model} @ {cfg.llm.endpoint}   |   graph: worms/{GRAPH}\n")
    tallies = {"notebook": Counter(), "pipeline": Counter()}
    lat = {"notebook": [], "pipeline": []}

    for i, (q, gold_cy, gp) in enumerate(GRADED, 1):
        print("=" * 80)
        print(f"[{i}/{len(GRADED)}] {q}")

        t0 = time.perf_counter()
        a = notebook.answer(q)
        lat["notebook"].append((time.perf_counter() - t0) * 1000)
        va, goldn = verdict(a.get("rows"), gold_cy, gp, arcade, timeout)
        tallies["notebook"][va] += 1
        arow = a.get("rows")
        print(f"  A notebook : {va:<8} rows={len(arow) if arow is not None else 'ERR'}"
              f"{'  (repaired)' if a.get('repaired') else ''}{'  ' + a['error'] if a.get('error') else ''}")
        print(f"             cypher: {a.get('cypher')}")

        t0 = time.perf_counter()
        b = pipeline.answer(QueryRequest(question=q, max_attempts=3, row_cap=200, timeout_s=timeout))
        lat["pipeline"].append((time.perf_counter() - t0) * 1000)
        brows = b.rows if b.status in ("ok", "empty") else None
        vb, _ = verdict(brows, gold_cy, gp, arcade, timeout)
        tallies["pipeline"][vb] += 1
        print(f"  B pipeline : {vb:<8} rows={len(b.rows or [])} status={b.status} attempts={len(b.attempts)}")
        print(f"             cypher: {b.final_cypher}")

    print("\n" + "#" * 80)
    n = len(GRADED)
    # STRICT grading: only EXACT counts as correct. SUPERSET (over-matching,
    # e.g. CONTAINS returning extra rows) is reported but NOT counted.
    for name in ("notebook", "pipeline"):
        t = tallies[name]
        lats = sorted(lat[name])
        print(f"{name:<9}  correct(EXACT)={t['EXACT']}/{n}  | SUPERSET(over-match, not counted)={t['SUPERSET']}"
              f"  WRONG={t['WRONG']}  | latency p50={lats[len(lats)//2]:.0f}ms")


if __name__ == "__main__":
    main()
