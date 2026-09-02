"""Researcher-scenario test: use the pipeline the way someone mining the concept
graph for research gaps would — searching evidence, finding structural holes,
spotting under-developed areas and open questions.

Runs each prompt through the pipeline (whatever model .env selects), and either
grades it against a hand-verified reference query (EXACT/SUPERSET/WRONG) or, for
open-ended discovery prompts, prints the query + rows to INSPECT.

    python scripts/researcher_gap_session.py
"""

from __future__ import annotations

import time
from collections import Counter

from text2cypher.clients.arcadedb import ArcadeDBClient
from text2cypher.config import load_config
from text2cypher.contracts import QueryRequest
from text2cypher.eval.compare import canonicalize
from text2cypher.pipeline import build_pipeline

# (theme, prompt, reference cypher | None=inspect, params)
PROMPTS = [
    # --- evidence / literature search ---
    ("evidence", "Which causal claims are only hypothesized, not yet established?",
     "MATCH (a:Node)-[r:causes]->(b:Node) WHERE r.graph='disgust-mollusca' AND r.status='hypothesized' RETURN a.name AS cause, b.name AS effect", {}),
    ("evidence", "What findings have been refuted?",
     "MATCH (a:Node)-[r]->(b:Node) WHERE r.graph='disgust-mollusca' AND r.status='refuted' RETURN a.name AS a, b.name AS b", {}),
    ("evidence", "What are the established effects of alien species?",
     "MATCH (a:Node)-[r]->(b:Node) WHERE a.name='alien species' AND r.status='established' RETURN b.name AS effect", {}),
    ("evidence", "What methods have been used to study morph frequencies?", None, {}),
    ("evidence", "Trace the causal chain from predators to shell colour phenotype.", None, {}),

    # --- structural gaps ---
    ("gap", "Which concepts are isolated, with no relationships at all?",
     "MATCH (n:Node) WHERE n.graph='disgust-mollusca' AND NOT (n)--() RETURN n.name AS name", {}),
    ("gap", "Which phenomena have no known cause in the graph?",
     "MATCH (p:Node) WHERE p.kind='Phenomenon' AND p.graph='disgust-mollusca' AND NOT (p)<-[:causes]-() RETURN p.name AS phenomenon", {}),
    ("gap", "Which measurement methods are not linked to anything they measure?",
     "MATCH (m:Node) WHERE m.kind='Method' AND m.graph='disgust-mollusca' AND NOT (m)-[:measures]->() RETURN m.name AS method", {}),
    ("gap", "Which processes have no downstream effects recorded?",
     "MATCH (p:Node) WHERE p.kind='Process' AND p.graph='disgust-mollusca' AND NOT (p)-->() RETURN p.name AS process", {}),
    ("gap", "Find concept pairs connected through an intermediate but not directly — candidate missing links.",
     None, {}),
    ("gap", "Which entities appear only as effects and never as a cause of anything?", None, {}),

    # --- under-developed areas / new fields ---
    ("frontier", "Which single category (kind) of concept is least represented?",
     "MATCH (n:Node) WHERE n.graph='disgust-mollusca' RETURN n.kind AS kind, count(*) AS n ORDER BY n ASC LIMIT 1", {}),
    ("frontier", "Which concepts are the biggest hubs — most total relationships in or out?", None, {}),
    ("frontier", "What downstream effects of climatic changes are still only hypothesized?", None, {}),
    ("frontier", "Which methods could potentially be applied to phenomena they have not yet measured?", None, {}),
    ("frontier", "Which single concept is the most contested — involved in the most contradictions?",
     "MATCH (n:Node)-[r:contradicts]-() WHERE n.graph='disgust-mollusca' RETURN n.name AS name, count(r) AS c ORDER BY c DESC LIMIT 1", {}),

    # --- aggregation / statistics ---
    ("aggregate", "What is the average confidence of claims, broken down by status?",
     "MATCH ()-[r]->() WHERE r.graph='disgust-mollusca' AND r.status IS NOT NULL RETURN r.status AS status, avg(r.confidence) AS avg_conf", {}),
    ("aggregate", "Which kind of concept participates in the most relationships?",
     "MATCH (n:Node)-[r]-() WHERE n.graph='disgust-mollusca' RETURN n.kind AS kind, count(r) AS c ORDER BY c DESC LIMIT 1", {}),

    # --- connectivity / reachability ---
    ("connectivity", "Is there any path connecting Cepaea nemoralis to climate?",
     "MATCH p=(a:Node {name:'Cepaea nemoralis'})-[*1..4]-(b:Node {name:'climate'}) RETURN a.name AS a, b.name AS b", {}),
    ("connectivity", "How is invasive species connected to genetic structure?", None, {}),

    # --- open questions ---
    ("open", "What contradictions exist in the current understanding?",
     "MATCH (a:Node)-[:contradicts]->(b:Node) WHERE a.graph='disgust-mollusca' RETURN a.name AS a, b.name AS b", {}),
    ("open", "Which concepts have both established and non-established (hypothesized/emerging/refuted) claims about them?",
     None, {}),

    # --- out of graph ---
    ("out-of-graph", "What genes are associated with Alzheimer's disease?",
     "MATCH (n:Node) WHERE toLower(n.name) CONTAINS 'alzheimer' RETURN n.name AS name", {}),
]


def _vals(row):
    items = [v for k, v in row.items() if not str(k).startswith("@")] if isinstance(row, dict) else [row]
    return Counter(str(canonicalize(v)) for v in items)


def verdict(pred_rows, gold_cypher, gp, arcade, timeout):
    if gold_cypher is None:
        return "INSPECT", None
    gold = arcade.query(gold_cypher, gp, timeout_s=timeout, limit=200)
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
    pipeline = build_pipeline(cfg)
    print(f"model: {cfg.llm.model} @ {cfg.llm.endpoint}   |   graph: worms/disgust-mollusca\n")

    tally = Counter()
    for i, (theme, q, gold_cy, gp) in enumerate(PROMPTS, 1):
        t0 = time.perf_counter()
        r = pipeline.answer(QueryRequest(question=q, max_attempts=3, row_cap=200, timeout_s=timeout))
        ms = (time.perf_counter() - t0) * 1000
        v, goldn = verdict(r.rows if r.status in ("ok", "empty") else None, gold_cy, gp, arcade, timeout)
        tally[v] += 1
        print("=" * 82)
        print(f"[{i}/{len(PROMPTS)}] ({theme}) {q}")
        print(f"  verdict: {v}   status: {r.status}   rows: {len(r.rows or [])}"
              f"{'/gold ' + str(goldn) if goldn is not None else ''}   attempts: {len(r.attempts)}   {ms:.0f}ms")
        print(f"  cypher: {r.final_cypher}")
        rows = [{k: v2 for k, v2 in row.items() if not str(k).startswith('@')} if isinstance(row, dict) else row
                for row in (r.rows or [])[:4]]
        print(f"  rows: {rows}")

    print("\n" + "#" * 82)
    print("verdicts:", dict(tally))
    graded = tally["EXACT"] + tally["SUPERSET"] + tally["WRONG"]
    if graded:
        # STRICT: only EXACT counts as correct; SUPERSET reported separately.
        print(f"graded correct (EXACT only): {tally['EXACT']}/{graded}"
              f"  | SUPERSET(not counted): {tally['SUPERSET']}  WRONG: {tally['WRONG']}"
              f"  | INSPECT (open-ended): {tally['INSPECT']}")


if __name__ == "__main__":
    main()
