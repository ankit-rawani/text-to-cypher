"""Simulate a researcher exploring the disgust-mollusca concept graph via
text2cypher. Runs a graded set of escalating-complexity questions through the
real pipeline (Anthropic gateway -> ArcadeDB), and grades each against a
hand-written reference query using an alias-insensitive value comparison.

    python scripts/research_session.py
"""

from __future__ import annotations

import time
from collections import Counter

from text2cypher.clients.arcadedb import ArcadeDBClient
from text2cypher.config import load_config
from text2cypher.contracts import QueryRequest
from text2cypher.eval.compare import canonicalize
from text2cypher.pipeline import build_pipeline

# q, stratum, reference cypher (None = open-ended, shown but not graded), params
QUESTIONS = [
    # --- 1-hop ---
    ("What does climatic changes cause?", "1-hop",
     "MATCH (a:Node)-[:causes]->(b:Node) WHERE a.name=$n RETURN b.name AS effect", {"n": "climatic changes"}),
    ("What is a component of Dreissena?", "1-hop",
     "MATCH (a:Node)-[:is_component_of]->(b:Node) WHERE b.name=$n RETURN a.name AS part", {"n": "Dreissena"}),
    ("What reduces the shell colour phenotype?", "1-hop",
     "MATCH (a:Node)-[:reduces]->(b:Node) WHERE b.name=$n RETURN a.name AS x", {"n": "shell colour phenotype"}),

    # --- alias / phrasing ---
    ("Which techniques are used to measure climatic changes?", "alias-phrased",
     "MATCH (m:Node)-[:measures]->(b:Node) WHERE b.name=$n RETURN m.name AS method", {"n": "climatic changes"}),

    # --- multi-hop ---
    ("What are the effects two causal steps downstream of climatic changes?", "multi-hop",
     "MATCH (a:Node)-[:causes]->()-[:causes]->(c:Node) WHERE a.name=$n RETURN DISTINCT c.name AS effect", {"n": "climatic changes"}),
    ("Through what intermediate concept do predators causally affect morph frequencies?", "multi-hop",
     "MATCH (a:Node)-[:causes]->(b:Node)-[:causes]->(c:Node) WHERE a.name=$x AND c.name=$y RETURN DISTINCT b.name AS intermediate",
     {"x": "predators", "y": "morph frequencies"}),
    ("Which concepts sit in the middle of a causal chain (something causes them and they cause something)?", "multi-hop",
     "MATCH (a:Node)-[:causes]->(x:Node)-[:causes]->(c:Node) RETURN DISTINCT x.name AS concept", {}),

    # --- filtered by edge attribute ---
    ("List the established causal relationships.", "filtered",
     "MATCH (a:Node)-[r:causes]->(b:Node) WHERE r.status='established' RETURN a.name AS cause, b.name AS effect", {}),
    ("What inhibits or reduces morph frequencies?", "filtered",
     "MATCH (a:Node)-[r]->(b:Node) WHERE b.name=$n AND type(r) IN ['inhibits','reduces'] RETURN a.name AS x", {"n": "morph frequencies"}),

    # --- contradiction ---
    ("Which pairs of concepts contradict each other?", "relation",
     "MATCH (a:Node)-[:contradicts]->(b:Node) RETURN a.name AS a, b.name AS b", {}),

    # --- aggregation ---
    ("Count the relationships of each type in the concept graph.", "aggregation",
     "MATCH (a:Node)-[r]->(b:Node) WHERE a.graph='disgust-mollusca' RETURN type(r) AS rel, count(*) AS n", {}),
    ("How many concepts are there of each kind?", "aggregation",
     "MATCH (n:Node) WHERE n.graph='disgust-mollusca' RETURN n.kind AS kind, count(*) AS n", {}),
    ("Which three concepts have the most outgoing relationships?", "aggregation",
     "MATCH (a:Node)-[r]->() WHERE a.graph='disgust-mollusca' RETURN a.name AS name, count(r) AS deg ORDER BY deg DESC LIMIT 3", {}),

    # --- open-ended (shown, not graded) ---
    ("How is visual selection connected to morph frequencies?", "open", None, {}),

    # --- out-of-graph (expect empty) ---
    ("What is the GDP of Japan?", "out-of-graph", "__EMPTY__", {}),
]


def _vals(row):
    """Multiset (Counter) of a row's values, ignoring column names and @-meta."""
    if isinstance(row, dict):
        items = [v for k, v in row.items() if not str(k).startswith("@")]
    else:
        items = [row]
    return Counter(str(canonicalize(v)) for v in items)


def _matches(pred_rows, gold_rows):
    """Projection-tolerant: same cardinality, and every gold row's values are a
    (multiset) subset of a distinct predicted row's values. Lets the model add
    extra descriptive columns (id/confidence/status/…) without being marked wrong."""
    pred = [_vals(r) for r in (pred_rows or [])]
    gold = [_vals(r) for r in (gold_rows or [])]
    if len(pred) != len(gold):
        return False
    used = [False] * len(pred)
    for g in gold:
        hit = next((i for i, p in enumerate(pred) if not used[i] and g <= p), None)
        if hit is None:
            return False
        used[hit] = True
    return True


def grade(status, pred_rows, gold_cypher, gold_params, arcade, timeout):
    if gold_cypher is None:
        return "n/a", None
    if gold_cypher == "__EMPTY__":
        return ("PASS" if status == "empty" else "FAIL"), 0
    try:
        gold = arcade.query(gold_cypher, gold_params, timeout_s=timeout, limit=200)
    except Exception as exc:
        return f"GOLD-ERR({exc})", None
    ok = status in ("ok", "empty") and _matches(pred_rows, gold)
    return ("PASS" if ok else "FAIL"), len(gold)


def main() -> None:
    cfg = load_config()
    pipeline = build_pipeline(cfg)
    arcade = ArcadeDBClient.from_config(cfg.arcadedb)
    timeout = cfg.arcadedb.timeout_s

    results = []
    for i, (q, stratum, gold_cypher, params) in enumerate(QUESTIONS, 1):
        t0 = time.perf_counter()
        resp = pipeline.answer(QueryRequest(question=q, max_attempts=3, row_cap=200, timeout_s=timeout))
        ms = (time.perf_counter() - t0) * 1000
        verdict, gold_n = grade(resp.status, resp.rows, gold_cypher, params, arcade, timeout)
        results.append((stratum, verdict, len(resp.attempts), ms))

        print("=" * 78)
        print(f"[{i}/{len(QUESTIONS)}] ({stratum}) {q}")
        print(f"  verdict: {verdict}   status: {resp.status}   attempts: {len(resp.attempts)}   {ms:.0f}ms")
        if resp.grounding:
            g = ", ".join(f"'{e.mention}'→{e.canonical_name}(${e.param_name})" for e in resp.grounding)
            print(f"  grounding: {g}")
        print(f"  cypher: {resp.final_cypher}")
        rows = resp.rows or []
        shown = [{k: v for k, v in r.items() if not str(k).startswith('@')} if isinstance(r, dict) else r for r in rows[:4]]
        print(f"  rows({len(rows)}){' / gold ' + str(gold_n) if gold_n is not None else ''}: {shown}")
        val_codes = [i2.code for v in resp.validations for i2 in v.issues]
        if val_codes:
            print(f"  validator: {val_codes}")

    print("\n" + "#" * 78)
    graded = [r for r in results if r[1] in ("PASS", "FAIL")]
    passed = sum(1 for r in graded if r[1] == "PASS")
    print(f"GRADED: {passed}/{len(graded)} correct  |  ungraded(open): {sum(1 for r in results if r[1]=='n/a')}")
    per = {}
    for stratum, verdict, _, _ in results:
        if verdict in ("PASS", "FAIL"):
            per.setdefault(stratum, [0, 0])
            per[stratum][0] += 1 if verdict == "PASS" else 0
            per[stratum][1] += 1
    for stratum, (p, n) in sorted(per.items()):
        print(f"  {stratum:<14} {p}/{n}")
    att = Counter(r[2] for r in results)
    lat = sorted(r[3] for r in results)
    p50 = lat[len(lat) // 2]
    print(f"attempts histogram: {dict(sorted(att.items()))}  |  latency p50={p50:.0f}ms max={lat[-1]:.0f}ms")


if __name__ == "__main__":
    main()
