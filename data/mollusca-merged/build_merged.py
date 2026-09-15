#!/usr/bin/env python3
"""Merge the WoRMS Mollusca graph export with the WoRMS REST enrichment.

Two input packages:

  --kg-package   the unzipped ``kg_mollusca-data-populate-script`` toolkit:
                 data/raw/{nodes,edges}.jsonl.gz   (worms-graph store export,
                 144,127 nodes / 144,129 edges) and data/clean/*.jsonl (the
                 K-Git KG upload shape produced by clean-worms.ts).
  --enriched     the untarred ``worms-enriched-full`` bundle: nodes.jsonl.gz
                 holding the 123,059 Species nodes with props.{vernaculars,
                 localities, attributes, external_ids} back-filled from the
                 WoRMS REST API (AphiaVernaculars / Distributions / Attributes /
                 ExternalID).

Every enriched node is one of the raw export's Species nodes (same node id),
so the merge is an overlay: the enrichment becomes node properties, the node
text (``description``) is extended with the new facts, and the two attribute
types whose values are AphiaIDs of *other* taxa (``Host``, ``Food source``)
are resolved into real graph edges (``has_host``, ``feeds_on``).

Outputs (under --out), all gzip JSONL unless noted:

  store/nodes.jsonl.gz          worms-graph store shape {id,kind,name,description,phylum,graph,props}
  store/edges.jsonl.gz          {id,kind,src,tgt,graph,phylum,attrs} — original + derived edges
  kg/nodes.jsonl.gz             K-Git KG shape {id,title,type,description,aliases,metadata,...}
  kg/edges.jsonl.gz             {source,target,type,status,directed,confidence,evidenceDois?,metadata?}
  kg/concept_nodes.jsonl.gz     the 240-node claims graph (unchanged, for parity with the toolkit)
  kg/concept_edges.jsonl.gz     the 271 claim edges (unchanged)
  documents/papers.jsonl.gz     evidence documents (papers/references) cited by nodes and edges
  documents/node_texts.jsonl.gz one retrieval document per store node (embedding / grounding text)
  vocab/attributes.json         controlled vocabularies of the enrichment (trait values, localities, ...)
  MANIFEST.json                 counts, coverage, provenance, per-file sha256

Stdlib only.  Re-runnable; output is deterministic for the same inputs.

    python build_merged.py --kg-package /path/to/kg_mollusca --enriched /path/to/worms-enriched-full --out .
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import sys
import time
import uuid
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
ENRICH_KEYS = ("vernaculars", "localities", "attributes", "external_ids", "_enriched")
ENRICHMENT_TAG = "worms-rest-attributes"
CONCEPT_GRAPH = "disgust-mollusca"
KG_DESC_MAX = 5000  # matches clean-worms.ts (graph-api document/node schema limit)

# Attribute types whose values are AphiaIDs of other taxa -> derived edge kinds.
TAXON_REF_ATTRS = {"Host": "has_host", "Food source": "feeds_on"}
# Constant on 98% of taxa (always ICZN) — pure noise in retrieval text.
SKIP_TEXT_ATTRS = {"Nomenclature code"}
EXT_LABEL = {"ncbi": "NCBI Taxonomy ID", "fishbase": "FishBase ID"}

# Flattened KG metadata keys (string -> string, values joined with " | ").
FLAT_META = {
    "Functional group": "functional_group",
    "Feeding method": "feeding_method",
    "Trophic level": "trophic_level",
    "Life stage": "life_stage",
    "Body size (qualitative)": "body_size_qualitative",
    "Body size": "body_size",
    "Unit": "body_size_unit",
    "Dimension": "body_size_dimension",
    "Type": "body_size_type",
    "Sex": "body_size_sex",
    "IUCN Red List Category": "iucn_category",
    "Year Assessed": "iucn_year_assessed",
    "Criteria": "iucn_criteria",
    "Ecological interactions": "ecological_interactions",
    "Development": "development",
    "Mobility": "mobility",
    "AMBI ecological group": "ambi_group",
    "Sociability": "sociability",
    "Composition": "composition",
    "Structure": "structure",
    "Supporting structure & enclosure": "supporting_structure",
    "Fossil range": "fossil_range",
    "CITES Annex": "cites_annex",
    "Species exhibits underwater soniferous behaviour": "soniferous_behaviour",
    "Species importance to society": "importance_to_society",
    "Paraphyletic group": "paraphyletic_group",
    "Brooding": "brooding",
    "Zonation": "zonation",
    "Environmental position": "environmental_position",
    "Etymology classification": "etymology",
    "Identifier": "identifier",
    "Locality (MRGID)": "locality_mrgids",
    "Host": "host_aphia_ids",
    "Food source": "food_source_aphia_ids",
}
VOCAB_CAP = 300  # per-attribute distinct values kept in vocab/attributes.json


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class Writer:
    """gzip JSONL writer that counts lines."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # mtime=0 and no embedded filename -> byte-identical output for identical inputs
        self._raw = open(path, "wb")
        self._gz = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, compresslevel=9, mtime=0)
        self.f = __import__("io").TextIOWrapper(self._gz, encoding="utf-8", newline="\n")
        self.n = 0

    def write(self, rec) -> None:
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n += 1

    def close(self) -> int:
        self.f.close()
        self._raw.close()
        return self.n


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def worms_uuid(aphia_id) -> str:
    """Same construction as clean-worms.ts: UUIDv5(DNS namespace, "worms:<aphia>")."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"worms:{aphia_id}"))


_ID_TAIL = re.compile(r"(\d+)$")


def aphia_of(node: dict):
    a = (node.get("props") or {}).get("aphia_id")
    if a is not None:
        return str(a)
    m = _ID_TAIL.search(node["id"])
    return m.group(1) if m else None


_WS = re.compile(r"\s+")


def clean_val(v) -> str:
    return _WS.sub(" ", str(v)).strip()


def uniq(seq):
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def base_hash(node: dict) -> str:
    """Hash of everything except the enrichment-bearing props — used to verify
    that an enriched record really is the raw record plus enrichment."""
    n = dict(node)
    p = dict(n.get("props") or {})
    for k in ENRICH_KEYS:
        p.pop(k, None)
    n["props"] = p
    return hashlib.sha1(json.dumps(n, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# enrichment -> text lines / metadata
# ---------------------------------------------------------------------------
def enrichment_lines(e: dict, names: dict) -> list[str]:
    """Human-readable facts to append to a node's retrieval text.

    One fact per line, mirroring the store's own description convention
    (name / lineage / citation / authority / vernaculars / localities / paper
    titles, one per line)."""
    lines: list[str] = []
    for v in e.get("vernaculars") or []:
        n = clean_val(v.get("name", ""))
        if n:
            lines.append(n)
    seen_loc = set()
    for loc in e.get("localities") or []:
        n = clean_val(loc.get("locality", ""))
        if not n or n in seen_loc:
            continue
        seen_loc.add(n)
        st = loc.get("status")
        lines.append(n if st in (None, "valid") else f"{n} ({st})")
    for attr, vals in (e.get("attributes") or {}).items():
        if attr in SKIP_TEXT_ATTRS:
            continue
        vals = uniq(clean_val(v) for v in vals)
        if not vals:
            continue
        if attr in TAXON_REF_ATTRS:
            rendered = []
            for a in vals:
                nm = names.get(a)
                rendered.append(f"{nm} [AphiaID {a}]" if nm else f"AphiaID {a}")
            vals = rendered
        lines.append(f"{attr}: {'; '.join(vals)}")
    for src, ids in (e.get("external_ids") or {}).items():
        ids = uniq(clean_val(i) for i in ids)
        if ids:
            lines.append(f"{EXT_LABEL.get(src, src)}: {', '.join(ids)}")
    return uniq(lines)


def merge_description(orig: str, lines: list[str]) -> str:
    have = {l.strip() for l in (orig or "").split("\n")}
    add = [l for l in lines if l not in have]
    if not add:
        return orig or ""
    base = (orig or "").rstrip()
    return (base + " \n" if base else "") + " \n".join(add)


def flat_metadata(e: dict) -> dict[str, str]:
    """String->string KG metadata for the enrichment (JSON blobs for the full
    structures plus flattened filterable keys)."""
    meta: dict[str, str] = {}
    if e.get("vernaculars"):
        meta["vernaculars"] = json.dumps(e["vernaculars"], ensure_ascii=False)
    if e.get("localities"):
        meta["localities"] = json.dumps(e["localities"], ensure_ascii=False)
        meta["locality_count"] = str(len({l.get("locality") for l in e["localities"]}))
    if e.get("attributes"):
        meta["attributes"] = json.dumps(e["attributes"], ensure_ascii=False)
        for attr, key in FLAT_META.items():
            vals = e["attributes"].get(attr)
            if vals:
                vals = uniq(clean_val(v) for v in vals)
                if vals:
                    meta[key] = " | ".join(vals)
    if e.get("external_ids"):
        meta["external_ids"] = json.dumps(e["external_ids"], ensure_ascii=False)
        for src, ids in e["external_ids"].items():
            ids = uniq(clean_val(i) for i in ids)
            if ids:
                meta[f"{src}_id"] = " | ".join(ids)
    meta["enrichment"] = ENRICHMENT_TAG
    return meta


# ---------------------------------------------------------------------------
# papers / evidence documents
# ---------------------------------------------------------------------------
class Papers:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.order: list[str] = []

    @staticmethod
    def key_of(e: dict):
        doi = (e.get("doi") or "").strip().lower()
        if doi:
            return "doi:" + doi
        if e.get("key"):
            return "key:" + str(e["key"])
        if e.get("url"):
            return "url:" + str(e["url"])
        if e.get("citation"):
            return "cit:" + hashlib.sha1(str(e["citation"]).encode("utf-8")).hexdigest()[:16]
        return None

    def add(self, e: dict, cited_by: dict) -> str | None:
        k = self.key_of(e)
        if k is None:
            return None
        d = self.docs.get(k)
        if d is None:
            d = {"id": k, "doi": None, "key": None, "title": None, "year": None, "venue": None,
                 "authors": [], "citation": None, "url": None, "abstract": None, "sources": [],
                 "tags": [], "cited_by": []}
            self.docs[k] = d
            self.order.append(k)
        for f_src, f_dst in (("doi", "doi"), ("key", "key"), ("title", "title"), ("year", "year"),
                             ("venue", "venue"), ("container", "venue"), ("citation", "citation"),
                             ("url", "url"), ("abstract", "abstract")):
            v = e.get(f_src)
            if v not in (None, "", []) and not d.get(f_dst):
                d[f_dst] = v
        for f in ("authors", "sources"):
            for v in e.get(f) or []:
                if v and v not in d[f]:
                    d[f].append(v)
        for v in e.get("zotero_tags") or []:
            if v and v not in d["tags"]:
                d["tags"].append(v)
        if cited_by not in d["cited_by"]:
            d["cited_by"].append(cited_by)
        return k


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------
def build(kg_pkg: Path, enriched: Path, out: Path) -> dict:
    t0 = time.time()
    raw_nodes = kg_pkg / "data" / "raw" / "nodes.jsonl.gz"
    raw_edges = kg_pkg / "data" / "raw" / "edges.jsonl.gz"
    clean_nodes = kg_pkg / "data" / "clean" / "clean_worms_nodes.jsonl"
    clean_edges = kg_pkg / "data" / "clean" / "clean_worms_edges.jsonl"
    concept_nodes = kg_pkg / "data" / "clean" / "concept_nodes.jsonl"
    concept_edges = kg_pkg / "data" / "clean" / "concept_edges.jsonl"
    enr_nodes = enriched / "nodes.jsonl.gz"
    for p in (raw_nodes, raw_edges, clean_nodes, clean_edges, concept_nodes, concept_edges, enr_nodes):
        if not p.exists():
            sys.exit(f"missing input: {p}")
    log = lambda *a: print(*a, file=sys.stderr, flush=True)

    # ---- pass 0: index the raw taxonomy (ids, names, aphia winners) --------
    log("pass 0: indexing raw nodes")
    name_by_aphia: dict[str, str] = {}
    winner_by_aphia: dict[str, tuple[str, int]] = {}   # aphia -> (node id, prop-key count)
    raw_hash: dict[str, str] = {}
    aphia_by_id: dict[str, str] = {}
    n_raw = 0
    for n in jsonl(raw_nodes):
        n_raw += 1
        if n.get("graph") is not None:
            continue
        a = aphia_of(n)
        aphia_by_id[n["id"]] = a
        raw_hash[n["id"]] = base_hash(n)
        nprops = len(n.get("props") or {})
        # clean-worms.ts rule: the record with the most props wins; ties keep file order
        if a not in winner_by_aphia or nprops > winner_by_aphia[a][1]:
            winner_by_aphia[a] = (n["id"], nprops)
            name_by_aphia[a] = n.get("name") or ""
    losers = {nid for nid, a in aphia_by_id.items() if winner_by_aphia[a][0] != nid}
    log(f"  raw nodes {n_raw:,}  taxa {len(aphia_by_id):,}  distinct aphia {len(winner_by_aphia):,}  duplicate stubs {len(losers)}")

    # ---- pass 1: load enrichment (overlay keyed by node id) ----------------
    log("pass 1: loading enrichment")
    enr: dict[str, dict] = {}
    stats = collections.Counter()
    mismatched_base: list[str] = []
    for n in jsonl(enr_nodes):
        nid = n["id"]
        stats["enriched_records"] += 1
        if nid not in aphia_by_id:
            stats["enriched_not_in_raw"] += 1
            continue
        if base_hash(n) != raw_hash[nid]:
            mismatched_base.append(nid)
        p = n.get("props") or {}
        e = {k: p[k] for k in ("vernaculars", "localities", "attributes", "external_ids") if p.get(k)}
        enr[nid] = e
    del raw_hash
    log(f"  enrichment overlays {len(enr):,}  base-record mismatches {len(mismatched_base)}  not-in-raw {stats['enriched_not_in_raw']}")

    # ---- pass 2: store nodes + node texts + papers + vocab -----------------
    log("pass 2: writing store/nodes + documents/node_texts")
    w_nodes = Writer(out / "store" / "nodes.jsonl.gz")
    w_texts = Writer(out / "documents" / "node_texts.jsonl.gz")
    papers = Papers()
    vocab_attr: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    vocab_loc = collections.Counter()
    vocab_lang = collections.Counter()
    vocab_ext = collections.Counter()
    derived: dict[tuple[str, str, str], dict] = {}      # (kind, src aphia, tgt aphia) -> attrs
    unresolved = collections.Counter()
    kinds = collections.Counter()
    cov = collections.Counter()
    enriched_species_ids: set[str] = set()

    for n in jsonl(raw_nodes):
        kinds[n["kind"]] += 1
        p = dict(n.get("props") or {})
        nid = n["id"]
        e = enr.get(nid)
        text_extra: list[str] = []
        if e is not None:
            enriched_species_ids.add(nid)
            cov["species_enriched"] += 1
            if not e:
                cov["species_without_any_field"] += 1
            for k in ("vernaculars", "localities", "attributes", "external_ids"):
                if e.get(k):
                    cov[f"with_{k}"] += 1
            # legacy records already carried richer localities (mrgid, establishment_means) —
            # keep them, under their own key, and normalise the shared key to the enrichment shape
            if isinstance(p.get("localities"), list) and p["localities"] and "name" in p["localities"][0]:
                p["localities_detail"] = p.pop("localities")
            if isinstance(p.get("vernaculars"), list) and e.get("vernaculars"):
                seen = {(v.get("name"), v.get("lang")) for v in e["vernaculars"]}
                e["vernaculars"] = e["vernaculars"] + [v for v in p["vernaculars"] if (v.get("name"), v.get("lang")) not in seen]
            p.update(e)
            p["_enriched"] = True
            p["enrichment_source"] = ENRICHMENT_TAG
            # resolved cross-taxon references
            attrs = e.get("attributes") or {}
            for attr, kind in TAXON_REF_ATTRS.items():
                resolved = []
                src_a = aphia_by_id[nid]
                for tgt_a in uniq(clean_val(v) for v in attrs.get(attr, [])):
                    if tgt_a in winner_by_aphia and tgt_a != src_a:
                        resolved.append({"aphia_id": int(tgt_a), "node_id": winner_by_aphia[tgt_a][0], "name": name_by_aphia[tgt_a]})
                        derived[(kind, src_a, tgt_a)] = {"origin": ENRICHMENT_TAG, "attribute": attr}
                    else:
                        unresolved[kind] += 1
                if resolved:
                    p["hosts" if kind == "has_host" else "food_sources"] = resolved
            # vocab
            for attr, vals in attrs.items():
                for v in vals:
                    vocab_attr[attr][clean_val(v)] += 1
            for loc in e.get("localities") or []:
                vocab_loc[clean_val(loc.get("locality", ""))] += 1
            for v in e.get("vernaculars") or []:
                vocab_lang[v.get("lang") or "?"] += 1
            for src, ids in (e.get("external_ids") or {}).items():
                vocab_ext[src] += len(ids)
            text_extra = enrichment_lines(e, name_by_aphia)
        desc = merge_description(n.get("description") or "", text_extra)
        rec = dict(n)
        rec["description"] = desc
        rec["props"] = p
        w_nodes.write(rec)

        # papers cited by this node
        for role, field in (("paper", "papers"), ("reference", "references"), ("evidence", "evidenced_by")):
            for pe in p.get(field) or []:
                if isinstance(pe, dict):
                    papers.add(pe, {"node_id": nid, "role": role})

        # retrieval document
        is_taxon = n.get("graph") is None
        a = aphia_by_id.get(nid)
        kg_id = worms_uuid(a) if is_taxon else str(uuid.uuid5(uuid.NAMESPACE_DNS, f"worms-concept:{nid}"))
        aliases = uniq(clean_val(v.get("name", "")) for v in (p.get("vernaculars") or []))
        doc = {
            "node_id": nid,
            "kg_id": None if nid in losers else kg_id,
            "canonical_name": n.get("name"),
            "node_type": (p.get("rank") if is_taxon else n["kind"]),
            "kind": n["kind"],
            "graph": n.get("graph"),
            "aphia_id": int(a) if (is_taxon and a) else None,
            "aliases": aliases,
            "definition": desc,
            "enriched": e is not None,
        }
        if nid in losers:
            doc["kg_merged_into"] = kg_id
        w_texts.write(doc)
    n_store_nodes = w_nodes.close()
    n_texts = w_texts.close()
    log(f"  store nodes {n_store_nodes:,}  node texts {n_texts:,}  derived edge candidates {len(derived):,}  unresolved {dict(unresolved)}")

    # ---- pass 3: store edges (original + derived) --------------------------
    log("pass 3: writing store/edges")
    w_edges = Writer(out / "store" / "edges.jsonl.gz")
    edge_kinds = collections.Counter()
    for e in jsonl(raw_edges):
        edge_kinds[e["kind"]] += 1
        w_edges.write(e)
        for pe in (e.get("attrs") or {}).get("evidenced_by") or []:
            if isinstance(pe, dict):
                papers.add(pe, {"edge_id": e.get("id") or f"{e['kind']}::{e['src']}::{e['tgt']}", "role": "evidence"})
    # derived edges: every raw record (winners and duplicate stubs) that carries the attribute
    derived_store = collections.Counter()
    for nid, a in aphia_by_id.items():
        e = enr.get(nid)
        if not e:
            continue
        attrs = e.get("attributes") or {}
        for attr, kind in TAXON_REF_ATTRS.items():
            for tgt_a in uniq(clean_val(v) for v in attrs.get(attr, [])):
                if tgt_a in winner_by_aphia and tgt_a != a:
                    tgt = winner_by_aphia[tgt_a][0]
                    w_edges.write({
                        "id": f"{kind}::{nid}::{tgt}", "kind": kind, "src": nid, "tgt": tgt,
                        "graph": None, "phylum": None,
                        "attrs": {"origin": ENRICHMENT_TAG, "attribute": attr, "confidence": 1.0},
                    })
                    derived_store[kind] += 1
                    edge_kinds[kind] += 1
    n_store_edges = w_edges.close()
    log(f"  store edges {n_store_edges:,}  derived {dict(derived_store)}")

    # ---- pass 4: KG nodes -------------------------------------------------
    log("pass 4: writing kg/nodes")
    w_kg = Writer(out / "kg" / "nodes.jsonl.gz")
    kg_ids: set[str] = set()
    kg_types = collections.Counter()
    kg_cov = collections.Counter()
    uuid_mismatch = 0
    kg_desc_truncated = 0
    for n in jsonl(clean_nodes):
        kg_types[n["type"]] += 1
        a = (n.get("metadata") or {}).get("aphia_id")
        if a is not None:
            if worms_uuid(a) != n["id"]:
                uuid_mismatch += 1
            win = winner_by_aphia.get(str(a))
            e = enr.get(win[0]) if win else None
            if e:
                kg_cov["species_enriched"] += 1
                lines = enrichment_lines(e, name_by_aphia)
                desc = merge_description(n.get("description") or "", lines)
                if len(desc) > KG_DESC_MAX:
                    desc = desc[: KG_DESC_MAX - 3] + "…"
                    kg_desc_truncated += 1
                if desc:
                    n["description"] = desc
                aliases = uniq(list(n.get("aliases") or []) + [clean_val(v.get("name", "")) for v in (e.get("vernaculars") or [])])
                if aliases:
                    n["aliases"] = aliases
                meta = dict(n.get("metadata") or {})
                meta.update(flat_metadata(e))
                hosts = [str(t) for (k, s, t) in derived if k == "has_host" and s == str(a)]
                foods = [str(t) for (k, s, t) in derived if k == "feeds_on" and s == str(a)]
                if hosts:
                    meta["host_names"] = " | ".join(name_by_aphia[t] for t in hosts)
                if foods:
                    meta["food_source_names"] = " | ".join(name_by_aphia[t] for t in foods)
                n["metadata"] = meta
                # keep key order stable: id,title,type,description,aliases,metadata
                ordered = {k: n[k] for k in ("id", "title", "type", "description", "aliases") if k in n}
                ordered.update({k: v for k, v in n.items() if k not in ordered})
                n = ordered
        kg_ids.add(n["id"])
        w_kg.write(n)
    n_kg_nodes = w_kg.close()
    log(f"  kg nodes {n_kg_nodes:,}  enriched {kg_cov['species_enriched']:,}  uuid mismatches {uuid_mismatch}  descriptions truncated {kg_desc_truncated}")

    # ---- pass 5: KG edges -------------------------------------------------
    log("pass 5: writing kg/edges")
    w_kge = Writer(out / "kg" / "edges.jsonl.gz")
    kg_edge_types = collections.Counter()
    for e in jsonl(clean_edges):
        kg_edge_types[e["type"]] += 1
        w_kge.write(e)
    derived_kg = collections.Counter()
    seen = set()
    for (kind, src_a, tgt_a), attrs in derived.items():
        s, t = worms_uuid(src_a), worms_uuid(tgt_a)
        if s in kg_ids and t in kg_ids and (s, kind, t) not in seen:
            seen.add((s, kind, t))
            w_kge.write({"source": s, "target": t, "type": kind, "status": "Established", "directed": True,
                         "confidence": 1.0, "metadata": {"origin": attrs["origin"], "attribute": attrs["attribute"]}})
            derived_kg[kind] += 1
            kg_edge_types[kind] += 1
    n_kg_edges = w_kge.close()
    log(f"  kg edges {n_kg_edges:,}  derived {dict(derived_kg)}")

    # ---- concept graph (unchanged copies) ---------------------------------
    for src, dst in ((concept_nodes, out / "kg" / "concept_nodes.jsonl.gz"), (concept_edges, out / "kg" / "concept_edges.jsonl.gz")):
        w = Writer(dst)
        for r in jsonl(src):
            w.write(r)
        w.close()

    # ---- documents/papers -------------------------------------------------
    log("pass 6: writing documents/papers + vocab")
    w_p = Writer(out / "documents" / "papers.jsonl.gz")
    paper_stats = collections.Counter()
    for k in papers.order:
        d = papers.docs[k]
        paper_stats["total"] += 1
        if d["doi"]:
            paper_stats["with_doi"] += 1
        if d["abstract"]:
            paper_stats["with_abstract"] += 1
        w_p.write(d)
    n_papers = w_p.close()

    # ---- vocab ------------------------------------------------------------
    def capped(counter: collections.Counter, cap: int) -> dict:
        items = counter.most_common()
        return {"distinct": len(counter), "total": sum(counter.values()), "truncated": len(items) > cap,
                "values": [{"value": v, "count": c} for v, c in items[:cap]]}
    vocab = {
        "attributes": {attr: capped(c, VOCAB_CAP) for attr, c in sorted(vocab_attr.items(), key=lambda kv: -sum(kv[1].values()))},
        "localities": capped(vocab_loc, 2000),
        "vernacular_languages": capped(vocab_lang, 500),
        "external_id_sources": dict(vocab_ext),
        "derived_edge_attributes": TAXON_REF_ATTRS,
        "kg_metadata_keys": {"flattened": FLAT_META, "json_blobs": ["vernaculars", "localities", "attributes", "external_ids"],
                              "external_ids": {s: f"{s}_id" for s in vocab_ext}},
    }
    (out / "vocab").mkdir(parents=True, exist_ok=True)
    (out / "vocab" / "attributes.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- manifest ---------------------------------------------------------
    log("writing MANIFEST.json")
    def load_json(p: Path):
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    out_files = [out / "store" / "nodes.jsonl.gz", out / "store" / "edges.jsonl.gz",
                 out / "kg" / "nodes.jsonl.gz", out / "kg" / "edges.jsonl.gz",
                 out / "kg" / "concept_nodes.jsonl.gz", out / "kg" / "concept_edges.jsonl.gz",
                 out / "documents" / "papers.jsonl.gz", out / "documents" / "node_texts.jsonl.gz",
                 out / "vocab" / "attributes.json"]
    n_taxa = len(aphia_by_id)
    pc = lambda x, d: f"{x}/{d} ({100 * x // max(d, 1)}%)"
    n_species = cov["species_enriched"]
    manifest = {
        "dataset": "WoRMS Mollusca graph — merged taxonomy + claims graph + WoRMS REST enrichment",
        "built": date.today().isoformat(),
        "builder": "data/mollusca-merged/build_merged.py",
        "sources": {
            "kg_package": {
                "name": "kg_mollusca-data-populate-script.zip",
                "raw_manifest": load_json(kg_pkg / "data" / "raw" / "MANIFEST.json"),
                "files": {str(p.relative_to(kg_pkg)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                          for p in (raw_nodes, raw_edges, clean_nodes, clean_edges, concept_nodes, concept_edges)},
            },
            "enriched": {
                "name": "worms-enriched-full.tar",
                "manifest": load_json(enriched / "MANIFEST.json"),
                "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in (enr_nodes,)},
            },
        },
        "merge": {
            "key": "store node id (enriched nodes are the raw export's Species nodes, same ids); KG nodes joined via metadata.aphia_id -> uuid5('worms:<aphia>')",
            "enriched_records": stats["enriched_records"],
            "overlaid_on_raw": len(enr),
            "enriched_not_in_raw": stats["enriched_not_in_raw"],
            "base_record_mismatches": len(mismatched_base),
            "duplicate_aphia_stubs_in_store": len(losers),
            "kg_uuid_mismatches": uuid_mismatch,
            "kg_descriptions_truncated_to_5000": kg_desc_truncated,
        },
        "counts": {
            "store": {"nodes": n_store_nodes, "edges": n_store_edges, "nodes_by_kind": dict(kinds), "edges_by_kind": dict(edge_kinds),
                      "derived_edges": dict(derived_store), "unresolved_taxon_refs": dict(unresolved)},
            "kg": {"nodes": n_kg_nodes, "edges": n_kg_edges, "nodes_by_type": dict(kg_types), "edges_by_type": dict(kg_edge_types),
                   "derived_edges": dict(derived_kg), "enriched_species": kg_cov["species_enriched"]},
            "documents": {"papers": n_papers, "papers_with_doi": paper_stats["with_doi"], "papers_with_abstract": paper_stats["with_abstract"],
                          "node_texts": n_texts},
        },
        "coverage": {
            "species_enriched": pc(n_species, n_taxa) + " of all taxa; " + pc(n_species, stats["enriched_records"]) + " of enriched records",
            "vernaculars": pc(cov["with_vernaculars"], n_species),
            "localities": pc(cov["with_localities"], n_species),
            "attributes": pc(cov["with_attributes"], n_species),
            "external_ids": pc(cov["with_external_ids"], n_species),
            "no_field_returned": pc(cov["species_without_any_field"], n_species),
        },
        "schemas": {
            "store/nodes": "{id,kind,name,description,phylum,graph,props} -> store.upsert_node; enrichment in props.{vernaculars,localities,attributes,external_ids,hosts,food_sources,_enriched,enrichment_source}; legacy rich localities kept as props.localities_detail",
            "store/edges": "{id,kind,src,tgt,graph,phylum,attrs} -> store.upsert_edge; derived kinds has_host/feeds_on carry attrs.origin='" + ENRICHMENT_TAG + "'",
            "kg/nodes": "{id,title,type,description?,aliases?,metadata,confidence?,evidenceDois?,fieldTags?} (graph-api CreateNodeBody); enrichment as metadata JSON blobs + flattened keys (see vocab/attributes.json kg_metadata_keys)",
            "kg/edges": "{source,target,type,status,directed,confidence,evidenceDois?,metadata?} (graph-api CreateEdgeBody); new edge types has_host, feeds_on must be registered",
            "documents/papers": "{id,doi,key,title,year,venue,authors,citation,url,abstract,sources,tags,cited_by[{node_id|edge_id,role}]}",
            "documents/node_texts": "{node_id,kg_id,canonical_name,node_type,kind,graph,aphia_id,aliases,definition,enriched} — directly consumable by text2cypher.ingest.index_nodes",
        },
        "files": {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in out_files},
        "build_seconds": round(time.time() - t0, 1),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"done in {manifest['build_seconds']}s")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kg-package", type=Path, required=True, help="unzipped kg_mollusca-data-populate-script dir (contains data/raw, data/clean)")
    ap.add_argument("--enriched", type=Path, required=True, help="untarred worms-enriched-full dir (contains nodes.jsonl.gz)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    a = ap.parse_args()
    m = build(a.kg_package, a.enriched, a.out)
    c = m["counts"]
    print(f"store: {c['store']['nodes']:,} nodes / {c['store']['edges']:,} edges   "
          f"kg: {c['kg']['nodes']:,} nodes / {c['kg']['edges']:,} edges   "
          f"documents: {c['documents']['papers']} papers / {c['documents']['node_texts']:,} node texts")
    print("coverage:", json.dumps(m["coverage"]))


if __name__ == "__main__":
    main()
