# WoRMS Mollusca — merged dataset

The complete WoRMS graph export (taxonomy backbone + the `disgust-mollusca`
claims graph) merged with the per-species enrichment pulled from the WoRMS REST
API (vernaculars, distributions, attributes/traits, external cross-reference
IDs). One dataset, delivered in both shapes the two upstream packages used, plus
the evidence documents, per-node retrieval text, controlled vocabularies and a
manifest with provenance and checksums.

Built by `build_merged.py` (stdlib only, deterministic). Exact counts, coverage
and per-file SHA-256 are in `MANIFEST.json`.

## What was merged

| input | what it is | records |
|---|---|---|
| `kg_mollusca-data-populate-script.zip` → `data/raw/` | worms-graph store export (2026-08-13): 143,887 `Taxon` + 240 concept nodes, 143,858 `parent_of` + 271 typed claim edges | 144,127 nodes / 144,129 edges |
| `kg_mollusca-data-populate-script.zip` → `data/clean/` | the same graph in K-Git KG upload shape, typed by rank (`clean-worms.ts` output) | 144,102 nodes / 144,126 edges |
| `worms-enriched-full.tar` | the export's 123,059 Species nodes with `props.{vernaculars,localities,attributes,external_ids}` back-filled from the WoRMS REST API (built 2026-09-12) | 123,059 nodes / 0 edges |

Every enriched node is one of the raw export's Species nodes with the **same
node id**, and the non-enrichment part of every enriched record hashes
identically to the raw record (0 mismatches), so the merge is a pure overlay.
KG records are joined through `metadata.aphia_id → uuid5("worms:<aphia>")`,
the same id construction `clean-worms.ts` uses (0 mismatches).

## Layout

```
store/nodes.jsonl.gz          144,127 nodes  — worms-graph store shape (Postgres → ArcadeDB → Qdrant)
store/edges.jsonl.gz          150,280 edges  — 144,129 original + 6,151 derived (has_host / feeds_on)
kg/nodes.jsonl.gz             144,102 nodes  — K-Git KG shape (migrate.ts), typed by rank / concept kind
kg/edges.jsonl.gz             150,277 edges  — 144,126 original + 6,151 derived
kg/concept_nodes.jsonl.gz         240 nodes  — the claims graph, unchanged (parity with the toolkit's data/clean)
kg/concept_edges.jsonl.gz         271 edges  — unchanged
documents/papers.jsonl.gz         389 docs   — evidence documents cited by nodes and edges (papers + references)
documents/node_texts.jsonl.gz 144,127 docs   — one retrieval document per store node, enrichment text included
vocab/attributes.json                        — trait vocabularies, locality names, vernacular languages, KG metadata key map
MANIFEST.json                                — counts, coverage, provenance (input checksums), per-file sha256
build_merged.py                              — the merge script
```

## Enrichment coverage (of the 123,059 Species)

| field | coverage |
|---|---|
| attributes / traits (≥1) | 121,499 (98%) |
| localities (distribution) | 56,918 (46%) |
| external IDs (NCBI 27,235 · FishBase 13,555) | 32,525 (26%) |
| vernaculars (109 languages) | 15,556 (12%) |

1,408 Species returned nothing from the API (no field at all); they are
flagged `_enriched: true` in `store/` (as in the source bundle) but carry no
`enrichment` key in `kg/`, which is why `kg/` reports 121,633 enriched Species
against 123,041 Species nodes.

## What the merge does to each record

### `store/nodes.jsonl.gz` — `{id, kind, name, description, phylum, graph, props}`

Species nodes get the enrichment merged into `props`:

- `vernaculars` `[{name, lang}]`, `localities` `[{locality, status}]`,
  `attributes` `{trait: [values]}`, `external_ids` `{ncbi: [...], fishbase: [...]}`
  — exactly as delivered by the enrichment bundle.
- `hosts` / `food_sources` `[{aphia_id, node_id, name}]` — the `Host` and
  `Food source` attributes hold AphiaIDs of *other* taxa; these are the ones
  that resolve to a node in this dataset.
- `_enriched: true`, `enrichment_source: "worms-rest-attributes"`.
- The 37 legacy `mollusca_*` records already carried a richer locality shape
  (MRGID, higher geography, establishment means). That list is preserved as
  `props.localities_detail`; `props.localities` is the uniform enrichment shape
  everywhere. Their legacy vernaculars are unioned with the API's.
- `description` (the store's BM25/embedding text, one fact per line) is
  extended with the new facts: vernacular names, locality names (non-`valid`
  records suffixed with their status), one `Trait: value; value` line per
  attribute (the constant `Nomenclature code` line is omitted), host/food taxa
  rendered as `Name [AphiaID n]`, and `NCBI Taxonomy ID` / `FishBase ID` lines.
  Lines already present in the original text are not repeated.

Concept nodes and non-Species taxa are copied unchanged. The 25 AphiaIDs that
the export carries twice (a rich `mollusca_*` record plus a bare `taxon_*`
hierarchy stub) stay as two records here, exactly as in the export; both get
the enrichment when they are Species.

### `store/edges.jsonl.gz` — `{id, kind, src, tgt, graph, phylum, attrs}`

All 144,129 original edges unchanged, then the derived edges:

| kind | direction | count | attrs |
|---|---|---|---|
| `has_host` | parasite/symbiont → host taxon | 6,138 | `{origin: "worms-rest-attributes", attribute: "Host", confidence: 1.0}` |
| `feeds_on` | consumer → food taxon | 13 | `{..., attribute: "Food source", ...}` |

Only references that resolve to a node in this dataset become edges (6,283
`Host` and 138 `Food source` references point at taxa outside it; they remain
available as attribute values and in the KG metadata). Edge ids are
`<kind>::<src>::<tgt>`, so re-loading is idempotent.

### `kg/nodes.jsonl.gz` — `{id, title, type, description?, aliases?, metadata, ...}`

Every record from the toolkit's `clean_worms_nodes.jsonl` with, for enriched
Species:

- `aliases` — existing aliases ∪ vernacular names (deduplicated). Titles are
  untouched, so the toolkit's title-collision handling still applies.
- `description` — same extension as the store text, capped at 5,000 chars as
  in `clean-worms.ts` (8 records hit the cap).
- `metadata` (string → string, the KG convention) — the full structures as
  JSON strings (`vernaculars`, `localities`, `attributes`, `external_ids`) plus
  flattened, filterable keys with multi-values joined by ` | `:
  `functional_group`, `feeding_method`, `trophic_level`, `life_stage`,
  `body_size_qualitative`, `body_size` / `body_size_unit` / `body_size_dimension`
  / `body_size_type` / `body_size_sex`, `iucn_category`, `iucn_year_assessed`,
  `iucn_criteria`, `ecological_interactions`, `development`, `mobility`,
  `ambi_group`, `sociability`, `composition`, `structure`,
  `supporting_structure`, `fossil_range`, `cites_annex`, `soniferous_behaviour`,
  `importance_to_society`, `paraphyletic_group`, `brooding`, `zonation`,
  `environmental_position`, `etymology`, `identifier`, `locality_mrgids`,
  `host_aphia_ids`, `food_source_aphia_ids`, `host_names`, `food_source_names`,
  `ncbi_id`, `fishbase_id`, `locality_count`, and `enrichment =
  "worms-rest-attributes"`. The full attribute → key map is in
  `vocab/attributes.json` under `kg_metadata_keys`.

Note that WoRMS delivers body-size measurements as parallel lists
(`Body size`, `Unit`, `Dimension`, `Type`, `Sex`) whose alignment is not
recoverable from the enrichment; they are kept as-is rather than re-paired.

### `kg/edges.jsonl.gz` — `{source, target, type, status, directed, confidence, evidenceDois?, metadata?}`

All 144,126 original edges unchanged, then the same 6,151 derived edges keyed
by KG uuid, with `status: "Established"`, `confidence: 1.0` and
`metadata: {origin, attribute}`.

**Two new edge types, `has_host` and `feeds_on`, must be registered** (Settings
UI, and the `EDGE_TYPES` allowlist in `migrate.ts`) before uploading
`kg/edges.jsonl.gz`. Suggested descriptions:

- `has_host` — "A parasitic or symbiotic taxon → the taxon it lives on or in
  (WoRMS `Host` attribute). Directed parasite → host."
- `feeds_on` — "A taxon → a taxon it is recorded feeding on (WoRMS `Food source`
  attribute). Directed consumer → food."

### `documents/papers.jsonl.gz`

`{id, doi, key, title, year, venue, authors, citation, url, abstract, sources, tags, cited_by}`

The 389 distinct evidence documents behind the graph, deduplicated by DOI, then
by source key, URL or citation: the 226 papers cited through `evidenced_by` on
concept nodes/edges and `papers` on taxa, plus the 163 `references` entries on
taxa that the KG format did not previously carry as documents. 63 have a DOI, 30
an abstract. `cited_by` lists `{node_id | edge_id, role}` with role
`evidence` (claims graph), `paper` or `reference` (taxa); ids are store ids.
KG `evidenceDois` values resolve to the documents whose `doi` matches.

### `documents/node_texts.jsonl.gz`

`{node_id, kg_id, canonical_name, node_type, kind, graph, aphia_id, aliases, definition, enriched}`

One document per store node with the complete merged text in `definition`
(uncapped). The field names are those `text2cypher.ingest.index_nodes` expects,
so the file can be embedded directly for grounding; `kg_id` is the K-Git uuid
(`null` for the 25 duplicate hierarchy stubs, which carry `kg_merged_into`
instead). Vectors are not included, by design — regenerate them with
`worms-graph embed` or `index_nodes`.

### `vocab/attributes.json`

Per-attribute distinct values with counts (58 attribute types; long-tailed ones
such as `Host`, `Identifier`, `Body size` are truncated to the top 300 and
flagged), the 4,207 distinct locality names, vernacular language codes, and
the KG metadata key map. Useful for grounding, filters and prompt hints.

## Loading

worms-graph store (Postgres → ArcadeDB → Qdrant), using the loader shipped in
the toolkit's `data/raw/`:

```bash
python load_backbone.py data/mollusca-merged/store --dry-run   # validates shape (passes)
python load_backbone.py data/mollusca-merged/store
worms-graph reproject && worms-graph embed
```

K-Git KG via the toolkit's `migrate.ts` (register `has_host` / `feeds_on`
first, and add them to `EDGE_TYPES`):

```bash
gunzip -k kg/nodes.jsonl.gz kg/edges.jsonl.gz
NODES_FILE=kg/nodes.jsonl EDGES_FILE=kg/edges.jsonl node --experimental-strip-types scripts/migrate.ts
```

For a KG that already holds the taxonomy, do not re-create: extract the
changed Species records and PATCH them (the playbook's §5 pattern), e.g.

```bash
zcat kg/nodes.jsonl.gz | grep '"enrichment":"worms-rest-attributes"' > kg/species_enriched.jsonl
```

then upload the derived edges alone (`grep -E '"type":"(has_host|feeds_on)"'`
over `kg/edges.jsonl`).

## Rebuilding

```bash
python data/mollusca-merged/build_merged.py \
    --kg-package /path/to/kg_mollusca-data-populate-script \
    --enriched   /path/to/worms-enriched-full \
    --out        data/mollusca-merged
```

The inputs are not committed (their names and SHA-256s are recorded in
`MANIFEST.json` under `sources`). Rebuilding from the same inputs reproduces
the same files byte for byte, except the `built` date in the manifest.
