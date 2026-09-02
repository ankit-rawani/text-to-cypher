"""Stage 2 — EntityGrounder.

Extract entity mentions from the question, embed them, search the node-vector
collection, and resolve them to ``canonical_name`` / node ids with similarity
scores. The output is a set of **parameter bindings** — grounded values always
travel as ``$params``, never as inlined strings (spec acceptance criterion 3).

Mention extraction is deterministic by default (quoted spans, capitalized runs,
content-word n-grams) so grounding works offline and reproducibly; an optional
callable can swap in an LLM-based extractor.
"""

from __future__ import annotations

import re
from typing import Callable

from ..clients.base import VectorSearchClient
from ..contracts import EntityCandidate, GroundedEntity, GroundingResult, SchemaContext
from ..embeddings.base import BaseEmbedder

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
_QUOTED_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'')

_STOPWORDS = {
    "which", "what", "who", "whom", "whose", "where", "when", "why", "how",
    "list", "show", "find", "give", "get", "tell", "return", "count", "many",
    "concept", "concepts", "node", "nodes", "entity", "entities", "related",
    "relate", "relates", "relation", "link", "links", "linked", "connect",
    "connects", "connected", "between", "and", "or", "the", "a", "an", "to",
    "of", "in", "for", "with", "that", "this", "these", "those", "are", "is",
    "was", "were", "be", "do", "does", "did", "by", "on", "from", "all", "any",
    "have", "has", "via", "through", "about", "into", "than", "then", "at",
    "as", "it", "its", "their", "there", "each", "some", "most", "more",
}

# Generic / schema-structural words that are not entities. Dropped ONLY as a
# standalone single-word mention (multi-word phrases that contain them survive),
# so we don't ground "relationships"/"type"/"method" to a node that merely
# contains the word — which produces spurious $params that mislead weak models.
_GENERIC = {
    "relationship", "relationships", "type", "types", "kind", "kinds",
    "effect", "effects", "method", "methods", "property", "properties",
    "process", "processes", "model", "models", "phenomenon", "phenomena",
    "thing", "things", "item", "items", "name", "names", "value", "values",
    "result", "results", "number", "numbers", "pair", "pairs", "group", "groups",
    # research/query meta words — never domain entities (they otherwise produce
    # borderline embedder false-matches, e.g. "contradictions" -> some node).
    "contradiction", "contradictions", "evidence", "finding", "findings",
    "claim", "claims", "hypothesis", "hypotheses", "understanding", "gap", "gaps",
    "category", "categories", "chain", "chains", "concept", "concepts",
}
_DROP_SINGLE = _STOPWORDS | _GENERIC

MentionExtractor = Callable[[str], "list[str]"]


def extract_mentions(question: str, max_candidates: int = 40) -> list[str]:
    """Deterministically produce candidate entity mentions (over-generates)."""
    seen: set[str] = set()
    ordered: list[str] = []

    def add(m: str) -> None:
        m = m.strip()
        if len(m) < 2:
            return
        key = m.lower()
        if key in seen:
            return
        # Drop a bare generic/stopword single-word mention (keep multi-word).
        if " " not in m and key in _DROP_SINGLE:
            return
        seen.add(key)
        ordered.append(m)

    # 1. Quoted spans (highest signal)
    for a, b in _QUOTED_RE.findall(question):
        add(a or b)

    tokens = [(m.group(0), m.start(), m.end()) for m in _WORD_RE.finditer(question)]
    words = [t[0] for t in tokens]

    # 2. Capitalized / acronym runs (proper-noun-ish phrases)
    run: list[str] = []
    for w in words:
        is_propery = (w[0].isupper() or any(ch.isdigit() for ch in w)) and w.lower() not in _STOPWORDS
        if is_propery:
            run.append(w)
        else:
            if run:
                add(" ".join(run))
                run = []
    if run:
        add(" ".join(run))

    # 3. Content-word n-grams (1..3) over non-stopword tokens
    content_idx = [i for i, w in enumerate(words) if w.lower() not in _STOPWORDS and len(w) > 1]
    for n in (3, 2, 1):
        for start in range(len(content_idx) - n + 1):
            idxs = content_idx[start:start + n]
            # require contiguity in the original token stream
            if idxs[-1] - idxs[0] != n - 1:
                continue
            add(" ".join(words[i] for i in idxs))

    return ordered[:max_candidates]


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not s:
        s = re.sub(r"[^a-z0-9]+", "_", fallback.lower()).strip("_") or "entity"
    if s[0].isdigit():
        s = "e_" + s
    return s[:40]


class EntityGrounder:
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorSearchClient,
        collection: str,
        *,
        sim_threshold: float = 0.55,
        max_candidates: int = 3,
        low_confidence_threshold: float = 0.65,
        enabled: bool = True,
        mention_extractor: MentionExtractor | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._collection = collection
        self._threshold = sim_threshold
        self._max_candidates = max_candidates
        self._low_conf = low_confidence_threshold
        self._enabled = enabled
        self._extract = mention_extractor or extract_mentions

    def ground(self, question: str, schema: SchemaContext | None = None) -> GroundingResult:
        if not self._enabled:
            return GroundingResult()

        mentions = self._extract(question)
        raw_entities: list[GroundedEntity] = []
        for mention in mentions:
            vector = self._embedder.embed_one(mention)
            hits = self._store.search(self._collection, vector, limit=self._max_candidates)
            above = [h for h in hits if h.score >= self._threshold]
            if not above:
                continue
            candidates = [
                EntityCandidate(
                    node_id=_str_or_none(h.payload.get("node_id", h.id)),
                    canonical_name=h.payload.get("canonical_name"),
                    node_type=h.payload.get("node_type"),
                    similarity=h.score,
                    aliases=list(h.payload.get("aliases", []) or []),
                )
                for h in above
            ]
            best = candidates[0]
            raw_entities.append(
                GroundedEntity(
                    mention=mention,
                    node_id=best.node_id,
                    canonical_name=best.canonical_name,
                    similarity=best.similarity,
                    ambiguous=len(candidates) > 1
                    and (candidates[0].similarity - candidates[1].similarity) < 0.05,
                    candidates=candidates,
                )
            )

        entities = self._dedupe(raw_entities)

        params: dict[str, object] = {}
        used_names: set[str] = set()
        for ent in entities:
            base = _slug(ent.canonical_name or ent.mention, ent.mention)
            name = base
            i = 1
            while name in used_names:
                name = f"{base}_{i}"
                i += 1
            used_names.add(name)
            ent.param_name = name
            value = ent.canonical_name if ent.canonical_name is not None else ent.node_id
            if value is not None:
                params[name] = value

        return GroundingResult(entities=entities, params=params)

    def _dedupe(self, entities: list[GroundedEntity]) -> list[GroundedEntity]:
        """Keep the best mention per resolved node; drop shorter overlaps."""
        entities = sorted(entities, key=lambda e: (e.similarity, len(e.mention)), reverse=True)
        kept: list[GroundedEntity] = []
        seen_nodes: set[str] = set()
        for ent in entities:
            key = ent.node_id or (ent.canonical_name or "").lower()
            if key and key in seen_nodes:
                continue
            # Skip a mention that is a substring of an already-kept mention ONLY
            # when they resolve to the same node (or this one is unresolved).
            # Distinct nodes whose mentions overlap ("Tesla" vs "Tesla Motors")
            # must both be kept so each gets a bound param.
            ml = ent.mention.lower()
            if any(
                ml in k.mention.lower()
                and ml != k.mention.lower()
                and (k.node_id == ent.node_id or ent.node_id is None)
                for k in kept
            ):
                continue
            if key:
                seen_nodes.add(key)
            kept.append(ent)
        # stable order by first appearance in the question is nicer, but score order is fine
        return kept


def _str_or_none(v) -> str | None:
    if v is None:
        return None
    return str(v)


__all__ = ["EntityGrounder", "extract_mentions"]
