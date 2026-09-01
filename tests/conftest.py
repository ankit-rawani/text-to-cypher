"""Shared test fixtures."""

from __future__ import annotations

import pytest

from text2cypher.testing import build_fake_pipeline, gen_json  # noqa: F401 (re-export)


@pytest.fixture(autouse=True)
def _hermetic_config(monkeypatch):
    """Keep tests independent of any developer's local `.env` — never autoload it."""
    monkeypatch.setattr("text2cypher.config._autoload_dotenv", lambda: None)

SAMPLE_NODES = [
    {
        "node_id": "n1",
        "canonical_name": "Type 2 Diabetes Mellitus",
        "node_type": "Disease",
        "aliases": ["T2DM", "adult-onset diabetes", "type 2 diabetes"],
        "definition": "a chronic metabolic disorder of high blood glucose",
    },
    {
        "node_id": "n2",
        "canonical_name": "Metformin",
        "node_type": "Drug",
        "aliases": ["Glucophage"],
        "definition": "first-line oral antidiabetic medication",
    },
    {
        "node_id": "n3",
        "canonical_name": "Insulin Resistance",
        "node_type": "Phenotype",
        "aliases": ["insulin insensitivity"],
        "definition": "reduced cellular response to insulin",
    },
    {
        "node_id": "n4",
        "canonical_name": "TCF7L2",
        "node_type": "Gene",
        "aliases": ["transcription factor 7 like 2"],
        "definition": "a gene strongly associated with type 2 diabetes risk",
    },
]


@pytest.fixture
def sample_nodes():
    return [dict(n) for n in SAMPLE_NODES]


@pytest.fixture
def make_pipeline(sample_nodes):
    def _make(llm_responses=None, llm_handler=None, **kwargs):
        return build_fake_pipeline(
            nodes=sample_nodes,
            llm_responses=llm_responses,
            llm_handler=llm_handler,
            grounding_threshold=kwargs.pop("grounding_threshold", 0.2),
            **kwargs,
        )

    return _make
