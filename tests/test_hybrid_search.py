"""Fixture-corpus tests for HybridSearcher.search end-to-end behavior."""

import uuid

import numpy as np
import pytest

from docoracle.core.hybrid_searcher import HybridSearcher
from docoracle.core.models import Chunk


@pytest.fixture
def fixture_searcher():
    """A small hybrid searcher with both indices populated.

    Chunks are deliberately constructed so that:
    - chunk 0 contains an exact term that BM25 should match strongly
    - chunk 1 contains a stem-equivalent that BM25 should also match
    - chunk 2 is unrelated to "konfiguration" (distractor)
    - chunk 3 is unrelated to "konfiguration" (distractor)
    """
    texts = [
        "Konfiguration der Datenbankverbindung in der config.yaml",
        "Beispiel für eine einfache Konfigurationsdatei",
        "Installationsanleitung für PostgreSQL auf Linux",
        "Allgemeine Hinweise zur Verwendung der API",
    ]
    chunks = [
        Chunk(
            text=t,
            chunk_id=str(uuid.uuid4()),
            module="m",
            component="c",
            version="v",
            page_id="p",
        )
        for t in texts
    ]
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(len(chunks), 64)).tolist()

    hs = HybridSearcher()
    hs.add_chunks(chunks, embeddings)
    return hs, chunks, embeddings, rng


def test_bm25_mode_finds_keyword_exactly(fixture_searcher):
    hs, chunks, _, _ = fixture_searcher
    results = hs.search(query="konfiguration", k=3, mode="bm25")
    assert len(results) > 0
    # The chunk containing the exact stem should be at or near the top.
    top_texts = [r.chunk.text for r in results]
    assert any("Konfiguration" in t for t in top_texts)


def test_hybrid_mode_returns_results_with_provenance(fixture_searcher):
    hs, chunks, _, rng = fixture_searcher
    results = hs.search(
        query="konfiguration",
        query_embedding=rng.normal(size=64),
        k=3,
        mode="hybrid",
    )
    assert len(results) > 0
    for r in results:
        assert r.chunk in chunks
        # At least one of the two retrievers should have contributed.
        assert "semantic" in r.sources or "bm25" in r.sources


def test_invalid_retrieval_mode_raises():
    hs = HybridSearcher()
    with pytest.raises(ValueError, match="Invalid retrieval mode"):
        hs.search(query="x", mode="nonsense")


def test_empty_searcher_returns_empty_list():
    hs = HybridSearcher()
    assert hs.search(query="anything", mode="bm25") == []
    assert hs.search(query="anything", mode="semantic", query_embedding=[0.0] * 4) == []
    assert hs.search(query="anything", mode="hybrid", query_embedding=[0.0] * 4) == []


def test_search_respects_module_filter(fixture_searcher):
    hs, chunks, _, _ = fixture_searcher
    # Mark chunk 1 with a unique module
    chunks[1].module = "special"
    hs._chunk_dict[chunks[1].chunk_id] = chunks[1]

    results = hs.search(query="konfiguration", k=10, mode="bm25", filters={"module": "special"})
    assert all(r.chunk.module == "special" for r in results)


def test_search_respects_module_filter_across_both_paths(fixture_searcher):
    hs, chunks, _, rng = fixture_searcher
    chunks[0].module = "special"
    hs._chunk_dict[chunks[0].chunk_id] = chunks[0]

    results = hs.search(
        query="konfiguration",
        query_embedding=rng.normal(size=64),
        k=10,
        mode="hybrid",
        filters={"module": "special"},
    )
    assert all(r.chunk.module == "special" for r in results)
