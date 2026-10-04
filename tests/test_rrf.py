"""Tests for Reciprocal Rank Fusion in HybridSearcher."""

import uuid

from docoracle.core.hybrid_searcher import HybridSearcher
from docoracle.core.models import Chunk
from docoracle.core.search_index import ScoredChunk


def _make_chunk(text: str) -> Chunk:
    return Chunk(
        text=text,
        chunk_id=str(uuid.uuid4()),
        module="m",
        component="c",
        version="v",
        page_id="p",
    )


def _seed(hs: HybridSearcher, *chunks: Chunk) -> None:
    """Add chunks to the canonical store without invoking the indices."""
    for c in chunks:
        hs.chunks.append(c)
        hs._chunk_dict[c.chunk_id] = c


def test_chunk_in_both_lists_sums_both_contributions():
    """rrf = 1/(60+rank_sem) + 1/(60+rank_bm25) when both find the chunk."""
    hs = HybridSearcher()
    rrf_k = 60
    hs.rrf_k = rrf_k

    chunk = _make_chunk("hello")
    _seed(hs, chunk)
    sem = [ScoredChunk(chunk=chunk, score=0.9, sources={"semantic": 1})]
    bm = [ScoredChunk(chunk=chunk, score=1.0, sources={"bm25": 3})]

    fused = hs._rrf_fuse(sem, bm)
    assert len(fused) == 1
    expected = 1.0 / (rrf_k + 1) + 1.0 / (rrf_k + 3)
    assert abs(fused[0].score - expected) < 1e-9
    assert fused[0].sources == {"semantic": 1, "bm25": 3}


def test_chunk_in_only_one_list_contributes_one_term():
    hs = HybridSearcher()
    rrf_k = 60
    hs.rrf_k = rrf_k

    chunk = _make_chunk("hello")
    _seed(hs, chunk)
    sem: list = []
    bm = [ScoredChunk(chunk=chunk, score=1.0, sources={"bm25": 5})]

    fused = hs._rrf_fuse(sem, bm)
    assert len(fused) == 1
    expected = 1.0 / (rrf_k + 5)
    assert abs(fused[0].score - expected) < 1e-9
    assert fused[0].sources == {"bm25": 5}


def test_custom_rrf_k_changes_damping():
    hs = HybridSearcher(rrf_k=30)
    chunk = _make_chunk("hi")
    _seed(hs, chunk)
    sem = [ScoredChunk(chunk=chunk, score=0.9, sources={"semantic": 1})]
    bm = [ScoredChunk(chunk=chunk, score=1.0, sources={"bm25": 1})]

    fused = hs._rrf_fuse(sem, bm)
    expected = 1.0 / 31 + 1.0 / 31
    assert abs(fused[0].score - expected) < 1e-9


def test_fused_results_are_sorted_descending_by_score():
    hs = HybridSearcher()
    c1, c2, c3 = _make_chunk("a"), _make_chunk("b"), _make_chunk("c")
    _seed(hs, c1, c2, c3)
    sem = [
        ScoredChunk(chunk=c1, score=0.9, sources={"semantic": 1}),
        ScoredChunk(chunk=c2, score=0.8, sources={"semantic": 2}),
    ]
    bm = [
        ScoredChunk(chunk=c2, score=1.0, sources={"bm25": 1}),
        ScoredChunk(chunk=c3, score=0.9, sources={"bm25": 2}),
    ]
    fused = hs._rrf_fuse(sem, bm)
    # c2 found by both should rank highest; c1 by semantic only; c3 by bm25 only
    assert [r.chunk.chunk_id for r in fused[:3]] == [c2.chunk_id, c1.chunk_id, c3.chunk_id]


def test_fusion_dedups_chunks_present_in_both_lists():
    hs = HybridSearcher()
    chunk = _make_chunk("shared")
    _seed(hs, chunk)
    sem = [ScoredChunk(chunk=chunk, score=0.9, sources={"semantic": 1})]
    bm = [ScoredChunk(chunk=chunk, score=1.0, sources={"bm25": 1})]
    fused = hs._rrf_fuse(sem, bm)
    assert len(fused) == 1


def test_weak_bm25_tail_is_dropped_before_fusion():
    """BM25 results far below the best BM25 score get no RRF vote."""
    hs = HybridSearcher(bm25_min_score_ratio=0.5)
    sem_chunk, weak_chunk = _make_chunk("sem"), _make_chunk("weak")
    _seed(hs, sem_chunk, weak_chunk)
    sem = [ScoredChunk(chunk=sem_chunk, score=0.9, sources={"semantic": 1})]
    bm = [
        ScoredChunk(chunk=sem_chunk, score=8.0, sources={"bm25": 1}),
        ScoredChunk(chunk=weak_chunk, score=1.0, sources={"bm25": 2}),
    ]
    fused = hs._rrf_fuse(sem, bm)
    by_id = {r.chunk.chunk_id: r for r in fused}
    # weak_chunk scores 1/8 of the BM25 top -> filtered out, semantic only
    assert weak_chunk.chunk_id not in by_id or by_id[weak_chunk.chunk_id].sources == {"semantic": 1}
    assert by_id[sem_chunk.chunk_id].sources == {"semantic": 1, "bm25": 1}


def test_zero_score_bm25_list_contributes_nothing():
    """An all-zero BM25 list (no token overlap) is dropped entirely."""
    hs = HybridSearcher(bm25_min_score_ratio=0.5)
    sem_chunk, junk_chunk = _make_chunk("sem"), _make_chunk("junk")
    _seed(hs, sem_chunk, junk_chunk)
    sem = [ScoredChunk(chunk=sem_chunk, score=0.9, sources={"semantic": 1})]
    bm = [
        ScoredChunk(chunk=junk_chunk, score=0.0, sources={"bm25": 1}),
        ScoredChunk(chunk=sem_chunk, score=0.0, sources={"bm25": 2}),
    ]
    fused = hs._rrf_fuse(sem, bm)
    assert len(fused) == 1
    assert fused[0].chunk.chunk_id == sem_chunk.chunk_id
    assert fused[0].sources == {"semantic": 1}


def test_bm25_min_score_ratio_zero_disables_filter():
    hs = HybridSearcher(bm25_min_score_ratio=0.0)
    sem_chunk, weak_chunk = _make_chunk("sem"), _make_chunk("weak")
    _seed(hs, sem_chunk, weak_chunk)
    sem = [ScoredChunk(chunk=sem_chunk, score=0.9, sources={"semantic": 1})]
    bm = [
        ScoredChunk(chunk=sem_chunk, score=8.0, sources={"bm25": 1}),
        ScoredChunk(chunk=weak_chunk, score=1.0, sources={"bm25": 2}),
    ]
    fused = hs._rrf_fuse(sem, bm)
    assert len(fused) == 2
