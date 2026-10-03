"""On-disk round-trip tests for HybridSearcher."""

import tempfile
import uuid
from pathlib import Path

import numpy as np

from docoracle.core.hybrid_searcher import HybridSearcher
from docoracle.core.models import Chunk


def _make_corpus(n: int = 5):
    rng = np.random.default_rng(42)
    chunks = [
        Chunk(
            text=f"chunk {i} content about something",
            chunk_id=str(uuid.uuid4()),
            module=f"m{i % 2}",
            component="c",
            version="v",
            page_id="p",
        )
        for i in range(n)
    ]
    embeddings = rng.normal(size=(n, 16)).tolist()
    return chunks, embeddings, rng


def test_save_and_load_roundtrip_preserves_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        chunks, embeddings, _ = _make_corpus()
        hs = HybridSearcher()
        hs.add_chunks(chunks, embeddings)
        hs.save(tmp)

        hs2 = HybridSearcher()
        loaded = hs2.load(tmp)
        assert loaded
        assert len(hs2.chunks) == len(chunks)
        assert {c.chunk_id for c in hs2.chunks} == {c.chunk_id for c in chunks}


def test_save_and_load_roundtrip_preserves_search_results():
    with tempfile.TemporaryDirectory() as tmp:
        chunks, embeddings, rng = _make_corpus()
        hs = HybridSearcher()
        hs.add_chunks(chunks, embeddings)

        # Use the SAME query embedding for both before and after so the
        # FAISS search produces deterministic results.
        query_emb = rng.normal(size=16)
        before = hs.search(
            query="chunk 1",
            query_embedding=query_emb,
            k=3,
            mode="hybrid",
        )
        hs.save(tmp)

        hs2 = HybridSearcher()
        hs2.load(tmp)
        after = hs2.search(
            query="chunk 1",
            query_embedding=query_emb,
            k=3,
            mode="hybrid",
        )

        assert len(before) == len(after)
        assert [r.chunk.chunk_id for r in before] == [r.chunk.chunk_id for r in after]


def test_save_writes_three_files():
    with tempfile.TemporaryDirectory() as tmp:
        chunks, embeddings, _ = _make_corpus()
        hs = HybridSearcher()
        hs.add_chunks(chunks, embeddings)
        hs.save(tmp)

        store = Path(tmp)
        assert (store / "chunks.pkl").exists()
        assert (store / "index.faiss").exists()
        assert (store / "bm25.pkl").exists()


def test_load_returns_false_when_no_chunks_file():
    with tempfile.TemporaryDirectory() as tmp:
        hs = HybridSearcher()
        assert hs.load(tmp) is False


def test_bm25_search_works_after_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        chunks, embeddings, _ = _make_corpus()
        hs = HybridSearcher()
        hs.add_chunks(chunks, embeddings)
        hs.save(tmp)

        hs2 = HybridSearcher()
        hs2.load(tmp)
        results = hs2.search(query="chunk", k=3, mode="bm25")
        assert len(results) > 0


def test_load_returns_false_on_incompatible_chunks_pickle(capsys):
    """Stale chunks.pkl from a previous Chunk schema must not crash load().

    Regression test: previously, loading a pickle whose Chunks lacked
    ``chunk_id`` raised a bare AttributeError deep inside the searcher.
    Now load() should detect the mismatch, warn, and return False so
    the CLI can re-ingest.
    """
    import pickle

    from docoracle.core.models import Chunk

    with tempfile.TemporaryDirectory() as tmp:
        # Build a Chunk via the current schema, then strip chunk_id from
        # its __dict__ to simulate a pickle produced by an older code
        # version where the field was absent.
        stale = Chunk(
            text="stale",
            chunk_id="ignored",
            module="m",
            component="c",
            version="v",
            page_id="p",
        )
        del stale.__dict__["chunk_id"]

        with open(Path(tmp) / "chunks.pkl", "wb") as f:
            pickle.dump([stale], f)

        hs = HybridSearcher()
        loaded = hs.load(tmp)

        assert loaded is False
        assert hs.chunks == []
        assert hs._chunk_dict == {}
        captured = capsys.readouterr()
        assert "incompatible" in captured.err.lower()


def test_load_returns_false_on_garbage_chunks_pickle(capsys):
    """A corrupted chunks.pkl must also fail gracefully."""
    import pickle

    with tempfile.TemporaryDirectory() as tmp:
        with open(Path(tmp) / "chunks.pkl", "wb") as f:
            pickle.dump([1, 2, 3], f)  # ints, not Chunks

        hs = HybridSearcher()
        loaded = hs.load(tmp)

        assert loaded is False
        assert hs.chunks == []
        captured = capsys.readouterr()
        assert "incompatible" in captured.err.lower()
