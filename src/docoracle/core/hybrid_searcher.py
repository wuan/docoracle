"""Hybrid searcher composing semantic and BM25 indices with RRF fusion.

The ``HybridSearcher`` is the single retrieval entry point used by the
Q&A engine. It owns the canonical chunks list (the source of truth) and
delegates searching to two ``SearchIndex`` implementations: a FAISS-backed
``SemanticIndex`` and a ``BM25Index``. Results are fused via Reciprocal
Rank Fusion (RRF) so the two paths don't need score normalization.

On disk the hybrid searcher owns ``chunks.pkl`` and orchestrates load/save
of all three files (``chunks.pkl``, ``index.faiss``, ``bm25.pkl``) under a
single store directory.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .bm25_index import BM25Index
from .models import Chunk
from .search_index import ScoredChunk
from .semantic_index import SemanticIndex

VALID_MODES = ("hybrid", "semantic", "bm25")


class HybridSearcher:
    """Composes semantic + BM25 indices and fuses results via RRF."""

    def __init__(
        self,
        semantic: SemanticIndex | None = None,
        bm25: BM25Index | None = None,
        rrf_k: int = 60,
        prefetch_k: int | None = None,
    ) -> None:
        self.semantic = semantic or SemanticIndex()
        self.bm25 = bm25 or BM25Index()
        self.rrf_k = rrf_k
        self.prefetch_k = prefetch_k  # if None, callers compute it as 4 * top_k

        self.chunks: list[Chunk] = []
        self._chunk_dict: dict[str, Chunk] = {}

    # ------------------------------------------------------------------
    # Canonical chunk management
    # ------------------------------------------------------------------

    def _resolve_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunk_dict.get(chunk_id)

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]] | None = None,
    ) -> None:
        """Add chunks to the canonical store and to every configured index.

        ``embeddings`` is required when the semantic index is in use and
        ``chunks`` is non-empty (the BM25 path does not need embeddings).
        """
        if not chunks:
            return
        self.chunks.extend(chunks)
        for c in chunks:
            self._chunk_dict[c.chunk_id] = c

        self.bm25.add(chunks)
        if embeddings is not None:
            if len(embeddings) != len(chunks):
                raise ValueError("embeddings length must match chunks length")
            self.semantic.add(chunks, embeddings)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        mode: str = "hybrid",
    ) -> list[ScoredChunk]:
        """Search using ``mode`` (``hybrid``/``semantic``/``bm25``) and return top-``k`` results.

        ``query`` is the raw text used by BM25. ``query_embedding`` is the
        pre-computed embedding used by the semantic path. For ``hybrid``
        both must be provided; for ``semantic`` only ``query_embedding``
        is required; for ``bm25`` only ``query`` is required.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid retrieval mode {mode!r}. Must be one of {VALID_MODES}.")
        if not self.chunks:
            return []

        prefetch = self.prefetch_k if self.prefetch_k is not None else max(k * 4, k)

        if mode == "semantic":
            if query_embedding is None:
                raise ValueError("query_embedding is required for mode='semantic'")
            return self.semantic.search(
                query_embedding, k=k, filters=filters, chunk_resolver=self._resolve_chunk
            )

        if mode == "bm25":
            return self.bm25.search(query, k=k, filters=filters, chunk_resolver=self._resolve_chunk)

        # mode == "hybrid"
        semantic_results: list[ScoredChunk] = []
        bm25_results: list[ScoredChunk] = []
        if query_embedding is not None:
            semantic_results = self.semantic.search(
                query_embedding,
                k=prefetch,
                filters=filters,
                chunk_resolver=self._resolve_chunk,
            )
        bm25_results = self.bm25.search(
            query,
            k=prefetch,
            filters=filters,
            chunk_resolver=self._resolve_chunk,
        )

        fused = self._rrf_fuse(semantic_results, bm25_results)
        return fused[:k]

    def _rrf_fuse(
        self,
        semantic_results: list[ScoredChunk],
        bm25_results: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        """Combine two ranked result lists using Reciprocal Rank Fusion.

        For each unique chunk id across both lists, the RRF score is

            rrf_score(d) = Σ_i  1 / (rrf_k + rank_i(d))

        where ``rank_i(d)`` is the 1-based rank of ``d`` in retriever
        ``i``'s list (a chunk absent from a list contributes 0).
        """
        scores: dict[str, float] = {}
        # Semantic-side contribution. ``ScoredChunk.sources["semantic"]``
        # is the 1-based rank assigned by ``SemanticIndex.search``.
        for r in semantic_results:
            rank = r.sources.get("semantic")
            if rank is None:
                continue
            scores[r.chunk.chunk_id] = scores.get(r.chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        # BM25-side contribution.
        for r in bm25_results:
            rank = r.sources.get("bm25")
            if rank is None:
                continue
            scores[r.chunk.chunk_id] = scores.get(r.chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        fused: list[ScoredChunk] = []
        for chunk_id, score in scores.items():
            chunk = self._chunk_dict.get(chunk_id)
            if chunk is None:
                continue
            sources: dict[str, int] = {}
            for r in semantic_results:
                if r.chunk.chunk_id == chunk_id:
                    rank = r.sources.get("semantic")
                    if rank is not None:
                        sources["semantic"] = rank
            for r in bm25_results:
                if r.chunk.chunk_id == chunk_id:
                    rank = r.sources.get("bm25")
                    if rank is not None:
                        sources["bm25"] = rank
            fused.append(ScoredChunk(chunk=chunk, score=score, sources=sources))

        fused.sort(key=lambda r: r.score, reverse=True)
        return fused

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, store_dir: str) -> None:
        """Persist chunks, semantic index, and BM25 index under ``store_dir``."""
        store_path = Path(store_dir)
        store_path.mkdir(parents=True, exist_ok=True)
        with open(store_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        if self.semantic.index is not None:
            self.semantic.save(store_path)
        if len(self.bm25) > 0:
            self.bm25.save(store_path)

    def load(self, store_dir: str) -> bool:
        """Restore from ``store_dir``. Returns True if everything loaded.

        If ``chunks.pkl`` exists but cannot be loaded as the current
        ``Chunk`` schema (e.g. stale pickle from a previous code
        version), a warning is printed and ``False`` is returned so the
        caller can re-ingest instead of crashing mid-load.
        """
        store_path = Path(store_dir)
        chunks_file = store_path / "chunks.pkl"
        if not chunks_file.exists():
            return False
        try:
            with open(chunks_file, "rb") as f:
                raw = pickle.load(f)
            # Re-validate through Pydantic so missing/renamed fields fail
            # loudly here (with a precise ValidationError) instead of
            # surfacing as a cryptic AttributeError on first access.
            self.chunks = [Chunk.model_validate(c) for c in raw]
            self._chunk_dict = {c.chunk_id: c for c in self.chunks}
        except (ValidationError, AttributeError, TypeError, pickle.UnpicklingError) as e:
            print(
                f"Warning: chunks.pkl at {store_path} is incompatible with the "
                f"current Chunk schema ({type(e).__name__}: {e}). "
                f"Re-ingest to rebuild the store.",
                file=sys.stderr,
            )
            self.chunks = []
            self._chunk_dict = {}
            return False

        # Semantic and BM25 are best-effort: if only some files exist the
        # caller may need to re-ingest. The hybrid search reports the
        # situation via length() per index rather than failing here.
        if (store_path / "index.faiss").exists():
            self.semantic.load(store_path)
        if (store_path / "bm25.pkl").exists():
            self.bm25.load(store_path)
        return True
