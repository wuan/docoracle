"""FAISS-backed semantic index.

Stores embedding vectors and retrieves chunks by L2 nearest-neighbor
search. The index does not own the canonical chunks list; it holds only
the mapping ``position -> chunk_id`` so the ``HybridSearcher`` can resolve
``ScoredChunk`` results back to the canonical chunks it owns.

On disk this class owns exactly one file: ``index.faiss`` (plus an
internal ``_id_map.pkl`` sidecar so chunk IDs survive FAISS position
reassignments).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

from .models import Chunk
from .search_index import ScoredChunk, matches_filters


class SemanticIndex:
    """FAISS-backed semantic similarity index."""

    def __init__(self) -> None:
        self.index: Any = None
        self._chunk_ids: list[str] = []
        self._nlist: int = 0

    # ------------------------------------------------------------------
    # SearchIndex interface
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Add chunks along with their pre-computed embedding vectors.

        Both lists must be the same length. ``chunks`` are referenced by ID;
        their full text/attributes are not stored here.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return

        vectors = np.array(embeddings, dtype=np.float32)

        if self.index is None:
            nlist = min(100, len(chunks))
            self._init_index(vectors.shape[1], nlist=nlist)

        if isinstance(self.index, faiss.IndexIVFFlat):  # type: ignore[arg-type]
            self._add_to_ivf(vectors)
        else:
            self.index.add(vectors)  # type: ignore[arg-type]

        self._chunk_ids.extend(c.chunk_id for c in chunks)

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
        chunk_resolver: Any | None = None,
    ) -> list[ScoredChunk]:
        """Return up to ``k`` chunks ranked by L2 distance to ``query_embedding``.

        ``chunk_resolver`` is a callable ``chunk_id -> Chunk`` (typically the
        ``HybridSearcher``) used to look up the canonical chunk for each
        returned ID. Without it, only chunk IDs are available and we return
        ``ScoredChunk(chunk=None, ...)`` which the searcher should not do in
        production — this is a fallback for tests.
        """
        if self.index is None or not self._chunk_ids:
            return []

        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query, k * 10)  # type: ignore[arg-type]

        results: list[ScoredChunk] = []
        for idx, dist in zip(indices[0], distances[0], strict=True):  # type: ignore[arg-type]
            if idx < 0 or idx >= len(self._chunk_ids):
                continue
            chunk_id = self._chunk_ids[idx]  # type: ignore[index]

            chunk: Chunk | None = None
            if chunk_resolver is not None:
                chunk = chunk_resolver(chunk_id)  # type: ignore[assignment]
            if chunk is None and not callable(chunk_resolver):
                continue

            if filters and chunk is not None and not matches_filters(chunk, filters):
                continue

            if chunk is not None:
                results.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=self._distance_to_score(float(dist)),
                        sources={"semantic": len(results) + 1},
                    )
                )

            if len(results) >= k:
                break

        results.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(results, 1):
            r.sources["semantic"] = i
        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise ValueError("No index to save. Call add() first.")
        faiss.write_index(self.index, str(path / "index.faiss"))  # type: ignore[arg-type]
        with open(path / "_id_map.pkl", "wb") as f:
            pickle.dump({"chunk_ids": self._chunk_ids, "nlist": self._nlist}, f)

    def load(self, path: Path) -> bool:
        path = Path(path)
        index_file = path / "index.faiss"
        id_map_file = path / "_id_map.pkl"
        if not index_file.exists() or not id_map_file.exists():
            return False
        self.index = faiss.read_index(str(index_file))  # type: ignore[arg-type]
        with open(id_map_file, "rb") as f:
            data = pickle.load(f)
        self._chunk_ids = data.get("chunk_ids", [])
        self._nlist = data.get("nlist", 0)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chunk_ids)

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        """Convert L2 distance to a similarity-like score (higher = better).

        Used so that ``ScoredChunk.score`` has a consistent direction across
        both retrieval paths. RRF fusion only looks at ranks, but the
        standalone ``--retrieval semantic`` path benefits from a readable
        score.
        """
        return 1.0 / (1.0 + distance)

    def _init_index(self, dimension: int, nlist: int) -> None:
        if nlist <= 1:
            self.index = faiss.IndexFlatL2(dimension)  # type: ignore[arg-type]
        else:
            quantizer = faiss.IndexFlatL2(dimension)  # type: ignore[arg-type]
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)  # type: ignore[arg-type]
        self._nlist = nlist

    def _add_to_ivf(self, vectors: np.ndarray) -> None:
        assert isinstance(self.index, faiss.IndexIVFFlat)  # type: ignore[arg-type]
        if not self.index.is_trained:  # type: ignore[arg-type]
            total = self.index.ntotal + len(vectors)  # type: ignore[arg-type]
            if total >= self._nlist:
                if self.index.ntotal == 0:  # type: ignore[arg-type]
                    self.index.train(vectors)  # type: ignore[arg-type]
                else:
                    existing = self.index.reconstruct_n(0, self.index.ntotal)  # type: ignore[arg-type]
                    self.index.train(np.vstack([existing, vectors]))  # type: ignore[arg-type]
                self.index.add(vectors)  # type: ignore[arg-type]
            else:
                old = self.index
                self.index = faiss.IndexFlatL2(old.d)  # type: ignore[arg-type]
                if old.ntotal > 0:  # type: ignore[arg-type]
                    self.index.add(old.reconstruct_n(0, old.ntotal))  # type: ignore[arg-type]
                self.index.add(vectors)  # type: ignore[arg-type]
        else:
            self.index.add(vectors)  # type: ignore[arg-type]
