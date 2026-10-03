"""Shared abstractions for the retrieval layer.

Defines:
- ``ScoredChunk``: the unified result type returned by every ``SearchIndex``
- ``SearchIndex``: a Protocol every index implementation must satisfy
- ``matches_filters``: the metadata filter primitive, lifted out of the
  legacy ``VectorStore`` so semantic and BM25 paths share the exact same
  semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .models import Chunk


class ScoredChunk(BaseModel):
    """A chunk returned from a search, with a score and per-retriever provenance.

    Replaces the previous dataclass implementation with Pydantic BaseModel.

    ``score`` is the fused score (RRF) when produced by ``HybridSearcher``,
    or the native score of the index that produced it (BM25 or converted
    semantic similarity) when produced by a single-path search.

    ``sources`` maps retriever name to the 1-based rank at which the chunk
    was found by that retriever. A chunk found by only one retriever has a
    single entry; a chunk found by both has two.
    """

    chunk: Chunk = Field(..., description="The actual chunk")
    score: float = Field(..., description="Fused or native score")
    sources: dict[str, int] = Field(
        default_factory=dict, description="Retriever name -> 1-based rank"
    )


class SearchIndex(Protocol):
    """The contract every retrieval index must satisfy.

    Indices are projections over the canonical chunks list owned by the
    ``HybridSearcher``. They must not maintain their own mutable copy of
    the chunks.
    """

    def add(self, chunks: list[Chunk]) -> None:
        """Index the given chunks so they become searchable."""
        ...

    def search(
        self,
        query: Any,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Return up to ``k`` results ranked by this index's native scoring.

        ``filters`` is a metadata filter dict (e.g. ``{"module": "api"}``)
        using the same semantics for every index.
        """
        ...

    def save(self, path: Path) -> None:
        """Persist this index's state under ``path``."""
        ...

    def load(self, path: Path) -> bool:
        """Restore this index's state from ``path``. Returns True on success."""
        ...


def matches_filters(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
    """Return True iff ``chunk`` matches every entry in ``filters``.

    Each filter entry maps a chunk attribute name (or a metadata key) to an
    expected value. Values may be a scalar (equality) or a list (membership).
    Multiple filter entries combine with AND semantics.
    """
    if not filters:
        return True
    for key, value in filters.items():
        chunk_value = getattr(chunk, key, None)
        if chunk_value is None:
            chunk_value = chunk.metadata.get(key)
        if isinstance(value, list):
            if chunk_value not in value:
                return False
        else:
            if chunk_value != value:
                return False
    return True
