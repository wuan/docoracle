"""Shared contract and result shaping for answer backends.

Every answer backend (the deterministic engine and the pydantic-ai agent)
implements :class:`AnswerBackend` and produces its detailed result through
:func:`build_answer_result` so callers see one consistent shape regardless of
which backend is configured.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..core.llm_outputs import AnswerResponse
from ..core.search_index import ScoredChunk


@runtime_checkable
class AnswerBackend(Protocol):
    """The interface the CLI and HTTP server depend on.

    ``default_mode`` is a mutable retrieval mode used to honor a per-request
    override without changing the backend's constructor.
    """

    default_mode: str

    def ask_detailed(
        self,
        question: str,
        module: str | None = None,
        component: str | None = None,
        version: str | None = None,
        k: int | None = None,
        retrieval: str | None = None,
    ) -> dict[str, Any]:
        """Answer ``question`` and return the detailed, engine-shaped result."""
        ...

    async def ask_detailed_async(
        self,
        question: str,
        module: str | None = None,
        component: str | None = None,
        version: str | None = None,
        k: int | None = None,
        retrieval: str | None = None,
    ) -> dict[str, Any]:
        """Async version of :meth:`ask_detailed`."""
        ...

    def get_related_chunks(
        self,
        text: str,
        k: int = 5,
        **filters: Any,
    ) -> list[ScoredChunk]:
        """Return scored chunks related to ``text`` using the default mode."""
        ...


def build_filters(
    module: str | None = None,
    component: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Build a metadata filter dict from the optional Antora selectors."""
    filters: dict[str, Any] = {}
    if module:
        filters["module"] = module
    if component:
        filters["component"] = component
    if version:
        filters["version"] = version
    return filters


def dedupe_scored_chunks(results: list[ScoredChunk]) -> list[ScoredChunk]:
    """Deduplicate by chunk id, preserving first-seen order.

    The agent can retrieve the same chunk across multiple tool calls; the
    engine can see the same chunk from both retrievers. Deduplicating here keeps
    ``sources``, ``chunk_details``, and ``retrieved_count`` in agreement.
    """
    seen: set[str] = set()
    unique: list[ScoredChunk] = []
    for scored in results:
        chunk_id = scored.chunk.chunk_id
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        unique.append(scored)
    return unique


def build_answer_result(
    *,
    question: str,
    response: AnswerResponse,
    results: list[ScoredChunk],
    mode: str,
    site_url: str | None,
) -> dict[str, Any]:
    """Shape an :class:`AnswerResponse` and its chunks into the detailed result.

    This is the single place that defines the detailed result contract, shared
    by every backend.
    """
    results = dedupe_scored_chunks(results)
    return {
        "question": question,
        "answer": response.answer,
        "confidence": response.confidence,
        "citations": list(response.citations),
        "reasoning": response.reasoning,
        "sources": [r.chunk.get_link() for r in results],
        "chunk_details": [
            {
                "link": r.chunk.get_link(),
                "url": r.chunk.site_url(site_url),
                "text": r.chunk.text[:200] + "..." if len(r.chunk.text) > 200 else r.chunk.text,
                "module": r.chunk.module,
                "component": r.chunk.component,
                "breadcrumb": r.chunk.breadcrumb,
                "section_title": r.chunk.section_title,
                "sources": r.sources,
            }
            for r in results
        ],
        "retrieved_count": len(results),
        "retrieval_mode": mode,
    }
