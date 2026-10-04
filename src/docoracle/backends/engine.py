"""Deterministic retrieve-then-answer backend.

The engine always retrieves once and generates once from the retrieved chunks,
honoring the configured retrieval mode and hyperparameters. It implements the
shared :class:`~docoracle.backends.protocol.AnswerBackend` interface so the CLI and
HTTP server can select it without knowing its concrete type.
"""

from __future__ import annotations

from typing import Any

from ..api.structured_llm_client import StructuredLLMClient
from ..core.config import Config, get_config
from ..core.hybrid_searcher import VALID_MODES, HybridSearcher
from ..core.llm_outputs import AnswerResponse
from ..core.models import Chunk
from ..core.search_index import ScoredChunk
from .protocol import build_answer_result, build_filters

DEFAULT_RETRIEVAL_MODE = "hybrid"


class QAEngine:
    """Orchestrates question answering with retrieval and generation."""

    def __init__(
        self,
        searcher: HybridSearcher,
        llm_client: StructuredLLMClient,
        config_path: str = "config.yaml",
        config: Config | None = None,
    ):
        """Initialize the engine.

        Args:
            searcher: Hybrid searcher holding the chunks and both indices.
            llm_client: LLM API client.
            config_path: Path to configuration file (used when ``config`` is
                not supplied).
            config: Pre-loaded configuration, preferred by the backend factory.
        """
        self.searcher = searcher
        self.llm_client = llm_client
        self.config: Config = config if config is not None else get_config(config_path)

        # Retrieval configuration.
        self.top_k = self.config.retrieval.top_k
        self.rrf_k = self.config.retrieval.rrf_k
        self.prefetch_k = self.config.retrieval.prefetch_k
        self.default_mode = self.config.retrieval.mode
        if self.default_mode not in VALID_MODES:
            self.default_mode = DEFAULT_RETRIEVAL_MODE
        # Apply RRF and prefetch config to the searcher so they're picked
        # up across CLI and HTTP entry points.
        self.searcher.rrf_k = self.rrf_k
        if self.prefetch_k is not None:
            self.searcher.prefetch_k = self.prefetch_k
        self.searcher.bm25_min_score_ratio = self.config.retrieval.bm25_min_score_ratio

        # Generation configuration.
        self.max_context_length = self.config.generation.max_context_length
        self.site_url = self.config.antora_site.url

    # ------------------------------------------------------------------
    # Retrieval helpers (shared by the sync and async answer paths)
    # ------------------------------------------------------------------

    def _resolve_mode(self, retrieval: str | None) -> str:
        mode = retrieval or self.default_mode
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid retrieval mode {mode!r}. Must be one of {VALID_MODES}.")
        return mode

    def _query_embedding(self, question: str, mode: str) -> list[float] | None:
        """Embed the query only when the mode needs the semantic path."""
        if mode in ("hybrid", "semantic"):
            return self.llm_client.embed([question])[0]
        return None

    def _retrieve(
        self,
        question: str,
        module: str | None,
        component: str | None,
        version: str | None,
        k: int | None,
        retrieval: str | None,
    ) -> tuple[str, list[ScoredChunk]]:
        mode = self._resolve_mode(retrieval)
        k = k or self.top_k
        filters = build_filters(module, component, version)
        results = self.searcher.search(
            query=question,
            query_embedding=self._query_embedding(question, mode),
            k=k,
            filters=filters or None,
            mode=mode,
        )
        return mode, results

    # ------------------------------------------------------------------
    # AnswerBackend interface
    # ------------------------------------------------------------------

    def ask_detailed(
        self,
        question: str,
        module: str | None = None,
        component: str | None = None,
        version: str | None = None,
        k: int | None = None,
        retrieval: str | None = None,
    ) -> dict[str, Any]:
        """Answer a question and return the detailed, backend-agnostic result."""
        mode, results = self._retrieve(question, module, component, version, k, retrieval)
        context = self._build_context([r.chunk for r in results])
        response = self._generate_answer(question, context)
        return build_answer_result(
            question=question,
            response=response,
            results=results,
            mode=mode,
            site_url=self.site_url,
        )

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
        mode, results = self._retrieve(question, module, component, version, k, retrieval)
        context = self._build_context([r.chunk for r in results])
        response = await self._generate_answer_async(question, context)
        return build_answer_result(
            question=question,
            response=response,
            results=results,
            mode=mode,
            site_url=self.site_url,
        )

    def get_related_chunks(
        self,
        text: str,
        k: int = 5,
        **filters: Any,
    ) -> list[ScoredChunk]:
        """Return chunks related to ``text`` using the default retrieval mode."""
        mode = self.default_mode
        return self.searcher.search(
            query=text,
            query_embedding=self._query_embedding(text, mode),
            k=k,
            filters=filters if filters else None,
            mode=mode,
        )

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        module: str | None = None,
        component: str | None = None,
        version: str | None = None,
        k: int | None = None,
        retrieval: str | None = None,
    ) -> dict[str, Any]:
        """Alias for :meth:`ask_detailed` (kept for existing callers)."""
        return self.ask_detailed(question, module, component, version, k, retrieval)

    async def ask_async(
        self,
        question: str,
        module: str | None = None,
        component: str | None = None,
        version: str | None = None,
        k: int | None = None,
        retrieval: str | None = None,
    ) -> dict[str, Any]:
        """Alias for :meth:`ask_detailed_async` (kept for existing callers)."""
        return await self.ask_detailed_async(question, module, component, version, k, retrieval)

    def batch_ask(
        self,
        questions: list[str],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Answer multiple questions in batch."""
        return [self.ask(q, **kwargs) for q in questions]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _build_context(self, chunks: list[Chunk]) -> str:
        """Build the LLM context string from retrieved chunks.

        Provenance fields (RRF score, per-retriever ranks) are deliberately
        excluded from the prompt sent to the LLM — only chunk text and
        ``[Source N]`` markers appear.
        """
        context_parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            source_ref = f"[Source {i}: {chunk.get_link()}]"
            if chunk.section_title:
                context_parts.append(f"\n=== {chunk.section_title} ({chunk.module}) ===")
            context_parts.append(chunk.text)
            context_parts.append(source_ref)
        context = "\n\n".join(context_parts)
        if len(context) > self.max_context_length:
            context = context[: self.max_context_length] + "...\n[Context truncated]"
        return context

    def _generate_answer(self, question: str, context: str) -> AnswerResponse:
        """Generate a validated answer through the client's native structured path."""
        return self.llm_client.ask(
            question=question,
            context=context,
            model=self.config.generation.model,
            temperature=self.config.generation.temperature,
            system=self.config.prompts.system,
        )  # type: ignore[return-value]

    async def _generate_answer_async(self, question: str, context: str) -> AnswerResponse:
        """Async version of :meth:`_generate_answer`."""
        return await self.llm_client.ask_async(
            question=question,
            context=context,
            model=self.config.generation.model,
            temperature=self.config.generation.temperature,
            system=self.config.prompts.system,
        )  # type: ignore[return-value]
