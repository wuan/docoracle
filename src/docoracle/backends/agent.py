"""Q&A agent built on pydantic-ai's Agent and Tool abstractions.

Unlike :class:`docoracle.backends.engine.QAEngine`, which always runs a fixed
retrieve-then-answer pipeline, this agent lets the model decide whether
to call the retrieval tool before producing a structured
:class:`AnswerResponse`.

The model is built with the same OpenAI-compatible provider as
:class:`docoracle.api.structured_llm_client.StructuredLLMClient`, so no
vendor-specific optional dependency is required.

Retrieved chunks are collected per run through pydantic-ai's dependency
injection (see :class:`RetrievalState`), so concurrent runs don't share
state and callers can still report retrieval provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, ModelSettings, RunContext, Tool
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from ..api.structured_llm_client import StructuredLLMClient, build_chat_model
from ..core.config import Config, get_config, resolve_api_key
from ..core.hybrid_searcher import VALID_MODES
from ..core.llm_outputs import AnswerResponse
from ..core.search_index import ScoredChunk
from .protocol import build_answer_result, build_filters, dedupe_scored_chunks

DEFAULT_TOP_K = 5


@dataclass
class RetrievalState:
    """Per-run dependency that collects chunks returned by the retrieval tool.

    It also carries the resolved retrieval inputs (``mode``, ``k``, ``filters``)
    so the tool applies request filters deterministically rather than relying on
    the model to pass them through.
    """

    results: list[ScoredChunk] = field(default_factory=list)
    mode: str | None = None
    k: int | None = None
    filters: dict[str, Any] = field(default_factory=dict)


class RetrievalTool:
    """Tool that retrieves documentation chunks for a query."""

    def __init__(self, searcher: Any, llm_client: Any, mode: str = "hybrid") -> None:
        self.searcher = searcher
        self.llm_client = llm_client
        self.mode = mode
        self.tool = Tool(self.search, name="retrieve_documents")

    def search(
        self,
        ctx: RunContext[RetrievalState],
        query: str,
        k: int | None = None,
        module: str | None = None,
        component: str | None = None,
        version: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the documentation for chunks relevant to ``query``.

        Use this whenever you need facts from the documentation before answering.

        Args:
            query: Natural-language search query.
            k: Maximum number of chunks to return.
            module: Optional Antora module filter.
            component: Optional Antora component filter.
            version: Optional component version filter.

        Returns:
            Matching chunks with their text, link, and Antora metadata.
        """
        tool_filters: dict[str, Any] = {}
        if module:
            tool_filters["module"] = module
        if component:
            tool_filters["component"] = component
        if version:
            tool_filters["version"] = version
        # Request-level filters (from RetrievalState) win over model-supplied
        # ones so a caller's filter is always honored.
        filters = {**tool_filters, **ctx.deps.filters}

        mode = ctx.deps.mode or self.mode
        effective_k = ctx.deps.k or k or DEFAULT_TOP_K
        query_embedding: list[float] | None = None
        if mode in ("hybrid", "semantic"):
            query_embedding = self.llm_client.embed([query])[0]

        results = self.searcher.search(
            query=query,
            query_embedding=query_embedding,
            k=effective_k,
            filters=filters or None,
            mode=mode,
        )
        ctx.deps.results.extend(results)
        return [
            {
                "link": r.chunk.get_link(),
                "module": r.chunk.module,
                "component": r.chunk.component,
                "section_title": r.chunk.section_title,
                "text": r.chunk.text,
            }
            for r in results
        ]

    def as_tool(self) -> Tool[RetrievalState]:
        """Return the pydantic-ai ``Tool`` instance."""
        return self.tool


class SummaryTool:
    """Tool that summarizes text before it is used as context."""

    def __init__(self, llm_client: Any) -> None:
        self.llm_client = llm_client
        self.tool = Tool(self.summarize, name="summarize")

    def summarize(self, text: str, max_length: int = 500) -> str:
        """Summarize ``text`` in at most ``max_length`` characters.

        Use this to condense long retrieval results before reasoning about them.

        Args:
            text: The text to summarize.
            max_length: Maximum length of the summary in characters.

        Returns:
            The summary text.
        """
        prompt = (
            f"Summarize the following text in {max_length} characters or less:\n\n"
            f"{text}\n\nSummary:"
        )
        return self.llm_client.chat([{"role": "user", "content": prompt}])

    def as_tool(self) -> Tool:
        """Return the pydantic-ai ``Tool`` instance."""
        return self.tool


class QAAgent:
    """Answer questions about documentation using a tool-using pydantic-ai agent.

    The agent has a retrieval tool (and a summarization tool) and returns a
    validated :class:`AnswerResponse`. It decides for itself whether retrieval
    is needed.

    Example:
        searcher = HybridSearcher(...)
        agent = QAAgent(searcher)
        response = agent.ask("How do I install the system?")
        print(response.answer)
    """

    def __init__(
        self,
        searcher: Any,
        config_path: str = "config.yaml",
        llm_client: Any | None = None,
        mode: str | None = None,
        config: Config | None = None,
    ) -> None:
        self.config: Config = config if config is not None else get_config(config_path)
        self.searcher = searcher
        self.llm_client = llm_client if llm_client is not None else StructuredLLMClient(config_path)
        # Default the retrieval mode to configuration; an explicit ``mode``
        # argument still overrides it.
        self.default_mode = mode or self.config.retrieval.mode
        if self.default_mode not in VALID_MODES:
            self.default_mode = "hybrid"

        # Mirror the engine: apply retrieval hyperparameters to the searcher.
        searcher.rrf_k = self.config.retrieval.rrf_k
        if self.config.retrieval.prefetch_k is not None:
            searcher.prefetch_k = self.config.retrieval.prefetch_k

        self.retrieval_tool = RetrievalTool(searcher, self.llm_client, mode=self.default_mode)
        self.summary_tool = SummaryTool(self.llm_client)

        model = build_chat_model(
            self.config.generation.model,
            self.config.llm.api_url,
            resolve_api_key(self.config),
        )
        self.agent: Agent[RetrievalState, AnswerResponse] = Agent(
            model=model,
            output_type=AnswerResponse,
            deps_type=RetrievalState,
            instructions=self.config.prompts.system,
            tools=[self.retrieval_tool.as_tool(), self.summary_tool.as_tool()],
        )

    def _model_settings(self, temperature: float | None, max_tokens: int | None) -> ModelSettings:
        settings = ModelSettings(
            temperature=(
                temperature if temperature is not None else self.config.generation.temperature
            )
        )
        if max_tokens is not None:
            settings["max_tokens"] = max_tokens
        return settings

    def _build_prompt(self, question: str) -> str:
        """Return the user prompt.

        Retrieval inputs travel through :class:`RetrievalState` rather than the
        prompt, so the model cannot ignore a caller's filters or ``k``.
        """
        return question

    def _resolve_mode(self, retrieval: str | None) -> str:
        """Validate ``retrieval`` against the supported modes (like QAEngine)."""
        mode = retrieval or self.default_mode
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid retrieval mode {mode!r}. Must be one of {VALID_MODES}.")
        return mode

    def _make_state(
        self,
        mode: str,
        module: str | None,
        component: str | None,
        version: str | None,
        k: int | None,
    ) -> RetrievalState:
        return RetrievalState(
            mode=mode,
            k=k or self.config.retrieval.top_k,
            filters=build_filters(module, component, version),
        )

    def _run(
        self,
        question: str,
        module: str | None,
        component: str | None,
        version: str | None,
        k: int | None,
        retrieval: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> tuple[AnswerResponse, list[ScoredChunk], str]:
        mode = self._resolve_mode(retrieval)
        state = self._make_state(mode, module, component, version, k)
        result = self.agent.run_sync(
            self._build_prompt(question),
            deps=state,
            model_settings=self._model_settings(temperature, max_tokens),
        )
        return result.output, dedupe_scored_chunks(state.results), mode

    async def _run_async(
        self,
        question: str,
        module: str | None,
        component: str | None,
        version: str | None,
        k: int | None,
        retrieval: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> tuple[AnswerResponse, list[ScoredChunk], str]:
        mode = self._resolve_mode(retrieval)
        state = self._make_state(mode, module, component, version, k)
        result = await self.agent.run(
            self._build_prompt(question),
            deps=state,
            model_settings=self._model_settings(temperature, max_tokens),
        )
        return result.output, dedupe_scored_chunks(state.results), mode

    def _format_result(
        self,
        question: str,
        response: AnswerResponse,
        results: list[ScoredChunk],
        mode: str,
    ) -> dict[str, Any]:
        """Shape the agent response like the engine's detailed result."""
        return build_answer_result(
            question=question,
            response=response,
            results=results,
            mode=mode,
            site_url=self.config.antora_site.url,
        )

    def ask(
        self,
        question: str,
        module: str | None = None,
        component: str | None = None,
        version: str | None = None,
        k: int | None = None,
        retrieval: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AnswerResponse:
        """Ask a question and return a validated structured answer."""
        response, _, _ = self._run(
            question, module, component, version, k, retrieval, max_tokens, temperature
        )
        return response

    async def ask_async(
        self,
        question: str,
        module: str | None = None,
        component: str | None = None,
        version: str | None = None,
        k: int | None = None,
        retrieval: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AnswerResponse:
        """Async version of :meth:`ask`."""
        response, _, _ = await self._run_async(
            question, module, component, version, k, retrieval, max_tokens, temperature
        )
        return response

    def ask_detailed(
        self,
        question: str,
        module: str | None = None,
        component: str | None = None,
        version: str | None = None,
        k: int | None = None,
        retrieval: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Like :meth:`ask`, but include retrieved chunks (QAEngine-shaped dict)."""
        response, results, mode = self._run(
            question, module, component, version, k, retrieval, max_tokens, temperature
        )
        return self._format_result(question, response, results, mode)

    async def ask_detailed_async(
        self,
        question: str,
        module: str | None = None,
        component: str | None = None,
        version: str | None = None,
        k: int | None = None,
        retrieval: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Async version of :meth:`ask_detailed`."""
        response, results, mode = await self._run_async(
            question, module, component, version, k, retrieval, max_tokens, temperature
        )
        return self._format_result(question, response, results, mode)

    def get_related_chunks(
        self,
        text: str,
        k: int = 5,
        **filters: Any,
    ) -> list[ScoredChunk]:
        """Return chunks related to ``text`` using the default retrieval mode."""
        mode = self.default_mode
        query_embedding: list[float] | None = None
        if mode in ("hybrid", "semantic"):
            query_embedding = self.llm_client.embed([text])[0]
        return self.searcher.search(
            query=text,
            query_embedding=query_embedding,
            k=k,
            filters=filters if filters else None,
            mode=mode,
        )

    def explain(
        self,
        question: str,
        module: str | None = None,
        component: str | None = None,
        version: str | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Answer a question and return the tool calls alongside the answer."""
        state = self._make_state(self.default_mode, module, component, version, k)
        result = self.agent.run_sync(
            self._build_prompt(question),
            deps=state,
            model_settings=self._model_settings(None, None),
        )
        tool_calls: list[dict[str, Any]] = []
        for message in result.all_messages():
            for part in getattr(message, "parts", []):
                if isinstance(part, ToolCallPart):
                    tool_calls.append({"tool": part.tool_name, "input": part.args})
                elif isinstance(part, ToolReturnPart):
                    tool_calls.append({"tool": part.tool_name, "output": part.content})
        return {
            "answer": result.output,
            "tool_calls": tool_calls,
            "retrieved_count": len(dedupe_scored_chunks(state.results)),
        }


def create_qa_agent(
    searcher: Any,
    config_path: str = "config.yaml",
    llm_client: Any | None = None,
    mode: str | None = None,
    config: Config | None = None,
) -> QAAgent:
    """Create a :class:`QAAgent` with sensible defaults."""
    return QAAgent(searcher, config_path, llm_client, mode, config)
