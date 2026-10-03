"""Tests for the pydantic-ai based Q&A agent and its tools."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai import Tool
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from docoracle.backends.agent import (
    QAAgent,
    RetrievalState,
    RetrievalTool,
    SummaryTool,
    create_qa_agent,
)
from docoracle.core.config import Config
from docoracle.core.llm_outputs import AnswerResponse
from docoracle.core.models import Chunk
from docoracle.core.search_index import ScoredChunk

# =============================================================================
# FIXTURES
# =============================================================================


class MockSearcher:
    """Searcher that records calls and returns one predictable chunk."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        mode: str = "hybrid",
    ) -> list[ScoredChunk]:
        self.calls.append(
            {
                "query": query,
                "query_embedding": query_embedding,
                "k": k,
                "filters": filters,
                "mode": mode,
            }
        )
        return [_mock_scored_chunk()]


class MockLLMClient:
    """LLM client double providing embeddings and chat."""

    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []
        self.chat_calls: list[list[dict[str, Any]]] = []

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [[0.1, 0.2] for _ in texts]

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.chat_calls.append(messages)
        return "Mock chat response"


def _mock_scored_chunk(chunk_id: str = "mock-1", text: str = "Mock chunk") -> ScoredChunk:
    chunk = Chunk(
        text=text,
        chunk_id=chunk_id,
        module="test-module",
        component="test-component",
        version="1.0",
        page_id="test:pages:index",
        section_title="Mock section",
    )
    return ScoredChunk(chunk=chunk, score=1.0, sources={"bm25": 1})


@pytest.fixture
def mock_searcher() -> MockSearcher:
    return MockSearcher()


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    return MockLLMClient()


def _ctx(state: RetrievalState | None = None) -> Any:
    """Minimal stand-in for a pydantic-ai ``RunContext``."""
    return SimpleNamespace(deps=state or RetrievalState())


# =============================================================================
# RETRIEVAL TOOL
# =============================================================================


class TestRetrievalTool:
    def test_init(self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient) -> None:
        tool = RetrievalTool(mock_searcher, mock_llm_client)

        assert tool.searcher is mock_searcher
        assert isinstance(tool.as_tool(), Tool)

    def test_search_returns_chunk_dicts(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        tool = RetrievalTool(mock_searcher, mock_llm_client)

        result = tool.search(_ctx(), query="test query", k=3)

        assert isinstance(result, list)
        assert result[0]["text"] == "Mock chunk"
        assert result[0]["link"] == "test:pages:index"
        assert result[0]["section_title"] == "Mock section"
        assert mock_searcher.calls[0]["k"] == 3

    def test_search_records_chunks_in_state(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        tool = RetrievalTool(mock_searcher, mock_llm_client)
        state = RetrievalState()

        tool.search(_ctx(state), query="test")

        assert len(state.results) == 1
        assert state.results[0].chunk.chunk_id == "mock-1"

    def test_search_hybrid_embeds_query(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        tool = RetrievalTool(mock_searcher, mock_llm_client, mode="hybrid")

        tool.search(_ctx(), query="test")

        assert mock_searcher.calls[0]["query_embedding"] == [0.1, 0.2]

    def test_search_bm25_skips_embedding(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        tool = RetrievalTool(mock_searcher, mock_llm_client, mode="bm25")

        tool.search(_ctx(), query="test")

        assert mock_searcher.calls[0]["query_embedding"] is None
        assert mock_llm_client.embed_calls == []

    def test_search_builds_filters(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        tool = RetrievalTool(mock_searcher, mock_llm_client)

        tool.search(_ctx(), query="q", module="m", component="c", version="1")

        assert mock_searcher.calls[0]["filters"] == {
            "module": "m",
            "component": "c",
            "version": "1",
        }

    def test_search_without_filters_passes_none(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        tool = RetrievalTool(mock_searcher, mock_llm_client)

        tool.search(_ctx(), query="q")

        assert mock_searcher.calls[0]["filters"] is None

    def test_search_state_mode_overrides_default(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        tool = RetrievalTool(mock_searcher, mock_llm_client, mode="hybrid")

        tool.search(_ctx(RetrievalState(mode="bm25")), query="q")

        assert mock_searcher.calls[0]["mode"] == "bm25"
        assert mock_llm_client.embed_calls == []

    def test_search_state_filters_override_tool_args(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        tool = RetrievalTool(mock_searcher, mock_llm_client)

        tool.search(_ctx(RetrievalState(filters={"module": "forced"})), query="q", module="ignored")

        assert mock_searcher.calls[0]["filters"] == {"module": "forced"}

    def test_search_state_k_overrides_tool_arg(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        tool = RetrievalTool(mock_searcher, mock_llm_client)

        tool.search(_ctx(RetrievalState(k=2)), query="q", k=9)

        assert mock_searcher.calls[0]["k"] == 2


# =============================================================================
# SUMMARY TOOL
# =============================================================================


class TestSummaryTool:
    def test_init(self, mock_llm_client: MockLLMClient) -> None:
        tool = SummaryTool(mock_llm_client)

        assert tool.llm_client is mock_llm_client
        assert isinstance(tool.as_tool(), Tool)

    def test_summarize_returns_chat_result(self, mock_llm_client: MockLLMClient) -> None:
        tool = SummaryTool(mock_llm_client)

        assert tool.summarize("some long text", max_length=100) == "Mock chat response"
        assert "100 characters" in mock_llm_client.chat_calls[0][0]["content"]


# =============================================================================
# Q&A AGENT
# =============================================================================


class FakeAgent:
    """Stand-in for the pydantic-ai Agent, avoiding network calls."""

    def __init__(self, output: AnswerResponse, results: list[ScoredChunk] | None = None) -> None:
        self.output = output
        self.results = results or []
        self.prompts: list[str] = []
        self.deps: list[Any] = []
        self.messages: list[Any] = []

    def run_sync(self, prompt: str, deps: Any = None, model_settings: Any = None) -> Any:
        self.prompts.append(prompt)
        self.deps.append(deps)
        if deps is not None:
            deps.results.extend(self.results)
        outer = self

        class Result:
            output = outer.output

            def all_messages(self) -> list[Any]:
                return outer.messages

        return Result()

    async def run(self, prompt: str, deps: Any = None, model_settings: Any = None) -> Any:
        return self.run_sync(prompt, deps, model_settings)


def _make_agent(mock_searcher: MockSearcher, mock_llm_client: MockLLMClient) -> QAAgent:
    with patch("docoracle.backends.agent.resolve_api_key", return_value="test-key"):
        return QAAgent(mock_searcher, llm_client=mock_llm_client)


class TestQAAgent:
    def test_ask_returns_structured_answer(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        qa = _make_agent(mock_searcher, mock_llm_client)
        expected = AnswerResponse(answer="42", confidence=0.9, citations=["a:b#c"])
        qa.agent = FakeAgent(expected)  # type: ignore[assignment]

        result = qa.ask("What is the answer?", module="m")

        assert result is expected
        # Retrieval inputs travel through RetrievalState, not the prompt.
        assert qa.agent.prompts[0] == "What is the answer?"  # type: ignore[attr-defined]
        assert qa.agent.deps[0].filters == {"module": "m"}  # type: ignore[attr-defined]
        assert qa.agent.deps[0].k == qa.config.retrieval.top_k  # type: ignore[attr-defined]

    def test_ask_async_returns_structured_answer(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        qa = _make_agent(mock_searcher, mock_llm_client)
        expected = AnswerResponse(answer="async", confidence=0.8)
        qa.agent = FakeAgent(expected)  # type: ignore[assignment]

        result = asyncio.run(qa.ask_async("Async question?"))

        assert result is expected

    def test_ask_detailed_includes_sources(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        qa = _make_agent(mock_searcher, mock_llm_client)
        expected = AnswerResponse(answer="detailed", confidence=0.7)
        results = [_mock_scored_chunk("a"), _mock_scored_chunk("a"), _mock_scored_chunk("b")]
        qa.agent = FakeAgent(expected, results=results)  # type: ignore[assignment]

        result = qa.ask_detailed("Question?")

        assert result["answer"] == "detailed"
        assert result["retrieved_count"] == 2  # duplicates deduped
        assert result["sources"] == ["test:pages:index", "test:pages:index"]
        assert result["chunk_details"][0]["module"] == "test-module"

    def test_explain_returns_tool_calls(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        qa = _make_agent(mock_searcher, mock_llm_client)
        fake = FakeAgent(AnswerResponse(answer="explained"))
        fake.messages = [
            MagicMock(
                parts=[
                    ToolCallPart(tool_name="retrieve_documents", args={"query": "q"}),
                    ToolReturnPart(tool_name="retrieve_documents", content="[chunk]"),
                ]
            )
        ]
        qa.agent = fake  # type: ignore[assignment]

        result = qa.explain("Explain this")

        assert result["answer"] == AnswerResponse(answer="explained")
        assert result["tool_calls"] == [
            {"tool": "retrieve_documents", "input": {"query": "q"}},
            {"tool": "retrieve_documents", "output": "[chunk]"},
        ]

    def test_ask_detailed_uses_requested_retrieval_mode(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        qa = _make_agent(mock_searcher, mock_llm_client)
        qa.agent = FakeAgent(AnswerResponse(answer="x"))  # type: ignore[assignment]

        result = qa.ask_detailed("Question?", retrieval="bm25")

        assert result["retrieval_mode"] == "bm25"

    def test_ask_detailed_async_includes_sources(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        qa = _make_agent(mock_searcher, mock_llm_client)
        expected = AnswerResponse(answer="async detailed", confidence=0.6)
        qa.agent = FakeAgent(expected, results=[_mock_scored_chunk("a")])  # type: ignore[assignment]

        result = asyncio.run(qa.ask_detailed_async("Question?"))

        assert result["answer"] == "async detailed"
        assert result["retrieved_count"] == 1
        assert result["sources"] == ["test:pages:index"]

    def test_ask_detailed_rejects_invalid_retrieval_mode(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        qa = _make_agent(mock_searcher, mock_llm_client)
        qa.agent = FakeAgent(AnswerResponse(answer="x"))  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Invalid retrieval mode"):
            qa.ask_detailed("Question?", retrieval="invalid")

    def test_temperature_defaults_to_config(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        qa = _make_agent(mock_searcher, mock_llm_client)

        assert qa._model_settings(None, None)["temperature"] == qa.config.generation.temperature
        assert qa._model_settings(0.9, 128) == {"temperature": 0.9, "max_tokens": 128}


# =============================================================================
# FACTORY
# =============================================================================


class TestQAAgentParity:
    """Agent retrieval defaults, filters and hyperparameters mirror the engine."""

    def _agent_with_config(
        self,
        mock_searcher: MockSearcher,
        mock_llm_client: MockLLMClient,
        config: Config,
    ) -> QAAgent:
        with patch("docoracle.backends.agent.resolve_api_key", return_value="test-key"):
            return QAAgent(mock_searcher, llm_client=mock_llm_client, config=config)

    def test_default_mode_and_k_come_from_config(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        config = Config(retrieval={"mode": "bm25", "top_k": 8})  # type: ignore[arg-type]
        qa = self._agent_with_config(mock_searcher, mock_llm_client, config)

        assert qa.default_mode == "bm25"
        state = qa._make_state("bm25", None, None, None, None)
        assert state.k == 8

    def test_rrf_and_prefetch_applied_to_searcher(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        config = Config(retrieval={"rrf_k": 12, "prefetch_k": 7})  # type: ignore[arg-type]
        self._agent_with_config(mock_searcher, mock_llm_client, config)

        assert mock_searcher.rrf_k == 12
        assert mock_searcher.prefetch_k == 7

    def test_explicit_mode_overrides_config(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        config = Config(retrieval={"mode": "hybrid"})  # type: ignore[arg-type]
        qa = self._agent_with_config(mock_searcher, mock_llm_client, config)

        assert qa._resolve_mode("semantic") == "semantic"

    def test_invalid_mode_rejected_even_without_retrieval_call(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        qa = _make_agent(mock_searcher, mock_llm_client)
        qa.agent = FakeAgent(AnswerResponse(answer="x"))  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Invalid retrieval mode"):
            asyncio.run(qa.ask_detailed_async("Question?", retrieval="invalid"))

    def test_get_related_chunks_uses_default_mode(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        qa = _make_agent(mock_searcher, mock_llm_client)
        qa.default_mode = "bm25"

        results = qa.get_related_chunks("q", k=2, module="m")

        assert results[0].chunk.chunk_id == "mock-1"
        call = mock_searcher.calls[0]
        assert call["mode"] == "bm25"
        assert call["k"] == 2
        assert call["filters"] == {"module": "m"}


# =============================================================================
# FACTORY
# =============================================================================


class TestCreateQAAgent:
    def test_create_qa_agent_returns_qa_agent(
        self, mock_searcher: MockSearcher, mock_llm_client: MockLLMClient
    ) -> None:
        with patch("docoracle.backends.agent.resolve_api_key", return_value="test-key"):
            qa = create_qa_agent(mock_searcher, llm_client=mock_llm_client)

        assert isinstance(qa, QAAgent)
        assert qa.llm_client is mock_llm_client
