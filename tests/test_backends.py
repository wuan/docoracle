"""Tests for the backend abstraction: protocol, shared result builder, config
selection, factory, and engine/agent parity."""

import asyncio
import logging
import uuid
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from docoracle.backends.agent import QAAgent
from docoracle.backends.engine import QAEngine
from docoracle.backends.factory import create_answer_backend
from docoracle.backends.protocol import (
    AnswerBackend,
    build_answer_result,
    build_filters,
    dedupe_scored_chunks,
)
from docoracle.core.bm25_index import BM25Index
from docoracle.core.config import Config
from docoracle.core.hybrid_searcher import HybridSearcher
from docoracle.core.llm_outputs import AnswerResponse
from docoracle.core.models import Chunk
from docoracle.core.semantic_index import SemanticIndex


class _LLM:
    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return "chat"

    def ask(self, question: str, context: str = "", **kwargs: Any) -> AnswerResponse:
        return AnswerResponse(answer="engine answer")


def _searcher() -> HybridSearcher:
    chunk = Chunk(
        text="alpha bravo charlie",
        chunk_id=str(uuid.uuid4()),
        module="m",
        component="c",
        version="v",
        page_id="p",
    )
    hs = HybridSearcher(semantic=SemanticIndex(), bm25=BM25Index())
    hs.add_chunks([chunk])
    return hs


def _scored(chunk_id: str, rank: int) -> Any:
    from docoracle.core.search_index import ScoredChunk

    chunk = Chunk(
        text="text",
        chunk_id=chunk_id,
        module="m",
        component="c",
        version="v",
        page_id="p",
    )
    return ScoredChunk(chunk=chunk, score=1.0, sources={"bm25": rank})


# =============================================================================
# PROTOCOL
# =============================================================================


def test_engine_satisfies_protocol() -> None:
    engine = QAEngine(_searcher(), _LLM(), config=Config())
    assert isinstance(engine, AnswerBackend)
    assert engine.default_mode == "hybrid"


def test_agent_satisfies_protocol() -> None:
    with patch("docoracle.backends.agent.resolve_api_key", return_value="test-key"):
        agent = QAAgent(_searcher(), llm_client=_LLM(), config=Config())
    assert isinstance(agent, AnswerBackend)


def test_plain_object_does_not_satisfy_protocol() -> None:
    assert not isinstance(object(), AnswerBackend)


# =============================================================================
# SHARED RESULT BUILDER
# =============================================================================


def test_build_answer_result_has_full_field_set() -> None:
    result = build_answer_result(
        question="q",
        response=AnswerResponse(answer="a", confidence=0.5, citations=["c"], reasoning="r"),
        results=[_scored("one", 1)],
        mode="bm25",
        site_url=None,
    )
    assert set(result) == {
        "question",
        "answer",
        "confidence",
        "citations",
        "reasoning",
        "sources",
        "chunk_details",
        "retrieved_count",
        "retrieval_mode",
    }
    assert result["retrieved_count"] == 1
    assert result["chunk_details"][0]["sources"] == {"bm25": 1}


def test_build_answer_result_dedupes() -> None:
    result = build_answer_result(
        question="q",
        response=AnswerResponse(answer="a"),
        results=[_scored("one", 1), _scored("one", 2), _scored("two", 3)],
        mode="hybrid",
        site_url=None,
    )
    assert result["retrieved_count"] == 2
    assert len(result["sources"]) == 2


def test_build_filters() -> None:
    assert build_filters(module="m", component="c", version="v") == {
        "module": "m",
        "component": "c",
        "version": "v",
    }
    assert build_filters() == {}


def test_dedupe_preserves_first_seen_order() -> None:
    unique = dedupe_scored_chunks([_scored("a", 1), _scored("b", 2), _scored("a", 3)])
    assert [r.chunk.chunk_id for r in unique] == ["a", "b"]


# =============================================================================
# CONFIG SELECTION
# =============================================================================


def test_backend_defaults_to_engine() -> None:
    assert Config().docoracle.backend == "engine"


def test_invalid_backend_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        Config(docoracle={"backend": "nope"})  # type: ignore[arg-type]
    assert "engine" in str(exc.value)
    assert "agent" in str(exc.value)


def test_backend_from_config_dict() -> None:
    assert Config(docoracle={"backend": "agent"}).docoracle.backend == "agent"  # type: ignore[arg-type]


# =============================================================================
# FACTORY
# =============================================================================


def test_factory_returns_engine_by_default() -> None:
    backend = create_answer_backend(_searcher(), _LLM(), Config())
    assert isinstance(backend, QAEngine)


def test_factory_returns_agent_when_configured() -> None:
    config = Config(docoracle={"backend": "agent"})  # type: ignore[arg-type]
    with patch("docoracle.backends.agent.resolve_api_key", return_value="test-key"):
        backend = create_answer_backend(_searcher(), _LLM(), config)
    assert isinstance(backend, QAAgent)


def test_factory_reports_active_backend(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="backends.factory"):
        create_answer_backend(_searcher(), _LLM(), Config())
    assert "engine" in caplog.text.lower()


# =============================================================================
# PARITY
# =============================================================================


class _FakeAgentBackend:
    """Drives the agent's detailed path without a real model."""

    def __init__(self, agent: QAAgent, response: AnswerResponse) -> None:
        from tests.test_agents import FakeAgent

        self._agent = agent
        self._response = response
        self._fake = FakeAgent(response)

    def run(self, **kwargs: Any) -> dict[str, Any]:
        self._agent.agent = self._fake  # type: ignore[assignment]
        return self._agent.ask_detailed(**kwargs)


def test_engine_and_agent_emit_same_field_set() -> None:
    searcher = _searcher()
    response = AnswerResponse(answer="same", confidence=0.5)

    engine = QAEngine(searcher, _LLM(), config=Config())
    engine_result = engine.ask_detailed("alpha", k=1, retrieval="bm25")

    with patch("docoracle.backends.agent.resolve_api_key", return_value="test-key"):
        agent = QAAgent(searcher, llm_client=_LLM(), config=Config())
    agent_result = _FakeAgentBackend(agent, response).run(question="alpha", k=1, retrieval="bm25")

    assert set(engine_result) == set(agent_result)
    assert engine_result["retrieval_mode"] == agent_result["retrieval_mode"] == "bm25"


def test_agent_backend_async_path_returns_result() -> None:
    """The server's async path must work for the agent backend inside an event loop."""
    from tests.test_agents import FakeAgent

    config = Config(docoracle={"backend": "agent"})  # type: ignore[arg-type]
    with patch("docoracle.backends.agent.resolve_api_key", return_value="test-key"):
        backend = create_answer_backend(_searcher(), _LLM(), config)

    backend.agent = FakeAgent(AnswerResponse(answer="async"))  # type: ignore[attr-defined]
    result = asyncio.run(backend.ask_detailed_async("alpha", k=1, retrieval="bm25"))

    assert result["answer"] == "async"
    assert result["retrieval_mode"] == "bm25"
