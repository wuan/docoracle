"""Tests for StructuredLLMClient, native structured output, and the Q&A engine.

The structured-answer path is provider-native (pydantic-ai ``Agent`` with
``output_type``), so these tests stub the ``Agent`` rather than a text response
parser.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from docoracle.api.structured_llm_client import StructuredLLMClient, _event_to_text
from docoracle.core.config import resolve_api_key
from docoracle.core.llm_outputs import AnswerResponse

# =============================================================================
# FIXTURES / HELPERS
# =============================================================================


@pytest.fixture
def mock_config():
    from docoracle.core.config import Config

    return Config(
        generation={"model": "mistral-small-latest", "temperature": 0.3},
        embedding={"api_key": None},
        llm={"api_url": "https://api.example.com", "timeout": 120, "api_key": None},
        retrieval={"top_k": 5},
    )


def _make_client(config):
    """Construct a client without running ``__init__`` (skips env/config resolution)."""
    client = StructuredLLMClient.__new__(StructuredLLMClient)
    client.config = config
    client.api_key = "test-key"
    client.api_url = config.llm.api_url
    client.timeout = config.llm.timeout
    client.embedding_model = config.embedding.model
    client.rate_limit_delay = config.embedding.rate_limit_delay
    client._cached_model_key = None
    client._cached_model = None
    client._cached_agent_key = None
    client._cached_agent = None
    return client


class FakeStream:
    """Sync context manager that yields events, mimicking ``model_request_stream_sync``."""

    def __init__(self, events):
        self._events = events

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeAsyncStream:
    """Async context manager that yields events, mimicking ``model_request_stream``."""

    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self._aiter()

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def _aiter(self):
        for event in self._events:
            yield event


class CustomResponse(BaseModel):
    summary: str = Field(..., description="Concise summary")
    tags: list[str] = Field(default_factory=list)


class FakeAgent:
    """Stand-in for pydantic-ai's ``Agent`` recording construction and run args."""

    last_init: dict = {}
    last_run: dict = {}
    output: BaseModel | None = None
    error: Exception | None = None

    def __init__(self, **kwargs) -> None:
        FakeAgent.last_init = kwargs

    def run_sync(self, prompt: str, model_settings=None):
        FakeAgent.last_run = {"prompt": prompt, "model_settings": model_settings}
        if FakeAgent.error is not None:
            raise FakeAgent.error
        return SimpleNamespace(output=FakeAgent.output)

    async def run(self, prompt: str, model_settings=None):
        return self.run_sync(prompt, model_settings)


@pytest.fixture(autouse=True)
def _reset_fake_agent():
    FakeAgent.output = None
    FakeAgent.error = None
    yield


# =============================================================================
# CONFIG / API-KEY TESTS
# =============================================================================


class TestAPIKeyResolution:
    """Direct tests of the shared ``resolve_api_key`` helper."""

    @patch.dict("os.environ", {"LLM_API_KEY": "env-key"}, clear=True)
    def test_llm_api_key_env(self, mock_config):
        assert resolve_api_key(mock_config) == "env-key"

    @patch.dict("os.environ", {"MISTRAL_API_KEY": "mistral-key"}, clear=True)
    def test_mistral_api_key_env_fallback(self, mock_config):
        assert resolve_api_key(mock_config) == "mistral-key"

    @patch.dict("os.environ", {"SOME_API_KEY": "from-env-var"}, clear=True)
    def test_config_env_var_syntax(self, mock_config):
        mock_config.embedding.api_key = "${SOME_API_KEY}"
        assert resolve_api_key(mock_config) == "from-env-var"

    def test_config_literal_api_key(self, mock_config):
        mock_config.embedding.api_key = "literal-key"
        assert resolve_api_key(mock_config) == "literal-key"

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_when_missing(self, mock_config):
        with pytest.raises(ValueError, match="API key not found"):
            resolve_api_key(mock_config)

    @patch.dict("os.environ", {"LLM_API_KEY": "from-llm-env"}, clear=True)
    def test_llm_config_env_var_syntax(self, mock_config):
        mock_config.embedding.api_key = None
        mock_config.llm.api_key = "${LLM_API_KEY}"
        assert resolve_api_key(mock_config) == "from-llm-env"

    def test_llm_config_literal_api_key(self, mock_config):
        mock_config.embedding.api_key = None
        mock_config.llm.api_key = "llm-literal-key"
        assert resolve_api_key(mock_config) == "llm-literal-key"


class TestStructuredLLMClientConfig:
    def test_uses_config_model(self, mock_config):
        client = _make_client(mock_config)
        assert client.config.generation.model == "mistral-small-latest"

    def test_uses_config_temperature(self, mock_config):
        client = _make_client(mock_config)
        assert client.config.generation.temperature == 0.3

    def test_empty_model_settings(self, mock_config):
        client = _make_client(mock_config)
        assert client._model_settings(temperature=None, max_tokens=None) == {}

    def test_model_settings_includes_overrides(self, mock_config):
        client = _make_client(mock_config)
        assert client._model_settings(temperature=0.7, max_tokens=200) == {
            "temperature": 0.7,
            "max_tokens": 200,
        }


# =============================================================================
# NATIVE STRUCTURED OUTPUT
# =============================================================================


class TestNativeStructuredOutput:
    def test_ask_returns_validated_output(self, mock_config):
        client = _make_client(mock_config)
        expected = AnswerResponse(answer="yes", confidence=0.9, citations=["a"])
        FakeAgent.output = expected
        with patch("docoracle.api.structured_llm_client.Agent", FakeAgent):
            result = client.ask("Q?", "ctx")

        assert result is expected

    def test_ask_uses_requested_output_model(self, mock_config):
        client = _make_client(mock_config)
        expected = CustomResponse(summary="short", tags=["x"])
        FakeAgent.output = expected
        with patch("docoracle.api.structured_llm_client.Agent", FakeAgent):
            result = client.ask("Q?", "ctx", output_model=CustomResponse)

        assert result is expected
        # The provider is told the requested schema via ``output_type``.
        assert FakeAgent.last_init["output_type"] is CustomResponse

    def test_ask_uses_configured_system_prompt_by_default(self, mock_config):
        client = _make_client(mock_config)
        FakeAgent.output = AnswerResponse(answer="x")
        with patch("docoracle.api.structured_llm_client.Agent", FakeAgent):
            client.ask("Q?", "ctx")

        assert FakeAgent.last_init["instructions"] == mock_config.prompts.system

    def test_ask_system_override(self, mock_config):
        client = _make_client(mock_config)
        FakeAgent.output = AnswerResponse(answer="x")
        with patch("docoracle.api.structured_llm_client.Agent", FakeAgent):
            client.ask("Q?", "ctx", system="You are a pirate.")

        assert FakeAgent.last_init["instructions"] == "You are a pirate."

    def test_ask_prompt_includes_question_and_context(self, mock_config):
        client = _make_client(mock_config)
        FakeAgent.output = AnswerResponse(answer="x")
        with patch("docoracle.api.structured_llm_client.Agent", FakeAgent):
            client.ask("What is X?", "the context")

        prompt = FakeAgent.last_run["prompt"]
        assert "What is X?" in prompt
        assert "the context" in prompt

    def test_ask_forwards_temperature(self, mock_config):
        client = _make_client(mock_config)
        FakeAgent.output = AnswerResponse(answer="x")
        with patch("docoracle.api.structured_llm_client.Agent", FakeAgent):
            client.ask("Q?", "ctx")

        assert FakeAgent.last_run["model_settings"]["temperature"] == 0.3

    def test_ask_async_returns_output(self, mock_config):
        client = _make_client(mock_config)
        expected = CustomResponse(summary="async ok")
        FakeAgent.output = expected
        with patch("docoracle.api.structured_llm_client.Agent", FakeAgent):
            result = asyncio.run(client.ask_async("Q?", "ctx", output_model=CustomResponse))

        assert result is expected

    def test_non_conforming_output_raises(self, mock_config):
        """No heuristic fallback: a model failure must propagate."""
        client = _make_client(mock_config)
        FakeAgent.error = RuntimeError("model did not conform to schema")
        with (
            patch("docoracle.api.structured_llm_client.Agent", FakeAgent),
            pytest.raises(RuntimeError, match="did not conform"),
        ):
            client.ask("Q?", "ctx")


# =============================================================================
# STREAMING TESTS
# =============================================================================


class TestStreaming:
    def test_stream_ask_yields_text(self, mock_config):
        from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta

        client = _make_client(mock_config)
        part = TextPart(content="Hello ")
        delta = TextPartDelta(
            content_delta="world",
            provider_name=None,
            provider_details=None,
            part_delta_kind="text",
        )
        events = [
            PartStartEvent(index=0, part=part, previous_part_kind=None, event_kind="part_start"),
            PartDeltaEvent(index=0, delta=delta, event_kind="part_delta"),
        ]
        with patch(
            "docoracle.api.structured_llm_client.model_request_stream_sync",
            return_value=FakeStream(events),
        ):
            chunks = list(client.stream_ask("Q", "ctx"))

        assert "".join(chunks) == "Hello world"

    def test_stream_ask_async_yields_text(self, mock_config):
        from pydantic_ai.messages import PartStartEvent, TextPart

        client = _make_client(mock_config)
        part = TextPart(content="async!")
        events = [
            PartStartEvent(index=0, part=part, previous_part_kind=None, event_kind="part_start"),
        ]
        with patch(
            "docoracle.api.structured_llm_client.model_request_stream",
            return_value=FakeAsyncStream(events),
        ):

            async def collect():
                chunks = []
                async for chunk in client.stream_ask_async("Q", "ctx"):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(collect())

        assert "".join(chunks) == "async!"


class TestEventToText:
    def test_part_start_with_text(self):
        from pydantic_ai.messages import PartStartEvent, TextPart

        part = TextPart(content="hi")
        event = PartStartEvent(index=0, part=part, previous_part_kind=None, event_kind="part_start")
        assert _event_to_text(event) == "hi"

    def test_part_delta_with_text(self):
        from pydantic_ai.messages import PartDeltaEvent, TextPartDelta

        delta = TextPartDelta(
            content_delta="yo",
            provider_name=None,
            provider_details=None,
            part_delta_kind="text",
        )
        event = PartDeltaEvent(index=0, delta=delta, event_kind="part_delta")
        assert _event_to_text(event) == "yo"

    def test_returns_empty_for_unknown_event(self):
        assert _event_to_text(object()) == ""


# =============================================================================
# QA ENGINE INTEGRATION
# =============================================================================


class FakeEngineLLM:
    """LLM client double capturing the generation call and returning a response."""

    def __init__(self, response: AnswerResponse | None = None, error: Exception | None = None):
        self.response = response or AnswerResponse(answer="default")
        self.error = error
        self.system: str | None = None

    def embed(self, texts, model=None):  # noqa: ANN001
        return [[0.0] * 4 for _ in texts]

    def ask(
        self,
        question,
        context,
        output_model=None,
        model=None,
        temperature=None,
        max_tokens=None,
        system=None,
    ):  # noqa: ANN001
        self.system = system
        if self.error is not None:
            raise self.error
        return self.response

    async def ask_async(
        self,
        question,
        context,
        output_model=None,
        model=None,
        temperature=None,
        max_tokens=None,
        system=None,
    ):  # noqa: ANN001
        return self.ask(question, context, output_model, model, temperature, max_tokens, system)


def _build_engine(llm_client):
    import uuid

    from docoracle.backends.engine import QAEngine
    from docoracle.core.bm25_index import BM25Index
    from docoracle.core.hybrid_searcher import HybridSearcher
    from docoracle.core.models import Chunk
    from docoracle.core.semantic_index import SemanticIndex

    chunk = Chunk(
        text="alpha bravo charlie",
        chunk_id=str(uuid.uuid4()),
        module="m",
        component="c",
        version="v",
        page_id="p",
    )
    searcher = HybridSearcher(semantic=SemanticIndex(), bm25=BM25Index())
    searcher.add_chunks([chunk])
    return QAEngine(searcher, llm_client, config_path="/nonexistent.yaml")


class TestQAEngineStructuredOutput:
    def test_surfaces_confidence_citations_and_reasoning(self):
        response = AnswerResponse(
            answer="yes",
            confidence=0.82,
            citations=["mod:page#sec"],
            reasoning="the docs say so",
        )
        engine = _build_engine(FakeEngineLLM(response))

        result = engine.ask_detailed("alpha", k=1, retrieval="bm25")

        assert result["answer"] == "yes"
        assert result["confidence"] == pytest.approx(0.82)
        assert result["citations"] == ["mod:page#sec"]
        assert result["reasoning"] == "the docs say so"
        assert result["retrieval_mode"] == "bm25"
        assert result["retrieved_count"] == 1

    def test_ask_async_surfaces_fields(self):
        response = AnswerResponse(answer="async", confidence=0.4)
        engine = _build_engine(FakeEngineLLM(response))

        result = asyncio.run(engine.ask_detailed_async("alpha", k=1, retrieval="bm25"))

        assert result["answer"] == "async"
        assert result["confidence"] == pytest.approx(0.4)

    def test_uses_configured_system_prompt(self):
        llm = FakeEngineLLM(AnswerResponse(answer="ok"))
        engine = _build_engine(llm)

        engine.ask_detailed("alpha", k=1, retrieval="bm25")

        assert llm.system == engine.config.prompts.system
        assert llm.system

    def test_non_conforming_output_raises(self):
        engine = _build_engine(FakeEngineLLM(error=RuntimeError("bad schema")))

        with pytest.raises(RuntimeError, match="bad schema"):
            engine.ask_detailed("alpha", k=1, retrieval="bm25")
