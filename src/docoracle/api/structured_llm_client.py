"""LLM client for embeddings and chat.

Embeddings go directly to the configured Mistral-compatible REST endpoint
(via :mod:`requests`) because pydantic-ai has no embedding API. Chat and
streaming go through pydantic-ai's direct API so we get type-safe structured
outputs and proper async support.
"""

import time
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import requests
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRequest, ModelSettings
from pydantic_ai.direct import (
    model_request,
    model_request_stream,
    model_request_stream_sync,
    model_request_sync,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from docoracle.core.config import Config, get_config, resolve_api_key
from docoracle.core.llm_outputs import AnswerResponse


def _model_name(value: str) -> str:
    """Strip any ``provider:`` prefix from a model identifier.

    ``pydantic-ai`` uses ``"provider:model"`` strings to dispatch to a
    specific provider implementation (e.g. ``"mistral:..."`` -> the optional
    Mistral client). We deliberately use the OpenAI provider against an
    OpenAI-compatible base URL instead, so we accept any incoming prefix
    and pass just the model name itself to the provider.
    """
    _, _, tail = value.rpartition(":")
    return tail or value


def build_chat_model(name: str, api_url: str, api_key: str) -> Model:
    """Return a ``Model`` for ``name`` using an OpenAI-compatible base URL.

    Shared by :class:`StructuredLLMClient` and the pydantic-ai agent so both
    use the same provider setup. We only depend on the bundled OpenAI
    provider, which already speaks to any OpenAI-compatible REST endpoint
    via ``base_url`` — Mistral's ``/v1`` API is one of those.
    """
    provider = OpenAIProvider(base_url=api_url, api_key=api_key)
    return OpenAIChatModel(_model_name(name), provider=provider)


class StructuredLLMClient:
    """LLM client with embeddings, chat, streaming, and structured outputs."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config: Config = get_config(config_path)
        self.api_key = resolve_api_key(self.config)
        self.api_url = self.config.llm.api_url
        self.timeout = self.config.llm.timeout
        self.embedding_model = self.config.embedding.model
        self.rate_limit_delay = self.config.embedding.rate_limit_delay
        # Cache for the lazily-built chat Model (see ``_resolve_model``).
        self._cached_model_key: str | None = None
        self._cached_model: Model | None = None
        # Cache for the lazily-built structured-output agent (see
        # ``_structured_agent``), keyed by (model, output model, system).
        self._cached_agent_key: tuple[Any, ...] | None = None
        self._cached_agent: Agent[Any, Any] | None = None

    # ------------------------------------------------------------------ helpers

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _resolve_model(self, name: str | None) -> Model:
        """Build (and cache) the ``Model`` instance for ``name``.

        Cached by the raw config string so we don't re-instantiate the
        provider on every request.
        """
        key = name or self.config.generation.model
        if key == self._cached_model_key:
            return self._cached_model  # type: ignore[return-value]
        model = self._build_chat_model(key)
        self._cached_model_key = key
        self._cached_model = model
        return model

    def _build_chat_model(self, name: str) -> Model:
        """Return a ``Model`` for ``name`` using our OpenAI-compatible base URL.

        ``pydantic-ai`` ships providers for each vendor; importing one
        (e.g. ``mistral:...``) requires its optional dependency. We only
        depend on the bundled OpenAI provider, which already speaks to any
        OpenAI-compatible REST endpoint via ``base_url`` — Mistral's
        ``/v1`` API is one of those, so ``OpenAIChatModel`` is enough.
        """
        return build_chat_model(name, self.api_url, self.api_key)

    def _model_settings(self, temperature: float | None, max_tokens: int | None) -> ModelSettings:
        settings: ModelSettings = {}
        if temperature is not None:
            settings["temperature"] = temperature
        if max_tokens is not None:
            settings["max_tokens"] = max_tokens
        return settings

    def _messages_from_dicts(self, messages: Sequence[dict[str, Any]]) -> list[ModelMessage]:
        """Convert OpenAI-style message dicts to pydantic-ai ``ModelMessage``s.

        A single ``user`` message is the most common shape, so that's
        represented directly as a ``ModelRequest``. Multi-message inputs are
        joined into one prompt (pydantic-ai's direct API doesn't carry chat
        history through, so this is the supported degradation).
        """
        if not messages:
            return [ModelRequest.user_text_prompt("")]
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        return [ModelRequest.user_text_prompt("\n\n".join(parts))]

    def _text_from_response(self, response: ModelResponse) -> str:
        return "".join(part.content for part in response.parts if isinstance(part, TextPart))

    # ------------------------------------------------------------------ chat

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return raw text."""
        response = model_request_sync(
            model=self._resolve_model(model),
            messages=self._messages_from_dicts(messages),
            model_settings=self._model_settings(
                temperature if temperature is not None else self.config.generation.temperature,
                max_tokens,
            ),
        )
        return self._text_from_response(response)

    async def chat_async(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Async version of :meth:`chat`."""
        response = await model_request(
            model=self._resolve_model(model),
            messages=self._messages_from_dicts(messages),
            model_settings=self._model_settings(
                temperature if temperature is not None else self.config.generation.temperature,
                max_tokens,
            ),
        )
        return self._text_from_response(response)

    # ------------------------------------------------------------------ structured

    def _structured_agent(
        self,
        output_model: type[BaseModel],
        system: str | None,
        model: str | None,
    ) -> Agent[Any, Any]:
        """Build (and cache) the structured-output agent for these settings.

        The response schema is enforced by the provider through pydantic-ai's
        native structured output (``output_type``), not by embedding JSON in the
        prompt and reparsing the model's text.
        """
        key = (model or self.config.generation.model, output_model, system)
        if key == self._cached_agent_key and self._cached_agent is not None:
            return self._cached_agent
        agent: Agent[Any, Any] = Agent(
            model=self._resolve_model(model),
            output_type=output_model,
            instructions=system or self.config.prompts.system,
        )
        self._cached_agent_key = key
        self._cached_agent = agent
        return agent

    def _structured_prompt(self, question: str, context: str) -> str:
        """Render the configured user-prompt template for two placeholders."""
        template = self.config.prompts.user
        try:
            return template.format(question=question, context=context)
        except (KeyError, IndexError):
            return f"QUESTION: {question}\n\nCONTEXT:\n{context}"

    def ask(
        self,
        question: str,
        context: str = "",
        output_model: type[BaseModel] = AnswerResponse,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> BaseModel:
        """Ask a question and return a provider-validated ``output_model`` instance."""
        agent = self._structured_agent(output_model, system, model)
        result = agent.run_sync(
            self._structured_prompt(question, context),
            model_settings=self._model_settings(
                temperature if temperature is not None else self.config.generation.temperature,
                max_tokens,
            ),
        )
        return result.output

    async def ask_async(
        self,
        question: str,
        context: str = "",
        output_model: type[BaseModel] = AnswerResponse,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> BaseModel:
        """Async version of :meth:`ask`."""
        agent = self._structured_agent(output_model, system, model)
        result = await agent.run(
            self._structured_prompt(question, context),
            model_settings=self._model_settings(
                temperature if temperature is not None else self.config.generation.temperature,
                max_tokens,
            ),
        )
        return result.output

    # ------------------------------------------------------------------ streaming

    def stream_ask(
        self,
        question: str,
        context: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield raw text chunks for ``question`` (sync)."""
        prompt = self._build_stream_prompt(question, context)
        with model_request_stream_sync(
            model=self._resolve_model(model),
            messages=[ModelRequest.user_text_prompt(prompt)],
            model_settings=self._model_settings(
                temperature if temperature is not None else self.config.generation.temperature,
                max_tokens,
            ),
        ) as stream:
            for event in stream:
                text = _event_to_text(event)
                if text:
                    yield text

    async def stream_ask_async(
        self,
        question: str,
        context: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield raw text chunks for ``question`` (async)."""
        prompt = self._build_stream_prompt(question, context)
        async with model_request_stream(
            model=self._resolve_model(model),
            messages=[ModelRequest.user_text_prompt(prompt)],
            model_settings=self._model_settings(
                temperature if temperature is not None else self.config.generation.temperature,
                max_tokens,
            ),
        ) as stream:
            async for event in stream:
                text = _event_to_text(event)
                if text:
                    yield text

    def _build_stream_prompt(self, question: str, context: str) -> str:
        if context:
            return (
                "You are a helpful assistant answering questions about technical "
                "documentation. Answer using only the provided context.\n\n"
                f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nAnswer:"
            )
        return (
            "You are a helpful assistant. Answer the following question.\n\n"
            f"QUESTION: {question}\n\nAnswer:"
        )

    # ------------------------------------------------------------------ embeddings

    def embed(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Embed ``texts`` one-by-one or as a single batch call.

        Try the batch call first, fall back to per-text calls if the API
        rejects the batch shape.
        """
        if not texts:
            return []
        if len(texts) == 1:
            return self._embed_single(texts[0], model or self.embedding_model)

        try:
            return self._embed_batch(texts, model or self.embedding_model)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 422:
                return [self._embed_single(t, model or self.embedding_model)[0] for t in texts]
            raise
        except requests.RequestException:
            # Network issues: fall back to single embedding
            return [self._embed_single(t, model or self.embedding_model)[0] for t in texts]

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        model: str | None = None,
    ) -> list[list[float]]:
        """Embed ``texts`` in batches of ``batch_size`` with a configurable rate-limit pause."""
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self.embed(batch, model=model))
            if len(batch) == batch_size:
                time.sleep(self.rate_limit_delay)
        return results

    def _embed_single(self, text: str, model: str) -> list[list[float]]:
        url = f"{self.api_url}/embeddings"
        for payload in ({"model": model, "input": text}, {"model": model, "inputs": text}):
            response = requests.post(
                url, headers=self._headers(), json=payload, timeout=self.timeout
            )
            if response.status_code == 422 and payload.get("input") is not None:
                continue
            response.raise_for_status()
            data = response.json()
            if "data" in data and data["data"]:
                first = data["data"][0]
                if isinstance(first, dict):
                    return [item["embedding"] for item in data["data"]]
                return data["data"]
            if "embedding" in data:
                return [data["embedding"]]
            raise ValueError(f"Unexpected embedding response format: {data}")
        raise ValueError("Embedding endpoint rejected both 'input' and 'inputs' payload shapes")

    def _embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        url = f"{self.api_url}/embeddings"
        for payload in ({"model": model, "inputs": texts}, {"model": model, "input": texts}):
            response = requests.post(
                url, headers=self._headers(), json=payload, timeout=self.timeout
            )
            if response.status_code == 422 and payload.get("input") is not None:
                continue
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("data", [])
            if not embeddings:
                if "embedding" in data:
                    return [data["embedding"]]
                raise ValueError(f"Unexpected embedding response format: {data}")
            if isinstance(embeddings[0], dict):
                return [item["embedding"] for item in embeddings]
            return embeddings
        raise ValueError("Embedding endpoint rejected both 'inputs' and 'input' payload shapes")


def _event_to_text(event: Any) -> str:
    """Pull text out of a streamed event, if any."""
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
        return event.part.content
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        return event.delta.content_delta
    return ""
