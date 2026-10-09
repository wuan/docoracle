"""Agent, entry-point, and integration tests for MCP tools.

Every MCP toolset here is a stub; no real MCP server is spawned and no real
endpoint is contacted. A ``FunctionModel`` drives the agent so no network model
is used either.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pydantic
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.abstract import ToolsetTool

from docoracle.backends.agent import QAAgent, RetrievalState
from docoracle.backends.factory import create_answer_backend
from docoracle.backends.mcp import MCPConnectionError, MCPToolsetManager
from docoracle.core.config import Config
from docoracle.core.llm_outputs import AnswerResponse
from docoracle.core.search_index import ScoredChunk
from docoracle.server import main as server_main
from docoracle.server.main import app

# =============================================================================
# MOCKED MCP TOOLSET
# =============================================================================


class StubMCPToolset(AbstractToolset[RetrievalState]):
    """In-process MCP toolset stand-in exposing one ``echo`` tool.

    ``fail_on_call`` mimics a server that connected but dies mid-tool-call: the
    call raises ``ModelRetry``, pydantic-ai's normal tool-error signal.
    """

    def __init__(self, name: str = "stub", fail_on_call: bool = False) -> None:
        self._name = name
        self.fail_on_call = fail_on_call
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.entered = 0
        self.exited = 0

    @property
    def id(self) -> str:
        return self._name

    async def __aenter__(self) -> StubMCPToolset:
        self.entered += 1
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exited += 1

    async def get_tools(self, ctx: Any) -> dict[str, ToolsetTool[RetrievalState]]:
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
        return {
            "echo": ToolsetTool(
                toolset=self,
                tool_def=ToolDefinition(
                    name="echo", description="Echo text", parameters_json_schema=schema
                ),
                max_retries=1,
                args_validator=pydantic.TypeAdapter(dict).validator,
            )
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: Any, tool: ToolsetTool[RetrievalState]
    ) -> Any:
        self.calls.append((name, tool_args))
        if self.fail_on_call:
            raise ModelRetry("MCP server is unavailable")
        return f"echoed:{tool_args.get('text', '')}"


class _StubSearcher:
    def search(self, **kwargs: Any) -> list[ScoredChunk]:
        return []


class _StubLLM:
    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        return [[0.0, 0.0] for _ in texts]

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return "chat"


def _function_model(*, call_tool: bool = True, tool_name: str = "echo") -> FunctionModel:
    """A model that calls ``tool_name`` once, then produces an AnswerResponse."""

    def fn(messages: list[Any], info: AgentInfo) -> ModelResponse:
        seen = [
            part
            for message in messages
            for part in getattr(message, "parts", [])
            if isinstance(part, (ToolReturnPart, RetryPromptPart))
        ]
        if call_tool and not seen:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args={"text": "hi"})])
        output_tool = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=output_tool, args={"answer": "done"})])

    return FunctionModel(fn)


def _make_agent(toolsets: list[AbstractToolset[RetrievalState]] | None = None) -> QAAgent:
    with patch("docoracle.backends.agent.resolve_api_key", return_value="test-key"):
        return QAAgent(_StubSearcher(), llm_client=_StubLLM(), config=Config(), toolsets=toolsets)


def _install_model(qa: QAAgent, *, call_tool: bool = True, tool_name: str = "echo") -> None:
    """Rebuild the inner agent around a function model, keeping the mounted toolsets."""
    qa.agent = Agent(
        _function_model(call_tool=call_tool, tool_name=tool_name),
        output_type=AnswerResponse,
        deps_type=RetrievalState,
        tools=[qa.retrieval_tool.as_tool(), qa.summary_tool.as_tool()],
        toolsets=qa.mcp_toolsets,
    )


# =============================================================================
# 3. AGENT WIRING
# =============================================================================


def test_retrieval_tool_always_mounted() -> None:
    qa = _make_agent()
    assert "retrieve_documents" in qa.agent._function_toolset.tools


def test_retrieval_tool_survives_mcp_configuration() -> None:
    qa = _make_agent([StubMCPToolset("s")])
    assert "retrieve_documents" in qa.agent._function_toolset.tools
    assert "summarize" in qa.agent._function_toolset.tools


def test_enabled_toolsets_registered_with_agent() -> None:
    first, second = StubMCPToolset("a"), StubMCPToolset("b")
    qa = _make_agent([first, second])
    assert qa.agent._user_toolsets == [first, second]


def test_disabled_server_never_connected_at_construction() -> None:
    # build_mcp_toolsets skips disabled servers, so an agent constructed with
    # only disabled servers has no toolsets and never contacts anything.
    from docoracle.backends.mcp import build_mcp_toolsets
    from docoracle.core.config import StreamableHTTPMCPServerConfig

    server = StreamableHTTPMCPServerConfig(
        name="off",
        transport="streamable-http",
        url="https://unreachable.invalid/mcp",
        enabled=False,
    )
    pairs = build_mcp_toolsets([server])
    qa = _make_agent([toolset for _, toolset in pairs])
    assert qa.mcp_toolsets == []


def test_factory_passes_toolsets_to_agent() -> None:
    config = Config(docoracle={"backend": "agent"})  # type: ignore[arg-type]
    stub = StubMCPToolset("s")
    with patch("docoracle.backends.agent.resolve_api_key", return_value="test-key"):
        backend = create_answer_backend(_StubSearcher(), _StubLLM(), config, toolsets=[stub])
    assert isinstance(backend, QAAgent)
    assert backend.mcp_toolsets == [stub]


def test_engine_backend_ignores_toolsets() -> None:
    from docoracle.backends.engine import QAEngine

    stub = StubMCPToolset("s")
    backend = create_answer_backend(_StubSearcher(), _StubLLM(), Config(), toolsets=[stub])
    assert isinstance(backend, QAEngine)
    assert not hasattr(backend, "mcp_toolsets")


class _FailingEntryToolset(StubMCPToolset):
    async def __aenter__(self) -> _FailingEntryToolset:
        raise RuntimeError("cannot start server")


def test_enabled_server_connect_failure_is_fail_fast() -> None:
    import asyncio

    manager = MCPToolsetManager([("broken", _FailingEntryToolset("broken"))])  # type: ignore[list-item]

    async def run() -> None:
        with pytest.raises(MCPConnectionError) as exc:
            async with manager:
                pass
        assert exc.value.server_name == "broken"
        assert "cannot start server" in str(exc.value)

    asyncio.run(run())


def test_server_startup_fails_fast_and_serves_no_request() -> None:
    failing = _FailingEntryToolset("broken")

    with (
        patch.object(server_main, "build_mcp_toolsets", return_value=[("broken", failing)]),
        pytest.raises(MCPConnectionError),
        TestClient(app) as client,
    ):
        # Startup aborted, so no request is ever served.
        client.get("/health")


def test_agent_construction_fails_fast_via_factory_path() -> None:
    import asyncio

    # A server that cannot connect means the agent is never constructed.
    manager = MCPToolsetManager([("broken", _FailingEntryToolset("broken"))])  # type: ignore[list-item]
    constructed: list[QAAgent] = []

    async def run() -> None:
        with pytest.raises(MCPConnectionError):
            async with manager:
                constructed.append(_make_agent(manager.toolsets))

    asyncio.run(run())
    assert constructed == []


# =============================================================================
# 5. INTEGRATION: MOCKED MCP TOOL
# =============================================================================


def test_agent_calls_mocked_mcp_tool_and_returns_answer() -> None:
    stub = StubMCPToolset("stub")
    qa = _make_agent([stub])
    _install_model(qa)

    result = qa.ask("ping")

    assert isinstance(result, AnswerResponse)
    assert result.answer == "done"
    assert stub.calls == [("echo", {"text": "hi"})]


def test_agent_survives_dying_mcp_server() -> None:
    stub = StubMCPToolset("stub", fail_on_call=True)
    qa = _make_agent([stub])
    _install_model(qa)

    result = qa.ask("ping")

    assert result.answer == "done"
    # The failing tool was invoked but the run completed instead of crashing.
    assert stub.calls == [("echo", {"text": "hi"})]


def test_existing_agent_behavior_without_mcp() -> None:
    qa = _make_agent()
    assert qa.mcp_toolsets == []
    _install_model(qa, call_tool=False)
    assert qa.ask("ping").answer == "done"


# =============================================================================
# 4. ENTRY POINT LIFECYCLE
# =============================================================================


def test_server_lifespan_connects_once_reused_across_requests() -> None:
    toolset = StubMCPToolset("s")
    with (
        patch.object(server_main, "build_mcp_toolsets", return_value=[("s", toolset)]),
        TestClient(app) as client,
    ):
        assert toolset.entered == 1
        client.get("/health")
        client.get("/health")
        # Connections are established once at startup, not per request.
        assert toolset.entered == 1
    # ... and closed at shutdown.
    assert toolset.exited == 1


def test_server_lifespan_without_servers_opens_nothing() -> None:
    with TestClient(app) as client:
        assert server_main._mcp_manager is not None
        assert server_main._mcp_manager.toolsets == []
        assert client.get("/health").status_code == 200


# =============================================================================
# CLI lifecycle
# =============================================================================


def test_cli_no_servers_starts_without_connecting(monkeypatch) -> None:
    from docoracle import cli as cli_module

    built: list[Any] = []

    def fake_build(servers: Any) -> list[Any]:
        built.append(servers)
        return []

    monkeypatch.setattr(cli_module, "build_mcp_toolsets", fake_build)

    class _Store:
        def load(self, path: str) -> bool:
            return True

        def __len__(self) -> int:
            return 1

        def search(self, **kwargs: Any) -> list[Any]:
            return []

    monkeypatch.setattr(cli_module, "HybridSearcher", _Store)
    monkeypatch.setattr(cli_module, "StructuredLLMClient", lambda *a, **k: _StubLLM())

    result = CliRunner().invoke(cli_module.cli, ["search", "q"])

    assert result.exit_code == 0
    # No servers configured: the builder is called with an empty list and the
    # lifecycle opens no connection.
    assert built == [[]]


def test_cli_connections_last_for_invocation(monkeypatch) -> None:
    from docoracle import cli as cli_module

    toolset = StubMCPToolset("s")
    monkeypatch.setattr(cli_module, "build_mcp_toolsets", lambda servers: [("s", toolset)])

    class _Store:
        def load(self, path: str) -> bool:
            return True

        def __len__(self) -> int:
            return 1

        def search(self, **kwargs: Any) -> list[Any]:
            # Inside the invocation the toolset is connected.
            assert toolset.entered == 1
            assert toolset.exited == 0
            return []

    monkeypatch.setattr(cli_module, "HybridSearcher", _Store)
    monkeypatch.setattr(cli_module, "StructuredLLMClient", lambda *a, **k: _StubLLM())

    result = CliRunner().invoke(cli_module.cli, ["search", "q"])

    assert result.exit_code == 0
    # The connection is closed when the command finishes.
    assert toolset.entered == 1
    assert toolset.exited == 1
