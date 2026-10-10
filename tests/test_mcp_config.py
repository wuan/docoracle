"""Tests for MCP server configuration models and validation.

These cover the ``agent.mcp_servers`` discriminated union: transport-specific
fields, defaults, uniqueness, and inherited ``${VAR}`` resolution. No MCP
server is ever contacted here.
"""

import pytest
import yaml
from pydantic import ValidationError

from docoracle.core.config import Config


def _config(mcp_servers: list[dict]) -> Config:
    return Config(docoracle={"backend": "agent", "agent": {"mcp_servers": mcp_servers}})  # type: ignore[arg-type]


def test_mcp_servers_default_to_empty_list() -> None:
    config = Config()
    assert config.docoracle.agent.mcp_servers == []


def test_stdio_entry_is_accepted() -> None:
    config = _config([{"name": "ha", "transport": "stdio", "command": "uvx", "args": ["mcp-ha"]}])
    server = config.docoracle.agent.mcp_servers[0]
    assert isinstance(server, object)
    assert server.name == "ha"
    assert server.transport == "stdio"
    assert server.args == ["mcp-ha"]  # type: ignore[union-attr]


def test_streamable_http_entry_is_accepted() -> None:
    config = _config(
        [{"name": "web", "transport": "streamable-http", "url": "https://example.com/mcp"}]
    )
    server = config.docoracle.agent.mcp_servers[0]
    assert server.transport == "streamable-http"
    assert server.url == "https://example.com/mcp"  # type: ignore[union-attr]


def test_enabled_defaults_to_true() -> None:
    config = _config([{"name": "ha", "transport": "stdio", "command": "uvx"}])
    assert config.docoracle.agent.mcp_servers[0].enabled is True


def test_stdio_entry_may_not_supply_url() -> None:
    with pytest.raises(ValidationError):
        _config([{"name": "ha", "transport": "stdio", "command": "uvx", "url": "https://x/mcp"}])


def test_streamable_http_entry_may_not_supply_command() -> None:
    with pytest.raises(ValidationError):
        _config(
            [
                {
                    "name": "web",
                    "transport": "streamable-http",
                    "url": "https://x/mcp",
                    "command": "uvx",
                }
            ]
        )


def test_unknown_transport_is_rejected_naming_valid_transports() -> None:
    with pytest.raises(ValidationError) as exc:
        _config([{"name": "x", "transport": "sse", "url": "https://x/mcp"}])
    message = str(exc.value)
    assert "stdio" in message
    assert "streamable-http" in message


def test_stdio_entry_missing_command_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        _config([{"name": "ha", "transport": "stdio"}])
    assert "command" in str(exc.value)


def test_streamable_http_entry_missing_url_is_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        _config([{"name": "web", "transport": "streamable-http"}])
    assert "url" in str(exc.value)


def test_duplicate_server_names_are_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        _config(
            [
                {"name": "dup", "transport": "stdio", "command": "uvx"},
                {"name": "dup", "transport": "streamable-http", "url": "https://x/mcp"},
            ]
        )
    assert "dup" in str(exc.value)


def test_env_var_resolved_in_stdio_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HA_URL", "http://ha.local:8123")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "docoracle": {
                    "backend": "agent",
                    "agent": {
                        "mcp_servers": [
                            {
                                "name": "ha",
                                "transport": "stdio",
                                "command": "uvx",
                                "env": {"HA_URL": "${HA_URL}"},
                            }
                        ]
                    },
                }
            }
        )
    )
    config = Config.from_yaml(str(config_path))
    server = config.docoracle.agent.mcp_servers[0]
    assert server.env == {"HA_URL": "http://ha.local:8123"}  # type: ignore[union-attr]


def test_env_var_resolved_in_streamable_http_headers(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOKEN", "secret-token")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "docoracle": {
                    "backend": "agent",
                    "agent": {
                        "mcp_servers": [
                            {
                                "name": "web",
                                "transport": "streamable-http",
                                "url": "https://example.com/mcp",
                                "headers": {"Authorization": "${TOKEN}"},
                            }
                        ]
                    },
                }
            }
        )
    )
    config = Config.from_yaml(str(config_path))
    server = config.docoracle.agent.mcp_servers[0]
    assert server.headers == {"Authorization": "secret-token"}  # type: ignore[union-attr]
