"""Tests for CLI invocation of `docoracle ask` with the --retrieval flag."""

from click.testing import CliRunner

from docoracle.cli import cli


def test_ask_with_invalid_retrieval_mode_reports_error():
    runner = CliRunner()
    result = runner.invoke(cli, ["ask", "some question", "--retrieval", "nonsense"])
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "hybrid" in result.output


def test_search_with_invalid_retrieval_mode_reports_error():
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "some text", "--retrieval", "nonsense"])
    assert result.exit_code != 0


def test_ask_help_includes_retrieval_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["ask", "--help"])
    assert result.exit_code == 0
    assert "--retrieval" in result.output
    assert "hybrid" in result.output
    assert "semantic" in result.output
    assert "bm25" in result.output


def test_search_help_includes_retrieval_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "--help"])
    assert result.exit_code == 0
    assert "--retrieval" in result.output
