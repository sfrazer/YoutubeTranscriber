"""Smoke tests for the CLI skeleton.

Real behavior tests live with their respective modules. These tests exist to
catch regressions in the CLI surface itself (argparse/typer schema, help
text, import-time side effects).
"""

from typer.testing import CliRunner

from youtubetranscriber.cli import app

runner = CliRunner()


def test_help_works():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Transcribe" in result.stdout


def test_transcribe_command_is_registered():
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.stdout
    assert "--diarize" in result.stdout
    assert "--summarize" in result.stdout


def test_transcribe_not_yet_implemented():
    result = runner.invoke(app, ["transcribe", "https://youtu.be/dQw4w9WgXcQ"])
    assert result.exit_code != 0
