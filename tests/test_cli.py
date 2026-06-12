"""Smoke tests for the CLI surface and end-to-end orchestration.

Real behavior tests live with their respective modules. These tests catch
regressions in the CLI surface (typer schema, help text, import-time
side effects) and verify the end-to-end pipeline wiring.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from youtubetranscriber.cli import app

runner = CliRunner()


# --- Help text and surface -------------------------------------------------


def test_help_works():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Transcribe" in result.stdout


def test_help_includes_all_options():
    result = runner.invoke(app, ["--help"])
    assert "--model" in result.stdout
    assert "--format" in result.stdout
    assert "--diarize" in result.stdout
    assert "--summarize" in result.stdout
    assert "--output-dir" in result.stdout


# --- Argument validation ---------------------------------------------------


def test_invalid_model_rejected():
    result = runner.invoke(app, ["https://youtu.be/dQw4w9WgXcQ", "--model", "huge"])
    assert result.exit_code != 0
    assert "Invalid model" in result.stdout or "Invalid model" in (result.output or "")


def test_invalid_format_rejected():
    result = runner.invoke(app, ["https://youtu.be/dQw4w9WgXcQ", "--format", "docx"])
    assert result.exit_code != 0


def test_invalid_url_rejected():
    """Bad URLs should fail with InvalidURLError, not crash."""
    result = runner.invoke(app, ["not a url"])
    assert result.exit_code != 0


# --- End-to-end with mocked pipeline ---------------------------------------


def test_end_to_end_with_mocked_pipeline(tmp_path: Path) -> None:
    """Full CLI invocation with audio + transcribe mocked."""
    fake_audio = tmp_path / "fake.m4a"
    fake_audio.write_bytes(b"fake")

    with (
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio),
        patch("youtubetranscriber.cli.do_transcribe", return_value=[]),
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0, result.stdout + (result.output or "")
    # Output file should have been created (even if empty)
    expected_output = tmp_path / "dQw4w9WgXcQ" / "dQw4w9WgXcQ.txt"
    assert expected_output.exists()
    assert "Transcript written to" in result.stdout


def test_end_to_end_writes_segments(tmp_path: Path) -> None:
    """The text from segments should actually land in the output file."""
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = tmp_path / "fake.m4a"
    fake_audio.write_bytes(b"fake")
    fake_segments = [
        TranscriptSegment(text="Hello world.", start=0.0, end=1.0),
        TranscriptSegment(text="Goodbye.", start=1.0, end=2.0),
    ]

    with (
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio),
        patch("youtubetranscriber.cli.do_transcribe", return_value=fake_segments),
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0, result.stdout
    out_file = tmp_path / "dQw4w9WgXcQ" / "dQw4w9WgXcQ.txt"
    content = out_file.read_text()
    assert "Hello world." in content
    assert "Goodbye." in content


def test_verbose_flag_prints_video_id(tmp_path: Path) -> None:
    """--verbose should print the video ID early so the user knows what's happening."""
    fake_audio = tmp_path / "fake.m4a"
    fake_audio.write_bytes(b"fake")

    with (
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio),
        patch("youtubetranscriber.cli.do_transcribe", return_value=[]),
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--verbose",
            ],
        )

    assert result.exit_code == 0
    assert "dQw4w9WgXcQ" in result.stdout
