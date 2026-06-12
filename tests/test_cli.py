"""Smoke tests for the CLI surface and end-to-end orchestration.

Real behavior tests live with their respective modules. These tests catch
regressions in the CLI surface (typer schema, help text, import-time
side effects) and verify the end-to-end pipeline wiring.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from youtubetranscriber.audio import DownloadResult
from youtubetranscriber.cli import app

runner = CliRunner()


def _fake_info(title: str = "Sample Video", video_id: str = "dQw4w9WgXcQ") -> DownloadResult:
    return DownloadResult(path=Path(""), title=title, video_id=video_id)


def _fake_audio_result(tmp_path: Path, video_id: str = "dQw4w9WgXcQ") -> DownloadResult:
    fake_audio = tmp_path / f"fake {video_id}.m4a"
    fake_audio.write_bytes(b"fake")
    return DownloadResult(path=fake_audio, title="Sample Video", video_id=video_id)


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
    assert "--interactive" in result.stdout


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


def test_end_to_end_uses_title_named_dir(tmp_path: Path) -> None:
    """Output directory should be named after the video title, not the ID."""
    fake_audio = _fake_audio_result(tmp_path)

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
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
    # The directory is named after the title, the file inside is named
    # after the video ID.
    assert (tmp_path / "Sample Video").exists()
    out_file = tmp_path / "Sample Video" / "dQw4w9WgXcQ.txt"
    assert out_file.exists()
    assert "Transcript written to" in result.stdout


def test_end_to_end_writes_segments(tmp_path: Path) -> None:
    """The text from segments should actually land in the output file."""
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [
        TranscriptSegment(text="Hello world.", start=0.0, end=1.0),
        TranscriptSegment(text="Goodbye.", start=1.0, end=2.0),
    ]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
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
    out_file = tmp_path / "Sample Video" / "dQw4w9WgXcQ.txt"
    content = out_file.read_text()
    assert "Hello world." in content
    assert "Goodbye." in content


def test_verbose_flag_prints_title(tmp_path: Path) -> None:
    """--verbose should print the title early so the user knows what's happening."""
    fake_audio = _fake_audio_result(tmp_path)

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
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
    assert "Sample Video" in result.stdout


def test_existing_dir_gets_indexed(tmp_path: Path, monkeypatch) -> None:
    """If the title-named dir already exists, the next one is auto-renamed."""
    # Pre-create the title-named dir to force a conflict
    (tmp_path / "Sample Video").mkdir()
    fake_audio = _fake_audio_result(tmp_path)

    # Stdin is not a TTY (default in test runner)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
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

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "Sample Video (1)").exists()
    # Original pre-existing dir is untouched (no transcript file added to it)
    assert not (tmp_path / "Sample Video" / "dQw4w9WgXcQ.txt").exists()
    out_file = tmp_path / "Sample Video (1)" / "dQw4w9WgXcQ.txt"
    assert out_file.exists()


def test_unsafe_title_gets_sanitized(tmp_path: Path) -> None:
    """Titles with slashes/colons should be sanitized for the filesystem."""
    fake_audio = _fake_audio_result(tmp_path)
    bad_info = _fake_info(title="Cool/Video: Part 1?")

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=bad_info),
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

    assert result.exit_code == 0, result.stdout
    # Should sanitize to "Cool_Video_ Part 1_" then strip trailing _ and collapse
    # We don't assert the exact string here; just that the dir was created
    # and didn't blow up.
    title_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(title_dirs) == 1
    assert "Cool_Video" in title_dirs[0].name
