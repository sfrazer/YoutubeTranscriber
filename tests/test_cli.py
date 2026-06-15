"""Smoke tests for the CLI surface and end-to-end orchestration.

Real behavior tests live with their respective modules. These tests catch
regressions in the CLI surface (typer schema, help text, import-time
side effects) and verify the end-to-end pipeline wiring.

Note: user-facing [ytx] status messages go to stderr (not stdout),
so we assert on result.stderr for those. stdout is reserved for
typer's --help screen and any future machine-readable output.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from youtubetranscriber.audio import DownloadResult, VideoInfo
from youtubetranscriber.cli import app

runner = CliRunner()


def _fake_info(title: str = "Sample Video", video_id: str = "dQw4w9WgXcQ") -> VideoInfo:
    return VideoInfo(title=title, video_id=video_id)


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
    # typer's BadParameter message comes via its own output stream
    assert "Invalid model" in (result.output or "")


def test_invalid_format_rejected():
    result = runner.invoke(app, ["https://youtu.be/dQw4w9WgXcQ", "--format", "docx"])
    assert result.exit_code != 0


def test_invalid_url_rejected():
    """Bad URLs should fail with InvalidURLError, not crash."""
    result = runner.invoke(app, ["not a url"])
    assert result.exit_code != 0
    # Status messages go to stderr — the user-actionable error should be there,
    # not a Python traceback.
    assert "Not a YouTube URL" in (result.stderr or "")


def test_invalid_url_no_traceback_in_real_subprocess(tmp_path: Path) -> None:
    """Real subprocess (no catch_exceptions) should show clean error, not traceback.

    CliRunner's catch_exceptions=True masks the original bug where
    InvalidURLError leaked as a Python traceback. This test runs the
    CLI as a real subprocess to verify the user-facing behavior.
    """
    result = subprocess.run(
        [sys.executable, "-m", "youtubetranscriber.cli", "not a url", "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    # Exit code is non-zero
    assert result.returncode != 0
    # Combined output should NOT contain a Python traceback
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined
    # The user-actionable error message should be visible
    assert "Not a YouTube URL" in combined


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

    assert result.exit_code == 0, result.stderr
    # The directory is named after the title, the file inside is named
    # after the video ID.
    assert (tmp_path / "Sample Video").exists()
    out_file = tmp_path / "Sample Video" / "dQw4w9WgXcQ.txt"
    assert out_file.exists()
    # Status messages go to stderr
    assert "Transcript written to" in (result.stderr or "")


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

    assert result.exit_code == 0, result.stderr
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
    # Title appears on stderr
    assert "Sample Video" in (result.stderr or "")


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

    assert result.exit_code == 0, result.stderr
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

    assert result.exit_code == 0, result.stderr
    # Should sanitize to "Cool_Video_ Part 1_" then strip trailing _ and collapse
    # We don't assert the exact string here; just that the dir was created
    # and didn't blow up.
    title_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(title_dirs) == 1
    assert "Cool_Video" in title_dirs[0].name


# --- --prefer-captions path ------------------------------------------------


def test_prefer_captions_uses_captions_when_available(tmp_path: Path) -> None:
    """With --prefer-captions, captions are used and Whisper is NOT called."""
    from youtubetranscriber.transcribe import TranscriptSegment

    caption_segments = [
        TranscriptSegment(text="From captions.", start=0.0, end=1.0),
    ]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch(
            "youtubetranscriber.cli.captions.fetch_captions",
            return_value=caption_segments,
        ) as fetch_mock,
        patch("youtubetranscriber.cli.audio.download_audio") as dl_mock,
        patch("youtubetranscriber.cli.do_transcribe") as whisper_mock,
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--prefer-captions",
            ],
        )

    assert result.exit_code == 0, result.stderr
    assert fetch_mock.called
    # Crucial: download_audio and do_transcribe should NOT have been called
    assert not dl_mock.called, "download_audio should be skipped with --prefer-captions"
    assert not whisper_mock.called, "Whisper should be skipped with --prefer-captions"

    out_file = tmp_path / "Sample Video" / "dQw4w9WgXcQ.txt"
    content = out_file.read_text()
    assert "From captions." in content


def test_prefer_captions_falls_back_to_whisper(tmp_path: Path) -> None:
    """If captions fail, fall through to audio download + Whisper."""
    from youtubetranscriber.captions import CaptionsUnavailableError
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    whisper_segments = [TranscriptSegment(text="From Whisper.", start=0.0, end=1.0)]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch(
            "youtubetranscriber.cli.captions.fetch_captions",
            side_effect=CaptionsUnavailableError("no captions"),
        ) as fetch_mock,
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio) as dl_mock,
        patch("youtubetranscriber.cli.do_transcribe", return_value=whisper_segments),
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--prefer-captions",
            ],
        )

    assert result.exit_code == 0, result.stderr
    assert fetch_mock.called
    # Fall through happened
    assert dl_mock.called
    # Status message on stderr
    assert "Falling back to Whisper" in (result.stderr or "")

    out_file = tmp_path / "Sample Video" / "dQw4w9WgXcQ.txt"
    content = out_file.read_text()
    assert "From Whisper." in content


def test_prefer_captions_falls_back_on_empty_captions(tmp_path: Path) -> None:
    """An empty caption list is treated as 'not available' and falls through."""
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    whisper_segments = [TranscriptSegment(text="From Whisper.", start=0.0, end=1.0)]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch("youtubetranscriber.cli.captions.fetch_captions", return_value=[]),
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio),
        patch("youtubetranscriber.cli.do_transcribe", return_value=whisper_segments),
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--prefer-captions",
            ],
        )

    assert result.exit_code == 0, result.stderr
    # Both messages (CaptionsUnavailable and empty captions) emit
    # "falling back to Whisper" (capitalization varies at the start
    # of the sentence, so we case-fold the whole comparison).
    assert "falling back to whisper" in (result.stderr or "").lower()


def test_without_prefer_captions_skips_captions(tmp_path: Path) -> None:
    """Default behavior: don't even try captions, go straight to Whisper."""
    fake_audio = _fake_audio_result(tmp_path)

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch("youtubetranscriber.cli.captions.fetch_captions") as fetch_mock,
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

    assert result.exit_code == 0, result.stderr
    # captions.fetch_captions should NOT have been called
    assert not fetch_mock.called


def test_help_includes_prefer_captions() -> None:
    result = runner.invoke(app, ["--help"])
    assert "--prefer-captions" in result.stdout
    assert "--format" in result.stdout


# --- SRT and JSON end-to-end ----------------------------------------------


def test_end_to_end_writes_srt(tmp_path: Path) -> None:
    """--format srt should produce a valid SRT file."""
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [
        TranscriptSegment(text="Hello.", start=0.0, end=1.5),
        TranscriptSegment(text="World.", start=1.5, end=3.0),
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
                "--format",
                "srt",
            ],
        )

    assert result.exit_code == 0, result.stderr
    out_file = tmp_path / "Sample Video" / "dQw4w9WgXcQ.srt"
    assert out_file.exists()
    content = out_file.read_text()
    assert "00:00:00,000 --> 00:00:01,500" in content
    assert "Hello." in content


def test_end_to_end_writes_json_with_metadata(tmp_path: Path) -> None:
    """--format json should include video_id and title in the output."""
    import json

    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [TranscriptSegment(text="Hello.", start=0.0, end=1.0)]

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
                "--format",
                "json",
            ],
        )

    assert result.exit_code == 0, result.stderr
    out_file = tmp_path / "Sample Video" / "dQw4w9WgXcQ.json"
    data = json.loads(out_file.read_text())
    assert data["video_id"] == "dQw4w9WgXcQ"
    assert data["title"] == "Sample Video"
    assert data["segments"][0]["text"] == "Hello."


# --- --diarize path -------------------------------------------------------


def test_diarize_assigns_speakers_to_whisper_segments(tmp_path: Path) -> None:
    """With --diarize, Whisper segments get speaker labels from pyannote."""
    from youtubetranscriber.merge import SpeakerSpan
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [
        TranscriptSegment(text="hi", start=0.0, end=1.0),
        TranscriptSegment(text="there", start=2.0, end=3.0),
    ]
    fake_spans = [
        SpeakerSpan(speaker="SPEAKER_00", start=0.0, end=1.0),
        SpeakerSpan(speaker="SPEAKER_01", start=2.0, end=3.0),
    ]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio),
        patch("youtubetranscriber.cli.do_transcribe", return_value=fake_segments),
        patch("youtubetranscriber.cli.diarize.diarize", return_value=fake_spans),
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--diarize",
            ],
        )

    assert result.exit_code == 0, result.stderr
    out_file = tmp_path / "Sample Video" / "dQw4w9WgXcQ.txt"
    content = out_file.read_text()
    # txt writer prefixes speakers in brackets
    assert "[SPEAKER_00] hi" in content
    assert "[SPEAKER_01] there" in content


def test_diarize_skips_when_segments_came_from_captions(tmp_path: Path) -> None:
    """If captions were used, --diarize is a no-op (with a notice)."""
    from youtubetranscriber.transcribe import TranscriptSegment

    caption_segments = [TranscriptSegment(text="caption text", start=0.0, end=1.0)]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch(
            "youtubetranscriber.cli.captions.fetch_captions",
            return_value=caption_segments,
        ),
        patch("youtubetranscriber.cli.audio.download_audio") as dl_mock,
        patch("youtubetranscriber.cli.do_transcribe") as whisper_mock,
        patch("youtubetranscriber.cli.diarize.diarize") as diarize_mock,
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--prefer-captions",
                "--diarize",
            ],
        )

    assert result.exit_code == 0, result.stderr
    # Diarization should NOT have been called
    assert not diarize_mock.called
    # Whisper and download also not called
    assert not whisper_mock.called
    assert not dl_mock.called
    # The notice should appear on stderr
    assert "can't be diarized" in (result.stderr or "")


def test_diarize_fails_clearly_when_hf_token_missing(tmp_path: Path) -> None:
    """Missing HF_TOKEN should fail with the user-actionable error."""
    from youtubetranscriber.diarize import HfTokenMissingError
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [TranscriptSegment(text="hi", start=0.0, end=1.0)]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio),
        patch("youtubetranscriber.cli.do_transcribe", return_value=fake_segments),
        patch(
            "youtubetranscriber.cli.diarize.diarize",
            side_effect=HfTokenMissingError("HF_TOKEN is not set"),
        ),
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--diarize",
            ],
        )

    # The exception should propagate to the user (non-zero exit)
    assert result.exit_code != 0
    # The user-actionable error should be printed to stderr
    assert "HF_TOKEN" in (result.stderr or "")


def test_diarize_passes_num_speakers(tmp_path: Path) -> None:
    """The --num-speakers flag should reach the diarize function."""
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [TranscriptSegment(text="hi", start=0.0, end=1.0)]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio),
        patch("youtubetranscriber.cli.do_transcribe", return_value=fake_segments),
        patch("youtubetranscriber.cli.diarize.diarize", return_value=[]) as diarize_mock,
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--diarize",
                "--num-speakers",
                "3",
            ],
        )

    assert result.exit_code == 0, result.stderr
    diarize_mock.assert_called_once()
    call_kwargs = diarize_mock.call_args.kwargs
    assert call_kwargs.get("num_speakers") == 3


def test_help_includes_diarize_options() -> None:
    result = runner.invoke(app, ["--help"])
    assert "--diarize" in result.stdout
    assert "--num-speakers" in result.stdout
    assert "--min-speakers" in result.stdout
    assert "--max-speakers" in result.stdout


# --- --summarize path -----------------------------------------------------


def test_summarize_calls_summarizer_and_writes_sidecar(tmp_path: Path) -> None:
    """With --summarize, the summarizer runs and a .summary.md is written."""
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [TranscriptSegment(text="Some transcript content.", start=0.0, end=1.0)]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio),
        patch("youtubetranscriber.cli.do_transcribe", return_value=fake_segments),
        patch(
            "youtubetranscriber.cli.summarize.summarize",
            return_value="## TL;DR\nTest summary.",
        ) as sum_mock,
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--summarize",
            ],
        )

    assert result.exit_code == 0, result.stderr
    assert sum_mock.called
    summary_file = tmp_path / "Sample Video" / "dQw4w9WgXcQ.summary.md"
    assert summary_file.exists()
    content = summary_file.read_text()
    assert "Test summary." in content
    # Header should include the title
    assert "Sample Video" in content


def test_summarize_passes_model_to_summarizer(tmp_path: Path) -> None:
    """--summary-model should reach the summarizer."""
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio),
        patch("youtubetranscriber.cli.do_transcribe", return_value=fake_segments),
        patch("youtubetranscriber.cli.summarize.summarize", return_value="ok") as sum_mock,
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--summarize",
                "--summary-model",
                "gpt-oss:20b",
            ],
        )

    assert result.exit_code == 0, result.stderr
    assert sum_mock.call_args.kwargs.get("model") == "gpt-oss:20b"


def test_summarize_uses_custom_prompt_file(tmp_path: Path) -> None:
    """--summary-prompt should be read and passed to the summarizer."""
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]
    custom_prompt = tmp_path / "my_prompt.txt"
    custom_prompt.write_text("CUSTOM PROMPT: {transcript}")

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio),
        patch("youtubetranscriber.cli.do_transcribe", return_value=fake_segments),
        patch("youtubetranscriber.cli.summarize.summarize", return_value="ok") as sum_mock,
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--summarize",
                "--summary-prompt",
                str(custom_prompt),
            ],
        )

    assert result.exit_code == 0, result.stderr
    passed_prompt = sum_mock.call_args.kwargs.get("prompt")
    assert passed_prompt == "CUSTOM PROMPT: {transcript}"


def test_missing_summary_prompt_fails_fast(tmp_path: Path) -> None:
    """A bad --summary-prompt path should fail at argument parsing,
    before any pipeline work is done. Regression test for a bug where
    a missing file surfaced as a FileNotFoundError traceback from
    deep inside _maybe_summarize, after audio download.
    """
    missing = tmp_path / "does_not_exist.txt"

    # The pipeline should NOT be called — validation must fail first.
    with (
        patch("youtubetranscriber.cli.audio.get_video_info") as info_mock,
        patch("youtubetranscriber.cli.audio.download_audio") as dl_mock,
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--summarize",
                "--summary-prompt",
                str(missing),
            ],
        )

    assert result.exit_code != 0
    assert not info_mock.called
    assert not dl_mock.called
    # typer's BadParameter message includes the path (possibly
    # wrapped across lines by the rich box drawing), so we normalize
    # whitespace before checking. macOS resolves tmp_path via /private/var
    # so we match on the filename rather than the full resolved path.
    combined = ((result.output or "") + (result.stderr or "")).replace("\n", " ")
    assert "summary prompt" in combined.lower()
    assert "not found" in combined.lower()
    assert missing.name in combined
    # Critically: no Python traceback
    assert "Traceback" not in combined


def test_summary_prompt_path_to_directory_rejected(tmp_path: Path) -> None:
    """--summary-prompt pointing at a directory (not a file) is rejected."""
    not_a_file = tmp_path / "a_directory"
    not_a_file.mkdir()

    with patch("youtubetranscriber.cli.audio.get_video_info") as info_mock:
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--summarize",
                "--summary-prompt",
                str(not_a_file),
            ],
        )

    assert result.exit_code != 0
    assert not info_mock.called


def test_summary_prompt_vanished_between_validate_and_read(
    tmp_path: Path, monkeypatch
) -> None:
    """TOCTOU: if the prompt file is deleted after argument validation
    but before _maybe_summarize reads it, we should fail cleanly with
    a [ytx] Error message rather than a traceback.
    """
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("PROMPT: {transcript}")

    # Simulate the file vanishing between the eager validate (in the
    # callback, which only calls .exists()/.is_file()) and the read
    # inside _maybe_summarize. The first read_text call is the one
    # we want to fail.
    real_read_text = Path.read_text
    call_count = {"n": 0}

    def maybe_vanish(self, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 1:
            raise FileNotFoundError(f"simulated disappearance: {self}")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", maybe_vanish)

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
                "--summarize",
                "--summary-prompt",
                str(prompt),
            ],
        )

    assert result.exit_code != 0
    # Clean user-facing error, no traceback
    assert "Traceback" not in (result.stderr or "")
    assert "disappeared" in (result.stderr or "").lower()


def test_summarize_skips_when_not_requested(tmp_path: Path) -> None:
    """Without --summarize, the summarizer is never called."""
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio),
        patch("youtubetranscriber.cli.do_transcribe", return_value=fake_segments),
        patch("youtubetranscriber.cli.summarize.summarize") as sum_mock,
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
            ],
        )

    assert result.exit_code == 0, result.stderr
    assert not sum_mock.called


def test_summarize_fails_clearly_when_ollama_key_missing(tmp_path: Path) -> None:
    """Missing OLLAMA_API_KEY produces a clear, actionable error."""
    from youtubetranscriber.summarize import OllamaApiKeyMissingError
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch("youtubetranscriber.cli.audio.download_audio", return_value=fake_audio),
        patch("youtubetranscriber.cli.do_transcribe", return_value=fake_segments),
        patch(
            "youtubetranscriber.cli.summarize.summarize",
            side_effect=OllamaApiKeyMissingError("OLLAMA_API_KEY is not set"),
        ),
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--summarize",
            ],
        )

    assert result.exit_code != 0
    assert "OLLAMA_API_KEY" in (result.stderr or "")


def test_help_includes_summarize_options() -> None:
    result = runner.invoke(app, ["--help"])
    assert "--summarize" in result.stdout
    assert "--summary-model" in result.stdout
    assert "--summary-prompt" in result.stdout


# --- --keep-audio flag ---------------------------------------------------


def test_default_deletes_audio_file(tmp_path: Path) -> None:
    """By default, the audio file is removed after successful pipeline."""
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]

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

    assert result.exit_code == 0, result.stderr
    # Audio file should be gone
    assert not fake_audio.path.exists()
    # But the transcript should still be there
    assert (tmp_path / "Sample Video" / "dQw4w9WgXcQ.txt").exists()


def test_keep_audio_preserves_audio_file(tmp_path: Path) -> None:
    """--keep-audio preserves the downloaded .m4a."""
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]

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
                "--keep-audio",
            ],
        )

    assert result.exit_code == 0, result.stderr
    # Audio file should still be there
    assert fake_audio.path.exists()
    # And the transcript too
    assert (tmp_path / "Sample Video" / "dQw4w9WgXcQ.txt").exists()


def test_no_keep_audio_explicit_works(tmp_path: Path) -> None:
    """--no-keep-audio explicitly deletes (same as default)."""
    from youtubetranscriber.transcribe import TranscriptSegment

    fake_audio = _fake_audio_result(tmp_path)
    fake_segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]

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
                "--no-keep-audio",
            ],
        )

    assert result.exit_code == 0, result.stderr
    assert not fake_audio.path.exists()


def test_captions_path_does_not_error_on_audio_cleanup(tmp_path: Path) -> None:
    """If captions were used (no audio file), cleanup should be a no-op."""
    from youtubetranscriber.transcribe import TranscriptSegment

    caption_segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]

    with (
        patch("youtubetranscriber.cli.audio.get_video_info", return_value=_fake_info()),
        patch(
            "youtubetranscriber.cli.captions.fetch_captions",
            return_value=caption_segments,
        ),
    ):
        result = runner.invoke(
            app,
            [
                "https://youtu.be/dQw4w9WgXcQ",
                "--output-dir",
                str(tmp_path),
                "--prefer-captions",
            ],
        )

    assert result.exit_code == 0, result.stderr
    # The cleanup should not have failed; transcript exists
    assert (tmp_path / "Sample Video" / "dQw4w9WgXcQ.txt").exists()
