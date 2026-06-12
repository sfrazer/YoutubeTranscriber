"""Tests for output.py: text transcript writer."""

from __future__ import annotations

from pathlib import Path

import pytest

from youtubetranscriber.output import write_transcript
from youtubetranscriber.transcribe import TranscriptSegment


def _make_segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(text="Hello world.", start=0.0, end=1.5),
        TranscriptSegment(text="This is a test.", start=1.5, end=3.0),
        TranscriptSegment(text="Goodbye.", start=3.0, end=4.0),
    ]


def test_write_txt_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "transcript.txt"
    write_transcript(_make_segments(), out, format="txt")
    assert out.exists()


def test_write_txt_contains_all_text(tmp_path: Path) -> None:
    out = tmp_path / "transcript.txt"
    write_transcript(_make_segments(), out, format="txt")
    content = out.read_text()
    assert "Hello world." in content
    assert "This is a test." in content
    assert "Goodbye." in content


def test_write_txt_one_segment_per_line(tmp_path: Path) -> None:
    out = tmp_path / "transcript.txt"
    write_transcript(_make_segments(), out, format="txt")
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    assert lines == ["Hello world.", "This is a test.", "Goodbye."]


def test_write_txt_empty_segments_produces_empty_file(tmp_path: Path) -> None:
    out = tmp_path / "transcript.txt"
    write_transcript([], out, format="txt")
    assert out.exists()
    assert out.read_text() == ""


def test_write_txt_with_speaker(tmp_path: Path) -> None:
    """When speakers are present, include them as a prefix on each line."""
    segments = [
        TranscriptSegment(text="Hi there.", start=0.0, end=1.0, speaker="SPEAKER_00"),
        TranscriptSegment(text="Hello!", start=1.0, end=2.0, speaker="SPEAKER_01"),
    ]
    out = tmp_path / "transcript.txt"
    write_transcript(segments, out, format="txt")
    lines = [ln for ln in out.read_text().splitlines() if ln.strip()]
    # The format should make speakers obvious without being noisy
    content = "\n".join(lines)
    assert "SPEAKER_00" in content
    assert "SPEAKER_01" in content
    assert "Hi there." in content
    assert "Hello!" in content


def test_write_txt_rejects_unknown_format(tmp_path: Path) -> None:
    out = tmp_path / "transcript.txt"
    with pytest.raises(ValueError, match="Unknown format"):
        write_transcript(_make_segments(), out, format="docx")
