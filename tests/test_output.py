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


# --- SRT writer -----------------------------------------------------------


def test_write_srt_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "transcript.srt"
    write_transcript(_make_segments(), out, format="srt")
    assert out.exists()


def test_write_srt_has_sequential_indices(tmp_path: Path) -> None:
    """SRT entries are numbered 1, 2, 3, ..."""
    out = tmp_path / "transcript.srt"
    write_transcript(_make_segments(), out, format="srt")
    content = out.read_text()
    assert content.startswith("1\n")
    assert "\n2\n" in content
    assert "\n3\n" in content


def test_write_srt_has_correct_timestamp_format(tmp_path: Path) -> None:
    """SRT uses HH:MM:SS,mmm with a comma, not a period."""
    out = tmp_path / "transcript.srt"
    write_transcript(_make_segments(), out, format="srt")
    content = out.read_text()
    assert "00:00:00,000 --> 00:00:01,500" in content
    assert "00:00:01,500 --> 00:00:03,000" in content
    assert "00:00:03,000 --> 00:00:04,000" in content


def test_write_srt_includes_text(tmp_path: Path) -> None:
    out = tmp_path / "transcript.srt"
    write_transcript(_make_segments(), out, format="srt")
    content = out.read_text()
    assert "Hello world." in content
    assert "This is a test." in content
    assert "Goodbye." in content


def test_write_srt_entries_separated_by_blank_lines(tmp_path: Path) -> None:
    """SRT spec: entries separated by exactly one blank line."""
    out = tmp_path / "transcript.srt"
    write_transcript(_make_segments(), out, format="srt")
    content = out.read_text()
    # Split into blocks by double newlines
    blocks = [b for b in content.split("\n\n") if b.strip()]
    assert len(blocks) == 3


def test_write_srt_handles_long_durations(tmp_path: Path) -> None:
    """Hours/minutes should be computed correctly from raw seconds."""
    segments = [TranscriptSegment(text="Long.", start=3661.5, end=3662.0)]  # 1h 1m 1.5s
    out = tmp_path / "transcript.srt"
    write_transcript(segments, out, format="srt")
    content = out.read_text()
    # 3661.5s = 1h 1m 1.5s = 01:01:01,500
    assert "01:01:01,500" in content


def test_write_srt_empty_produces_empty_file(tmp_path: Path) -> None:
    out = tmp_path / "transcript.srt"
    write_transcript([], out, format="srt")
    assert out.exists()
    assert out.read_text() == ""


def test_write_srt_with_speaker(tmp_path: Path) -> None:
    """Speakers should be visible in SRT output, prefixed to the text."""
    segments = [
        TranscriptSegment(text="Hi.", start=0.0, end=1.0, speaker="Alice"),
        TranscriptSegment(text="Hello!", start=1.0, end=2.0, speaker="Bob"),
    ]
    out = tmp_path / "transcript.srt"
    write_transcript(segments, out, format="srt")
    content = out.read_text()
    assert "Alice:" in content or "Alice" in content
    assert "Bob:" in content or "Bob" in content


# --- JSON writer ----------------------------------------------------------


def test_write_json_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "transcript.json"
    write_transcript(_make_segments(), out, format="json")
    assert out.exists()


def test_write_json_is_valid_json(tmp_path: Path) -> None:
    """Output must parse as valid JSON."""
    import json

    out = tmp_path / "transcript.json"
    write_transcript(_make_segments(), out, format="json")
    data = json.loads(out.read_text())
    assert isinstance(data, dict)


def test_write_json_has_video_metadata(tmp_path: Path) -> None:
    """Top-level dict should have video_id, title, and segments."""
    import json

    out = tmp_path / "transcript.json"
    write_transcript(_make_segments(), out, format="json")
    data = json.loads(out.read_text())
    assert "video_id" in data
    assert "title" in data
    assert "segments" in data
    assert isinstance(data["segments"], list)


def test_write_json_segments_have_required_fields(tmp_path: Path) -> None:
    import json

    out = tmp_path / "transcript.json"
    write_transcript(_make_segments(), out, format="json")
    data = json.loads(out.read_text())
    for seg in data["segments"]:
        assert "text" in seg
        assert "start" in seg
        assert "end" in seg
        # speaker may be null/absent; we always include it
        assert "speaker" in seg


def test_write_json_preserves_segment_data(tmp_path: Path) -> None:
    import json

    out = tmp_path / "transcript.json"
    write_transcript(_make_segments(), out, format="json")
    data = json.loads(out.read_text())
    assert data["segments"][0]["text"] == "Hello world."
    assert data["segments"][0]["start"] == 0.0
    assert data["segments"][0]["end"] == 1.5


def test_write_json_preserves_speaker_info(tmp_path: Path) -> None:
    import json

    segments = [
        TranscriptSegment(text="Hi.", start=0.0, end=1.0, speaker="Alice"),
    ]
    out = tmp_path / "transcript.json"
    write_transcript(segments, out, format="json")
    data = json.loads(out.read_text())
    assert data["segments"][0]["speaker"] == "Alice"


def test_write_json_optional_metadata(tmp_path: Path) -> None:
    """Caller can pass video metadata that gets included in the JSON."""
    import json

    out = tmp_path / "transcript.json"
    write_transcript(
        _make_segments(),
        out,
        format="json",
        video_id="dQw4w9WgXcQ",
        title="Sample Video",
    )
    data = json.loads(out.read_text())
    assert data["video_id"] == "dQw4w9WgXcQ"
    assert data["title"] == "Sample Video"


def test_write_json_empty_produces_valid_object(tmp_path: Path) -> None:
    import json

    out = tmp_path / "transcript.json"
    write_transcript([], out, format="json")
    data = json.loads(out.read_text())
    assert data["segments"] == []
