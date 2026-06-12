"""Transcript output: write TranscriptSegment list to txt/srt/json files.

Pure functions, no IO side effects beyond the file write itself.
"""

from __future__ import annotations

import json
from pathlib import Path

from youtubetranscriber.transcribe import TranscriptSegment

VALID_FORMATS = ("txt", "srt", "json")


def _format_txt_line(segment: TranscriptSegment) -> str:
    """One line per segment. Prefix with speaker label if present."""
    if segment.speaker:
        return f"[{segment.speaker}] {segment.text}"
    return segment.text


def _format_txt(segments: list[TranscriptSegment]) -> str:
    return "\n".join(_format_txt_line(seg) for seg in segments)


def _format_srt_timestamp(seconds: float) -> str:
    """Format a number of seconds as SRT's HH:MM:SS,mmm.

    SRT uses comma as the fractional separator, not period.
    """
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_srt_entry(index: int, segment: TranscriptSegment) -> str:
    """Format a single SRT entry: index, timestamp range, text, blank."""
    start_ts = _format_srt_timestamp(segment.start)
    end_ts = _format_srt_timestamp(segment.end)
    text = segment.text
    if segment.speaker:
        # SRT doesn't have a standard speaker convention; prefix the text
        # in a vtt-ish way that's also valid SRT.
        text = f"{segment.speaker}: {text}"
    return f"{index}\n{start_ts} --> {end_ts}\n{text}\n"


def _format_srt(segments: list[TranscriptSegment]) -> str:
    if not segments:
        return ""
    entries = [_format_srt_entry(i, seg) for i, seg in enumerate(segments, start=1)]
    # SRT spec: entries separated by a single blank line, trailing
    # newline at the very end.
    return "\n".join(entries) + "\n"


def _format_json(
    segments: list[TranscriptSegment],
    *,
    video_id: str | None = None,
    title: str | None = None,
) -> str:
    """Format a transcript as a structured JSON document.

    Shape:
        {
          "video_id": "...",
          "title": "...",
          "segments": [
            {"text": "...", "start": 0.0, "end": 1.5, "speaker": null},
            ...
          ]
        }

    `speaker` is always present (null if unset) so consumers don't have
    to deal with missing keys.
    """
    payload = {
        "video_id": video_id,
        "title": title,
        "segments": [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "speaker": seg.speaker,
            }
            for seg in segments
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def write_transcript(
    segments: list[TranscriptSegment],
    path: Path,
    format: str = "txt",
    *,
    video_id: str | None = None,
    title: str | None = None,
) -> None:
    """Write a transcript to disk in the requested format.

    Args:
        segments: The transcript segments to write.
        path: Destination file path. Extension is up to the caller.
        format: One of "txt", "srt", "json".
        video_id: Optional video ID, included in JSON output.
        title: Optional video title, included in JSON output.

    Raises:
        ValueError: if format is not recognized.
    """
    if format not in VALID_FORMATS:
        raise ValueError(f"Unknown format {format!r}. Must be one of: {', '.join(VALID_FORMATS)}")

    if format == "txt":
        content = _format_txt(segments)
    elif format == "srt":
        content = _format_srt(segments)
    elif format == "json":
        content = _format_json(segments, video_id=video_id, title=title)
    else:
        # Unreachable given the check above, but keeps the type checker happy.
        raise ValueError(f"Unknown format {format!r}")

    # Ensure parent dir exists
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
