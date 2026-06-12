"""Transcript output: write TranscriptSegment list to txt/srt/json files.

Pure functions, no IO side effects beyond the file write itself.
"""

from __future__ import annotations

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


def write_transcript(segments: list[TranscriptSegment], path: Path, format: str = "txt") -> None:
    """Write a transcript to disk in the requested format.

    Args:
        segments: The transcript segments to write.
        path: Destination file path. Extension is up to the caller.
        format: One of "txt", "srt", "json".

    Raises:
        ValueError: if format is not recognized.
    """
    if format not in VALID_FORMATS:
        raise ValueError(f"Unknown format {format!r}. Must be one of: {', '.join(VALID_FORMATS)}")

    # Phase 1 only implements txt. Other formats raise clearly so the
    # user knows the feature is forthcoming (phase 2), not silently broken.
    if format == "txt":
        content = _format_txt(segments)
    elif format == "srt":
        raise NotImplementedError("SRT output is part of phase 2.")
    elif format == "json":
        raise NotImplementedError("JSON output is part of phase 2.")
    else:
        # Unreachable given the check above, but keeps the type checker happy.
        raise ValueError(f"Unknown format {format!r}")

    # Ensure parent dir exists
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
