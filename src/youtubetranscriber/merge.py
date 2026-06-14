"""Merge speaker diarization spans with transcript segments.

Pure functions. No I/O, no model calls. The algorithm is the
foundation of phase 3's diarization feature.

Algorithm: for each transcript segment, find the diarization span
with maximum temporal overlap. Assign that span's speaker label to
the segment. If no span overlaps, speaker is left as None.
"""

from __future__ import annotations

from dataclasses import dataclass

from youtubetranscriber.transcribe import TranscriptSegment


@dataclass
class SpeakerSpan:
    """A time range during which a single speaker is talking.

    Attributes:
        speaker: The speaker label (e.g. "SPEAKER_00", "ALICE").
        start: Start time in seconds.
        end: End time in seconds.
    """

    speaker: str
    start: float
    end: float


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Compute the length of overlap between two [start, end] intervals.

    Returns 0.0 if the intervals don't overlap (including touching
    at a single point).
    """
    overlap_start = max(a_start, b_start)
    overlap_end = min(a_end, b_end)
    return max(0.0, overlap_end - overlap_start)


def assign_speakers(
    segments: list[TranscriptSegment],
    spans: list[SpeakerSpan],
) -> list[TranscriptSegment]:
    """Assign a speaker label to each segment based on maximum overlap.

    Pure function. Does not mutate the input list or its segments;
    returns a new list of new TranscriptSegment objects with the
    `speaker` field populated (or left as None if no span overlaps).

    Args:
        segments: Whisper transcript segments.
        spans: pyannote diarization spans.

    Returns:
        New list of TranscriptSegment, one per input segment, with
        speaker labels filled in where possible.
    """
    result: list[TranscriptSegment] = []
    for seg in segments:
        best_speaker: str | None = None
        best_overlap = 0.0
        for span in spans:
            ovl = _overlap(seg.start, seg.end, span.start, span.end)
            if ovl > best_overlap:
                best_overlap = ovl
                best_speaker = span.speaker
        # Build a new segment with the speaker filled in
        result.append(
            TranscriptSegment(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                speaker=best_speaker,
            )
        )
    return result
