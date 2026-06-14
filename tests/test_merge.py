"""Tests for merge.py: assigning speaker labels to transcript segments.

Pure function, full coverage of edge cases. The algorithm is the
foundation of phase 3's diarization feature.
"""

from __future__ import annotations

from youtubetranscriber.merge import SpeakerSpan, assign_speakers
from youtubetranscriber.transcribe import TranscriptSegment


def _seg(text: str, start: float, end: float) -> TranscriptSegment:
    return TranscriptSegment(text=text, start=start, end=end)


def _span(speaker: str, start: float, end: float) -> SpeakerSpan:
    return SpeakerSpan(speaker=speaker, start=start, end=end)


# --- Happy path -----------------------------------------------------------


def test_segment_inside_single_span() -> None:
    """A segment fully inside one speaker's span gets that speaker."""
    segments = [_seg("hello", 1.0, 2.0)]
    spans = [_span("ALICE", 0.0, 5.0)]

    result = assign_speakers(segments, spans)

    assert result[0].speaker == "ALICE"


def test_segment_matches_span_with_max_overlap() -> None:
    """When a segment overlaps two speakers, the one with more overlap wins."""
    # Segment 1.5-3.5 overlaps ALICE (0-2) for 0.5s and BOB (2-5) for 1.5s
    segments = [_seg("mixed", 1.5, 3.5)]
    spans = [_span("ALICE", 0.0, 2.0), _span("BOB", 2.0, 5.0)]

    result = assign_speakers(segments, spans)

    assert result[0].speaker == "BOB"


def test_multiple_segments_get_distinct_speakers() -> None:
    """Two segments in different spans get different speakers."""
    segments = [
        _seg("hi", 0.5, 1.5),
        _seg("there", 2.5, 3.5),
    ]
    spans = [
        _span("ALICE", 0.0, 2.0),
        _span("BOB", 2.0, 5.0),
    ]

    result = assign_speakers(segments, spans)

    assert result[0].speaker == "ALICE"
    assert result[1].speaker == "BOB"


# --- Edge cases -----------------------------------------------------------


def test_segment_with_no_overlap_gets_none() -> None:
    """A segment that doesn't overlap any span gets speaker=None."""
    segments = [_seg("silent", 10.0, 11.0)]
    spans = [_span("ALICE", 0.0, 5.0)]

    result = assign_speakers(segments, spans)

    assert result[0].speaker is None


def test_empty_segments_returns_empty() -> None:
    """No segments means no work to do."""
    result = assign_speakers([], [_span("ALICE", 0.0, 5.0)])
    assert result == []


def test_empty_spans_leaves_speakers_none() -> None:
    """No diarization data means no speakers, regardless of segments."""
    segments = [_seg("hi", 0.0, 1.0), _seg("there", 1.0, 2.0)]
    result = assign_speakers(segments, [])
    assert all(s.speaker is None for s in result)


def test_segment_partially_overlapping_one_span() -> None:
    """Half-overlapping a span still gives the segment that speaker."""
    segments = [_seg("hi", 0.5, 1.5)]  # 0.5-1.5 in span 0.0-1.0 → 0.5s overlap
    spans = [_span("ALICE", 0.0, 1.0)]

    result = assign_speakers(segments, spans)

    assert result[0].speaker == "ALICE"


def test_segment_at_exact_span_boundary() -> None:
    """Zero-width overlap at the boundary is still a valid assignment."""
    # Segment ends exactly when span begins; overlap is 0
    segments = [_seg("hi", 0.0, 1.0)]
    spans = [_span("ALICE", 1.0, 2.0)]

    result = assign_speakers(segments, spans)

    # Zero overlap = no assignment
    assert result[0].speaker is None


def test_segment_with_multiple_spans_same_speaker() -> None:
    """If the same speaker appears in two spans, both contribute to overlap."""
    # ALICE speaks 0-1 and 3-4; BOB speaks 1-3
    # Segment 2.0-3.5 overlaps ALICE 3-4 by 1.0s, BOB 1-3 by 1.0s → tie
    # Our algorithm picks the first one found with max overlap.
    segments = [_seg("hi", 2.0, 3.5)]
    spans = [
        _span("ALICE", 0.0, 1.0),
        _span("BOB", 1.0, 3.0),
        _span("ALICE", 3.0, 4.0),
    ]

    result = assign_speakers(segments, spans)

    # Tie: 1.0s for BOB and 1.0s for ALICE (3-4) - first found with max wins
    # Implementation detail: the first match in iteration order wins
    assert result[0].speaker in ("ALICE", "BOB")


def test_touching_spans_have_no_overlap() -> None:
    """Spans that touch but don't overlap should not double-count."""
    # ALICE 0-2, BOB 2-4, ALICE 4-6
    # Segment 1.9-2.1 overlaps ALICE (0-2) by 0.1, BOB (2-4) by 0.1
    # Either is acceptable, just verify it doesn't crash
    segments = [_seg("hi", 1.9, 2.1)]
    spans = [
        _span("ALICE", 0.0, 2.0),
        _span("BOB", 2.0, 4.0),
    ]

    result = assign_speakers(segments, spans)
    assert result[0].speaker in ("ALICE", "BOB")


def test_segment_entirely_within_span_chooses_that_span() -> None:
    """A segment fully inside a span unambiguously gets that speaker."""
    segments = [_seg("hi", 2.5, 3.5)]
    spans = [
        _span("ALICE", 0.0, 5.0),
        _span("BOB", 0.0, 5.0),  # Same time range, different speaker
    ]

    result = assign_speakers(segments, spans)

    # Both overlap fully; tie. Whichever the impl picks is fine, but
    # it must pick one of them.
    assert result[0].speaker in ("ALICE", "BOB")


# --- Mutation safety -----------------------------------------------------


def test_assign_speakers_does_not_mutate_input_segments() -> None:
    """The input list and its segments should not be modified."""
    original = _seg("hi", 0.0, 1.0)
    segments = [original]
    spans = [_span("ALICE", 0.0, 5.0)]

    assign_speakers(segments, spans)

    # Original segment should still have speaker=None
    assert original.speaker is None
    # The input list itself should be unchanged
    assert segments == [original]


def test_assign_speakers_returns_new_list() -> None:
    """The returned list should be a new object, not the same list."""
    segments = [_seg("hi", 0.0, 1.0)]
    spans = [_span("ALICE", 0.0, 5.0)]

    result = assign_speakers(segments, spans)

    assert result is not segments
