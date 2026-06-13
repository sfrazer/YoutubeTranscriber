"""Tests for captions.py: YouTubeTranscriptApi wrapper.

Mocked at the API level — we don't hit YouTube in unit tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from youtubetranscriber.captions import (
    CaptionsUnavailableError,
    fetch_captions,
)
from youtubetranscriber.transcribe import TranscriptSegment


def _make_snippet(text: str, start: float, duration: float) -> MagicMock:
    """Build a FetchedTranscriptSnippet-like object."""
    s = MagicMock()
    s.text = text
    s.start = start
    s.duration = duration
    return s


def test_fetch_captions_returns_segments() -> None:
    """fetch_captions should return a list of TranscriptSegment."""
    fake_snippets = [
        _make_snippet(" Hello.", 0.0, 1.5),
        _make_snippet(" World.", 1.5, 2.0),
    ]

    fake_api = MagicMock()
    fake_api.fetch.return_value = fake_snippets
    fake_api_class = MagicMock(return_value=fake_api)

    with patch("youtubetranscriber.captions.YouTubeTranscriptApi", fake_api_class):
        result = fetch_captions("dQw4w9WgXcQ")

    assert isinstance(result, list)
    assert all(isinstance(s, TranscriptSegment) for s in result)
    assert len(result) == 2


def test_fetch_captions_strips_whitespace() -> None:
    """YouTube's snippets have leading whitespace; we strip it."""
    fake_snippets = [
        _make_snippet("  Hello.  ", 0.0, 1.5),
    ]
    fake_api = MagicMock()
    fake_api.fetch.return_value = fake_snippets
    fake_api_class = MagicMock(return_value=fake_api)

    with patch("youtubetranscriber.captions.YouTubeTranscriptApi", fake_api_class):
        result = fetch_captions("dQw4w9WgXcQ")

    assert result[0].text == "Hello."


def test_fetch_captions_computes_end_from_duration() -> None:
    """Caption snippets have start+duration; we expose start+end."""
    fake_snippets = [
        _make_snippet("Hello.", 2.0, 1.5),
    ]
    fake_api = MagicMock()
    fake_api.fetch.return_value = fake_snippets
    fake_api_class = MagicMock(return_value=fake_api)

    with patch("youtubetranscriber.captions.YouTubeTranscriptApi", fake_api_class):
        result = fetch_captions("dQw4w9WgXcQ")

    assert result[0].start == 2.0
    assert result[0].end == 3.5  # 2.0 + 1.5


def test_fetch_captions_speaker_is_none() -> None:
    """Captions never have speaker labels; the field is None."""
    fake_snippets = [_make_snippet("Hi.", 0.0, 1.0)]
    fake_api = MagicMock()
    fake_api.fetch.return_value = fake_snippets
    fake_api_class = MagicMock(return_value=fake_api)

    with patch("youtubetranscriber.captions.YouTubeTranscriptApi", fake_api_class):
        result = fetch_captions("dQw4w9WgXcQ")

    assert result[0].speaker is None


def test_fetch_captions_passes_video_id_to_api() -> None:
    """The video ID should be passed to the underlying API call."""
    fake_api = MagicMock()
    fake_api.fetch.return_value = []
    fake_api_class = MagicMock(return_value=fake_api)

    with patch("youtubetranscriber.captions.YouTubeTranscriptApi", fake_api_class):
        fetch_captions("dQw4w9WgXcQ")

    call_args = fake_api.fetch.call_args
    assert "dQw4w9WgXcQ" in call_args.args


def test_fetch_captions_passes_languages() -> None:
    """Languages parameter should reach the API."""
    fake_api = MagicMock()
    fake_api.fetch.return_value = []
    fake_api_class = MagicMock(return_value=fake_api)

    with patch("youtubetranscriber.captions.YouTubeTranscriptApi", fake_api_class):
        fetch_captions("dQw4w9WgXcQ", languages=["en", "es"])

    call_args = fake_api.fetch.call_args
    languages = call_args.kwargs.get("languages") or call_args.args[1]
    assert "en" in languages
    assert "es" in languages


def test_fetch_captions_raises_on_no_transcripts() -> None:
    """If the API signals no transcripts, raise CaptionsUnavailableError."""
    from youtube_transcript_api._transcripts import NoTranscriptFound

    fake_api = MagicMock()
    fake_api.fetch.side_effect = NoTranscriptFound("dQw4w9WgXcQ", ["en"], [])
    fake_api_class = MagicMock(return_value=fake_api)

    with (
        patch("youtubetranscriber.captions.YouTubeTranscriptApi", fake_api_class),
        pytest.raises(CaptionsUnavailableError),
    ):
        fetch_captions("dQw4w9WgXcQ")


def test_fetch_captions_raises_when_disabled() -> None:
    """If transcripts are disabled, raise CaptionsUnavailableError."""
    from youtube_transcript_api._transcripts import TranscriptsDisabled

    fake_api = MagicMock()
    fake_api.fetch.side_effect = TranscriptsDisabled("dQw4w9WgXcQ")
    fake_api_class = MagicMock(return_value=fake_api)

    with (
        patch("youtubetranscriber.captions.YouTubeTranscriptApi", fake_api_class),
        pytest.raises(CaptionsUnavailableError),
    ):
        fetch_captions("dQw4w9WgXcQ")


def test_fetch_captions_raises_on_video_unavailable() -> None:
    """Video unavailable (private/deleted) -> CaptionsUnavailableError."""
    from youtube_transcript_api._transcripts import VideoUnavailable

    fake_api = MagicMock()
    fake_api.fetch.side_effect = VideoUnavailable("dQw4w9WgXcQ")
    fake_api_class = MagicMock(return_value=fake_api)

    with (
        patch("youtubetranscriber.captions.YouTubeTranscriptApi", fake_api_class),
        pytest.raises(CaptionsUnavailableError),
    ):
        fetch_captions("dQw4w9WgXcQ")


def test_fetch_captions_wraps_unexpected_errors() -> None:
    """Any other API error should bubble up as CaptionsUnavailableError too,
    so the CLI can fall through to Whisper without crashing."""
    fake_api = MagicMock()
    fake_api.fetch.side_effect = RuntimeError("network blip")
    fake_api_class = MagicMock(return_value=fake_api)

    with (
        patch("youtubetranscriber.captions.YouTubeTranscriptApi", fake_api_class),
        pytest.raises(CaptionsUnavailableError, match="network blip"),
    ):
        fetch_captions("dQw4w9WgXcQ")


def test_fetch_captions_empty_list_returns_empty() -> None:
    """A successful fetch with no snippets returns an empty list."""
    fake_api = MagicMock()
    fake_api.fetch.return_value = []
    fake_api_class = MagicMock(return_value=fake_api)

    with patch("youtubetranscriber.captions.YouTubeTranscriptApi", fake_api_class):
        result = fetch_captions("dQw4w9WgXcQ")

    assert result == []
