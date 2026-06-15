"""YouTube caption fetching via youtube-transcript-api.

Wraps the API to return our own TranscriptSegment type and convert
errors to a single CaptionsUnavailable exception so the CLI can
fall through to Whisper without coupling to the API's exception
hierarchy.
"""

from __future__ import annotations

from youtube_transcript_api import YouTubeTranscriptApi

from youtubetranscriber.transcribe import TranscriptSegment


class CaptionsUnavailableError(Exception):
    """Raised when YouTube captions can't be fetched for any reason.

    This is the single signal to the CLI that it should fall through
    to Whisper. We collapse all of youtube-transcript-api's error
    types (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable,
    VideoUnplayable, network errors) into this one so callers don't
    need to know the API's exception hierarchy.
    """


def fetch_captions(
    video_id: str,
    *,
    languages: list[str] | None = None,
) -> list[TranscriptSegment]:
    """Fetch YouTube's auto/manual captions for a video.

    Args:
        video_id: 11-character YouTube video ID (not the full URL).
        languages: Preferred languages, in priority order. Defaults to
            English ("en"). The first available transcript in this
            list is returned.

    Returns:
        List of TranscriptSegment with text (whitespace-stripped),
        start, end (= start + duration), and speaker=None.

    Raises:
        CaptionsUnavailable: if no transcript is available, captions
            are disabled, the video is private/unavailable, or any
            other network/API error occurs.
    """
    if languages is None:
        languages = ["en"]

    # youtube-transcript-api exposes its exception hierarchy only via
    # the private _errors submodule, which is not part of the public
    # API and could change between minor releases. We deliberately
    # catch every failure mode (specific transcript errors, network
    # errors, HTTP errors, parsing errors) and collapse it into
    # CaptionsUnavailableError. Any new failure type the library
    # adds will land here too, which is the intended behavior —
    # any failure means "fall through to Whisper."
    try:
        api = YouTubeTranscriptApi()
        snippets = api.fetch(video_id, languages=languages)
    except Exception as e:
        raise CaptionsUnavailableError(
            f"Failed to fetch captions for {video_id} (languages={languages}): {e}"
        ) from e

    return [
        TranscriptSegment(
            text=snippet.text.strip(),
            start=float(snippet.start),
            end=float(snippet.start) + float(snippet.duration),
        )
        for snippet in snippets
    ]
