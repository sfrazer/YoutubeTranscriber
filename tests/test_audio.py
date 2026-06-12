"""Tests for audio.py: extract_video_id (pure function) and
download_audio (mocked yt-dlp)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youtubetranscriber.audio import (
    AudioDownloadError,
    InvalidURLError,
    download_audio,
    extract_video_id,
)

# --- extract_video_id -------------------------------------------------------


@pytest.mark.parametrize(
    "url,expected",
    [
        # Standard watch URL
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        # Short URL
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        # Embed URL
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        # Shorts URL
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        # With extra query params
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
        # With www stripped
        ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        # Mobile URL
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        # HTTP (not HTTPS)
        ("http://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id_valid(url: str, expected: str) -> None:
    assert extract_video_id(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "",  # empty
        "not a url at all",  # no scheme
        "https://example.com/watch?v=dQw4w9WgXcQ",  # not youtube
        "https://www.youtube.com/",  # no video id
        "https://www.youtube.com/watch",  # no query
        "https://www.youtube.com/watch?foo=bar",  # no v param
        "https://www.youtube.com/watch?v=tooshort",  # 8 chars, not 11
        "https://www.youtube.com/watch?v=waytoolongtobevalid",  # 23 chars
    ],
)
def test_extract_video_id_invalid(url: str) -> None:
    with pytest.raises(InvalidURLError):
        extract_video_id(url)


# --- download_audio (mocked) -----------------------------------------------


def test_download_audio_returns_path_to_downloaded_file(tmp_path: Path) -> None:
    """download_audio should return the path to the .m4a file yt-dlp produced."""

    expected_path = tmp_path / "Some Video [dQw4w9WgXcQ].m4a"

    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    # Simulate yt-dlp writing the file
    fake_ydl_class.return_value.__enter__.return_value.prepare_filename.return_value = str(
        expected_path.with_suffix("")
    )

    with patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class):
        # Create the file so existence check passes
        expected_path.write_bytes(b"fake audio")
        result = download_audio("https://youtu.be/dQw4w9WgXcQ", tmp_path)

    assert result == expected_path
    assert result.exists()


def test_download_audio_wraps_yt_dlp_errors(tmp_path: Path) -> None:
    """A yt-dlp downloader error should become AudioDownloadError.

    We use a valid-looking URL so we exercise the yt-dlp code path
    rather than failing fast on URL validation.
    """

    from yt_dlp.utils import DownloadError

    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    fake_ydl_class.return_value.__enter__.side_effect = DownloadError("video unavailable")

    with (
        patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class),
        pytest.raises(AudioDownloadError, match="video unavailable"),
    ):
        download_audio("https://youtu.be/dQw4w9WgXcQ", tmp_path)


def test_download_audio_uses_m4a_format(tmp_path: Path) -> None:
    """We should request m4a audio specifically for quality + size balance."""

    expected_path = tmp_path / "video [dQw4w9WgXcQ].m4a"
    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    fake_ydl_class.return_value.__enter__.return_value.prepare_filename.return_value = str(
        expected_path.with_suffix("")
    )

    with patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class) as ydl_cls:
        expected_path.write_bytes(b"fake")
        download_audio("https://youtu.be/dQw4w9WgXcQ", tmp_path)

    # Inspect the params dict passed to YoutubeDL
    call_args = ydl_cls.call_args
    params = call_args.args[0] if call_args.args else call_args.kwargs.get("params", {})
    # yt-dlp's format string should prefer m4a
    assert "m4a" in params.get("format", "")
    # outtmpl should write into tmp_path
    outtmpl = params.get("outtmpl", "")
    assert str(tmp_path) in outtmpl
    # We should be running an audio-only postprocessor
    postprocessors = params.get("postprocessors", [])
    assert any(p.get("key") == "FFmpegExtractAudio" for p in postprocessors)
