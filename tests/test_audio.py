"""Tests for audio.py: extract_video_id (pure function) and
download_audio (mocked yt-dlp)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youtubetranscriber.audio import (
    AudioDownloadError,
    DownloadResult,
    InvalidURLError,
    download_audio,
    extract_video_id,
    get_video_info,
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


# --- paste normalization ---------------------------------------------------
# Real-world cases where the URL arrives wrapped in shell metacharacters
# (surrounding quotes from zsh bracketed-paste), with leading/trailing
# whitespace (from a select-and-paste that grabbed an extra space), or
# with HTML-entity ampersands (Firefox quirk when copying from rendered
# HTML). All should be normalized to the same canonical ID.


@pytest.mark.parametrize(
    "url,expected",
    [
        # Surrounding double quotes
        ('"https://www.youtube.com/watch?v=dQw4w9WgXcQ"', "dQw4w9WgXcQ"),
        ('"https://youtu.be/dQw4w9WgXcQ"', "dQw4w9WgXcQ"),
        # Surrounding single quotes
        ("'https://www.youtube.com/watch?v=dQw4w9WgXcQ'", "dQw4w9WgXcQ"),
        # Trailing whitespace
        ("https://youtu.be/dQw4w9WgXcQ ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ  ", "dQw4w9WgXcQ"),
        # Leading whitespace
        (" https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        # Leading and trailing whitespace
        ("  https://youtu.be/dQw4w9WgXcQ  ", "dQw4w9WgXcQ"),
        # Leading newline (common from copy of address bar)
        ("\nhttps://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        # HTML-entity ampersand on a watch URL (the ?v= is matched
        # before the unescaped &t= is even reached)
        (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&amp;t=42s",
            "dQw4w9WgXcQ",
        ),
        # Shell-escaped metacharacters: zsh can insert a backslash
        # before '?' and '=' on paste. Reported in the wild.
        (
            "https://www.youtube.com/watch\\?v\\=wykPErJ8M-8",
            "wykPErJ8M-8",
        ),
        # Escaped '&' (also shell-glob-adjacent in some configs)
        (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ\\&t=42s",
            "dQw4w9WgXcQ",
        ),
        # Mixed: shell-escaped '?' and HTML-entity '&' together
        (
            "https://www.youtube.com/watch\\?v=dQw4w9WgXcQ&amp;t=42s",
            "dQw4w9WgXcQ",
        ),
    ],
)
def test_extract_video_id_normalizes_paste_artifacts(url: str, expected: str) -> None:
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
    """download_audio should return a DownloadResult with path and title."""

    expected_path = tmp_path / "Some Video [dQw4w9WgXcQ].m4a"
    expected_title = "Some Video"

    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    # Simulate yt-dlp writing the file and providing metadata
    fake_ydl_class.return_value.__enter__.return_value.prepare_filename.return_value = str(
        expected_path.with_suffix("")
    )
    fake_ydl_class.return_value.__enter__.return_value.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        "title": expected_title,
    }

    with patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class):
        # Create the file so existence check passes
        expected_path.write_bytes(b"fake audio")
        result = download_audio("https://youtu.be/dQw4w9WgXcQ", tmp_path)

    assert isinstance(result, DownloadResult)
    assert result.path == expected_path
    assert result.path.exists()
    assert result.title == expected_title
    assert result.video_id == "dQw4w9WgXcQ"


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


def test_download_audio_detects_gate_page_by_id_shape(tmp_path: Path) -> None:
    """yt-dlp returns a 24-char channel id when YouTube hands back a gate page.

    Real video ids are exactly 11 chars. A 24-char id starting with
    'UC' is a channel id and signals that the response is a gate,
    not the video. We should raise a clear AudioDownloadError
    rather than fall through to 'file not found'.
    """
    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    fake_ydl_class.return_value.__enter__.return_value.extract_info.return_value = {
        "id": "UCK5afNL1HiwIYiNloPN4rsg",  # 24-char channel id
        "title": "⬇️ Press Subscribe to continue.",
    }

    with (
        patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class),
        pytest.raises(AudioDownloadError, match="gate page"),
    ):
        download_audio("https://www.youtube.com/watch?v=wykPErJ8M-8", tmp_path)


def test_download_audio_detects_gate_page_by_title(tmp_path: Path) -> None:
    """A real-looking 11-char id can still be a gate page; check the title too.

    Some gate variants return an 11-char id with a recognizable
    phrase in the title. The phrase check catches those.
    """
    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    fake_ydl_class.return_value.__enter__.return_value.extract_info.return_value = {
        "id": "wykPErJ8M-8",  # 11 chars, would pass the id check
        "title": "Sign in to confirm your age",
    }

    with (
        patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class),
        pytest.raises(AudioDownloadError, match="gate page"),
    ):
        download_audio("https://www.youtube.com/watch?v=wykPErJ8M-8", tmp_path)


def test_download_audio_does_not_flag_real_video(tmp_path: Path) -> None:
    """A real-looking id and title should NOT be flagged as a gate page.

    Negative test for the gate detection — make sure we're not
    false-positive on real videos with unusual titles.
    """
    expected_path = tmp_path / "Some Video [dQw4w9WgXcQ].m4a"
    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    fake_ydl_class.return_value.__enter__.return_value.prepare_filename.return_value = str(
        expected_path.with_suffix("")
    )
    fake_ydl_class.return_value.__enter__.return_value.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        "title": "Some Video",
    }

    with patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class):
        expected_path.write_bytes(b"fake audio")
        result = download_audio("https://youtu.be/dQw4w9WgXcQ", tmp_path)

    assert result.video_id == "dQw4w9WgXcQ"

def test_get_video_info_returns_metadata(tmp_path: Path) -> None:
    """get_video_info should fetch title/id without downloading."""
    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    fake_ydl_class.return_value.__enter__.return_value.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        "title": "Never Gonna Give You Up",
    }

    with patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class) as ydl_cls:
        result = get_video_info("https://youtu.be/dQw4w9WgXcQ")

    assert result.title == "Never Gonna Give You Up"
    assert result.video_id == "dQw4w9WgXcQ"
    # Verify we asked yt-dlp NOT to download
    params = ydl_cls.call_args.args[0]
    assert params.get("skip_download") is True


def test_get_video_info_falls_back_to_id_for_missing_title(tmp_path: Path) -> None:
    """If yt-dlp returns no title, fall back to the video ID."""
    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    fake_ydl_class.return_value.__enter__.return_value.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        # no 'title' key
    }

    with patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class):
        result = get_video_info("https://youtu.be/dQw4w9WgXcQ")

    assert result.title == "dQw4w9WgXcQ"


def test_download_audio_uses_m4a_format(tmp_path: Path) -> None:
    """We should request m4a audio specifically for quality + size balance."""

    expected_path = tmp_path / "video [dQw4w9WgXcQ].m4a"
    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    fake_ydl_class.return_value.__enter__.return_value.prepare_filename.return_value = str(
        expected_path.with_suffix("")
    )
    fake_ydl_class.return_value.__enter__.return_value.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        "title": "video",
    }

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


# --- cookie options --------------------------------------------------------


def test_get_video_info_passes_cookies_from_browser(tmp_path: Path) -> None:
    """get_video_info should plumb cookiesfrombrowser into yt-dlp's opts."""
    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    fake_ydl_class.return_value.__enter__.return_value.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        "title": "Some Video",
    }

    with patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class) as ydl_cls:
        get_video_info(
            "https://youtu.be/dQw4w9WgXcQ",
            cookies_from_browser="firefox",
        )

    params = ydl_cls.call_args.args[0]
    # yt-dlp expects a tuple for cookiesfrombrowser
    assert params.get("cookiesfrombrowser") == ("firefox",)


def test_download_audio_passes_cookies_from_browser(tmp_path: Path) -> None:
    """download_audio should plumb cookiesfrombrowser into yt-dlp's opts."""
    expected_path = tmp_path / "Some Video [dQw4w9WgXcQ].m4a"
    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    fake_ydl_class.return_value.__enter__.return_value.prepare_filename.return_value = str(
        expected_path.with_suffix("")
    )
    fake_ydl_class.return_value.__enter__.return_value.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        "title": "Some Video",
    }

    with patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class) as ydl_cls:
        expected_path.write_bytes(b"fake")
        download_audio(
            "https://youtu.be/dQw4w9WgXcQ",
            tmp_path,
            cookies_from_browser="firefox",
        )

    params = ydl_cls.call_args.args[0]
    assert params.get("cookiesfrombrowser") == ("firefox",)


def test_download_audio_passes_cookies_file(tmp_path: Path) -> None:
    """download_audio should plumb cookiefile into yt-dlp's opts."""
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")

    expected_path = tmp_path / "Some Video [dQw4w9WgXcQ].m4a"
    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    fake_ydl_class.return_value.__enter__.return_value.prepare_filename.return_value = str(
        expected_path.with_suffix("")
    )
    fake_ydl_class.return_value.__enter__.return_value.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        "title": "Some Video",
    }

    with patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class) as ydl_cls:
        expected_path.write_bytes(b"fake")
        download_audio(
            "https://youtu.be/dQw4w9WgXcQ",
            tmp_path,
            cookies_file=cookies,
        )

    params = ydl_cls.call_args.args[0]
    # yt-dlp expects a string path for cookiefile
    assert params.get("cookiefile") == str(cookies)


def test_download_audio_no_cookie_options_by_default(tmp_path: Path) -> None:
    """Without explicit cookie kwargs, neither option is set.

    Negative test: make sure the default is no cookie config,
    not an accidental empty tuple or empty string.
    """
    expected_path = tmp_path / "Some Video [dQw4w9WgXcQ].m4a"
    fake_ydl = MagicMock()
    fake_ydl_class = MagicMock(return_value=fake_ydl)
    fake_ydl_class.return_value.__enter__.return_value.prepare_filename.return_value = str(
        expected_path.with_suffix("")
    )
    fake_ydl_class.return_value.__enter__.return_value.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        "title": "Some Video",
    }

    with patch("youtubetranscriber.audio.yt_dlp.YoutubeDL", fake_ydl_class) as ydl_cls:
        expected_path.write_bytes(b"fake")
        download_audio("https://youtu.be/dQw4w9WgXcQ", tmp_path)

    params = ydl_cls.call_args.args[0]
    assert "cookiesfrombrowser" not in params
    assert "cookiefile" not in params
