"""Audio acquisition via yt-dlp.

Depends on: ffmpeg (system binary), yt-dlp (pip).
Side effects: writes .m4a files to disk.
Pure functions: extract_video_id()
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import yt_dlp
from yt_dlp.utils import DownloadError

# Module-level regex constants. The two matchers are deliberately
# stricter than a bare "[a-zA-Z0-9_-]{11}" — they require anchoring
# context (the v= query param or a known path prefix) so that a
# random 11-char substring elsewhere in a URL can't be mistaken
# for a video ID.
_WATCH_ID_RE = re.compile(r"[?&]v=([a-zA-Z0-9_-]{11})(?:[&#]|$)")
_PATH_ID_RE = re.compile(r"/(?:shorts/|embed/)?([a-zA-Z0-9_-]{11})(?:[/?#]|$)")


class InvalidURLError(ValueError):
    """Raised when a YouTube URL is malformed or doesn't contain a video ID."""


class AudioDownloadError(RuntimeError):
    """Raised when yt-dlp fails to download the audio."""


@dataclass
class VideoInfo:
    """Metadata for a YouTube video, without the downloaded media.

    Returned by get_video_info() when you want to make decisions
    (e.g. where to put files) before committing to a download.

    Attributes:
        title: The video's title as reported by yt-dlp.
        video_id: The 11-character YouTube video ID.
    """

    title: str
    video_id: str


@dataclass
class DownloadResult:
    """The result of downloading a video's audio.

    Attributes:
        path: Path to the downloaded .m4a file.
        title: The video's title as reported by yt-dlp.
        video_id: The 11-character YouTube video ID.
    """

    path: Path
    title: str
    video_id: str


def _normalize_url(url: str) -> str:
    """Normalize a URL pasted from a browser/clipboard before parsing.

    Four classes of fix-up that a user pasting into a terminal hits:
        - Surrounding quote characters survive bracketed-paste
          (mostly with zsh/fish, less so with bash).
        - Shell-escaped metacharacters: some zsh configurations
          insert a backslash before characters like '?' and '='
          when a URL is pasted. The user copies a normal watch URL
          but the paste buffer arrives with backslashes inserted
          before '?' and '=' (because zsh treats them as
          glob/history-expansion characters).
        - HTML-entity-encoded ampersands from URLs copied out of
          rendered HTML (Firefox used to do this in some paths).
        - Leading/trailing whitespace from select-and-paste that
          grabbed an extra space or newline from the address bar.

    We deliberately do NOT try to fix other classes of malformed
    URLs (wrong scheme, missing host, etc.) — those should fail
    with a clear error so the user notices, not be silently
    corrected.
    """
    normalized = url.strip().strip('"').strip("'").strip()
    # Strip a backslash that is followed by something that isn't a
    # 'safe' URL character. This mirrors the set of characters that
    # shells typically escape, and leaves any legitimate backslashes
    # alone (which in YouTube URLs would have to be percent-encoded
    # as %5C anyway, so this is safe in practice).
    normalized = re.sub(r"\\([^a-zA-Z0-9._/:])", r"\1", normalized)
    normalized = html.unescape(normalized)
    return normalized


def extract_video_id(url: str) -> str:
    """Parse a YouTube URL and return the 11-character video ID.

    Supports:
        - https://www.youtube.com/watch?v=ID
        - https://youtu.be/ID
        - https://www.youtube.com/embed/ID
        - https://www.youtube.com/shorts/ID
        - https://m.youtube.com/watch?v=ID
        - http:// (not just https://)
        - Extra query params (e.g. &t=42s)
        - URLs pasted with surrounding quotes or HTML-entity ampersands
          (normalized before parsing — see _normalize_url)

    Raises:
        InvalidURLError: if the URL is not a recognized YouTube URL or
            has no valid 11-character video ID.
    """
    if not url or not isinstance(url, str):
        raise InvalidURLError(f"URL must be a non-empty string, got: {url!r}")

    url = _normalize_url(url)

    # Normalize: must contain "youtube" or be a youtu.be short link
    lowered = url.lower()
    is_youtube = "youtube.com" in lowered or "youtu.be" in lowered
    if not is_youtube:
        raise InvalidURLError(f"Not a YouTube URL: {url!r}")

    # The path component (after the host) is where the ID lives for
    # short, embed, and shorts URLs. For watch URLs, it's the v= query.
    # Try the query string first since it's the most common case.
    # The ID must be exactly 11 chars and terminated by &, #, or end of string.
    match = _WATCH_ID_RE.search(url)
    if match:
        return match.group(1)

    # Fall back to a path-based match: /ID or /shorts/ID or /embed/ID
    # We require a path segment that is *exactly* 11 chars, not just
    # any 11-char substring (which could match garbage in unrelated paths).
    path_match = _PATH_ID_RE.search(url)
    if path_match:
        return path_match.group(1)

    raise InvalidURLError(f"Could not extract video ID from: {url!r}")


def get_video_info(url: str) -> VideoInfo:
    """Fetch video metadata (title, id) without downloading the media.

    Cheaper than download_audio() — useful when you want to decide
    where to put files before committing to a download.

    Raises:
        InvalidURLError: if the URL is malformed.
        AudioDownloadError: if yt-dlp can't extract info for the URL.
    """
    video_id = extract_video_id(url)
    opts = {
        "quiet": True,
        "no_warnings": True,
        # Don't actually download; extract_info with download=False
        # only fetches the metadata.
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as e:
        raise AudioDownloadError(f"yt-dlp failed to extract info for {url}: {e}") from e

    title = (info or {}).get("title") or video_id
    return_id = (info or {}).get("id") or video_id
    return VideoInfo(title=title, video_id=return_id)


def download_audio(url: str, output_dir: Path) -> DownloadResult:
    """Download audio only from a YouTube URL to output_dir.

    Returns a DownloadResult containing the path to the .m4a file and
    the video's title and ID. The file will be named
    "<video title> [<video_id>].m4a" by default.

    Raises:
        InvalidURLError: if the URL is not a valid YouTube URL.
        AudioDownloadError: if yt-dlp fails to download (video unavailable,
            network error, etc.).
    """
    # Validate URL up front so we fail fast with a clear error.
    video_id = extract_video_id(url)

    output_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        # Prefer m4a (AAC) for size + quality. Fall back to any best audio.
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        # Write into output_dir with a predictable template.
        "outtmpl": str(output_dir / "%(title)s [%(id)s].%(ext)s"),
        # Don't download video, just audio. The postprocessor extracts
        # audio from the chosen format and converts to m4a if needed.
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "192",
            }
        ],
        # Quieter output unless verbose
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            # extract_info(download=True) returns the metadata dict and
            # triggers the download. We use this rather than ydl.download()
            # because it gives us the title.
            info = ydl.extract_info(url, download=True)
            title = info.get("title") or video_id
            # After FFmpegExtractAudio, the file lives at the same path
            # with .m4a extension (preferredcodec).
            base = ydl.prepare_filename(info)
            downloaded = Path(base).with_suffix(".m4a")
    except DownloadError as e:
        raise AudioDownloadError(f"yt-dlp failed to download {url}: {e}") from e

    if not downloaded.exists():
        # Defensive: yt-dlp succeeded but we can't find the file.
        # This can happen with unusual format combos.
        raise AudioDownloadError(
            f"yt-dlp reported success but file not found: expected {downloaded}"
        )

    return DownloadResult(path=downloaded, title=title, video_id=video_id)
