"""Audio acquisition via yt-dlp.

Depends on: ffmpeg (system binary), yt-dlp (pip).
Side effects: writes .m4a files to disk.
Pure functions: extract_video_id()
"""

from __future__ import annotations

import re
from pathlib import Path

import yt_dlp
from yt_dlp.utils import DownloadError

# YouTube video IDs are exactly 11 characters from [a-zA-Z0-9_-]
_VIDEO_ID_PATTERN = re.compile(r"[a-zA-Z0-9_-]{11}")


class InvalidURLError(ValueError):
    """Raised when a YouTube URL is malformed or doesn't contain a video ID."""


class AudioDownloadError(RuntimeError):
    """Raised when yt-dlp fails to download the audio."""


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

    Raises:
        InvalidURLError: if the URL is not a recognized YouTube URL or
            has no valid 11-character video ID.
    """
    if not url or not isinstance(url, str):
        raise InvalidURLError(f"URL must be a non-empty string, got: {url!r}")

    # Normalize: must contain "youtube" or be a youtu.be short link
    lowered = url.lower()
    is_youtube = "youtube.com" in lowered or "youtu.be" in lowered
    if not is_youtube:
        raise InvalidURLError(f"Not a YouTube URL: {url!r}")

    # The path component (after the host) is where the ID lives for
    # short, embed, and shorts URLs. For watch URLs, it's the v= query.
    # Try the query string first since it's the most common case.
    # The ID must be exactly 11 chars and terminated by &, #, or end of string.
    match = re.search(r"[?&]v=([a-zA-Z0-9_-]{11})(?:[&#]|$)", url)
    if match:
        return match.group(1)

    # Fall back to a path-based match: /ID or /shorts/ID or /embed/ID
    # We require a path segment that is *exactly* 11 chars, not just
    # any 11-char substring (which could match garbage in unrelated paths).
    path_match = re.search(r"/(?:shorts/|embed/)?([a-zA-Z0-9_-]{11})(?:[/?#]|$)", url)
    if path_match:
        return path_match.group(1)

    raise InvalidURLError(f"Could not extract video ID from: {url!r}")


def download_audio(url: str, output_dir: Path) -> Path:
    """Download audio only from a YouTube URL to output_dir.

    Returns the path to the downloaded .m4a file. The file will be named
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
            # download() returns True on success
            ydl.download([url])
            # The file's actual name comes from yt-dlp's prepare_filename
            # *before* postprocessing. We need the postprocessed name.
            info = ydl.extract_info(url, download=True)
            # After FFmpegExtractAudio, the file lives at the same path
            # with .m4a extension (preferredcodec).
            base = ydl.prepare_filename(info)
            # Strip the original extension, append .m4a
            downloaded = Path(base).with_suffix(".m4a")
    except DownloadError as e:
        raise AudioDownloadError(f"yt-dlp failed to download {url}: {e}") from e

    if not downloaded.exists():
        # Defensive: yt-dlp succeeded but we can't find the file.
        # This can happen with unusual format combos.
        raise AudioDownloadError(
            f"yt-dlp reported success but file not found: expected {downloaded}"
        )

    # Reference video_id to keep linter happy and document the validation.
    _ = video_id
    return downloaded
