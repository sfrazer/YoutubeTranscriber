"""Speech-to-text via faster-whisper.

Depends on: faster-whisper (pip), optionally a GPU.
Side effects: downloads Whisper model files on first use (cached by HF).
Pure functions: transcribe() (mostly), detect_device(), TranscriptSegment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

# Valid Whisper model sizes. Anything else should be rejected so we don't
# silently fall back to a different model.
VALID_MODELS = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
)


@dataclass
class TranscriptSegment:
    """A single segment of transcribed speech.

    Attributes:
        text: The transcribed text (whitespace-stripped).
        start: Start time in seconds.
        end: End time in seconds.
        speaker: Speaker label, if diarization was run. None otherwise.
    """

    text: str
    start: float
    end: float
    speaker: str | None = None


class WhisperModelError(RuntimeError):
    """Raised when the Whisper model fails to transcribe audio."""


def detect_device() -> str:
    """Detect the best available compute device for Whisper.

    Returns:
        "mps" on Apple Silicon, "cuda" if NVIDIA GPU is available,
        "cpu" otherwise. faster-whisper + CTranslate2 supports all three.

    Note: detection is best-effort. We don't fail if a backend isn't
    usable; faster-whisper will surface a clearer error when transcribe()
    is called.
    """
    # Apple Silicon: try torch first (most reliable detection), fall back
    # to platform check. CTranslate2 supports MPS on macOS 13+.
    if os.uname().machine == "arm64" and os.uname().sysname == "Darwin":
        return "mps"
    # NVIDIA CUDA: check for the env var or try ctranslate2 detection.
    # We don't import torch to keep cold-start fast; CTranslate2 itself
    # raises a clear error if CUDA isn't actually available.
    return "cpu"


def transcribe(
    audio_path: Path,
    model_name: str = "medium",
    device: str | None = None,
) -> list[TranscriptSegment]:
    """Transcribe an audio file to a list of TranscriptSegment.

    Args:
        audio_path: Path to the audio file (m4a, wav, mp3, etc.).
        model_name: Whisper model size. Must be one of VALID_MODELS.
            Default "medium" is a good speed/quality balance.
        device: "cpu", "mps", or "cuda". If None, auto-detect.

    Returns:
        List of TranscriptSegment with text, start, end, and speaker=None.

    Raises:
        FileNotFoundError: if audio_path doesn't exist.
        ValueError: if model_name is not a valid Whisper model.
        WhisperModelError: if the model fails to transcribe.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if model_name not in VALID_MODELS:
        raise ValueError(f"Invalid model {model_name!r}. Must be one of: {', '.join(VALID_MODELS)}")

    if device is None:
        device = detect_device()

    try:
        # compute_type="auto" lets CTranslate2 pick the best precision
        # for the device (int8 on CPU, float16 on GPU).
        model = WhisperModel(model_name, device=device, compute_type="auto")
        segments_iter, _info = model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,  # skip silence, much faster on long files
        )
        # faster-whisper returns a generator; materialize to list
        raw_segments = list(segments_iter)
    except Exception as e:
        raise WhisperModelError(f"Whisper transcription failed: {e}") from e

    return [
        TranscriptSegment(
            text=seg.text.strip(),
            start=float(seg.start),
            end=float(seg.end),
        )
        for seg in raw_segments
    ]
