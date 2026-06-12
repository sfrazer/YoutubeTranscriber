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
        "cpu" otherwise.

    Note: detection is best-effort. We don't fail if a backend isn't
    usable; faster-whisper will surface a clearer error when transcribe()
    is called. Callers should handle the case where the device the model
    was constructed with doesn't actually work (e.g. CTranslate2 4+
    dropped MPS support for Whisper).
    """
    # CTranslate2 v4+ no longer supports MPS for Whisper. We still return
    # "mps" here for backwards compatibility and discoverability, but
    # transcribe() will fall back to CPU if MPS construction fails.
    # Check for NVIDIA GPU via CUDA.
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    # Apple Silicon: report MPS (will fall back to CPU if unsupported).
    if os.uname().machine == "arm64" and os.uname().sysname == "Darwin":
        return "mps"
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

    # Try the requested device first. If model construction fails (e.g.
    # CTranslate2 4+ doesn't support MPS for Whisper), fall back to CPU.
    # We do this by attempting construction and catching the ValueError.
    try:
        # compute_type="auto" lets CTranslate2 pick the best precision
        # for the device (int8 on CPU, float16 on GPU).
        model = WhisperModel(model_name, device=device, compute_type="auto")
    except (ValueError, RuntimeError) as e:
        if device != "cpu":
            # Fall back to CPU and warn the user.
            import warnings

            warnings.warn(
                f"Device {device!r} not supported for Whisper ({e}); falling back to 'cpu'.",
                stacklevel=2,
            )
            device = "cpu"
            model = WhisperModel(model_name, device=device, compute_type="auto")
        else:
            raise WhisperModelError(f"Whisper transcription failed: {e}") from e

    try:
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
