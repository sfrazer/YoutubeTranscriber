"""Speaker diarization via pyannote.audio.

Wraps the pyannote Pipeline to return our own SpeakerSpan list.
The pyannote model requires:
  1. HF_TOKEN env var set to a Hugging Face access token
  2. The user to have accepted the EULA at
     https://huggingface.co/pyannote/speaker-diarization-3.1

We surface missing/invalid tokens as HfTokenMissingError so the
CLI can produce a clear, actionable error.

The model is loaded lazily (only when diarize() is called) and
cached at module level so subsequent calls don't re-download.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pyannote.audio import Pipeline

from youtubetranscriber.merge import SpeakerSpan

# The pretrained pipeline to use. pyannote-audio 4.x exposes
# speaker-diarization-3.1 as the current state-of-the-art.
DEFAULT_PIPELINE = "pyannote/speaker-diarization-3.1"


class HfTokenMissingError(RuntimeError):
    """Raised when HF_TOKEN is not set but diarization was requested.

    The error message is designed to be user-actionable: tells the
    user exactly what to do to fix the problem.
    """


@lru_cache(maxsize=1)
def _load_pipeline() -> Pipeline:
    """Load (and cache) the pyannote diarization pipeline.

    Raises:
        HfTokenMissingError: if HF_TOKEN env var is not set.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise HfTokenMissingError(
            "Diarization requires a Hugging Face access token.\n"
            "\n"
            "To fix this:\n"
            "  1. Create a free account at https://huggingface.co/join\n"
            "  2. Accept the EULAs at ALL three gated pyannote repos:\n"
            "     - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "     - https://huggingface.co/pyannote/segmentation-3.0\n"
            "     - https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "  3. Create an access token at https://huggingface.co/settings/tokens\n"
            "  4. Set HF_TOKEN in your environment, e.g.:\n"
            "       export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx\n"
        )

    pipeline = Pipeline.from_pretrained(DEFAULT_PIPELINE, token=token)
    if pipeline is None:
        # from_pretrained returns None when the user hasn't accepted
        # the EULA for the model OR when a gated dependency is
        # inaccessible. Surface this as a token-missing error so
        # the user gets the same actionable guidance.
        raise HfTokenMissingError(
            "Could not load the diarization pipeline. This usually means\n"
            "your HF_TOKEN doesn't have access to the model or one of\n"
            "its three gated dependencies. Please:\n"
            "  1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "     and click 'Agree and access repository' to accept the EULA\n"
            "  2. Visit https://huggingface.co/pyannote/segmentation-3.0\n"
            "     and request access (it's a gated dependency)\n"
            "  3. Visit https://huggingface.co/pyannote/speaker-diarization-community-1\n"
            "     and request access (another gated dependency)\n"
            "  4. Make sure your HF_TOKEN is from the same Hugging Face account\n"
        )
    return pipeline


def _annotation_to_spans(annotation) -> list[SpeakerSpan]:
    """Convert a pyannote Annotation to a list of our SpeakerSpan."""
    spans: list[SpeakerSpan] = []
    for segment, _track, label in annotation.itertracks(yield_label=True):
        spans.append(
            SpeakerSpan(
                speaker=str(label),
                start=float(segment.start),
                end=float(segment.end),
            )
        )
    return spans


def diarize(
    audio_path: Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[SpeakerSpan]:
    """Run speaker diarization on an audio file.

    Args:
        audio_path: Path to the audio file (m4a, wav, etc.).
        num_speakers: If set, tell pyannote exactly how many speakers
            to expect. If None, pyannote auto-detects.
        min_speakers: Minimum number of speakers to consider.
        max_speakers: Maximum number of speakers to consider.

    Returns:
        List of SpeakerSpan, one per (speaker, time-range) detected
        by the diarization model.

    Raises:
        FileNotFoundError: if audio_path doesn't exist.
        HfTokenMissingError: if HF_TOKEN isn't set or doesn't grant
            access to the diarization model.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    pipeline = _load_pipeline()

    # Build kwargs for the pipeline call. Only pass parameters that
    # the user actually specified — pyannote's defaults are good for
    # the unspecified cases.
    kwargs: dict = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    annotation = pipeline(str(audio_path), **kwargs)
    return _annotation_to_spans(annotation)
