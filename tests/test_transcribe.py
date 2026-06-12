"""Tests for transcribe.py.

Pure conversion logic gets full unit tests. The model invocation is
mocked in most tests, with one integration test marked slow that runs
the real model on a small synthetic fixture.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youtubetranscriber.transcribe import (
    TranscriptSegment,
    WhisperModelError,
    detect_device,
    transcribe,
)

# --- TranscriptSegment (data class shape) ---------------------------------


def test_segment_defaults_speaker_none() -> None:
    seg = TranscriptSegment(text="hello", start=0.0, end=1.0)
    assert seg.text == "hello"
    assert seg.start == 0.0
    assert seg.end == 1.0
    assert seg.speaker is None


def test_segment_accepts_speaker() -> None:
    seg = TranscriptSegment(text="hi", start=1.0, end=2.0, speaker="SPEAKER_00")
    assert seg.speaker == "SPEAKER_00"


# --- detect_device ---------------------------------------------------------


def test_detect_device_returns_string() -> None:
    """detect_device should always return a non-empty string."""
    device = detect_device()
    assert device in ("cpu", "mps", "cuda")
    # On this machine (Apple Silicon with no CUDA), expect mps or cpu.
    # We don't assert on the exact value because of the device-fallback
    # behavior in transcribe().


# --- transcribe (mocked) ---------------------------------------------------


def _fake_segments() -> list[MagicMock]:
    """Build a list of fake segment objects as faster-whisper returns them."""
    s1 = MagicMock()
    s1.start = 0.0
    s1.end = 2.5
    s1.text = " Hello world."
    s1.words = None

    s2 = MagicMock()
    s2.start = 2.5
    s2.end = 5.0
    s2.text = " This is a test."
    s2.words = None

    return [s1, s2]


def _fake_info() -> MagicMock:
    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.98
    info.duration = 5.0
    return info


def test_transcribe_returns_list_of_segments(tmp_path: Path) -> None:
    """transcribe should return TranscriptSegment objects with text cleaned."""
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (_fake_segments(), _fake_info())

    with patch("youtubetranscriber.transcribe.WhisperModel", return_value=fake_model):
        result = transcribe(audio, model_name="tiny", device="cpu")

    assert len(result) == 2
    assert all(isinstance(s, TranscriptSegment) for s in result)
    # Text should be stripped of leading whitespace
    assert result[0].text == "Hello world."
    assert result[1].text == "This is a test."


def test_transcribe_passes_audio_path_to_model(tmp_path: Path) -> None:
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (_fake_segments(), _fake_info())

    with patch("youtubetranscriber.transcribe.WhisperModel", return_value=fake_model):
        transcribe(audio, model_name="small", device="cpu")

    # Model should be called with the path (as string)
    call_args = fake_model.transcribe.call_args
    assert str(audio) in call_args.args or str(audio) in str(call_args.kwargs)


def test_transcribe_uses_device_parameter(tmp_path: Path) -> None:
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    fake_model = MagicMock()
    fake_model.transcribe.return_value = (_fake_segments(), _fake_info())

    with patch("youtubetranscriber.transcribe.WhisperModel", return_value=fake_model) as wm_cls:
        transcribe(audio, model_name="medium", device="mps")

    # WhisperModel should be constructed with device="mps"
    assert wm_cls.call_args.kwargs.get("device") == "mps"


def test_transcribe_wraps_model_errors(tmp_path: Path) -> None:
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    fake_model = MagicMock()
    fake_model.transcribe.side_effect = RuntimeError("model crashed")

    with (
        patch("youtubetranscriber.transcribe.WhisperModel", return_value=fake_model),
        pytest.raises(WhisperModelError, match="model crashed"),
    ):
        transcribe(audio, model_name="tiny", device="cpu")


def test_transcribe_falls_back_to_cpu_on_unsupported_device(tmp_path: Path) -> None:
    """If the requested device isn't supported, fall back to CPU."""
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"fake")

    fake_cpu_model = MagicMock()
    fake_cpu_model.transcribe.return_value = (_fake_segments(), _fake_info())

    call_count = {"n": 0}

    def fake_constructor(model_name, **kwargs):
        call_count["n"] += 1
        if kwargs.get("device") == "mps":
            raise ValueError("unsupported device mps")
        return fake_cpu_model

    with (
        patch("youtubetranscriber.transcribe.WhisperModel", side_effect=fake_constructor),
        pytest.warns(UserWarning, match="falling back to 'cpu'"),
    ):
        result = transcribe(audio, model_name="tiny", device="mps")

    assert len(result) == 2
    # Should have tried mps, then cpu
    assert call_count["n"] == 2


def test_transcribe_raises_if_audio_missing(tmp_path: Path) -> None:
    audio = tmp_path / "does_not_exist.m4a"
    with pytest.raises(FileNotFoundError):
        transcribe(audio, model_name="tiny", device="cpu")


# --- Integration test (slow) -----------------------------------------------


@pytest.mark.slow
def test_transcribe_real_model_on_synthetic_audio(tmp_path: Path) -> None:
    """End-to-end: generate synthetic audio, transcribe with 'tiny' model.

    This test is slow (~30s first run, downloads tiny model). Mark with
    'slow' and skip by default. Run with: uv run pytest -m slow
    """
    import subprocess

    audio = tmp_path / "sine.m4a"
    # Generate 3 seconds of 440Hz tone (sine wave). Whisper will probably
    # return empty or nonsense, but it should not crash.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-c:a",
            "aac",
            str(audio),
        ],
        check=True,
        capture_output=True,
    )
    assert audio.exists()

    result = transcribe(audio, model_name="tiny", device="cpu")
    # We don't assert on the content (sine waves don't produce speech),
    # only that we got back a list of segments.
    assert isinstance(result, list)
    for seg in result:
        assert isinstance(seg, TranscriptSegment)
        assert seg.start >= 0
        assert seg.end >= seg.start
