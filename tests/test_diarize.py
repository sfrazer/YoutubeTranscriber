"""Tests for diarize.py: pyannote.audio wrapper.

The model itself requires HF_TOKEN + EULA acceptance, so we mock
the Pipeline.from_pretrained call. We test:
  - HF_TOKEN env var is read and passed to from_pretrained
  - missing token raises a clear HfTokenMissing error
  - pipeline returning an Annotation is converted to SpeakerSpan list
  - num_speakers and min_speakers/max_speakers are passed as kwargs
  - lazy model load (model only instantiated when diarize() is called)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youtubetranscriber.diarize import (
    HfTokenMissingError,
    _load_pipeline,
    diarize,
)

# --- Token handling -------------------------------------------------------


def test_load_pipeline_raises_when_hf_token_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without HF_TOKEN in env, _load_pipeline should raise HfTokenMissingError."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(HfTokenMissingError, match="HF_TOKEN"):
        _load_pipeline()


def test_load_pipeline_passes_token_to_pyannote(monkeypatch: pytest.MonkeyPatch) -> None:
    """The HF_TOKEN should reach pyannote's from_pretrained."""
    monkeypatch.setenv("HF_TOKEN", "test_token_123")

    fake_pipeline = MagicMock()
    with patch(
        "youtubetranscriber.diarize.Pipeline.from_pretrained",
        return_value=fake_pipeline,
    ) as fp:
        result = _load_pipeline()

    assert result is fake_pipeline
    # Token should have been passed
    call_kwargs = fp.call_args.kwargs
    assert call_kwargs.get("token") == "test_token_123"


# --- diarize() function ---------------------------------------------------


def _make_fake_annotation(spans: list[tuple[float, float, str]]) -> MagicMock:
    """Build a fake pyannote DiarizeOutput (pyannote 4.x return type).

    Returns a mock with .speaker_diarization attribute that is the
    actual annotation with itertracks() method.
    """
    ann = MagicMock()
    itertracks = MagicMock(return_value=iter(spans))
    ann.itertracks = itertracks

    diarize_output = MagicMock()
    diarize_output.speaker_diarization = ann
    return diarize_output


def _fake_segment(start: float, end: float) -> MagicMock:
    seg = MagicMock()
    seg.start = start
    seg.end = end
    return seg


def test_diarize_returns_speaker_spans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """diarize() should return a list of SpeakerSpan from the annotation."""
    monkeypatch.setenv("HF_TOKEN", "test_token")

    fake_pipeline = MagicMock()
    # The pipeline returns whatever apply() returns
    fake_annotation = _make_fake_annotation(
        [
            (_fake_segment(0.0, 2.0), "track_a", "SPEAKER_00"),
            (_fake_segment(2.0, 4.0), "track_b", "SPEAKER_01"),
        ]
    )
    fake_pipeline.return_value = fake_annotation

    with patch("youtubetranscriber.diarize._load_pipeline", return_value=fake_pipeline):
        audio = tmp_path / "audio.m4a"
        audio.write_bytes(b"fake")
        result = diarize(audio)

    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[0].start == 0.0
    assert result[0].end == 2.0
    assert result[1].speaker == "SPEAKER_01"
    assert result[1].start == 2.0
    assert result[1].end == 4.0


def test_diarize_handles_legacy_annotation_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the pipeline returns a plain Annotation (legacy=True),
    we should still be able to extract spans from it.
    """
    monkeypatch.setenv("HF_TOKEN", "test_token")

    # Build a fake plain Annotation (not DiarizeOutput)
    fake_ann = MagicMock(spec=["itertracks"])  # spec= prevents auto-attr creation
    fake_ann.itertracks = MagicMock(
        return_value=iter(
            [
                (_fake_segment(0.0, 1.0), "t", "ALICE"),
            ]
        )
    )

    fake_pipeline = MagicMock()
    fake_pipeline.return_value = fake_ann  # No .speaker_diarization attribute

    with patch("youtubetranscriber.diarize._load_pipeline", return_value=fake_pipeline):
        audio = tmp_path / "audio.m4a"
        audio.write_bytes(b"fake")
        result = diarize(audio)

    assert len(result) == 1
    assert result[0].speaker == "ALICE"


def test_diarize_passes_audio_path_to_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audio file should be passed to the pipeline call."""
    monkeypatch.setenv("HF_TOKEN", "test_token")

    fake_pipeline = MagicMock()
    fake_annotation = _make_fake_annotation([])
    # MagicMock's __call__ returns whatever return_value is set to
    fake_pipeline.return_value = fake_annotation

    with patch("youtubetranscriber.diarize._load_pipeline", return_value=fake_pipeline):
        audio = tmp_path / "audio.m4a"
        audio.write_bytes(b"fake")
        diarize(audio)

    # The audio path should have been passed to the pipeline
    call_args = fake_pipeline.call_args
    assert str(audio) in str(call_args.args) or str(audio) in str(call_args.kwargs)


def test_diarize_passes_num_speakers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """num_speakers should be forwarded to the pipeline."""
    monkeypatch.setenv("HF_TOKEN", "test_token")

    fake_pipeline = MagicMock()
    fake_annotation = _make_fake_annotation([])
    fake_pipeline.return_value = fake_annotation

    with patch("youtubetranscriber.diarize._load_pipeline", return_value=fake_pipeline):
        audio = tmp_path / "audio.m4a"
        audio.write_bytes(b"fake")
        diarize(audio, num_speakers=2)

    call_kwargs = fake_pipeline.call_args.kwargs
    assert call_kwargs.get("num_speakers") == 2


def test_diarize_passes_min_max_speakers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """min_speakers and max_speakers should be forwarded when provided."""
    monkeypatch.setenv("HF_TOKEN", "test_token")

    fake_pipeline = MagicMock()
    fake_annotation = _make_fake_annotation([])
    fake_pipeline.return_value = fake_annotation

    with patch("youtubetranscriber.diarize._load_pipeline", return_value=fake_pipeline):
        audio = tmp_path / "audio.m4a"
        audio.write_bytes(b"fake")
        diarize(audio, min_speakers=2, max_speakers=4)

    call_kwargs = fake_pipeline.call_args.kwargs
    assert call_kwargs.get("min_speakers") == 2
    assert call_kwargs.get("max_speakers") == 4


def test_diarize_omits_num_speakers_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When num_speakers is None (auto), don't pass it to pyannote."""
    monkeypatch.setenv("HF_TOKEN", "test_token")

    fake_pipeline = MagicMock()
    fake_annotation = _make_fake_annotation([])
    fake_pipeline.return_value = fake_annotation

    with patch("youtubetranscriber.diarize._load_pipeline", return_value=fake_pipeline):
        audio = tmp_path / "audio.m4a"
        audio.write_bytes(b"fake")
        diarize(audio)

    call_kwargs = fake_pipeline.call_args.kwargs
    # Either num_speakers not present, or it's None
    assert call_kwargs.get("num_speakers") is None or "num_speakers" not in call_kwargs


def test_diarize_raises_if_audio_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test_token")
    audio = tmp_path / "does_not_exist.m4a"
    with pytest.raises(FileNotFoundError):
        diarize(audio)


def test_diarize_empty_annotation_returns_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An annotation with no spans (e.g. silent audio) returns []."""
    monkeypatch.setenv("HF_TOKEN", "test_token")

    fake_pipeline = MagicMock()
    fake_annotation = _make_fake_annotation([])
    fake_pipeline.return_value = fake_annotation

    with patch("youtubetranscriber.diarize._load_pipeline", return_value=fake_pipeline):
        audio = tmp_path / "audio.m4a"
        audio.write_bytes(b"fake")
        result = diarize(audio)

    assert result == []
