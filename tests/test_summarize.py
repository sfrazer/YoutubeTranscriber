"""Tests for summarize.py: Ollama cloud wrapper + transcript formatting.

The pure formatting logic gets full coverage. The Ollama client is
mocked — we don't hit the cloud in tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youtubetranscriber.summarize import (
    OllamaApiKeyMissingError,
    format_segments_as_text,
    load_default_prompt,
    summarize,
)
from youtubetranscriber.transcribe import TranscriptSegment

# --- format_segments_as_text (pure) ---------------------------------------


def test_format_empty_segments() -> None:
    assert format_segments_as_text([]) == ""


def test_format_single_segment_no_speaker() -> None:
    segments = [TranscriptSegment(text="Hello world.", start=0.0, end=1.0)]
    assert format_segments_as_text(segments) == "Hello world."


def test_format_segment_with_speaker() -> None:
    segments = [
        TranscriptSegment(text="Hi there.", start=0.0, end=1.0, speaker="SPEAKER_00"),
    ]
    text = format_segments_as_text(segments)
    assert "SPEAKER_00" in text
    assert "Hi there." in text


def test_format_multiple_segments() -> None:
    segments = [
        TranscriptSegment(text="First.", start=0.0, end=1.0),
        TranscriptSegment(text="Second.", start=1.0, end=2.0),
    ]
    text = format_segments_as_text(segments)
    # All text should be present
    assert "First." in text
    assert "Second." in text
    # Should be readable prose, not all on one giant line
    assert text.count(".") >= 2


def test_format_preserves_speaker_boundaries() -> None:
    """When speakers change, the new speaker should be marked."""
    segments = [
        TranscriptSegment(text="A.", start=0.0, end=1.0, speaker="ALICE"),
        TranscriptSegment(text="B.", start=1.0, end=2.0, speaker="BOB"),
    ]
    text = format_segments_as_text(segments)
    assert "ALICE" in text
    assert "BOB" in text
    # The two should be distinguishable in the output
    assert text.index("ALICE") < text.index("BOB")


def test_format_mixed_speaker_and_no_speaker() -> None:
    """Some segments with speakers, some without, should both render."""
    segments = [
        TranscriptSegment(text="Caption text.", start=0.0, end=1.0),
        TranscriptSegment(text="Whisper text.", start=1.0, end=2.0, speaker="SPEAKER_00"),
    ]
    text = format_segments_as_text(segments)
    assert "Caption text." in text
    assert "SPEAKER_00" in text
    assert "Whisper text." in text


# --- load_default_prompt --------------------------------------------------


def test_load_default_prompt_returns_text() -> None:
    prompt = load_default_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    # Should contain a placeholder for the transcript
    assert "{transcript}" in prompt


# --- summarize() function -------------------------------------------------


def _fake_ollama_response(content: str) -> MagicMock:
    response = MagicMock()
    response.message.content = content
    return response


def test_summarize_calls_ollama_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """summarize() should call the ollama client with a chat message."""
    monkeypatch.setenv("OLLAMA_API_KEY", "test_key")

    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response("This is the summary.")

    segments = [TranscriptSegment(text="hi", start=0.0, end=1.0)]

    with patch("youtubetranscriber.summarize._get_client", return_value=fake_client):
        result = summarize(segments, model="gpt-oss:120b")

    assert result == "This is the summary."
    fake_client.chat.assert_called_once()


def test_summarize_passes_segments_in_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The transcript text should be embedded in the user message."""
    monkeypatch.setenv("OLLAMA_API_KEY", "test_key")

    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response("summary")

    segments = [
        TranscriptSegment(text="Hello world.", start=0.0, end=1.0),
    ]

    with patch("youtubetranscriber.summarize._get_client", return_value=fake_client):
        summarize(segments, model="gpt-oss:120b")

    # Inspect the call
    call_kwargs = fake_client.chat.call_args.kwargs
    messages = call_kwargs.get("messages") or fake_client.chat.call_args.args[1]
    # The user message should contain the segment text
    user_msg = messages[0] if isinstance(messages[0], dict) else messages[0].dict()
    assert "Hello world." in user_msg["content"]


def test_summarize_passes_model_to_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The model parameter should reach client.chat()."""
    monkeypatch.setenv("OLLAMA_API_KEY", "test_key")

    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response("ok")

    segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]

    with patch("youtubetranscriber.summarize._get_client", return_value=fake_client):
        summarize(segments, model="gpt-oss:20b")

    model = fake_client.chat.call_args.kwargs.get("model") or fake_client.chat.call_args.args[0]
    assert model == "gpt-oss:20b"


def test_summarize_raises_on_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without OLLAMA_API_KEY, summarize() should raise a clear error."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]
    with pytest.raises(OllamaApiKeyMissingError, match="OLLAMA_API_KEY"):
        summarize(segments, model="gpt-oss:120b")


def test_summarize_uses_custom_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom prompt template should be used instead of the default."""
    monkeypatch.setenv("OLLAMA_API_KEY", "test_key")

    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response("ok")

    segments = [TranscriptSegment(text="Hello.", start=0.0, end=1.0)]
    custom_prompt = "CUSTOM: {transcript} END"

    with patch("youtubetranscriber.summarize._get_client", return_value=fake_client):
        summarize(segments, model="gpt-oss:120b", prompt=custom_prompt)

    messages = fake_client.chat.call_args.kwargs.get("messages")
    user_msg = messages[0] if isinstance(messages[0], dict) else messages[0].dict()
    # The custom prompt should wrap the transcript
    assert user_msg["content"].startswith("CUSTOM: ")
    assert user_msg["content"].endswith(" END")
    assert "Hello." in user_msg["content"]


def test_summarize_wraps_ollama_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A generic Ollama error should be surfaced as SummarizationError."""
    from youtubetranscriber.summarize import SummarizationError

    monkeypatch.setenv("OLLAMA_API_KEY", "test_key")

    fake_client = MagicMock()
    fake_client.chat.side_effect = RuntimeError("network down")

    segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]

    with (
        patch("youtubetranscriber.summarize._get_client", return_value=fake_client),
        pytest.raises(SummarizationError, match="network down"),
    ):
        summarize(segments, model="gpt-oss:120b")


def test_summarize_strips_whitespace_from_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The returned summary should be stripped of leading/trailing whitespace."""
    monkeypatch.setenv("OLLAMA_API_KEY", "test_key")

    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_ollama_response("   summary text  \n\n")

    segments = [TranscriptSegment(text="x", start=0.0, end=1.0)]

    with patch("youtubetranscriber.summarize._get_client", return_value=fake_client):
        result = summarize(segments, model="gpt-oss:120b")

    assert result == "summary text"
