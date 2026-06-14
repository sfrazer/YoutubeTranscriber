"""Summarization via the Ollama cloud API.

Wraps the ollama Python client to take a list of TranscriptSegment
and produce a summary string. The pure formatting logic
(segments -> prompt text) is separated from the IO (calling the
cloud) for testability.

The OLLAMA_API_KEY env var is auto-detected by the ollama client,
but we validate it explicitly and raise OllamaApiKeyMissingError
with actionable guidance when missing.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib import resources

from ollama import Client

from youtubetranscriber.transcribe import TranscriptSegment

# Default cloud model. Strong, well-known, available on Ollama cloud.
# See https://ollama.com/search?c=cloud for the full list.
DEFAULT_MODEL = "gpt-oss:120b"

# Ollama cloud endpoint. Pass this to Client(host=...) to point at
# the cloud rather than a local Ollama install.
CLOUD_HOST = "https://ollama.com"


class OllamaApiKeyMissingError(RuntimeError):
    """Raised when OLLAMA_API_KEY is not set but summarization was requested.

    The error message is designed to be user-actionable: tells the
    user exactly what to do to fix the problem.
    """


class SummarizationError(RuntimeError):
    """Raised when the Ollama API call fails for any reason.

    Wraps the underlying exception with a user-actionable message.
    """


def format_segments_as_text(segments: list[TranscriptSegment]) -> str:
    """Format transcript segments as readable plain text.

    Each segment becomes one line. Speaker labels are prefixed in
    [brackets] when available. The output is intended to be passed
    as the {transcript} placeholder in a prompt template.

    Pure function, no I/O.
    """
    if not segments:
        return ""

    lines: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if seg.speaker:
            lines.append(f"[{seg.speaker}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def load_default_prompt() -> str:
    """Load the default summary prompt shipped with the package.

    Returns the contents of prompts/default_summary.txt, which
    contains a {transcript} placeholder.
    """
    prompt_path = resources.files("youtubetranscriber.prompts").joinpath("default_summary.txt")
    return prompt_path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _get_client() -> Client:
    """Build (and cache) an Ollama client pointed at the cloud.

    The OLLAMA_API_KEY env var is auto-detected by the ollama
    client constructor. We still validate it here so we can give
    a user-actionable error.

    Raises:
        OllamaApiKeyMissingError: if OLLAMA_API_KEY is not set.
    """
    if not os.environ.get("OLLAMA_API_KEY"):
        raise OllamaApiKeyMissingError(
            "Summarization requires an Ollama cloud API key.\n"
            "\n"
            "To fix this:\n"
            "  1. Sign in or create an account at https://ollama.com\n"
            "  2. Create an API key at https://ollama.com/settings/keys\n"
            "  3. Set OLLAMA_API_KEY in your environment, e.g.:\n"
            "       export OLLAMA_API_KEY=your_key_here\n"
        )
    return Client(host=CLOUD_HOST)


def _build_messages(prompt_template: str, transcript_text: str) -> list[dict]:
    """Build the messages list for the ollama chat API.

    The template is expected to contain a {transcript} placeholder.
    We substitute the transcript text into the user message.
    """
    user_content = prompt_template.format(transcript=transcript_text)
    return [{"role": "user", "content": user_content}]


def summarize(
    segments: list[TranscriptSegment],
    *,
    model: str = DEFAULT_MODEL,
    prompt: str | None = None,
) -> str:
    """Summarize transcript segments using the Ollama cloud API.

    Args:
        segments: The transcript to summarize.
        model: Ollama cloud model name. Default: gpt-oss:120b.
        prompt: Custom prompt template. Must contain {transcript}.
            If None, uses the default from prompts/default_summary.txt.

    Returns:
        The summary text returned by the model.

    Raises:
        OllamaApiKeyMissingError: if OLLAMA_API_KEY is not set.
        SummarizationError: if the API call fails.
    """
    if prompt is None:
        prompt = load_default_prompt()

    transcript_text = format_segments_as_text(segments)
    messages = _build_messages(prompt, transcript_text)

    client = _get_client()

    try:
        response = client.chat(model=model, messages=messages)
    except Exception as e:
        raise SummarizationError(f"Ollama API call failed (model={model!r}): {e}") from e

    # The response is a ChatResponse object with .message.content
    content = response.message.content
    if not isinstance(content, str):
        # Defensive: if the client returns something unexpected
        raise SummarizationError(
            f"Ollama returned unexpected content type: {type(content).__name__}"
        )
    return content.strip()
