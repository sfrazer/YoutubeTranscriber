"""Command-line interface for YoutubeTranscriber.

This is the only module that should have side effects on import. All real
work lives in the other modules so they can be imported and tested in
isolation.

Phases:
    1. MVP — fetch audio + transcribe + write txt
    2. Multiple output formats + caption fallback
    3. Speaker diarization (opt-in)
    4. Summarization via Ollama cloud
    5. Polish
"""

import typer

app = typer.Typer(
    name="ytx",
    help="Transcribe (and optionally summarize) YouTube videos.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def transcribe(
    url: str = typer.Argument(..., help="YouTube video URL"),
    model: str = typer.Option(
        "medium",
        "--model",
        "-m",
        help="Whisper model size: tiny, base, small, medium, large-v3",
    ),
    format: str = typer.Option(
        "txt",
        "--format",
        "-f",
        help="Output format: txt, srt, json",
    ),
    diarize: bool = typer.Option(
        False,
        "--diarize/--no-diarize",
        help="Identify speakers (opt-in, requires HF_TOKEN).",
    ),
    summarize: bool = typer.Option(
        False,
        "--summarize/--no-summarize",
        help="Generate a summary using Ollama cloud.",
    ),
    prefer_captions: bool = typer.Option(
        False,
        "--prefer-captions/--no-prefer-captions",
        help="Try YouTube auto-captions first, fall back to Whisper.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
) -> None:
    """Transcribe a YouTube video to text."""
    raise NotImplementedError("Phase 1: not yet implemented")


if __name__ == "__main__":
    app()
