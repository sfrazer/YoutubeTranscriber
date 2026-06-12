"""Command-line interface for YoutubeTranscriber.

This is the only module that should have side effects at import. All real
work lives in the other modules so they can be imported and tested in
isolation.

Phases:
    1. MVP — fetch audio + transcribe + write txt
    2. Multiple output formats + caption fallback
    3. Speaker diarization (opt-in)
    4. Summarization via Ollama cloud
    5. Polish
"""

from __future__ import annotations

from pathlib import Path

import typer

from youtubetranscriber import audio, output
from youtubetranscriber.transcribe import VALID_MODELS, TranscriptSegment
from youtubetranscriber.transcribe import transcribe as do_transcribe

# Default output directory. Computed once at import time (path is fixed
# for a given user, and Typer needs a concrete default for --help output).
DEFAULT_OUTPUT_DIR = Path.home() / "ytx-output"

app = typer.Typer(
    name="ytx",
    help="Transcribe (and optionally summarize) YouTube videos.",
    no_args_is_help=True,
    add_completion=False,
)


def _validate_model(value: str) -> str:
    if value not in VALID_MODELS:
        raise typer.BadParameter(
            f"Invalid model {value!r}. Must be one of: {', '.join(VALID_MODELS)}"
        )
    return value


def _validate_format(value: str) -> str:
    valid = ("txt", "srt", "json")
    if value not in valid:
        raise typer.BadParameter(f"Invalid format {value!r}. Must be one of: {', '.join(valid)}")
    return value


@app.command()
def transcribe(
    url: str = typer.Argument(..., help="YouTube video URL"),
    model: str = typer.Option(
        "medium",
        "--model",
        "-m",
        help="Whisper model size.",
        callback=_validate_model,
    ),
    format: str = typer.Option(
        "txt",
        "--format",
        "-f",
        help="Output format.",
        callback=_validate_format,
    ),
    diarize: bool = typer.Option(
        False,
        "--diarize/--no-diarize",
        help="Identify speakers (opt-in, requires HF_TOKEN). [phase 3]",
    ),
    summarize: bool = typer.Option(
        False,
        "--summarize/--no-summarize",
        help="Generate a summary using Ollama cloud. [phase 4]",
    ),
    prefer_captions: bool = typer.Option(
        False,
        "--prefer-captions/--no-prefer-captions",
        help="Try YouTube auto-captions first. [phase 2]",
    ),
    output_dir: Path = typer.Option(  # noqa: B008  (Typer idiom: option must be a default-arg call)
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directory to write transcript and audio files.",
    ),
    keep_audio: bool = typer.Option(
        False,
        "--keep-audio/--no-keep-audio",
        help="Keep downloaded audio file after transcription. [phase 5]",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging.",
    ),
) -> None:
    """Transcribe a YouTube video to text."""
    _run_pipeline(
        url=url,
        model=model,
        format=format,
        output_dir=output_dir,
        verbose=verbose,
    )


def _run_pipeline(
    *,
    url: str,
    model: str,
    format: str,
    output_dir: Path,
    verbose: bool,
) -> Path:
    """The actual work: download → transcribe → write. Returns output path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-video subdirectory so audio and transcript live together.
    video_id = audio.extract_video_id(url)
    work_dir = output_dir / video_id
    work_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        typer.echo(f"[ytx] Video ID: {video_id}")
        typer.echo(f"[ytx] Output dir: {work_dir}")

    typer.echo(f"[ytx] Downloading audio from {url} ...")
    audio_path = audio.download_audio(url, work_dir)
    typer.echo(f"[ytx] Audio saved to: {audio_path}")

    typer.echo(f"[ytx] Transcribing with model={model!r} ...")
    segments: list[TranscriptSegment] = do_transcribe(audio_path, model_name=model)
    typer.echo(f"[ytx] Got {len(segments)} segments")

    output_path = work_dir / f"{video_id}.{format}"
    output.write_transcript(segments, output_path, format=format)
    typer.echo(f"[ytx] Transcript written to: {output_path}")

    return output_path


if __name__ == "__main__":
    app()
