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

from youtubetranscriber import audio, captions, output, paths
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
        help=("Try YouTube auto-captions first; fall back to Whisper if unavailable."),
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
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Prompt before overwriting an existing output directory.",
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
        prefer_captions=prefer_captions,
        interactive=interactive,
        verbose=verbose,
    )


def _run_pipeline(
    *,
    url: str,
    model: str,
    format: str,
    output_dir: Path,
    prefer_captions: bool,
    interactive: bool,
    verbose: bool,
) -> Path:
    """The actual work: download → transcribe → write. Returns output path.

    When `prefer_captions` is True, tries YouTube's auto/manual captions
    first; on CaptionsUnavailableError, falls through to Whisper.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        typer.echo(f"[ytx] Output root: {output_dir}")

    typer.echo(f"[ytx] Fetching video info from {url} ...")
    info = audio.get_video_info(url)
    typer.echo(f"[ytx] Title: {info.title}")

    work_dir = paths.resolve_unique_dir(output_dir, info.title, interactive=interactive)
    work_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        typer.echo(f"[ytx] Work dir: {work_dir}")

    segments = _obtain_segments(
        url=url,
        info=info,
        work_dir=work_dir,
        model=model,
        prefer_captions=prefer_captions,
        verbose=verbose,
    )
    typer.echo(f"[ytx] Got {len(segments)} segments")

    output_path = work_dir / f"{info.video_id}.{format}"
    output.write_transcript(
        segments, output_path, format=format, video_id=info.video_id, title=info.title
    )
    typer.echo(f"[ytx] Transcript written to: {output_path}")

    return output_path


def _obtain_segments(
    *,
    url: str,
    info: audio.DownloadResult,
    work_dir: Path,
    model: str,
    prefer_captions: bool,
    verbose: bool,
) -> list[TranscriptSegment]:
    """Get transcript segments from captions or Whisper.

    If `prefer_captions` is True, try captions first and fall back to
    Whisper on CaptionsUnavailableError. Otherwise go straight to
    Whisper.
    """
    if prefer_captions:
        try:
            typer.echo("[ytx] Trying YouTube captions ...")
            segments = captions.fetch_captions(info.video_id)
            if segments:
                typer.echo(f"[ytx] Got {len(segments)} caption segments")
                return segments
            # Empty list is technically not an error, but a video with
            # no caption segments is useless. Fall through to Whisper.
            typer.echo("[ytx] No caption segments found; falling back to Whisper")
        except captions.CaptionsUnavailableError as e:
            typer.echo(f"[ytx] Captions unavailable: {e}")
            typer.echo("[ytx] Falling back to Whisper")
    result = audio.download_audio(url, work_dir)
    typer.echo(f"[ytx] Audio saved to: {result.path}")

    typer.echo(f"[ytx] Transcribing with model={model!r} ...")
    return do_transcribe(result.path, model_name=model)


if __name__ == "__main__":
    app()
