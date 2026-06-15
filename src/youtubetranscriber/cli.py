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

User-facing status output ("[ytx] ...") goes to stderr so that stdout
remains free for any future machine-readable output. Transcript content
is always written to a file (under --output-dir), never to stdout.
"""

from __future__ import annotations

from pathlib import Path

import typer

from youtubetranscriber import audio, captions, diarize, merge, output, paths, progress, summarize
from youtubetranscriber.audio import AudioDownloadError, InvalidURLError, VideoInfo
from youtubetranscriber.diarize import HfTokenMissingError
from youtubetranscriber.summarize import OllamaApiKeyMissingError, SummarizationError
from youtubetranscriber.transcribe import VALID_MODELS, TranscriptSegment, WhisperModelError
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


def _status(msg: str) -> None:
    """Print a user-facing status message to stderr.

    All [ytx] chatter goes to stderr so stdout is reserved for any
    future machine-readable output and `ytx URL > file.txt` produces
    an empty file rather than a file full of status noise.
    """
    typer.echo(msg, err=True)


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


def _validate_summary_prompt(value: Path | None) -> Path | None:
    """Validate --summary-prompt eagerly so a typo fails before any work.

    Without this, a missing prompt file would only surface as a
    FileNotFoundError raised from deep inside _maybe_summarize(),
    after the pipeline has already done network IO. Checking up front
    also lets typer format the error with its standard BadParameter
    presentation, consistent with _validate_model / _validate_format.

    The callback receives None when --summary-prompt is not supplied
    (the default), so we must short-circuit before touching the path.
    """
    if value is None:
        return None
    if not value.exists():
        raise typer.BadParameter(f"Summary prompt file not found: {value}")
    if not value.is_file():
        raise typer.BadParameter(f"Summary prompt path is not a file: {value}")
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
    diarize_flag: bool = typer.Option(
        False,
        "--diarize/--no-diarize",
        help=(
            "Identify speakers (requires HF_TOKEN; only works with "
            "Whisper-sourced transcripts, not captions)."
        ),
    ),
    num_speakers: int | None = typer.Option(
        None,
        "--num-speakers",
        help="Exact number of speakers (passed to pyannote). Default: auto-detect.",
    ),
    min_speakers: int | None = typer.Option(
        None,
        "--min-speakers",
        help="Minimum number of speakers for pyannote to consider.",
    ),
    max_speakers: int | None = typer.Option(
        None,
        "--max-speakers",
        help="Maximum number of speakers for pyannote to consider.",
    ),
    summarize_flag: bool = typer.Option(
        False,
        "--summarize/--no-summarize",
        help="Generate a summary using Ollama cloud (requires OLLAMA_API_KEY).",
    ),
    summary_model: str = typer.Option(
        "gpt-oss:120b",
        "--summary-model",
        help="Ollama cloud model for summarization.",
    ),
    summary_prompt: Path | None = typer.Option(  # noqa: B008  (Typer idiom)
        None,
        "--summary-prompt",
        help=(
            "Path to a custom prompt template (must contain "
            "{transcript}). Default: the bundled prompt."
        ),
        callback=_validate_summary_prompt,
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
        help=(
            "Keep the downloaded .m4a audio file alongside the "
            "transcript. Default: delete it (the audio is intermediate; "
            "use this flag if you plan to re-transcribe with a different "
            "model)."
        ),
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
    try:
        _run_pipeline(
            url=url,
            model=model,
            format=format,
            output_dir=output_dir,
            prefer_captions=prefer_captions,
            diarize_flag=diarize_flag,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            summarize_flag=summarize_flag,
            summary_model=summary_model,
            summary_prompt=summary_prompt,
            keep_audio=keep_audio,
            interactive=interactive,
            verbose=verbose,
        )
    except InvalidURLError as e:
        # Bad URL — surface to the user with a clean message, no traceback.
        _status(f"[ytx] Error: {e}")
        raise typer.Exit(code=1) from e
    except (AudioDownloadError, WhisperModelError) as e:
        # yt-dlp or Whisper failed — user-actionable (network, video
        # unavailable, model load failure, etc.). Clean message, exit 1.
        _status(f"[ytx] Error: {e}")
        raise typer.Exit(code=1) from e


def _run_pipeline(
    *,
    url: str,
    model: str,
    format: str,
    output_dir: Path,
    prefer_captions: bool,
    diarize_flag: bool,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    summarize_flag: bool,
    summary_model: str,
    summary_prompt: Path | None,
    keep_audio: bool,
    interactive: bool,
    verbose: bool,
) -> Path:
    """The actual work: fetch → (captions | transcribe | diarize) → write."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        _status(f"[ytx] Output root: {output_dir}")

    with progress.step_progress(f"Fetching video info from {url}"):
        info = audio.get_video_info(url)
    _status(f"[ytx] Title: {info.title}")

    work_dir = paths.resolve_unique_dir(output_dir, info.title, interactive=interactive)
    work_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        _status(f"[ytx] Work dir: {work_dir}")

    segments, audio_path = _obtain_segments(
        url=url,
        info=info,
        work_dir=work_dir,
        model=model,
        prefer_captions=prefer_captions,
        verbose=verbose,
    )
    _status(f"[ytx] Got {len(segments)} segments")

    if diarize_flag:
        segments = _maybe_diarize(
            segments=segments,
            audio_path=audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    output_path = work_dir / f"{info.video_id}.{format}"
    output.write_transcript(
        segments, output_path, format=format, video_id=info.video_id, title=info.title
    )
    _status(f"[ytx] Transcript written to: {output_path}")

    if summarize_flag:
        _maybe_summarize(
            segments=segments,
            work_dir=work_dir,
            video_id=info.video_id,
            title=info.title,
            model=summary_model,
            prompt_path=summary_prompt,
        )

    if not keep_audio and audio_path is not None:
        _cleanup_audio(audio_path)

    return output_path


def _cleanup_audio(audio_path: Path) -> None:
    """Delete the intermediate audio file. Best-effort."""
    try:
        audio_path.unlink(missing_ok=True)
        _status(f"[ytx] Removed audio file: {audio_path}")
    except OSError as e:
        # Don't fail the whole run just because we couldn't clean up
        _status(f"[ytx] Warning: could not remove {audio_path}: {e}")


def _obtain_segments(
    *,
    url: str,
    info: VideoInfo,
    work_dir: Path,
    model: str,
    prefer_captions: bool,
    verbose: bool,
) -> tuple[list[TranscriptSegment], Path | None]:
    """Get transcript segments and the audio path that produced them.

    Returns (segments, audio_path). The audio_path is None when segments
    came from YouTube captions (no local audio file). This matters for
    diarization, which requires a local audio file.
    """
    if prefer_captions:
        try:
            _status("[ytx] Trying YouTube captions ...")
            segments = captions.fetch_captions(info.video_id)
            if segments:
                _status(f"[ytx] Got {len(segments)} caption segments")
                return segments, None
            # Empty list is technically not an error, but a video with
            # no caption segments is useless. Fall through to Whisper.
            _status("[ytx] No caption segments found; falling back to Whisper")
        except captions.CaptionsUnavailableError as e:
            _status(f"[ytx] Captions unavailable: {e}")
            _status("[ytx] Falling back to Whisper")
    with progress.step_progress("Downloading audio"):
        result = audio.download_audio(url, work_dir)
    _status(f"[ytx] Audio saved to: {result.path}")

    # Enable faster-whisper's internal tqdm progress for segment-level
    # granularity. log_progress=True only takes effect when stdout is a
    # TTY (tqdm checks isatty), so it's safe to always pass True.
    with progress.step_progress(f"Transcribing with model={model!r}"):
        segments = do_transcribe(result.path, model_name=model, log_progress=True)
    return segments, result.path


def _maybe_diarize(
    *,
    segments: list[TranscriptSegment],
    audio_path: Path | None,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> list[TranscriptSegment]:
    """Run diarization and assign speakers to segments.

    If audio_path is None (segments came from captions), skip with a
    notice. If HF_TOKEN is missing, surface a clear, user-actionable
    error.
    """
    if audio_path is None:
        _status(
            "[ytx] Note: --diarize requires audio; caption-sourced "
            "transcripts can't be diarized. Skipping."
        )
        return segments

    try:
        with progress.step_progress("Running speaker diarization"):
            spans = diarize.diarize(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
    except HfTokenMissingError as e:
        # Print the user-actionable error message and exit non-zero.
        # The HfTokenMissingError message itself explains how to fix it.
        _status(f"[ytx] Error: {e}")
        raise typer.Exit(code=1) from e

    if not spans:
        _status("[ytx] Diarization found no speakers")
        return segments

    return merge.assign_speakers(segments, spans)


def _maybe_summarize(
    *,
    segments: list[TranscriptSegment],
    work_dir: Path,
    video_id: str,
    title: str,
    model: str,
    prompt_path: Path | None,
) -> None:
    """Run summarization and write a sidecar .summary.md file.

    Errors are surfaced to the user (printed to stderr, exit 1) so
    a failed summary doesn't silently leave a missing file.
    """
    try:
        custom_prompt: str | None = None
        if prompt_path is not None:
            # TOCTOU defense: the path was validated eagerly by
            # _validate_summary_prompt at argument-parsing time, but
            # the file could have been deleted between then and now.
            try:
                custom_prompt = prompt_path.read_text(encoding="utf-8")
            except FileNotFoundError as e:
                _status(f"[ytx] Error: summary prompt file disappeared: {prompt_path}")
                raise typer.Exit(code=1) from e
        with progress.step_progress(f"Generating summary with model={model!r}"):
            summary = summarize.summarize(segments, model=model, prompt=custom_prompt)
    except OllamaApiKeyMissingError as e:
        _status(f"[ytx] Error: {e}")
        raise typer.Exit(code=1) from e
    except SummarizationError as e:
        _status(f"[ytx] Error: {e}")
        raise typer.Exit(code=1) from e

    summary_path = work_dir / f"{video_id}.summary.md"
    summary_path.write_text(
        f"# Summary: {title}\n\n{summary}\n",
        encoding="utf-8",
    )
    _status(f"[ytx] Summary written to: {summary_path}")


if __name__ == "__main__":
    app()
