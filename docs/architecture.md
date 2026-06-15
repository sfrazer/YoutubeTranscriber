# Architecture

## Pipeline

```
URL
  → fetch info (yt-dlp, no download)
  → resolve output dir (title + conflict-aware rename)
  → --prefer-captions?
     yes → try YouTube captions
       ok    → use captions
       fail  → fall through to Whisper
  → download audio (yt-dlp)
  → transcribe (faster-whisper)
  → --diarize? (pyannote, opt-in)
     yes → assign speaker labels (merge.py) to Whisper segments
     no  → skip
  → output (txt / srt / json)
  → --summarize? (Ollama cloud, opt-in)
     yes → call ollama.Client.chat() with prompt + segments
       ok    → write <video_id>.summary.md sidecar
       fail  → user-actionable error, exit 1
```

## Modules

| Module | Responsibility | Pure functions |
|---|---|---|
| `cli.py` | Typer entry point, orchestrator only | — |
| `audio.py` | Download audio with yt-dlp, parse URLs, get video info | `extract_video_id()` |
| `paths.py` | Sanitize titles, resolve unique output dirs | `sanitize_title()`, `resolve_unique_dir()` |
| `captions.py` | Pull YouTube auto/manual captions | `fetch_captions()` |
| `transcribe.py` | Run faster-whisper, return segments | `transcribe()` |
| `diarize.py` | Speaker diarization via pyannote | `diarize()` |
| `merge.py` | Segment/span overlap math; assigns speaker labels | `assign_speakers()` |
| `output.py` | Write txt/srt/json files | `write_transcript()` |
| `summarize.py` | Call Ollama cloud LLM | `format_segments_as_text()`, `summarize()` |
| `progress.py` | Rich progress bar wrapper for long steps | `step_progress()` |
| `prompts/` | Default prompt templates | — |

## Design principles

1. **Pure functions get thorough unit tests.** IO-bound functions get
   one happy-path integration test plus mocked unit tests.
2. **Exceptions are signals**, not bugs. `CaptionsUnavailable` means
   "fall through to Whisper." `HfTokenMissing` means "user must set
   `HF_TOKEN` to use `--diarize`."
3. **The CLI is a thin orchestrator.** All real logic lives in modules
   so it can be tested without invoking subprocesses.
4. **Config via flags + env vars, not config files.** Personal tool, no
   need for layered config yet.

## Data flow

The canonical internal representation is `list[TranscriptSegment]`
where each segment has `text`, `start`, `end`, and optional `speaker`.
JSON output is a direct serialization. TXT and SRT are projections.

The output directory is named after the sanitized video title (e.g.
`~/ytx-output/Me at the zoo/`). On conflict, the path is incremented
(`Me at the zoo (1)`, `Me at the zoo (2)`, ...). In interactive mode
with a TTY, the user is prompted to overwrite or rename; otherwise
the tool always renames. The transcript file inside is named
`<video_id>.<format>`.
