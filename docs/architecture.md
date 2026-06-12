# Architecture

## Pipeline

```
URL
  → fetch audio (yt-dlp)
  → captions? (YouTubeTranscriptApi)
  → whisper (faster-whisper)
  → diarize? (pyannote, opt-in)
  → output (txt / srt / json)
  → summarize? (Ollama cloud, opt-in)
```

## Modules

| Module | Responsibility | Pure functions |
|---|---|---|
| `cli.py` | Typer entry point, orchestrator only | — |
| `audio.py` | Download audio with yt-dlp, parse URLs | `extract_video_id()` |
| `captions.py` | Pull YouTube auto/manual captions | `fetch_captions()` |
| `transcribe.py` | Run faster-whisper, return segments | `transcribe()` |
| `diarize.py` | Speaker diarization + assignment | `assign_speakers()` |
| `merge.py` | Segment/span overlap math | `assign_speakers()` helper |
| `output.py` | Write txt/srt/json files | `write_transcript()` |
| `summarize.py` | Call Ollama cloud LLM | `summarize()` |
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
