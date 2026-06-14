# TODO

Living document of known issues, deferred decisions, and "this works but
isn't great yet." Read this first when resuming work.

## Current phase: 4 (summarization via Ollama cloud)

Adds `--summarize` flag that calls the Ollama cloud API to
summarize a transcript after writing it.

Order of work:
  1. prompts/default_summary.txt: default technical-talk summary prompt
  2. summarize.py: wraps ollama.Client for cloud chat
     - pure: segments-to-prompt-text formatting (TDD)
     - ollama.Client wrapper with clear errors for missing
       OLLAMA_API_KEY, missing model, network errors
  3. CLI: --summarize / --summary-model / --summary-prompt flags
     - writes <video_id>.summary.md next to the transcript
     - speaker-aware formatting when diarization ran
  4. Smoke test on a real transcript

Key dependency: ollama (pip). OLLAMA_API_KEY must be set in env
to authenticate. Default model: gpt-oss:120b. Cloud endpoint:
https://ollama.com. The user is on a Pro subscription with
usage-based limits, so calls are fine to make.

## Phase 3 status (DONE, merged as PR #5)

Phase 3 complete:
  - merge.py: pure function for assigning speakers to segments
  - diarize.py: pyannote.audio wrapper with HF_TOKEN handling
    and lazy model load
  - --diarize / --num-speakers / --min-speakers / --max-speakers
  - Captions skip diarization with a clear notice
  - HF_TOKEN missing: user-actionable error, exit 1
  - Smoke tested on real videos (solo + 3-speaker)
  - 127 tests passing (was 99 before phase 3)

## Phase 2 status (DONE, merged as PR #4)

Phase 2 complete:
  - SRT writer (HH:MM:SS,mmm timestamps, sequential indices)
  - JSON writer (canonical structured form with video_id, title, segments)
  - Captions module wrapping youtube-transcript-api 1.2.4
  - --prefer-captions flag with auto-fallback to Whisper on
    CaptionsUnavailableError or empty caption list
  - 99 tests passing (was 65 before phase 2)

## Phase 1 status

Phase 1 MVP is complete and verified. The tool can:
  - Parse YouTube URLs (8 URL shapes supported)
  - Download audio via yt-dlp (m4a, ffmpeg postprocess)
  - Transcribe with faster-whisper (any model, auto device detect
    with CPU fallback for unsupported devices like MPS in CTranslate2 4+)
  - Write a clean .txt file with one segment per line
  - Run end-to-end via 'uv run ytx URL'
  - Output dir is named after the sanitized video title
  - Conflict-aware auto-rename on duplicate runs
  - --interactive flag for TTY prompt

Merged: PR #1 (phase-1-mvp), PR #2 (title-based dirs), PR #3 (uv doc cleanup).

## Title-based output dirs (this fix branch)

Changes:
  - New `paths.py` module with `sanitize_title()` and
    `resolve_unique_dir()`
  - `audio.py` now returns `DownloadResult(path, title, video_id)`
    and exposes `get_video_info()` for the title-fetch pass
  - `cli.py` uses title for the dir name; prompts in TTY mode,
    auto-renames otherwise
  - Transcript file inside is still named `<video_id>.<format>`

Conflict resolution: 'Sample Video' -> 'Sample Video (1)' -> 'Sample
Video (2)' etc. Default to rename, 'o'/'overwrite' in interactive
mode to overwrite.

Next: phase 2 (multiple output formats + caption fallback).

### Phase 1 backlog

- [x] Add deps: yt-dlp, faster-whisper
- [x] `audio.py`: `extract_video_id()` with TDD
- [x] `audio.py`: `download_audio()` with mocked yt-dlp tests
- [x] `transcribe.py`: `transcribe()` with mocked + slow integration tests
- [x] `output.py`: `write_transcript()` for txt format with TDD
- [x] Wire `cli.py` end-to-end
- [x] Manual smoke test on a real short YouTube video
- [x] Commit and push branch

## Open questions deferred

- Speaker diarization prompt for summary: do we want speaker labels
  in the summary input? (Yes, by default. Confirm in phase 4.)
- Caching of downloaded audio: not in MVP, defer to phase 5.
- Auto-detection of Mac vs Mac mini: out of scope, user picks via flag.

## Known issues

- Hugging Face token storage: env var only for now (`HF_TOKEN`).
- Python 3.12 chosen over 3.14 for `pyannote.audio` compatibility.
  This means we *must* use `uv run` and not system Python.
- CTranslate2 4+ dropped MPS support for Whisper. `transcribe()`
  auto-falls-back to CPU and emits a UserWarning. Performance on
  Apple Silicon is therefore ~CPU-only. If we need GPU acceleration
  in the future, options are: (a) downgrade to CTranslate2 3.x,
  (b) wait for upstream MPS support, (c) use a non-Apple GPU.
- Whisper on CPU is slow for the medium/large models on long videos.
  For phase 1 manual testing, use 'tiny' or 'base'.
