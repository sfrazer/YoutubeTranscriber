# TODO

Living document of known issues, deferred decisions, and "this works but
isn't great yet." Read this first when resuming work.

## Current phase: 3 (speaker diarization, opt-in)

The hardest phase. Add `--diarize` flag that runs pyannote.audio
in parallel with Whisper and assigns speaker labels to each
transcript segment. Captions are NOT diarized (no audio-derived
segments to work with).

Order of work:
  1. merge.py: pure function `assign_speakers()` with full TDD
     (handles overlap, multi-speaker, no-overlap edge cases)
  2. diarize.py: pyannote.audio wrapper
     - lazy model load (don't download on first import)
     - clear errors for missing/invalid HF_TOKEN
     - returns SpeakerSpan list
  3. cli.py: --diarize flag, --num-speakers flag (default auto)
     - friendly errors when HF_TOKEN missing
     - integration with existing pipeline
  4. Real-video smoke test on a multi-speaker video
  5. Commit and push branch

Key dependencies to add:
  - pyannote.audio (heavy: ~500MB with torch/torchaudio)
  - Note: requires user to have HF_TOKEN set and to have accepted
    the model EULA at huggingface.co/pyannote/speaker-diarization-3.1

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
