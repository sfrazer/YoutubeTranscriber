# TODO

Living document of known issues, deferred decisions, and "this works but
isn't great yet." Read this first when resuming work.

## Current phase: 1 (MVP)

Phase 1 MVP is complete and verified. The tool can:
  - Parse YouTube URLs (8 URL shapes supported)
  - Download audio via yt-dlp (m4a, ffmpeg postprocess)
  - Transcribe with faster-whisper (any model, auto device detect
    with CPU fallback for unsupported devices like MPS in CTranslate2 4+)
  - Write a clean .txt file with one segment per line
  - Run end-to-end via 'uv run ytx URL'

Verified: transcribed 'Me at the zoo' (jNQXAC9IVRw) with tiny model,
got the famous first words on YouTube correctly.

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
