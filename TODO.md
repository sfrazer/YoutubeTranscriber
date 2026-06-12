# TODO

Living document of known issues, deferred decisions, and "this works but
isn't great yet." Read this first when resuming work.

## Current phase: 1 (MVP)

Phase 0 is complete and merged to main. Now building the smallest
useful tool: fetch audio → transcribe → write text file.

Order of work:
1. Add deps: `yt-dlp`, `faster-whisper`
2. `audio.py`: `extract_video_id()` with TDD (pure, full coverage)
3. `audio.py`: `download_audio()` with mocked yt-dlp tests
4. `transcribe.py`: `transcribe()` with integration test on synthetic audio
5. `output.py`: `write_transcript()` for txt format with TDD
6. Wire `cli.py` to call audio → transcribe → output
7. Manual smoke test on a real short YouTube video
8. Commit and push branch

### Phase 0 retrospective

Done:
- [x] Empty repo bootstrapped on `chore/bootstrap`
- [x] `.gitignore` with Python, uv, ffmpeg, project-specific patterns
- [x] `pyproject.toml` with `typer`, `pytest`, `ruff`
- [x] Package skeleton + CLI stub + smoke tests
- [x] `ffmpeg` installed via Homebrew
- [x] `uv` installed (was already present)
- [x] Python 3.12 pinned via uv
- [x] `uv sync` works, tests pass, CLI --help works
- [x] First commit on `chore/bootstrap`, pushed to remote
- [x] Merged to main

## Open questions deferred

- Speaker diarization prompt for summary: do we want speaker labels
  in the summary input? (Yes, by default. Confirm in phase 4.)
- Caching of downloaded audio: not in MVP, defer to phase 5.
- Auto-detection of Mac vs Mac mini: out of scope, user picks via flag.

## Known issues

- Hugging Face token storage: env var only for now (`HF_TOKEN`).
- Python 3.12 chosen over 3.14 for `pyannote.audio` compatibility.
  This means we *must* use `uv run` and not system Python.
