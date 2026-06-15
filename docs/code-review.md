# Code Review

Reviewed as of commit `96922bd`. All 155 tests pass; ruff is clean.
Overall the code is well-structured with clear module boundaries and good test coverage. Issues below are grouped by severity.

---

## Bugs

### 1. `InvalidURLError` and `AudioDownloadError` reach the user as Python tracebacks

**File:** `cli.py`, `_run_pipeline()`

Neither `InvalidURLError` nor `AudioDownloadError` is caught in the CLI pipeline. In real terminal use (not the test runner), Typer/Click in standalone mode only catches `ClickException` subclasses — raw `ValueError`/`RuntimeError` subclasses propagate to the top level and print a full Python traceback. The user sees:

```
Traceback (most recent call last):
  ...
youtubetranscriber.audio.InvalidURLError: Not a YouTube URL: 'not a url'
```

The test `test_invalid_url_rejected` passes because Typer's `CliRunner` has `catch_exceptions=True` by default, masking this. The test's docstring even says "not crash" — but it does crash in real use.

**Fix:** In `_run_pipeline`, wrap the call to `audio.get_video_info()` (and the download+transcribe block) with `except (InvalidURLError, AudioDownloadError) as e: typer.echo(f"[ytx] Error: {e}", err=True); raise typer.Exit(1)`.

---

### 2. Dead module-level constant in `audio.py`

**File:** `audio.py:18`

```python
_VIDEO_ID_PATTERN = re.compile(r"[a-zA-Z0-9_-]{11}")
```

This pattern is defined at module level but never referenced. The actual matching in `extract_video_id()` uses two separate inline patterns (lines 73 and 80). Ruff doesn't flag unused module-level names, so it slips through. It's misleading — a reader might think it's used somewhere.

**Fix:** Delete the constant, or replace the inline patterns with references to it (ensuring the semantics actually match first — the inline patterns are more specific, adding anchoring context).

---

### 3. Importing from a private module (`youtube_transcript_api._errors`)

**File:** `captions.py:12–17`

```python
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
)
```

The leading underscore marks `_errors` as a library-internal module with no stability guarantee. A minor version bump of `youtube-transcript-api` could rename or move these classes and break this silently (the broad `except Exception` catch-all on line 66 would then silently swallow new error types). The library may expose these publicly elsewhere.

**Fix:** Check whether the library exports these from its top-level `__init__`. If not, either import from the public module if one exists, or collapse the specific catches into the `except Exception` fallback and document the intent (we're converting all failures into `CaptionsUnavailableError` anyway).

---

## Design Issues

### 4. `DownloadResult` returned with a dummy `path` from `get_video_info()`

**File:** `audio.py:113`

```python
return DownloadResult(path=Path(""), title=title, video_id=return_id)
```

`DownloadResult` has a `path` field that represents the downloaded audio file, but `get_video_info()` returns one with `path=Path("")` — a path that doesn't exist and can't be used. The two callers in `cli.py` never use `info.path`, so it doesn't cause a bug today. But the type says "this has a path" and that's false.

**Fix:** Introduce a separate `VideoInfo` dataclass with only `title` and `video_id`. `DownloadResult` keeps its `path`. This makes the distinction explicit and lets the type checker enforce correct usage.

---

### 5. `summarize` parameter name shadows the `summarize` module in `cli.py`

**File:** `cli.py:94` and `cli.py:21`

```python
from youtubetranscriber import ... summarize   # module
...
summarize: bool = typer.Option(...)            # parameter in transcribe()
```

Inside `transcribe()` and `_run_pipeline()`, `summarize` is the boolean flag. The module is only reachable from `_maybe_summarize()`, which is a separate function so it works — but a reader scanning `_run_pipeline` sees `if summarize:` and has to remember that `summarize` is the bool, not the module. This naming collision has already been called out by the `# noqa: A002` comment in the ruff config (though A-rules aren't in the selected set, so it's a no-op annotation).

**Fix:** Rename the parameter to `do_summarize` or `summarize_flag` (matching the pattern used for `diarize_flag`).

---

### 6. `lru_cache` on env-var-dependent functions bakes in the env at first call

**Files:** `summarize.py:81`, `diarize.py:39`

Both `_get_client()` and `_load_pipeline()` are decorated with `@lru_cache(maxsize=1)` but read from `os.environ` at call time. After the first successful call, the cached object is returned regardless of env changes. This is fine for a short-lived CLI process, but surprising if this code is ever used as a library or in tests that set/unset env vars without process isolation. The `test_summarize_raises_on_missing_api_key` test uses `monkeypatch.delenv` correctly, but the test could interact with cached state from a prior test if ordering changes.

**Fix:** For the CLI use case this is acceptable — add a comment noting the assumption. For robustness, consider making the cache key include the env var value, or simply not caching (the clients are cheap to reconstruct).

---

## Minor Issues

### 7. Architecture doc has wrong function names in the module table

**File:** `docs/architecture.md`, module table

The `diarize.py` row lists `assign_speakers()` as a pure function — but `assign_speakers()` lives in `merge.py`, not `diarize.py`. `diarize.py`'s actual public function is `diarize()`.

---

### 8. `_next_indexed_path` has no upper bound

**File:** `paths.py:63–71`

The `while True` loop could theoretically run forever if every `Name (N)` slot is taken. Astronomically unlikely, but a safety cap (e.g., raise after 10,000 iterations) would convert a silent infinite loop into a loud, diagnosable error.

---

### 9. `per-file-ignores` for `A002` and `ARG` in `pyproject.toml` are a no-op

**File:** `pyproject.toml:53`

```toml
per-file-ignores = { "src/youtubetranscriber/cli.py" = ["ARG", "A002"] }
```

Neither the `A` (flake8-builtins) nor `ARG` (flake8-unused-arguments) rule prefixes are in the global `select` list, so these ignores suppress nothing. The comment references "forward-compat flags" but the actual issue is the `format` parameter shadowing the builtin (an A002 concern), which ruff currently won't flag anyway. This is just dead config that implies more lint coverage than exists.

---

### 10. Status messages on stdout contaminate piped output

**File:** `cli.py` throughout

All `[ytx]` status messages (`Title:`, `Got N segments`, `Transcript written to:`) go to stdout. If a caller pipes the output (`ytx URL > file.txt`), these status lines end up in the file alongside the transcript content (in the non-redirect case the transcript goes to a separate file, so this is only an issue if someone tries to use stdout). If the design intent is that stdout is only for human-readable status, that's fine, but it should be documented or the messages should move to stderr.

---

## Questions for the author

Before filing these as fixes I want to understand your intent on two things:

1. **`get_video_info` returning `DownloadResult`** (issue #4): Was there a plan to fold metadata + audio into a single result type later, or is the `path=Path("")` sentinel just a "I needed something to fill the field"?

2. **The `_VIDEO_ID_PATTERN` constant** (issue #2): Was this an earlier implementation that got replaced by the inline patterns, or is it intended as a shared reference for the patterns below it (and you just forgot to wire it up)?
