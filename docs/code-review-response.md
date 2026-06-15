# Code Review Response

Reviewed the feedback in `docs/code-review.md` (as of commit `96922bd`).
Going issue by issue, with my take and a concrete plan. No source files
were modified as part of writing this response — these are proposed
actions for a follow-up commit.

---

## Bugs

### 1. `InvalidURLError` / `AudioDownloadError` leak as tracebacks — **agree, will fix**

You're right, and the framing in the review is the part I want to
acknowledge: this is a *test gap masking a real bug*, not just a
stylistic complaint. `CliRunner(catch_exceptions=True)` makes
`test_invalid_url_rejected` green even though a real user would see a
traceback. That test is doing what its docstring says, but its
docstring is wrong about what "not crash" means in the real CLI.

Plan:

- Wrap the `audio.get_video_info()` call in `_run_pipeline()` with
  `try/except (InvalidURLError, AudioDownloadError)`, `typer.echo` the
  message to stderr, and `raise typer.Exit(1)`.
- Also wrap the download + transcribe block in `_obtain_segments()` for
  the same pair. (These are the two `audio.py` call sites in the CLI
  and both are user-actionable: bad URL or video unavailable.)
- Add a regression test that runs the CLI via `subprocess` (not
  `CliRunner`) with a bogus URL and asserts the exit code is 1 and the
  output does *not* contain the string `"Traceback"`. `CliRunner` masks
  this; `subprocess` doesn't. I'll keep the existing `CliRunner` test
  too — it's still useful as a unit test for the happy path.

I'd also consider doing this at a single chokepoint — e.g. a small
`_user_facing_errors` decorator on `_run_pipeline` — so future
user-actionable errors don't get forgotten the same way. Not required,
but worth a thought once there are three or four.

### 2. Dead `_VIDEO_ID_PATTERN` constant — **agree, will fix**

Honest answer: it's dead. It was an earlier attempt at the regex
before I tightened the matchers to require anchoring context
(`[?&]v=...` and `/shorts/|embed/`). The inline patterns are stricter
on purpose — a bare `[a-zA-Z0-9_-]{11}` will happily match inside
unrelated URL substrings. So the right fix is **delete it**, not wire
it up. Wiring it up would weaken the parser.

Plan: delete the constant from `audio.py:18`. While I'm there, I'll
extract the two regexes to module-level `_WATCH_ID_RE` and
`_PATH_ID_RE` named constants so the *current* implementation isn't
prone to the same "is this used?" question.

### 3. Importing from `youtube_transcript_api._errors` — **partly agree, will fix**

I checked. The library does not re-export these from its top-level
`__init__` in the version we depend on (`>=1.2.4`); they're only
importable from the `_errors` submodule. So switching to a public path
isn't an option today.

What I'll do: collapse the specific `except (NoTranscriptFound,
TranscriptsDisabled, VideoUnavailable, VideoUnplayable)` block into
the existing `except Exception` catch-all, and leave a comment that
explains the collapse. This is the option the review suggests as a
fallback, and I think it's actually the *cleanest* one here: we
genuinely don't care which specific subclass raised — they all
become "captions unavailable, fall through to Whisper." Today the
specific catch is load-bearing only as documentation; it doesn't
change behavior because everything funnels to
`CaptionsUnavailableError` either way.

That does mean we lose the *signal* that a new error type was added
upstream. To compensate I'll add a short comment near the catch-all:

```python
# youtube-transcript-api's exception hierarchy is private (_errors),
# so we collapse everything to CaptionsUnavailableError. If the
# library adds a new failure mode, it lands here too — which is the
# intended behavior (any failure = fall through to Whisper).
```

---

## Design issues

### 4. `DownloadResult` with `path=Path("")` — **agree, will fix**

This one is the "I needed something to fill the field" answer to
your question 1. I built `DownloadResult` first for `download_audio()`
and didn't want a second type when I added `get_video_info()`, so I
shoehorned it. That was the wrong call — the type system is
correctly flagging that the two return different shapes.

Plan: introduce `VideoInfo` with just `title` and `video_id`. Make
`get_video_info()` return `VideoInfo`. Keep `DownloadResult` for
`download_audio()`. Update the two callers in `cli.py` (`_run_pipeline`
and `_obtain_segments`) — they use `info.title` and `info.video_id`
only, so the change is mechanical.

The downstream `_obtain_segments` takes `info: audio.DownloadResult`
in its signature; that becomes `info: audio.VideoInfo`. Cheap fix.

### 5. `summarize` parameter shadows `summarize` module — **agree, will fix**

You're right, and I noticed this and left the `# noqa: A002` comment
intending to "fix it later." That comment is also part of issue #9 —
it's a no-op because A002 isn't in the selected rule set. Two birds.

Plan: rename the parameter to `summarize_flag`, matching the existing
`diarize_flag` naming. Update `_run_pipeline`'s parameter, the call
site in `transcribe()`, and the two references inside `_run_pipeline`
(`if summarize_flag:` and the `_maybe_summarize` call). Mechanical but
touches a few places.

The `from youtubetranscriber import ... summarize` import stays as-is;
the module import is what `_maybe_summarize` needs.

### 6. `lru_cache` on env-var-dependent functions — **partial agree**

For the CLI use case as it exists today, this is fine. A CLI process
is short-lived, `os.environ` doesn't change after the first call, and
the clients/pipelines are genuinely expensive to reconstruct (the
pyannote download alone is hundreds of MB). Caching is the right call.

But I take your point that the *behavior* is surprising and
*test isolation* is fragile if test ordering shifts. The
`test_summarize_raises_on_missing_api_key` test happens to work
today because `monkeypatch.delenv` runs before any client is cached,
but that's a load-bearing assumption I haven't documented.

Plan: add a one-line comment above each `@lru_cache` noting the
assumption:

```python
# NOTE: caches the client at first call. For the CLI use case this is
# fine (process is short-lived, env doesn't change). If this is ever
# used as a library, drop the cache or key it on os.environ values.
```

I considered removing the cache and accepting the cost, but the
pyannote pipeline in particular is *very* slow to re-instantiate.
Not worth it for a hypothetical future use case. Documenting the
assumption is.

I won't add env-var-keyed caching — that's complexity for a
non-existent problem.

---

## Minor issues

### 7. Architecture doc has wrong function names — **agree, will fix**

Good catch. The table is wrong on two rows: `diarize.py` is listed as
having `assign_speakers()` (it has `diarize()`), and `merge.py` is
listed as having `assign_speakers()` as "a helper" (it's the actual
public function, not a helper). Plan: fix the table to match the
code, and add `diarize()` to the diarize row.

### 8. `_next_indexed_path` has no upper bound — **agree, will fix**

Agreed on principle. The "astronomically unlikely" framing is correct
— you'd need 10,000 sibling directories all named `Title (N)` — but
"silently hangs" is a worse failure mode than "loudly errors." Plan:
add a cap (10,000 is fine), raise `FileExistsError` with a message
explaining what happened. No test needed; the function's existing
tests cover the normal path and 10,000 siblings isn't a realistic
test scenario worth a CI minute.

### 9. Dead `per-file-ignores` for A002/ARG — **agree, will fix**

You're right, this is dead config. The `A` and `ARG` rule families
aren't in the global `select` list, so the ignores suppress nothing.
Plan: remove the two entries from `per-file-ignores`. Also remove
the comment that references them, since it was the only justification.
After this fix the `per-file-ignores` table will be empty; I'll
delete the key entirely rather than leave an empty `{}` behind.

Note: this is also where the `format` parameter shadowing `format`
(the builtin) lives. After issue #5 lands (renaming `summarize` to
`summarize_flag`), the only remaining builtin-shadowing name in cli.py
is `format`. I have two options: (a) leave it — `format` is a
ubiquitous CLI flag name and renaming it would hurt UX more than the
shadowing hurts readability; (b) add the `A` rules to the select list
so the existing shadowing is actually checked. I'll go with (a) —
`format` as a CLI flag is the kind of convention that's worth more
than the lint signal. I'll add a comment in the config explaining
this explicit choice.

### 10. Status messages on stdout — **defer with rationale**

I read this one carefully and I'm going to push back, but
respectfully — let me know if you still disagree after my reasoning.

The review says: "If the design intent is that stdout is only for
human-readable status, that's fine, but it should be documented or
the messages should move to stderr."

The current design intent is exactly that: stdout is for human-
readable status when run interactively, and the actual transcript
output is *always* written to a file (the `-o` directory), never to
stdout. There's no code path that writes transcript content to
stdout. So `ytx URL > file.txt` produces a file full of `[ytx]`
status lines and no transcript — which is, arguably, correct
because the user didn't ask for stdout output.

That said, your underlying point is right: nothing in the code or
docs tells the user this. A user who tries to capture the transcript
by redirecting stdout will get a confusing result. Two cheap fixes:

- Document the behavior in `--help` (one line in the `transcribe`
  command docstring) and in the README. This is the minimum.
- If you want the belt-and-suspenders version: route all `[ytx]`
  messages through `typer.echo(..., err=True)`. This costs nothing
  for TTY use and makes `ytx URL > file.txt` produce an empty file,
  which is at least *less confusing* than a file full of status
  noise.

I lean toward doing the doc fix now and leaving the stderr change
as a follow-up if anyone actually trips on it. But I'm not opposed to
the stderr change; it's a one-line per-message edit.

---

## Answers to the explicit questions

### Q1. `get_video_info` returning `DownloadResult` — was this a plan or a sentinel?

Sentinel, not a plan. There was no intent to fold metadata + audio
into a single type — I just didn't want a second dataclass for two
fields. After your pushback I agree the type system is doing its job
here, and the right fix is the `VideoInfo` split from issue #4. Going
to do that.

### Q2. `_VIDEO_ID_PATTERN` — earlier implementation or intended shared reference?

Earlier implementation that I forgot to delete. The inline patterns
are deliberately stricter (they require the `[?&]v=` prefix and the
`/shorts/|embed/` path context) and can't be replaced by a shared
`[a-zA-Z0-9_-]{11}` without weakening the parser. The right fix is
deletion, not wiring — same conclusion as the review. (See issue #2.)

---

## Summary of planned changes

| # | Issue | Action |
|---|---|---|
| 1 | Tracebacks leak to user | Wrap + add subprocess test |
| 2 | Dead `_VIDEO_ID_PATTERN` | Delete; extract current regexes to named constants |
| 3 | Private `_errors` import | Collapse into single `except Exception` with comment |
| 4 | `DownloadResult` with empty `path` | Split into `VideoInfo` and `DownloadResult` |
| 5 | `summarize` parameter shadow | Rename to `summarize_flag` |
| 6 | `lru_cache` + env vars | Add comment documenting the assumption |
| 7 | Wrong function names in architecture doc | Fix the table |
| 8 | Unbounded `_next_indexed_path` | Add 10,000-iter cap, raise on overflow |
| 9 | Dead `per-file-ignores` | Remove the entries; remove the key |
| 10 | Status messages on stdout | Document behavior in `--help` and README; defer stderr change |

Bug fixes (1–3, 5, 8) and the cheap design fix (4) all going in one
commit. The doc fixes (7, 10-doc) in a separate commit so the diff
isn't noisy. Issue 6 (comment-only) and issue 9 (config cleanup) can
ride along with #5 since they touch the same files.
