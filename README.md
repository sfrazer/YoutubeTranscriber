# YoutubeTranscriber

A CLI tool for transcribing (and optionally summarizing) YouTube videos
locally with Whisper, with optional speaker diarization and LLM
summarization via Ollama cloud.

## What it does

Given a YouTube URL, `ytx` will:

1. Download the audio with `yt-dlp` (m4a, ~10 MB for an 11-min video)
2. Transcribe with `faster-whisper` (any model size, CPU or GPU)
3. Optionally identify speakers with `pyannote.audio` (`--diarize`)
4. Optionally summarize with the Ollama cloud API (`--summarize`)
5. Write a transcript (txt/srt/json) and sidecar summary.md

You get live progress bars in a real terminal; nothing is printed in
non-TTY contexts (safe for scripts and CI).

## Quick start

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.12 and sync dependencies
uv python install 3.12
uv sync

# Transcribe a YouTube video
uv run ytx "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Output goes to `~/ytx-output/<video title>/<video id>.txt` by default.
A second run on the same video is auto-renamed to `~/ytx-output/<video title> (1)/`
so you don't lose old transcripts.

## Requirements

- macOS or Linux
- `ffmpeg` (`brew install ffmpeg` on macOS)
- Python 3.12 (managed by `uv`)

Optional:
- An Apple Silicon Mac for best CPU performance (Intel Macs work but slower)
- A CUDA GPU for fastest transcription (not tested on this project)

## Usage

```bash
# Basic transcript (txt, Whisper medium)
uv run ytx "https://youtu.be/..."

# Faster model
uv run ytx "URL" --model small

# Higher quality
uv run ytx "URL" --model large-v3

# SRT subtitle output
uv run ytx "URL" --format srt

# JSON output (includes video_id, title, segments with timestamps)
uv run ytx "URL" --format json

# Transcribe a local audio file instead of a YouTube URL
uv run ytx --audio-file recording.m4a

# Separate two speakers in a local recording
uv run ytx --audio-file interview.m4a --diarize --num-speakers 2

# Use YouTube's auto-captions instead of Whisper (fast, less accurate)
uv run ytx "URL" --prefer-captions

# Identify speakers (requires HF_TOKEN — see "Optional features" below)
uv run ytx "URL" --diarize

# Generate a structured summary (requires OLLAMA_API_KEY)
uv run ytx "URL" --summarize

# Full pipeline: diarize + summarize, output as JSON
uv run ytx "URL" --diarize --summarize --format json
```

### All flags

```
--audio-file          Transcribe a local audio file (m4a, wav, mp3, ...)
                      instead of a YouTube URL. Skips yt-dlp; the file is
                      never deleted. Omit the URL when using this.
--model, -m           Whisper model size (tiny, base, small, medium, large-v3)
                      Default: medium
--format, -f          Output format: txt, srt, json
                      Default: txt
--output-dir, -o      Output directory
                      Default: ~/ytx-output
--interactive, -i     Prompt before overwriting an existing output directory
--verbose, -v         Enable verbose logging
--keep-audio          Keep the .m4a file after transcription (default: delete it)

--prefer-captions     Try YouTube's auto-captions first; fall back to Whisper

--diarize             Identify speakers (requires HF_TOKEN)
--num-speakers N      Hint the speaker count to pyannote (default: auto)
--min-speakers N      Minimum number of speakers
--max-speakers N      Maximum number of speakers

--summarize           Generate a summary using Ollama cloud
                      (requires OLLAMA_API_KEY)
--summary-model NAME  Ollama model (default: gpt-oss:120b)
--summary-prompt PATH Use a custom prompt template file
                      (must contain {transcript})
```

## Optional features

### Speaker diarization (`--diarize`)

Identifies who said what in multi-speaker videos (interviews, panel
discussions, etc.). Requires:

1. A free Hugging Face account at <https://huggingface.co/join>
2. Accept the EULA at **all three** gated pyannote repos:
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
   - <https://huggingface.co/pyannote/speaker-diarization-community-1>
3. An access token at <https://huggingface.co/settings/tokens>
4. `HF_TOKEN` set in your environment

If anything is missing, the tool produces a clear, actionable error
pointing to the exact step that needs to be done.

### Summarization (`--summarize`)

Calls the Ollama cloud API to generate a structured summary of the
transcript (TL;DR, key claims, technical concepts, open questions,
notable quotes). Requires:

1. An Ollama account at <https://ollama.com>
2. An API key at <https://ollama.com/settings/keys>
3. `OLLAMA_API_KEY` set in your environment

Default model: `gpt-oss:120b`. Override with `--summary-model` to use
a different cloud model (e.g. `gpt-oss:20b` for faster, smaller
summaries; `deepseek-v4-flash`, etc.).

The default prompt lives at
`src/youtubetranscriber/prompts/default_summary.txt`. Pass
`--summary-prompt PATH` to use your own (must contain `{transcript}`).

## Output examples

### `Me at the zoo/.txt` (the first YouTube video, 19 seconds)

```
All right, so here we are, in front of the
elephants
the cool thing about these guys is that they
have really...
really really long trunks
and that's cool
(baaaaaaaaaaahhh!!)
and that's pretty much all there is to
say
```

### `.srt` (same content, with timestamps)

```
1
00:00:01,200 --> 00:00:03,360
All right, so here we are, in front of the
elephants

2
00:00:05,318 --> 00:00:07,974
the cool thing about these guys is that they
have really...
```

### `.json` (with metadata)

```json
{
  "video_id": "jNQXAC9IVRw",
  "title": "Me at the zoo",
  "segments": [
    {
      "text": "All right, so here we are, in front of the\nelephants",
      "start": 1.2,
      "end": 3.36,
      "speaker": null
    }
  ]
}
```

### With `--diarize` (multi-speaker video)

```
[SPEAKER_01] This is it. Welcome to Dub Dub Daily, the final day of WWDC26.
[SPEAKER_01] What a week it has been.
[SPEAKER_00] Holly Borla, an engineering manager on the Swift team.
[SPEAKER_00] A lot's going on for you this year.
[SPEAKER_02] Yeah. To me, that means two things.
```

## Performance notes

On Apple Silicon, Whisper currently runs on CPU (CTranslate2 4+
dropped MPS support for Whisper). Expect roughly 0.5-1x realtime for
`medium` on a 16GB M-series Mac. For a 10-min video, that's
5-20 minutes of transcription time.

Use `--model small` or `--model base` for much faster runs at the
cost of accuracy. `tiny` is good for quick previews.

The Ollama cloud summary is typically 5-30 seconds regardless of
transcript length.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the module
layout, pipeline diagram, and design principles.

For `uv` setup, see [`docs/uv.md`](docs/uv.md).

## License

[MIT](LICENSE) © 2026 Scott Frazer
