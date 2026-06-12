# YoutubeTranscriber

A CLI tool for transcribing (and optionally summarizing) YouTube videos.

## Status

Early development. See `TODO.md` for the current state of the work.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — module layout, data flow, design principles
- [`docs/uv.md`](docs/uv.md) — `uv` primer for newcomers

## Quick start

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.12 and sync dependencies
uv python install 3.12
uv sync

# Run the tool
uv run ytx "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

## Requirements

- macOS (Apple Silicon recommended for `faster-whisper` performance)
- `ffmpeg` (`brew install ffmpeg`)
- Python 3.12 (managed by `uv`)

## Configuration

For speaker diarization, set `HF_TOKEN` in your environment. Get a free token
at <https://huggingface.co/settings/tokens> and accept the pyannote model EULA
at <https://huggingface.co/pyannote/speaker-diarization-3.1>.

For Ollama cloud summarization, configure credentials per the
[Ollama Python client docs](https://github.com/ollama/ollama-python).

## License

TBD
