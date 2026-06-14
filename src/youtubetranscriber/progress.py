"""Progress bar helpers using rich.

The CLI uses a multi-step pipeline (fetch, transcribe, diarize,
summarize). For each long-running step, we want:

  - A spinner/progress bar visible in a real terminal
  - Silent (no output) in non-TTY contexts (CI, tests, scripts)
  - A descriptive label that stays the same for the step's duration

We use rich's Progress with a single SpinnerColumn + TextColumn.
In a TTY the user sees a live spinner; otherwise the context
manager is a no-op visually but still safe to use.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator


@contextlib.contextmanager
def step_progress(description: str) -> Generator[None, None, None]:
    """Context manager that shows a spinner with the given description.

    Yields None. The spinner runs for the duration of the context
    block. In non-TTY contexts (CI, tests), the rich output is
    suppressed automatically by rich.

    Usage:
        with step_progress("Downloading audio"):
            audio.download_audio(...)
    """
    # Imported here so the module-level import doesn't pull rich
    # into the unit-test cold path more than necessary.
    from rich.progress import Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description, total=None)
        yield
