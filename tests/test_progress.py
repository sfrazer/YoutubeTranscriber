"""Tests for progress.py: rich progress bar wrapper.

The actual progress bar is hard to test (TTY-dependent), so we
mostly verify that the context manager is a no-op in non-TTY
contexts (which is what the test runner uses) and that it
doesn't crash on edge cases.
"""

from __future__ import annotations

from youtubetranscriber.progress import step_progress


def test_step_progress_runs_without_error() -> None:
    """The context manager should yield and complete without errors."""
    with step_progress("Test step"):
        pass


def test_step_progress_runs_with_work_inside() -> None:
    """Code inside the context manager executes normally."""
    result = []
    with step_progress("Working"):
        result.append("did work")
    assert result == ["did work"]


def test_step_progress_handles_exceptions() -> None:
    """If code inside raises, the context manager still cleans up.

    We catch the exception outside the with block to verify it
    propagates normally, and the context manager's __exit__
    should not interfere.
    """
    raised = False
    try:
        with step_progress("Failing step"):
            raise ValueError("test error")
    except ValueError:
        raised = True
    assert raised


def test_step_progress_nested() -> None:
    """Nested context managers (one for each step in a pipeline)
    should each get their own progress display.
    """
    steps = []
    with step_progress("Step 1"):
        steps.append("1")
        with step_progress("Step 2"):
            steps.append("2")
    assert steps == ["1", "2"]
