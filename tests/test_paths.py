"""Tests for paths.py: sanitize_title and resolve_unique_dir.

The conflict-resolution logic is the interesting part: we need to
test both the "prompt" and "auto-rename" paths, and the
edge cases (empty title, pure-punctuation title, very long title,
existing files vs existing directories, etc.).
"""

from __future__ import annotations

from pathlib import Path

from youtubetranscriber.paths import (
    MAX_TITLE_LENGTH,
    resolve_unique_dir,
    sanitize_title,
)

# --- sanitize_title --------------------------------------------------------


def test_sanitize_title_passthrough_clean() -> None:
    assert sanitize_title("Sample Video") == "Sample Video"


def test_sanitize_title_replaces_slashes() -> None:
    """Forward slashes would create subdirectories we don't want."""
    assert sanitize_title("a/b/c") == "a_b_c"


def test_sanitize_title_replaces_colons_and_question_marks() -> None:
    """These are problematic on Windows; treat as unsafe everywhere.

    Trailing underscores from replacement get stripped by the edge-strip
    pass, which is desirable (trailing underscores are confusing).
    """
    assert sanitize_title("Hello: World?") == "Hello_ World"


def test_sanitize_title_replaces_control_characters() -> None:
    """Newlines, tabs, null bytes should all become underscores."""
    assert sanitize_title("a\nb\tc\0d") == "a_b_c_d"


def test_sanitize_title_preserves_unicode() -> None:
    """Non-ASCII letters are valid filenames; keep them."""
    assert sanitize_title("Café résumé 日本語") == "Café résumé 日本語"


def test_sanitize_title_collapses_runs() -> None:
    """Multiple unsafe chars in a row should not create runs of underscores."""
    assert sanitize_title("a///b") == "a_b"
    assert sanitize_title("a ?/b") == "a _b"


def test_sanitize_title_strips_dangerous_edges() -> None:
    """Leading dots create hidden files; trailing dots/spaces confuse Windows."""
    assert sanitize_title("...hello...") == "hello"
    assert sanitize_title("   spaced   ") == "spaced"


def test_sanitize_title_truncates_long_titles() -> None:
    """Cap at MAX_TITLE_LENGTH chars to avoid OS path limits."""
    long = "a" * 500
    result = sanitize_title(long)
    assert len(result) == MAX_TITLE_LENGTH


def test_sanitize_title_empty_falls_back_to_placeholder() -> None:
    """All-punctuation titles should not produce empty strings."""
    assert sanitize_title("???") == "untitled"
    assert sanitize_title("   ") == "untitled"
    assert sanitize_title("///") == "untitled"


def test_sanitize_title_preserves_emoji() -> None:
    """Emoji are valid in modern filesystems; keep them."""
    assert sanitize_title("Cool video 🎬") == "Cool video 🎬"


# --- resolve_unique_dir ----------------------------------------------------


def test_resolve_unique_dir_no_conflict(tmp_path: Path) -> None:
    """If the target doesn't exist, return it unchanged."""
    target = tmp_path / "Sample Video"
    assert resolve_unique_dir(tmp_path, "Sample Video", interactive=False) == target


def test_resolve_unique_dir_increments_on_conflict(tmp_path: Path) -> None:
    """If the target exists, append (1), (2), etc."""
    (tmp_path / "Sample Video").mkdir()
    result = resolve_unique_dir(tmp_path, "Sample Video", interactive=False)
    assert result == tmp_path / "Sample Video (1)"


def test_resolve_unique_dir_skips_existing_indices(tmp_path: Path) -> None:
    """If 'Sample Video (1)' also exists, jump to (2)."""
    (tmp_path / "Sample Video").mkdir()
    (tmp_path / "Sample Video (1)").mkdir()
    result = resolve_unique_dir(tmp_path, "Sample Video", interactive=False)
    assert result == tmp_path / "Sample Video (2)"


def test_resolve_unique_dir_uses_file_in_conflict_check(tmp_path: Path) -> None:
    """A *file* (not a directory) with the same name should also trigger rename."""
    (tmp_path / "Sample Video").touch()
    result = resolve_unique_dir(tmp_path, "Sample Video", interactive=False)
    assert result == tmp_path / "Sample Video (1)"


def test_resolve_unique_dir_handles_many_conflicts(tmp_path: Path) -> None:
    """Even if 0..99 all exist, we should find a free number."""
    for i in range(100):
        if i == 0:
            (tmp_path / "Sample Video").mkdir()
        else:
            (tmp_path / f"Sample Video ({i})").mkdir()
    result = resolve_unique_dir(tmp_path, "Sample Video", interactive=False)
    assert result == tmp_path / "Sample Video (100)"


def test_resolve_unique_dir_prompts_in_interactive_mode(tmp_path: Path, monkeypatch) -> None:
    """In interactive mode and conflict, prompt the user."""
    (tmp_path / "Sample Video").mkdir()

    # User types 'r' to rename
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "r")

    result = resolve_unique_dir(tmp_path, "Sample Video", interactive=True)
    assert result == tmp_path / "Sample Video (1)"


def test_resolve_unique_dir_overwrites_in_interactive_mode(tmp_path: Path, monkeypatch) -> None:
    """In interactive mode, user can choose to overwrite."""
    existing = tmp_path / "Sample Video"
    existing.mkdir()

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "o")

    result = resolve_unique_dir(tmp_path, "Sample Video", interactive=True)
    assert result == existing


def test_resolve_unique_dir_interactive_default_is_rename(tmp_path: Path, monkeypatch) -> None:
    """Pressing enter (empty input) should default to rename, not overwrite."""
    (tmp_path / "Sample Video").mkdir()

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "")

    result = resolve_unique_dir(tmp_path, "Sample Video", interactive=True)
    assert result == tmp_path / "Sample Video (1)"


def test_resolve_unique_dir_non_tty_never_prompts(tmp_path: Path, monkeypatch) -> None:
    """Even with interactive=True, no TTY means no prompt — just auto-rename."""
    (tmp_path / "Sample Video").mkdir()

    # Stdin is not a TTY
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    # input() would raise EOFError if called, so this test fails if we prompt
    result = resolve_unique_dir(tmp_path, "Sample Video", interactive=True)
    assert result == tmp_path / "Sample Video (1)"
