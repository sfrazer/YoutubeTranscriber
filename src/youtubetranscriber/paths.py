"""Filesystem path helpers: title sanitization and conflict resolution.

Pure functions, no side effects beyond reading the filesystem (to detect
conflicts) and potentially creating directories.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Maximum length for a sanitized title. Most filesystems support 255 chars
# in a single path component; we leave headroom for any future suffix.
MAX_TITLE_LENGTH = 200

# Characters that are unsafe in filenames on at least one major OS.
# We replace these with underscores rather than stripping them, so
# "AC/DC" doesn't become "ACDC" (which would be confusing).
# Includes: / \ : * ? " < > | and control characters (0x00-0x1F, 0x7F).
_UNSAFE_CHARS = re.compile(r'[/\\:*?"<>|\x00-\x1f\x7f]')

# Multiple underscores in a row should collapse to one.
_MULTI_UNDERSCORE = re.compile(r"_+")

# Leading/trailing characters that confuse shells, Windows, or that
# would create hidden files on Unix.
_EDGE_CHARS = re.compile(r"^[\s._]+|[\s._]+$")

# Placeholder used when a title sanitizes to an empty string.
_UNTITLED = "untitled"


def sanitize_title(title: str) -> str:
    """Make a YouTube video title safe to use as a directory name.

    Transformations:
        - Replace unsafe characters (/\\:*?"<>| and control chars) with _
        - Collapse runs of _ to a single _
        - Strip leading/trailing whitespace, dots, underscores
        - Truncate to MAX_TITLE_LENGTH characters
        - If nothing remains, return "untitled"

    Unicode (emoji, accented letters, CJK) is preserved as-is.
    """
    if not isinstance(title, str):
        return _UNTITLED

    # Replace unsafe chars with underscore
    cleaned = _UNSAFE_CHARS.sub("_", title)
    # Collapse runs of underscores
    cleaned = _MULTI_UNDERSCORE.sub("_", cleaned)
    # Strip dangerous edges
    cleaned = _EDGE_CHARS.sub("", cleaned)
    # Truncate
    if len(cleaned) > MAX_TITLE_LENGTH:
        cleaned = cleaned[:MAX_TITLE_LENGTH]
        # Re-strip edges in case truncation left a trailing underscore
        cleaned = _EDGE_CHARS.sub("", cleaned)
    return cleaned or _UNTITLED


# Safety cap on the indexed-path search. If this many sibling
# directories all match the same base name, something is very
# wrong — bail out with a clear error rather than spinning forever.
_MAX_INDEX_ATTEMPTS = 10_000


def _next_indexed_path(base: Path) -> Path:
    """Find the next free 'Name (N)' path after base.

    Assumes base itself already exists.

    Raises:
        RuntimeError: if _MAX_INDEX_ATTEMPTS consecutive candidates
            are all taken, which indicates an absurd filesystem
            state rather than a normal conflict.
    """
    for n in range(1, _MAX_INDEX_ATTEMPTS + 1):
        candidate = base.parent / f"{base.name} ({n})"
        if not candidate.exists():
            return candidate
    raise RuntimeError(
        f"Could not find a free indexed path for {base} after "
        f"{_MAX_INDEX_ATTEMPTS} attempts. This usually means an "
        f"absurd number of conflicting directories exist; please "
        f"clean up {base.parent}."
    )


def resolve_unique_dir(parent: Path, title: str, *, interactive: bool) -> Path:
    """Return a directory path under `parent` for the given title.

    If a directory (or file) with the sanitized title already exists:
        - In interactive mode AND stdin is a TTY: prompt the user.
          'o' (or 'overwrite') returns the existing path.
          'r' (or 'rename'), or just Enter, returns a numbered path.
        - Otherwise: return a numbered path (e.g. 'Title (1)').
    """
    safe = sanitize_title(title)
    target = parent / safe

    if not target.exists():
        return target

    # Conflict. Can we prompt?
    if interactive and sys.stdin.isatty():
        prompt = (
            f"'{safe}' already exists at {target}. "
            f"Overwrite (o) or rename to '{safe} (1)' (r, default)? "
        )
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            answer = ""

        if answer in ("o", "overwrite", "y", "yes"):
            return target
        # Default and explicit 'r' / 'rename' both fall through to rename
        return _next_indexed_path(target)

    # Non-interactive: always auto-rename
    return _next_indexed_path(target)
