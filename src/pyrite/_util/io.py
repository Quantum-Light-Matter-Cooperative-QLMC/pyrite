"""Atomic file replacement for text and JSON state."""

import json
import os
import tempfile
from pathlib import Path


def atomic_write_text(path: str | os.PathLike[str], text: str, *, encoding: str = "utf-8") -> None:
    """Replace ``path`` with ``text`` so readers never see a partial file.

    Writes a hidden temporary sibling (``.<name>.<random>.tmp``), creates the
    parent directory if needed, and renames it into place with
    :func:`os.replace`. The temporary is removed if anything fails, including
    ``KeyboardInterrupt``. The file is created with :func:`tempfile.mkstemp`
    permissions (owner read/write only).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as stream:
            stream.write(text)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | os.PathLike[str], payload: object, *, indent: int = 2) -> None:
    """Atomically replace ``path`` with ``payload`` as indented JSON."""
    atomic_write_text(path, json.dumps(payload, indent=indent))
