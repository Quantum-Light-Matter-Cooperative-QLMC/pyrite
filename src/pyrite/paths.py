"""Installed-safe package and user-state path constants.

A leaf: nothing here reads configuration. The workspace root does -- it
resolves through the config store -- so it lives in
:mod:`pyrite.console.config`.
"""

import os
import tempfile
from pathlib import Path

import click
from platformdirs import user_cache_path, user_data_path


def data_dir() -> Path:
    """Return the packaged read-only data directory."""
    return Path(__file__).parent / "data"


def app_dir() -> Path:
    """Return the packaged marimo application and resource directory."""
    return Path(__file__).parent / "apps"


def state_dir() -> Path:
    """Return the canonical platform-specific mutable PyRITE state directory."""
    return Path(click.get_app_dir("pyrite"))


def state_path_for_read(name: str) -> Path:
    """Return one path in the canonical mutable-state directory."""
    return state_dir() / name


def cache_dir() -> Path:
    """Return the canonical platform-specific user cache directory."""
    return Path(user_cache_path("pyrite", appauthor=False))


def user_data_dir() -> Path:
    """Return the reserved canonical platform-specific user data directory."""
    return Path(user_data_path("pyrite", appauthor=False))


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace one mutable user-state text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=f"{path.suffix}.tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
