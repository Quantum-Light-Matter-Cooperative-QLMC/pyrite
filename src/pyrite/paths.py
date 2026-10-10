"""Installed-safe package and user-state path constants.

A leaf: nothing here reads configuration. The workspace root does -- it
resolves through the config store -- so it lives in
:mod:`pyrite.console.config`.
"""

import os
import tempfile
from pathlib import Path

import click
from platformdirs import user_cache_path, user_data_path, user_state_path


def data_dir() -> Path:
    """Return the packaged read-only data directory."""
    return Path(__file__).parent / "data"


def app_dir() -> Path:
    """Return the packaged marimo application and resource directory."""
    return Path(__file__).parent / "apps"


def config_dir() -> Path:
    """Return the platform user config directory: the config store and user profiles."""
    return Path(click.get_app_dir("pyrite"))


def state_dir() -> Path:
    """Return the platform user state directory: remembered app defaults and servers."""
    return Path(user_state_path("pyrite", appauthor=False))


def migrate_legacy_state(path: Path) -> Path:
    """Move a state file left in :func:`config_dir` by PyRITE <= 0.6 to ``path``.

    State files used to share the config directory. A no-op unless ``path``
    lives directly in :func:`state_dir`, is absent, and its legacy twin exists.
    """
    legacy = config_dir() / path.name
    if path.parent == state_dir() and legacy != path and legacy.is_file() and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(legacy, path)
        except OSError:
            return legacy
    return path


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
