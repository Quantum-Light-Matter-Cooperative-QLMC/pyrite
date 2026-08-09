"""Installed-safe package, workspace, and user-state path resolution."""

from __future__ import annotations

import os
import tempfile
from os import PathLike
from pathlib import Path

import click
from platformdirs import user_cache_path, user_data_path


def data_dir() -> Path:
    """Return the packaged read-only data directory."""
    return Path(__file__).parent / "data"


def app_dir() -> Path:
    """Return the packaged marimo application and resource directory."""
    return Path(__file__).parent / "apps"


def workspace_root(explicit: str | PathLike[str] | None = None) -> Path:
    """Resolve explicit > ``PYRITE_HOME``/``CXR_HOME`` > stores > cwd."""
    from .cli._config import resolve

    value = resolve("workspace.root", None if explicit is None else str(explicit)).value
    return Path(value).expanduser().resolve()


def state_dir() -> Path:
    """Return the canonical platform-specific mutable PyRITE state directory."""
    return Path(click.get_app_dir("pyrite"))


def legacy_state_dir() -> Path:
    """Return the retained cxr-mc mutable-state directory."""
    return Path(click.get_app_dir("cxr-mc"))


def state_path_for_read(name: str) -> Path:
    """Resolve one state file canonical-first, then through the legacy store."""
    canonical = state_dir() / name
    if canonical.exists():
        return canonical
    legacy = legacy_state_dir() / name
    return legacy if legacy.exists() else canonical


def cache_dir() -> Path:
    """Return the canonical platform-specific user cache directory."""
    return Path(user_cache_path("pyrite", appauthor=False))


def legacy_cache_dir() -> Path:
    """Reproduce the exact pre-PyRITE cache-root algorithm."""
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "cxr-mc"


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
