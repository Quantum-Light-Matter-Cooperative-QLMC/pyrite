"""Installed-safe package, workspace, and user-state path resolution."""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import click


def data_dir() -> Path:
    """Return the packaged read-only data directory."""
    return Path(__file__).parent / "data"


def workspace_root(explicit: str | PathLike[str] | None = None) -> Path:
    """Resolve explicit argument > ``CXR_HOME`` > config store > current directory."""
    from .cli._config import resolve

    value = resolve("workspace.root", None if explicit is None else str(explicit)).value
    return Path(value).expanduser().resolve()


def state_dir() -> Path:
    """Return the platform-specific directory for mutable cxr-mc user state."""
    return Path(click.get_app_dir("cxr-mc"))
