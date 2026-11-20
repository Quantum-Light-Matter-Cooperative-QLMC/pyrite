"""Installed-safe package and user-state path constants.

A leaf: nothing here reads configuration. The workspace root does -- it
resolves through the config store -- so it lives in
:mod:`pyrite.console.config`.
"""

import os
from pathlib import Path

import click
from platformdirs import user_cache_path, user_config_path, user_data_path, user_state_path

from ._util.io import atomic_write_text as atomic_write_text


def data_dir() -> Path:
    """Return the packaged read-only data directory."""
    return Path(__file__).parent / "data"


def app_dir() -> Path:
    """Return the packaged marimo application and resource directory."""
    return Path(__file__).parent / "apps"


def _root(env_name: str, default: Path) -> Path:
    """Honor an explicit PyRITE root before the OS-specific platform default."""
    value = os.environ.get(env_name)
    return Path(value).expanduser().resolve() if value else default


def config_dir() -> Path:
    """Return the config-store directory (PYRITE_CONFIG_DIR overrides it)."""
    return _root("PYRITE_CONFIG_DIR", user_config_path("pyrite", appauthor=False))


def legacy_config_dir() -> Path:
    """Old Click location (roaming on Windows), for read-only compatibility."""
    return Path(click.get_app_dir("pyrite"))


def state_dir() -> Path:
    """Return the application-state directory (PYRITE_STATE_DIR overrides it)."""
    return _root("PYRITE_STATE_DIR", user_state_path("pyrite", appauthor=False))


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
    return _root("PYRITE_CACHE_DIR", user_cache_path("pyrite", appauthor=False))


def user_data_dir() -> Path:
    """Return the reserved canonical platform-specific user data directory."""
    return _root("PYRITE_DATA_DIR", user_data_path("pyrite", appauthor=False))
