"""Small environment helpers shared by PyRITE runtime entry points."""

from __future__ import annotations

import os
from typing import overload


@overload
def env_value(name: str, default: str) -> str: ...


@overload
def env_value(name: str, default: None = None) -> str | None: ...


def env_value(name: str, default: str | None = None) -> str | None:
    """Return one PyRITE environment value or its default."""
    return os.environ.get(name, default)


def set_canonical_env(name: str, value: str) -> None:
    """Set one PyRITE environment value for a generated process."""
    os.environ[name] = value
