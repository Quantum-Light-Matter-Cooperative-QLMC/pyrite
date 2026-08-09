"""Compatibility helpers for the additive PyRITE identity migration."""

from __future__ import annotations

import os
import sys
from typing import overload

_WARNED_ENV_CONFLICTS: set[tuple[str, str]] = set()


def canonical_env_name(legacy_name: str) -> str:
    """Return the exact ``PYRITE_*`` counterpart of one ``CXR_*`` name."""
    if not legacy_name.startswith("CXR_"):
        raise ValueError(f"not a CXR environment name: {legacy_name}")
    return f"PYRITE_{legacy_name.removeprefix('CXR_')}"


def completion_active() -> bool:
    """Whether either canonical or compatibility Click completion is active."""
    return bool(os.environ.get("_PYRITE_COMPLETE") or os.environ.get("_CXR_COMPLETE"))


@overload
def env_value(legacy_name: str, default: str) -> str: ...


@overload
def env_value(legacy_name: str, default: None = None) -> str | None: ...


def env_value(legacy_name: str, default: str | None = None) -> str | None:
    """Resolve ``PYRITE_*`` before its ``CXR_*`` compatibility alias."""
    canonical_name = canonical_env_name(legacy_name)
    canonical = os.environ.get(canonical_name)
    legacy = os.environ.get(legacy_name)
    if canonical is not None:
        pair = (canonical_name, legacy_name)
        if canonical != legacy and legacy is not None and pair not in _WARNED_ENV_CONFLICTS:
            _WARNED_ENV_CONFLICTS.add(pair)
            if not completion_active():
                print(
                    f"warning: {canonical_name} and {legacy_name} differ; using {canonical_name}",
                    file=sys.stderr,
                )
        return canonical
    return legacy if legacy is not None else default


def set_canonical_env(legacy_name: str, value: str) -> None:
    """Set only the canonical environment spelling for a generated process."""
    os.environ[canonical_env_name(legacy_name)] = value


def warn_legacy_command(legacy: str, canonical: str) -> None:
    """Warn once for a normal compatibility-executable invocation."""
    if not completion_active():
        print(
            f"warning: `{legacy}` is a compatibility command; use `{canonical}` "
            "(removal no earlier than 0.4.0)",
            file=sys.stderr,
        )
