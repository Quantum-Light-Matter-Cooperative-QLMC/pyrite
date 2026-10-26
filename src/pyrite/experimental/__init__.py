"""Discoverable experimental interfaces, organized by owning domain.

Experimental is a compatibility status, not a separate scientific owner.
Importing this package does not import research models or optional backends.
See ``docs/guides/experimental-api.md`` for the admission and promotion policy.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import observations as observations
    from . import quantum as quantum
    from . import radiation as radiation

__all__ = ["radiation", "observations", "quantum"]


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(f"{__name__}.{name}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
