"""Detector-free directional radiation: per-electron spectral densities, not counts.

Importable, documented, provisional contract; existing domain imports remain valid.
Resolving an attribute imports the original implementation without wrapping it.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyrite.directional import directional_source as directional_source
    from pyrite.directional import directional_sources as directional_sources
    from pyrite.results.directional import DirectionalSpectrum as DirectionalSpectrum

_EXPORTS = {
    "directional_source": ("pyrite.directional", "directional_source"),
    "directional_sources": ("pyrite.directional", "directional_sources"),
    "DirectionalSpectrum": ("pyrite.results.directional", "DirectionalSpectrum"),
}

__all__ = [
    "directional_source",
    "directional_sources",
    "DirectionalSpectrum",
]


def __getattr__(name: str) -> Any:
    try:
        module, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
