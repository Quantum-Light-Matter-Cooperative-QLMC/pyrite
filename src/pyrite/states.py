"""Quantum transverse entrance states: portable specs and legacy runtime factories.

``TransverseStateSpec`` is JSON-safe and import-light. Existing #508 runtime
functions are lazily re-exported for diagnostics; using them directly does not
create a portable, serializable source description.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .quantum_state import TransverseStateSpec

_RUNTIME = frozenset({
    "TransverseState", "SampledState", "Superposition", "MixedState", "Translated",
    "EntryState", "gaussian", "hermite_gauss", "laguerre_gauss", "donut",
    "bessel_gauss", "airy", "aperture_probe", "from_array", "from_abtem",
    "superpose", "mixture", "translate",
})


def __getattr__(name: str) -> Any:
    if name not in _RUNTIME:
        raise AttributeError(name)
    value = getattr(import_module("pyrite.montecarlo.channeling.states"), name)
    globals()[name] = value
    return value


__all__ = ["TransverseStateSpec", *sorted(_RUNTIME)]
