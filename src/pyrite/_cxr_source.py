"""Validation of the characteristic-line opt-out ``Physics(cxr_source="none")`` (#1327).

``"none"`` scores no CXR lines: no reflection selection, no line grid, and no
line, coherent, diffracted-bremsstrahlung or Kossel pass. ``Result.cxr`` is an
explicit zero spectrum on a two-point placeholder grid (or the explicit
``EnergyBins.cxr``). The lattice-share correction (#1250) is switched off: no
CBS lines score the share it removes, so the continuum is the standard full
BremsLib one. It therefore differs from a default run's continuum by that
share and equals a lines-on run with the correction disabled. Fixed
sampling ignores ``cxr_histories``; the run transports ``brem_histories``
electrons. Coherent emission, temporal profiles, line roulette, channeling,
transition radiation and adaptive CXR precision targets are refused. The case
key is absent unless ``"none"``, so every existing identity is unchanged.
"""

from __future__ import annotations

from typing import Any

CXR_SOURCES = ("mc", "none")


def validate_cxr_source(physics: Any) -> None:
    """Normalize ``None`` to ``"none"`` and refuse line-only settings beside it."""
    if physics.cxr_source is None:
        object.__setattr__(physics, "cxr_source", "none")
    if physics.cxr_source not in CXR_SOURCES:
        raise ValueError("cxr_source must be 'mc', None, or 'none'")
    if physics.cxr_source == "mc":
        return
    if physics.coherent_emission:
        raise ValueError("cxr_source='none' scores no lines; it refuses coherent emission")
    if physics.transition_radiation != "none":
        raise ValueError(
            "cxr_source='none' scores no line grid; it refuses transition_radiation "
            f"{physics.transition_radiation!r}"
        )
