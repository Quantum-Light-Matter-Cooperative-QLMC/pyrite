"""Returned value for the public single-shot simulation API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ..montecarlo import Case


@dataclass(frozen=True)
class Result:
    """Intrinsic spectral arrays and resolved simulation provenance.

    ``spectrum`` and ``background`` are photon-density arrays per incident
    electron per eV per sr. ``energy_eV`` and ``background_energy_eV`` are their
    respective photon-energy coordinates in eV.
    """

    energy_eV: np.ndarray
    spectrum: np.ndarray
    background_energy_eV: np.ndarray
    background: np.ndarray
    case: Case
    provenance: Mapping[str, Any]
    coherent_spectrum: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy_eV", np.asarray(self.energy_eV))
        object.__setattr__(self, "spectrum", np.asarray(self.spectrum))
        object.__setattr__(self, "background_energy_eV", np.asarray(self.background_energy_eV))
        object.__setattr__(self, "background", np.asarray(self.background))
        if self.coherent_spectrum is not None:
            object.__setattr__(self, "coherent_spectrum", np.asarray(self.coherent_spectrum))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


__all__ = ["Result"]
