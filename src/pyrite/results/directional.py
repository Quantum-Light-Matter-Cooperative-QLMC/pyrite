"""Directional source components with their own energy coordinates."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.constants import elementary_charge

from .store import beam_current_na


@dataclass(frozen=True)
class DirectionalSpectrum:
    """Escaped incoherent PXR/CBS density, including PXR–CBS interference.

    ``density`` has shape (angle, energy), in ph/incident-electron/eV/sr.
    ``supported`` flags sample-face grazing directions; their density is NaN.
    Angles are periodic point samples, not solid-angle bins. Provenance and
    case retain resolved sampling, geometry, grids, seed and beam pulse data.
    Future components retain independent energy coordinates and identities;
    this object does not imply that arbitrary components can be added.
    """

    angle_deg: np.ndarray
    energy_eV: np.ndarray
    direction_lab: np.ndarray
    direction_sample: np.ndarray
    density: np.ndarray
    supported: np.ndarray
    case: Mapping[str, Any]
    provenance: Mapping[str, Any]
    component: str = "incoherent-pxr-cbs"
    units: str = "ph/incident-electron/eV/sr"

    @property
    def current_na(self) -> float:
        """Resolved mean electron current magnitude in nA (Q[pC] f[Hz]/1000).

        Validation: directional-source-rate-normalization
        """
        return beam_current_na(self.case)

    def rate(self, current_na: float | None = None) -> np.ndarray:
        """Return ph/s/eV/sr at the resolved or explicit mean current [nA].

        Equation: R(E,theta) = Y(E,theta) I[nA] 1e-9 / |e|, where
        Y is already normalized per incident electron, eV and sr. Assumes
        independent incident electrons and an incoherent source; intra-electron
        PXR–CBS interference remains in Y. No angular-bin/solid-angle factor.
        At zero current all supported bins vanish. Current must be a finite
        nonnegative magnitude; pulse defaults reuse the beam Q*f convention.
        Source: BIPM SI ampere, https://www.bipm.org/en/si-base-units/ampere
        Validation: directional-source-rate-normalization
        """
        current = self.current_na if current_na is None else float(current_na)
        if not np.isfinite(current) or current < 0:
            raise ValueError("current_na must be a finite nonnegative mean current in nA")
        return self.density * (current * 1e-9 / elementary_charge)
