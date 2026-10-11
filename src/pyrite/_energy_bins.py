"""Sampled true-photon grids shared below policy, transport, and detectors."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class EnergyBins:
    """Sampled true-photon energy grids for :class:`~pyrite.Numerics`.

    ``CXR`` is fine and narrow because coherent-CXR evaluation is expensive
    and kinematically bounded. ``brem`` is coarse and wide because the smooth,
    cheap continuum must extend to the beam energy. ``cxr_by_energy`` selects
    a fine CXR grid per beam energy when one fixed CXR grid is insufficient.

    Parameters
    ----------
    CXR
        Optional shared one-dimensional CXR photon-energy grid in eV.
    cxr_by_energy
        Optional mapping from electron energy in keV to CXR grids in eV.
        Mutually exclusive with ``CXR`` at catalog resolution.
    brem
        Optional one-dimensional bremsstrahlung photon-energy grid in eV.
    """

    cxr: np.ndarray | None = None
    cxr_by_energy: Mapping[float, np.ndarray] | None = None
    brem: np.ndarray | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnergyBins):
            return NotImplemented
        if (self.cxr is None) != (other.cxr is None) or (self.brem is None) != (other.brem is None):
            return False
        if (
            self.cxr is not None
            and other.cxr is not None
            and not np.array_equal(self.cxr, other.cxr)
        ):
            return False
        if (
            self.brem is not None
            and other.brem is not None
            and not np.array_equal(self.brem, other.brem)
        ):
            return False
        if (self.cxr_by_energy is None) != (other.cxr_by_energy is None):
            return False
        if self.cxr_by_energy is None:
            return True
        assert other.cxr_by_energy is not None
        return self.cxr_by_energy.keys() == other.cxr_by_energy.keys() and all(
            np.array_equal(grid, other.cxr_by_energy[energy])
            for energy, grid in self.cxr_by_energy.items()
        )


# Keep existing pickle records loadable during the public-object relocation.
EnergyBins.__module__ = "pyrite.detectors.spec"
