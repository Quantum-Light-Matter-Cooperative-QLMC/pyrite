"""Tabulated dimensionless efficiency curves for detector and grating models.

A :class:`TabulatedEfficiency` holds one dimensionless response (detector
quantum efficiency, absolute grating efficiency) sampled on an ascending
photon-energy grid. It evaluates by linear interpolation inside its band and
returns zero outside it: a tabulated curve makes no claim beyond its samples,
so it is never extrapolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..materials import MaterialCatalog


@dataclass(frozen=True)
class TabulatedEfficiency:
    """A dimensionless efficiency curve on an ascending energy grid.

    Parameters
    ----------
    energy_eV
        Strictly increasing, finite, positive photon energies in eV; at least
        two samples.
    value
        Matching dimensionless efficiencies in ``[0, 1]``.
    name
        Optional catalog curve name, kept for provenance and identity.

    Notes
    -----
    Calling the curve interpolates linearly in energy. Energies outside
    ``[energy_eV[0], energy_eV[-1]]`` return exactly zero (no extrapolation);
    callers that need a hard failure check :meth:`covers` first. The sample
    values, not only the name, are dataclass fields, so they enter any
    response or observation identity built from the owning object.
    """

    energy_eV: tuple[float, ...]
    value: tuple[float, ...]
    name: str | None = None

    def __post_init__(self) -> None:
        energy = tuple(float(item) for item in self.energy_eV)
        value = tuple(float(item) for item in self.value)
        if len(energy) < 2 or len(energy) != len(value):
            raise ValueError("TabulatedEfficiency needs two or more equal-length samples")
        if not all(math.isfinite(item) and item > 0.0 for item in energy):
            raise ValueError("TabulatedEfficiency.energy_eV must be finite and positive")
        if any(b <= a for a, b in zip(energy, energy[1:], strict=False)):
            raise ValueError("TabulatedEfficiency.energy_eV must be strictly increasing")
        if not all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in value):
            raise ValueError("TabulatedEfficiency.value must lie in [0, 1]")
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "value", value)

    @classmethod
    def from_catalog(cls, name: str, catalog: MaterialCatalog | None = None) -> TabulatedEfficiency:
        """Build from a loaded catalog curve.

        Parameters
        ----------
        name
            Curve name referenced by a catalog object (``qe_curve`` or
            ``efficiency_curve``).
        catalog
            Loaded catalog; ``None`` loads the selected catalog with its user
            layer.
        """
        if catalog is None:
            from ..materials import load_material_catalog

            catalog = load_material_catalog()
        try:
            curve = catalog.curves[name]
        except KeyError:
            raise KeyError(
                f"no catalog curve {name!r}; referenced curves: {sorted(catalog.curves)}"
            ) from None
        return cls(curve.energy_eV, curve.value, name=name)

    @property
    def band_eV(self) -> tuple[float, float]:
        """Inclusive ``(low, high)`` energy band of the samples in eV."""
        return self.energy_eV[0], self.energy_eV[-1]

    def covers(self, E_eV) -> np.ndarray:
        """Boolean mask of energies inside the tabulated band."""
        E = np.asarray(E_eV, dtype=float)
        low, high = self.band_eV
        return (E >= low) & (E <= high)

    def __call__(self, E_eV) -> np.ndarray:
        """Interpolated efficiency at ``E_eV``; zero outside the band."""
        E = np.asarray(E_eV, dtype=float)
        return np.interp(E, self.energy_eV, self.value, left=0.0, right=0.0)


__all__ = ["TabulatedEfficiency"]
