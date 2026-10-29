"""Sampled true-photon grids shared below policy, transport, and detectors."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np

from ._units import _is_pint_quantity, normalize_unit

if TYPE_CHECKING:
    from pint import Quantity


def _grid_eV(value: object, *, field: str) -> np.ndarray | None:
    """Own an explicit one-dimensional energy-grid quantity as numeric eV, or ``None``."""
    if value is None:
        return None
    if not _is_pint_quantity(value):
        raise TypeError(f"{field} requires a Pint quantity with energy units")
    original = np.asarray(cast(Any, value).magnitude)
    if original.ndim != 1 or original.dtype.kind not in "iuf":
        raise TypeError(f"{field} must be a real one-dimensional energy quantity")
    magnitude = np.asarray(normalize_unit(value, "eV", field=field, expected="energy"))
    with np.errstate(over="ignore", under="ignore"):
        stored = magnitude.astype(float)
    if not np.all(np.isfinite(stored)) or np.any((magnitude != 0) & (stored == 0)):
        raise ValueError(f"{field} must be finite and representable in numeric eV storage")
    return stored


def _ev_quantity(grid: np.ndarray | None) -> "Quantity | None":
    if grid is None:
        return None
    from .units import Q_

    return Q_(np.array(grid, dtype=float), "eV")


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

    Notes
    -----
    The unsuffixed fields predate the unit naming policy and keep returning
    numeric eV arrays. :attr:`cxr_energies` and :attr:`brem_energies` return
    fresh eV quantities; :meth:`with_cxr_energies` and
    :meth:`with_brem_energies` replace a grid from an explicit
    one-dimensional energy quantity (or ``None``). ``cxr_by_energy`` stays
    numeric: keys in keV, grids in eV.
    """

    cxr: np.ndarray | None = None
    cxr_by_energy: Mapping[float, np.ndarray] | None = None
    brem: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("cxr", "brem"):
            value = getattr(self, name)
            if value is not None:
                numeric = normalize_unit(
                    value,
                    "eV",
                    field=f"EnergyBins.{name}",
                    expected="energy",
                )
                object.__setattr__(self, name, np.asarray(numeric, dtype=float))
        if self.cxr_by_energy is not None and any(
            _is_pint_quantity(energy) or _is_pint_quantity(grid)
            for energy, grid in self.cxr_by_energy.items()
        ):
            normalized = {
                float(
                    normalize_unit(
                        energy,
                        "keV",
                        field="EnergyBins.cxr_by_energy key",
                        expected="energy",
                    )
                ): np.asarray(
                    normalize_unit(
                        grid,
                        "eV",
                        field="EnergyBins.cxr_by_energy value",
                        expected="energy",
                    ),
                    dtype=float,
                )
                for energy, grid in self.cxr_by_energy.items()
            }
            object.__setattr__(self, "cxr_by_energy", normalized)

    @property
    def cxr_energies(self) -> "Quantity | None":
        """Shared CXR photon-energy grid as a fresh eV quantity, or ``None``."""
        return _ev_quantity(self.cxr)

    @property
    def brem_energies(self) -> "Quantity | None":
        """Bremsstrahlung photon-energy grid as a fresh eV quantity, or ``None``."""
        return _ev_quantity(self.brem)

    def with_cxr_energies(self, energies: object) -> Self:
        """Return bins with a validated CXR energy-grid quantity, or ``None``."""
        return replace(self, cxr=_grid_eV(energies, field="EnergyBins.cxr_energies"))

    def with_brem_energies(self, energies: object) -> Self:
        """Return bins with a validated bremsstrahlung energy-grid quantity, or ``None``."""
        return replace(self, brem=_grid_eV(energies, field="EnergyBins.brem_energies"))

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
