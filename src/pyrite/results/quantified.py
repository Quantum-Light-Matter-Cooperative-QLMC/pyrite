"""Opt-in quantity-aware spectra, separate from numeric result storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Self

import numpy as np

from .._units import _check_energy_unit

if TYPE_CHECKING:
    from pint import Quantity


@dataclass(frozen=True, slots=True, eq=False)
class QuantifiedSpectrum:
    """A node-sampled spectrum with coupled energy/density conversion.

    Obtain via :meth:`pyrite.results.SpectralDensity.quantify`. The energy
    coordinate and density are *Pint* quantities on the component's original
    nodes. ``to_energy()`` converts both together, never interpolates, and
    returns a new spectrum.

    ``normalization`` is either response-free ``"source"`` or filtered,
    solid-angle-weighted ``"observation-average"`` (still per steradian).
    ``measure`` and ``count_basis`` preserve the physical interpretation that
    cannot be enforced by Pint's dimensional algebra alone. No acceptance,
    beam-current, or response scaling takes place here.

    Internal arrays are isolated from the source and private. Each property
    access returns a fresh quantity: caller mutation cannot desynchronize the
    paired values or alter the source, this object, or another conversion.
    """

    _energy_eV: np.ndarray = field(repr=False)
    _density_per_eV_sr: np.ndarray = field(repr=False)
    normalization: Literal["source", "observation-average"]
    _energy_unit: str = field(default="eV", repr=False)
    sampling: Literal["nodes"] = "nodes"
    measure: Literal["per_energy_per_steradian"] = "per_energy_per_steradian"
    count_basis: Literal["photons_per_incident_electron"] = "photons_per_incident_electron"

    def __post_init__(self) -> None:
        _check_energy_unit(self._energy_unit)
        if self.normalization not in {"source", "observation-average"}:
            raise ValueError("normalization must be source or observation-average")
        if (
            self.sampling != "nodes"
            or self.measure != "per_energy_per_steradian"
            or self.count_basis != "photons_per_incident_electron"
        ):
            raise ValueError("unsupported spectrum measure or normalization metadata")

        energy = np.array(self._energy_eV, copy=True)
        density = np.array(self._density_per_eV_sr, copy=True)
        if energy.ndim != 1 or density.shape != energy.shape:
            raise ValueError("density must match a one-dimensional photon-energy grid")
        energy.setflags(write=False)
        density.setflags(write=False)
        object.__setattr__(self, "_energy_eV", energy)
        object.__setattr__(self, "_density_per_eV_sr", density)

    @property
    def photon_energy(self) -> Quantity:
        """A fresh Pint photon-energy coordinate on the original nodes."""
        from ..units import Q_

        return Q_(self._energy_eV.copy(), "eV").to(self._energy_unit)

    @property
    def density(self) -> Quantity:
        """A fresh Pint density per displayed energy and steradian."""
        from ..units import Q_

        return Q_(self._density_per_eV_sr.copy(), "1 / eV / steradian").to(
            f"1 / ({self._energy_unit}) / steradian"
        )

    def to_energy(self, unit: str) -> Self:
        """Convert energy **and** spectral density; preserve integrated yield.

        This is an energy-axis change, *not* a source/observation, current or
        solid-angle normalization change. It retains every native sample and
        its physical meaning, and never modifies this spectrum.
        """
        return type(self)(
            self._energy_eV,
            self._density_per_eV_sr,
            self.normalization,
            unit,
        )
