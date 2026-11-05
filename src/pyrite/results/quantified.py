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
    """A sampled spectrum with coupled energy/density conversion.

    Obtain via :meth:`pyrite.results.SpectralDensity.quantify`. The energy
    coordinate and density are *Pint* quantities on the component's original
    nodes. ``sampling`` keeps the producer's convention: ``nodes`` (pointwise
    samples), ``bin-mean`` (means over :attr:`photon_energy_edges`) or
    ``mixed``. ``to_energy()`` converts coordinates, edges and density
    together, never interpolates, preserves every integral, and returns a new
    spectrum.

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
    sampling: Literal["nodes", "bin-mean", "mixed"] = "nodes"
    measure: Literal["per_energy_per_steradian"] = "per_energy_per_steradian"
    count_basis: Literal["photons_per_incident_electron"] = "photons_per_incident_electron"
    _bin_edges_eV: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _check_energy_unit(self._energy_unit)
        if self.normalization not in {"source", "observation-average"}:
            raise ValueError("normalization must be source or observation-average")
        if (
            self.sampling not in {"nodes", "bin-mean", "mixed"}
            or self.measure != "per_energy_per_steradian"
            or self.count_basis != "photons_per_incident_electron"
        ):
            raise ValueError("unsupported spectrum measure or normalization metadata")

        energy = np.array(self._energy_eV, copy=True)
        density = np.array(self._density_per_eV_sr, copy=True)
        if energy.ndim != 1 or density.shape != energy.shape:
            raise ValueError("density must match a one-dimensional photon-energy grid")
        if (self.sampling == "bin-mean") != (self._bin_edges_eV is not None):
            raise ValueError("bin-mean sampling requires, and only it accepts, bin edges")
        edges = None
        if self._bin_edges_eV is not None:
            edges = np.array(self._bin_edges_eV, dtype=float, copy=True)
            if edges.shape != (energy.size + 1,):
                raise ValueError("bin edges must have one more entry than the energy grid")
            edges.setflags(write=False)
        energy.setflags(write=False)
        density.setflags(write=False)
        object.__setattr__(self, "_energy_eV", energy)
        object.__setattr__(self, "_density_per_eV_sr", density)
        object.__setattr__(self, "_bin_edges_eV", edges)

    @property
    def photon_energy(self) -> Quantity:
        """A fresh Pint photon-energy coordinate on the original nodes."""
        from ..units import Q_

        return Q_(self._energy_eV.copy(), "eV").to(self._energy_unit)

    @property
    def photon_energy_edges(self) -> Quantity:
        """Fresh Pint native bin edges of a ``bin-mean`` spectrum."""
        from ..units import Q_

        if self._bin_edges_eV is None:
            raise ValueError("only bin-mean spectra carry native bin edges")
        return Q_(self._bin_edges_eV.copy(), "eV").to(self._energy_unit)

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
            self.sampling,
            _bin_edges_eV=self._bin_edges_eV,
        )
