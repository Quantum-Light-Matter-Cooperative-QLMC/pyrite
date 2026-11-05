"""Opt-in quantity-aware spectra, separate from numeric result storage."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Self

import numpy as np

from .._units import _check_energy_unit
from ._spectral_rate import (
    NormalizationBasis,
    count_basis,
    per_current_factor,
    quantity_columns,
    rate_factor,
    rate_units,
    table_metadata,
)

if TYPE_CHECKING:
    import polars as pl
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
    solid-angle-weighted ``"observation-average"``. Initial views are per
    steradian; Result.accepted_spectra() supplies accepted per-energy views.
    ``measure`` and ``count_basis`` preserve the physical interpretation that
    cannot be enforced by Pint's dimensional algebra alone. The initial
    quantify() view has no acceptance or current scaling; explicit rate/current
    methods change its count basis without response replay.

    Internal arrays are isolated from the source and private. Each property
    access returns a fresh quantity: caller mutation cannot desynchronize the
    paired values or alter the source, this object, or another conversion.
    """

    _energy_eV: np.ndarray = field(repr=False)
    _density_per_eV_sr: np.ndarray = field(repr=False)
    normalization: Literal["source", "observation-average"]
    _energy_unit: str = field(default="eV", repr=False)
    sampling: Literal["nodes", "bin-mean", "mixed"] = "nodes"
    measure: Literal["per_energy_per_steradian", "per_energy"] = "per_energy_per_steradian"
    _bin_edges_eV: np.ndarray | None = field(default=None, repr=False)

    acceptance_applied: bool = field(default=False, kw_only=True)
    accepted_solid_angle_sr: float | None = field(default=None, kw_only=True)

    normalization_basis: NormalizationBasis = field(default="per_incident_electron", kw_only=True)

    def __post_init__(self) -> None:
        _check_energy_unit(self._energy_unit)
        count_basis(self.normalization_basis)
        for value in (self._energy_eV, self._density_per_eV_sr, self._bin_edges_eV):
            if value is not None and not isinstance(value, (np.ndarray, list, tuple)):
                raise TypeError("quantity spectra require CPU NumPy arrays or numeric sequences")
        if self.normalization not in {"source", "observation-average"}:
            raise ValueError("normalization must be source or observation-average")
        if self.sampling not in {"nodes", "bin-mean", "mixed"} or self.measure not in {
            "per_energy_per_steradian",
            "per_energy",
        }:
            raise ValueError("unsupported spectrum measure or normalization metadata")

        if self.acceptance_applied != (self.measure == "per_energy"):
            raise ValueError("acceptance and spectral measure must agree")
        if self.acceptance_applied:
            if (
                self.accepted_solid_angle_sr is None
                or not np.isfinite(self.accepted_solid_angle_sr)
                or self.accepted_solid_angle_sr <= 0
            ):
                raise ValueError("accepted spectra require a finite positive matching solid angle")
        elif self.accepted_solid_angle_sr is not None:
            raise ValueError("unaccepted spectra cannot carry applied solid angle")

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
        """Fresh density in the displayed energy unit and declared angular/current measure."""
        from ..units import Q_

        angular = " / steradian" if self.measure == "per_energy_per_steradian" else ""
        temporal = rate_units(self.normalization_basis)
        return Q_(self._density_per_eV_sr.copy(), f"1 / eV{angular}{temporal}").to(
            f"1 / ({self._energy_unit}){angular}{temporal}"
        )

    def to_energy(self, unit: str) -> Self:
        """Convert energy **and** spectral density; preserve integrated yield.

        This is an energy-axis change, *not* a source/observation, current or
        solid-angle normalization change. It retains every native sample and
        its physical meaning, and never modifies this spectrum.
        """
        return replace(self, _energy_unit=unit)

    @property
    def count_basis(self) -> str:
        """Photon normalization, independent of Pint's dimensional algebra."""
        return count_basis(self.normalization_basis)

    def rate(self, *, current: Quantity) -> Self:
        """Return photons per second using explicit mean current I/|e|.

        Current must be a finite nonnegative scalar quantity. Rejects repeated
        current transforms; preserves acceptance, response and native sampling.
        Validation: spectral-explicit-rate-normalization
        """
        factor = rate_factor(self.normalization_basis, current)
        return replace(
            self,
            _density_per_eV_sr=self._density_per_eV_sr * factor,
            normalization_basis="per_second",
        )

    def rate_per_current(self) -> Self:
        """Return photons per second per ampere, including at zero beam current.

        Applies 1/|e| directly to per-primary yields, never divides by current.
        Validation: spectral-explicit-rate-normalization
        """
        factor = per_current_factor(self.normalization_basis)
        return replace(
            self,
            _density_per_eV_sr=self._density_per_eV_sr * factor,
            normalization_basis="per_current",
        )

    def _table_quantities(self) -> dict[str, Quantity]:
        columns = {"photon_energy": self.photon_energy, "density": self.density}
        if self._bin_edges_eV is not None:
            edges = self.photon_energy_edges
            columns.update(photon_energy_left=edges[:-1], photon_energy_right=edges[1:])
        return columns

    def to_dataframe(self) -> pl.DataFrame:
        """Numeric native-grid table with explicit measure and normalization labels.

        Coordinates and densities use the paired display energy unit. No
        interpolation, total, response replay or acquisition scaling is added.
        """
        import polars as pl

        columns = self._table_quantities()
        frame = pl.DataFrame({name: q.magnitude for name, q in columns.items()})
        return frame.with_columns(
            *[pl.lit(value).alias(name) for name, value in table_metadata(self).items()]
        )

    def dataframe_units(self) -> dict[str, dict[str, str | None]]:
        """Physical column units and count bases for :meth:`to_dataframe`."""
        return quantity_columns(self._table_quantities(), self.count_basis)
