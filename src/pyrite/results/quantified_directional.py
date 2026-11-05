"""Experimental quantity view of directional source densities on native nodes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import numpy as np

from ._spectral_rate import (
    NormalizationBasis,
    count_basis,
    per_current_factor,
    quantity_columns,
    rate_factor,
    rate_units,
    table_metadata,
)
from .quantified import _check_energy_unit
from .spectral_support import _checked_bin_edges

if TYPE_CHECKING:
    import polars as pl
    from pint import Quantity


@dataclass(frozen=True, slots=True, eq=False)
class QuantifiedDirectionalSpectrum:
    """Independent CPU view from :meth:`DirectionalSpectrum.quantify`.

    ``density`` has axes (angle, photon energy). Angles are periodic point
    samples, never solid-angle bins. Energy conversion scales density inversely;
    converting the returned angle quantity does not change the per-sr measure.
    ``supported`` flags valid directions, retaining unsupported density NaNs.
    Each array property returns a fresh copy. The view snapshots native NumPy
    arrays, retains component identity, and does no scoring, resampling, device
    transfer. Explicit rate methods change only current normalization. Case/provenance stay on the source.
    Numeric arrays and private constructor fields use degrees, eV and per-eV/sr.
    ``sampling`` keeps the producer's convention: ``nodes`` (pointwise) or
    ``bin-mean`` (means over :attr:`photon_energy_edges`).
    """

    _angle_deg: np.ndarray = field(repr=False)
    _energy_eV: np.ndarray = field(repr=False)
    _density_per_eV_sr: np.ndarray = field(repr=False)
    _supported: np.ndarray = field(repr=False)
    component: str
    _energy_unit: str = field(default="eV", repr=False)
    sampling: Literal["nodes", "bin-mean", "mixed"] = "nodes"
    _bin_edges_eV: np.ndarray | None = field(default=None, repr=False)
    axis_names: ClassVar[tuple[str, str]] = ("angle", "photon_energy")
    measure: ClassVar[Literal["per_energy_per_steradian"]] = "per_energy_per_steradian"

    normalization: ClassVar[Literal["source"]] = "source"

    acceptance_applied: ClassVar[bool] = False

    normalization_basis: NormalizationBasis = field(default="per_incident_electron", kw_only=True)

    def __post_init__(self) -> None:
        _check_energy_unit(self._energy_unit)
        count_basis(self.normalization_basis)
        for name in ("_angle_deg", "_energy_eV", "_density_per_eV_sr", "_supported"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a CPU NumPy array")
            if value.dtype.kind not in "fiu" and not (
                name == "_supported" and value.dtype.kind == "b"
            ):
                raise TypeError(f"{name} must contain real numeric values")
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if self._angle_deg.ndim != 1 or self._energy_eV.ndim != 1:
            raise ValueError("angle and photon-energy coordinates must be one-dimensional")
        if self._density_per_eV_sr.shape != (self._angle_deg.size, self._energy_eV.size):
            raise ValueError("density must have shape (angle, photon_energy)")
        if self._supported.shape != self._angle_deg.shape or self._supported.dtype.kind != "b":
            raise ValueError("supported must be a boolean mask on the angle axis")
        edges = _checked_bin_edges(self.sampling, self._bin_edges_eV, self._energy_eV.size)
        object.__setattr__(self, "_bin_edges_eV", edges)

    @property
    def angle(self) -> Quantity:
        """Fresh angular point coordinates in degrees, independent of measure."""
        from ..units import Q_

        return Q_(self._angle_deg.copy(), "degree")

    @property
    def photon_energy(self) -> Quantity:
        """Fresh photon-energy nodes in the selected display unit."""
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
        """Fresh density per display energy and sr; count basis stays semantic."""
        from ..units import Q_

        return Q_(
            self._density_per_eV_sr.copy(),
            f"1 / eV / steradian{rate_units(self.normalization_basis)}",
        ).to(f"1 / ({self._energy_unit}) / steradian{rate_units(self.normalization_basis)}")

    @property
    def supported(self) -> np.ndarray:
        """Fresh boolean mask, one entry per angular point."""
        return self._supported.copy()

    def to_energy(self, unit: str) -> Self:
        """Return a new view with coupled energy/density conversion.

        Native nodes, bin edges, support, sampling, angular measure and current count
        basis are retained.
        Non-energy unit strings raise ValueError; other inputs raise TypeError.
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
        return {"photon_energy": self.photon_energy, "angle": self.angle, "density": self.density}

    def to_dataframe(self) -> pl.DataFrame:
        """Numeric native-grid table with explicit measure and normalization labels.

        Coordinates and densities use the paired display energy unit. No
        interpolation, total, response replay or acquisition scaling is added.
        """
        import polars as pl

        frame = pl.DataFrame(
            {
                "angle": np.repeat(self.angle.magnitude, self._energy_eV.size),
                "photon_energy": np.tile(self.photon_energy.magnitude, self._angle_deg.size),
                "density": self.density.magnitude.reshape(-1),
                "supported": np.repeat(self.supported, self._energy_eV.size),
            }
        ).with_columns(pl.lit(self.component).alias("component"))
        return frame.with_columns(
            *[pl.lit(value).alias(name) for name, value in table_metadata(self).items()]
        )

    def dataframe_units(self) -> dict[str, dict[str, str | None]]:
        """Physical column units and count bases for :meth:`to_dataframe`."""
        return quantity_columns(self._table_quantities(), self.count_basis)
