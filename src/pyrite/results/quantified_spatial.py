"""Opt-in Pint spectra for selected physical-detector pixels."""

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
class QuantifiedSpatialSpectrum:
    """Detached, accepted-flux spectra for explicitly selected pixels.

    The axes are ``(pixel, photon_energy)``; ``pixel_coordinates`` contains
    the corresponding ``(row, column)`` indices. ``density`` has units of
    inverse energy, *not* inverse energy per steradian: the numerical spatial
    scorer already multiplied by each pixel's solid angle and transmission.
    ``measured`` reports whether detector response was applied during the
    original, bounded scoring operation. Unit conversion never scores again.

    ``sampling`` keeps the producer's convention: ``nodes`` (pointwise),
    ``bin-mean`` (means over :attr:`photon_energy_edges`) or ``mixed`` (a sum
    of conventions). A measured spectrum is always ``nodes``: the response
    reads and returns density at the energy coordinates.

    The private numerical snapshot is CPU-only. All public array read-backs
    are fresh copies; source spatial factors, saved observations and numeric
    APIs remain unchanged. Explicit current methods scale stored yield; no exposure is applied.
    """

    _energy_eV: np.ndarray = field(repr=False)
    _density_per_eV: np.ndarray = field(repr=False)
    _pixel_coordinates: np.ndarray = field(repr=False)
    component: str
    measured: bool
    fwhm_eV: float | None = None
    _energy_unit: str = field(default="eV", repr=False)
    sampling: Literal["nodes", "bin-mean", "mixed"] = "nodes"
    _bin_edges_eV: np.ndarray | None = field(default=None, repr=False)

    axis_names: ClassVar[tuple[str, str]] = ("pixel", "photon_energy")
    measure: ClassVar[Literal["per_energy"]] = "per_energy"
    normalization: ClassVar[Literal["pixel-accepted"]] = "pixel-accepted"

    acceptance_applied: ClassVar[bool] = True

    normalization_basis: NormalizationBasis = field(default="per_incident_electron", kw_only=True)

    def __post_init__(self) -> None:
        from .spatial import COMPONENTS

        _check_energy_unit(self._energy_unit)
        count_basis(self.normalization_basis)
        if self.component not in COMPONENTS:
            raise ValueError("unsupported spatial spectral component")
        if not isinstance(self.measured, bool):
            raise TypeError("measured must be boolean")
        for name in ("_energy_eV", "_density_per_eV", "_pixel_coordinates"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a CPU NumPy array")
            if name == "_pixel_coordinates":
                if value.dtype.kind not in "iu":
                    raise TypeError("pixel coordinates must contain integers")
            elif value.dtype.kind != "f":
                raise TypeError(f"{name} must contain real floating-point values")
            snapshot = value.copy()
            snapshot.setflags(write=False)
            object.__setattr__(self, name, snapshot)

        energy, density, pixels = self._energy_eV, self._density_per_eV, self._pixel_coordinates
        if energy.ndim != 1 or not np.all(np.isfinite(energy)):
            raise ValueError("photon energy must be a finite one-dimensional node grid")
        if energy.size == 0 or np.any(energy < 0) or np.any(np.diff(energy) <= 0):
            raise ValueError("photon energy nodes must be strictly increasing and nonnegative")
        if pixels.ndim != 2 or pixels.shape[1] != 2 or np.any(pixels < 0):
            raise ValueError("pixel coordinates must have nonnegative shape (n_pixel, 2)")
        if density.shape != (pixels.shape[0], energy.size):
            raise ValueError("density must have shape (n_pixel, n_energy)")
        if self.measured and self.sampling != "nodes":
            raise ValueError("measured spectra are node-sampled")
        edges = _checked_bin_edges(self.sampling, self._bin_edges_eV, energy.size)
        object.__setattr__(self, "_bin_edges_eV", edges)

    @property
    def pixel_coordinates(self) -> np.ndarray:
        """Fresh array of integer ``(row, column)`` pairs in selection order."""
        return self._pixel_coordinates.copy()

    @property
    def photon_energy(self) -> Quantity:
        """Fresh Pint photon-energy nodes in the selected display unit."""
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
        """Fresh accepted per-pixel spectrum per displayed photon energy."""
        from ..units import Q_

        return Q_(self._density_per_eV.copy(), f"1 / eV{rate_units(self.normalization_basis)}").to(
            f"1 / ({self._energy_unit}){rate_units(self.normalization_basis)}"
        )

    def to_energy(self, unit: str) -> Self:
        """Convert energy, bin edges and reciprocal density without rescoring pixels."""
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
            _density_per_eV=self._density_per_eV * factor,
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
            _density_per_eV=self._density_per_eV * factor,
            normalization_basis="per_current",
        )

    def _table_quantities(self) -> dict[str, Quantity]:
        return {"photon_energy": self.photon_energy, "density": self.density}

    def to_dataframe(self) -> pl.DataFrame:
        """Numeric native-grid table with explicit measure and normalization labels.

        Coordinates and densities use the paired display energy unit. No
        interpolation, total, response replay or acquisition scaling is added.
        """
        import polars as pl

        frame = pl.DataFrame(
            {
                "pixel_row": np.repeat(self._pixel_coordinates[:, 0], self._energy_eV.size),
                "pixel_column": np.repeat(self._pixel_coordinates[:, 1], self._energy_eV.size),
                "photon_energy": np.tile(
                    self.photon_energy.magnitude, self._pixel_coordinates.shape[0]
                ),
                "density": self.density.magnitude.reshape(-1),
            }
        ).with_columns(
            pl.lit(self.component).alias("component"), pl.lit(self.measured).alias("measured")
        )
        return frame.with_columns(
            *[pl.lit(value).alias(name) for name, value in table_metadata(self).items()]
        )

    def dataframe_units(self) -> dict[str, dict[str, str | None]]:
        """Physical column units and count bases for :meth:`to_dataframe`."""
        return quantity_columns(self._table_quantities(), self.count_basis)


@dataclass(frozen=True, slots=True, eq=False)
class QuantifiedSpatialImage:
    """Detached energy-window-integrated per-primary photon yields per pixel.

    ``photon_yield`` is dimensionless in Pint because photon counts per
    incident primary are dimensionless, but it is *not* a detector acquisition
    count or an exposure-normalized rate. Each image pixel has already been
    weighted by solid-angle acceptance and transmission; ``measured`` records
    whether the original scoring applied detector response. There is no
    remaining per-energy or per-steradian density.

    The native two-dimensional CPU image is copied once. Every read-back
    creates another independent array; display energy conversion changes only
    the reported integration window, never the integrated photon yields.
    """

    _yield_per_primary: np.ndarray = field(repr=False)
    _energy_window_eV: np.ndarray = field(repr=False)
    component: str
    measured: bool
    fwhm_eV: float | None = None
    _energy_unit: str = field(default="eV", repr=False)

    axis_names: ClassVar[tuple[str, str]] = ("pixel_row", "pixel_column")
    sampling: ClassVar[Literal["integrated_window"]] = "integrated_window"
    measure: ClassVar[Literal["per_pixel"]] = "per_pixel"
    normalization: ClassVar[Literal["pixel-accepted"]] = "pixel-accepted"
    count_basis: ClassVar[Literal["photons_per_incident_electron"]] = (
        "photons_per_incident_electron"
    )

    def __post_init__(self) -> None:
        from .spatial import COMPONENTS

        _check_energy_unit(self._energy_unit)
        if self.component not in COMPONENTS:
            raise ValueError("unsupported spatial image component")
        if not isinstance(self.measured, bool):
            raise TypeError("measured must be boolean")
        for name, ndim in (("_yield_per_primary", 2), ("_energy_window_eV", 1)):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.ndim != ndim:
                raise TypeError(f"{name} must be a {ndim}D CPU NumPy array")
            if value.dtype.kind != "f":
                raise TypeError(f"{name} must contain real floating-point values")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            snapshot = value.copy()
            snapshot.setflags(write=False)
            object.__setattr__(self, name, snapshot)

        if not all(self._yield_per_primary.shape):
            raise ValueError("spatial image must contain at least one pixel on each axis")
        if (
            self._energy_window_eV.shape != (2,)
            or self._energy_window_eV[0] >= self._energy_window_eV[1]
        ):
            raise ValueError("photon-energy window must contain two increasing bounds")

    @property
    def photon_energy_window(self) -> Quantity:
        """Requested inclusive selection bounds, not interpolated node edges."""
        from ..units import Q_

        return Q_(self._energy_window_eV.copy(), "eV").to(self._energy_unit)

    @property
    def photon_yield(self) -> Quantity:
        """Independent (row, column) photon yield per incident electron."""
        from ..units import Q_

        return Q_(self._yield_per_primary.copy(), "dimensionless")

    def to_energy(self, unit: str) -> Self:
        """Convert the requested window labels; never reintegrate or change yield."""
        return replace(self, _energy_unit=unit)
