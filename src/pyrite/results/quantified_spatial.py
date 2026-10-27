"""Opt-in Pint spectra for selected physical-detector pixels."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import numpy as np

from .quantified import _check_energy_unit

if TYPE_CHECKING:
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

    The private numerical snapshot is CPU-only. All public array read-backs
    are fresh copies; source spatial factors, saved observations and numeric
    APIs remain unchanged. No beam-current or exposure normalization occurs.
    """

    _energy_eV: np.ndarray = field(repr=False)
    _density_per_eV: np.ndarray = field(repr=False)
    _pixel_coordinates: np.ndarray = field(repr=False)
    component: str
    measured: bool
    fwhm_eV: float | None = None
    _energy_unit: str = field(default="eV", repr=False)

    axis_names: ClassVar[tuple[str, str]] = ("pixel", "photon_energy")
    sampling: ClassVar[Literal["nodes"]] = "nodes"
    measure: ClassVar[Literal["per_energy"]] = "per_energy"
    normalization: ClassVar[Literal["pixel-accepted"]] = "pixel-accepted"
    count_basis: ClassVar[Literal["photons_per_incident_electron"]] = (
        "photons_per_incident_electron"
    )

    def __post_init__(self) -> None:
        from .spatial import COMPONENTS

        _check_energy_unit(self._energy_unit)
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
    def density(self) -> Quantity:
        """Fresh accepted per-pixel spectrum per displayed photon energy."""
        from ..units import Q_

        return Q_(self._density_per_eV.copy(), "1 / eV").to(f"1 / ({self._energy_unit})")

    def to_energy(self, unit: str) -> Self:
        """Convert node energy and reciprocal density without rescoring pixels."""
        return type(self)(
            self._energy_eV,
            self._density_per_eV,
            self._pixel_coordinates,
            self.component,
            self.measured,
            self.fwhm_eV,
            unit,
        )


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
