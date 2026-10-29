"""Opt-in Pint view of aperture-integrated, bin-mean photon spectra."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import numpy as np

from .quantified import _check_energy_unit

if TYPE_CHECKING:
    from pint import Quantity

    from ..photon_event_observation import PhotonEventTally
    from ..photon_observation import PhotonContinuumTally


@dataclass(frozen=True, slots=True, eq=False)
class QuantifiedApertureSpectrum:
    """Detached quantity view of a physical-aperture energy histogram.

    ``density`` and its direct/redistributed components are *bin means* per
    incident electron and energy, already integrated over the acceptance (no
    ``/sr`` factor). ``counts`` are the in-range bin integrals. Detected counts
    also include independent unbinned/out-of-range detections; converting the
    displayed energy axis does not change any counts.

    Obtained from ``PhotonContinuumTally.quantify()`` or
    ``PhotonEventTally.quantify()``. No current, response or exposure scaling
    is performed. Stored tally arrays, event budgets and identity remain numeric.
    Every returned Pint quantity owns a fresh magnitude; no view exposes its
    private source snapshot or performs resampling or transport.
    """

    _edges_eV: np.ndarray = field(repr=False)
    _total_per_primary_eV: np.ndarray = field(repr=False)
    _direct_per_primary_eV: np.ndarray = field(repr=False)
    _redistributed_per_primary_eV: np.ndarray = field(repr=False)
    _detected_per_primary: float
    _unbinned_detected_per_primary: float
    source_component: str
    _energy_unit: str = field(default="eV", repr=False)

    axis_names: ClassVar[tuple[str, ...]] = ("photon_energy",)
    sampling: ClassVar[Literal["bin_means"]] = "bin_means"
    measure: ClassVar[Literal["per_energy"]] = "per_energy"
    normalization: ClassVar[Literal["aperture-integrated"]] = "aperture-integrated"
    count_basis: ClassVar[Literal["photons_per_incident_electron"]] = (
        "photons_per_incident_electron"
    )

    @classmethod
    def from_tally(cls, tally: PhotonContinuumTally | PhotonEventTally) -> Self:
        """Snapshot the native CPU tally without touching its stored representation."""
        return cls(
            tally.edges_eV,
            tally.total_per_primary_eV,
            tally.direct_per_primary_eV,
            tally.redistributed_per_primary_eV,
            tally.detected_per_primary,
            tally.unbinned_detected_per_primary,
            getattr(tally, "source_component", "continuum"),
        )

    def __post_init__(self) -> None:
        _check_energy_unit(self._energy_unit)
        if self.source_component not in {"continuum", "characteristic", "annihilation"}:
            raise ValueError("unsupported aperture source component")

        names = (
            "_edges_eV",
            "_total_per_primary_eV",
            "_direct_per_primary_eV",
            "_redistributed_per_primary_eV",
        )
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype.kind != "f":
                raise TypeError(f"{name} must be a one-dimensional floating-point CPU array")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

        edges = self._edges_eV
        if edges.size < 2 or np.any(edges < 0) or np.any(np.diff(edges) <= 0):
            raise ValueError("aperture energy-bin edges must increase and be nonnegative")
        total, direct, redistributed = (
            self._total_per_primary_eV,
            self._direct_per_primary_eV,
            self._redistributed_per_primary_eV,
        )
        if any(
            a.shape != (edges.size - 1,) or np.any(a < 0) for a in (total, direct, redistributed)
        ):
            raise ValueError("aperture densities must be nonnegative and match energy bins")
        if not np.allclose(total, direct + redistributed, rtol=1e-12, atol=0):
            raise ValueError("aperture categories must partition the total")
        for name in ("_detected_per_primary", "_unbinned_detected_per_primary"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float, np.float32, np.float64)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative scalar")
        accounted = float(np.sum(total * np.diff(edges))) + self._unbinned_detected_per_primary
        if not np.isclose(self._detected_per_primary, accounted, rtol=1e-12, atol=0):
            raise ValueError("detected counts must equal binned plus unbinned counts")

    @property
    def photon_energy_edges(self) -> Quantity:
        """Fresh N+1-bin-edge energy quantity, never N interpolated nodes."""
        from ..units import Q_

        return Q_(self._edges_eV.copy(), "eV").to(self._energy_unit)

    @property
    def bin_widths(self) -> Quantity:
        """Fresh positive widths in the chosen energy unit."""
        from ..units import Q_

        return Q_(np.diff(self._edges_eV), "eV").to(self._energy_unit)

    def _density(self, native: np.ndarray) -> Quantity:
        from ..units import Q_

        return Q_(native.copy(), "1 / eV").to(f"1 / ({self._energy_unit})")

    @property
    def density(self) -> Quantity:
        """Total aperture-integrated bin-mean density in 1/display energy."""
        return self._density(self._total_per_primary_eV)

    @property
    def direct_density(self) -> Quantity:
        """Unscattered/direct bin-mean density."""
        return self._density(self._direct_per_primary_eV)

    @property
    def redistributed_density(self) -> Quantity:
        """Transport-redistributed bin-mean density."""
        return self._density(self._redistributed_per_primary_eV)

    @staticmethod
    def _counts(density: np.ndarray, edges: np.ndarray) -> Quantity:
        from ..units import Q_

        return Q_(density * np.diff(edges), "dimensionless")

    @property
    def counts(self) -> Quantity:
        """In-range photon counts per original primary, one per energy bin."""
        return self._counts(self._total_per_primary_eV, self._edges_eV)

    @property
    def direct_counts(self) -> Quantity:
        return self._counts(self._direct_per_primary_eV, self._edges_eV)

    @property
    def redistributed_counts(self) -> Quantity:
        return self._counts(self._redistributed_per_primary_eV, self._edges_eV)

    @property
    def unbinned_counts(self) -> Quantity:
        """Detected photon counts outside the reported energy bins."""
        from ..units import Q_

        return Q_(float(self._unbinned_detected_per_primary), "dimensionless")

    @property
    def detected_counts(self) -> Quantity:
        """All detected photons: in-range bins plus unbinned detections."""
        from ..units import Q_

        return Q_(float(self._detected_per_primary), "dimensionless")

    def to_energy(self, unit: str) -> Self:
        """Couple bin-edge/width/density conversion; preserve counts and context."""
        return type(self)(
            self._edges_eV,
            self._total_per_primary_eV,
            self._direct_per_primary_eV,
            self._redistributed_per_primary_eV,
            self._detected_per_primary,
            self._unbinned_detected_per_primary,
            self.source_component,
            unit,
        )
