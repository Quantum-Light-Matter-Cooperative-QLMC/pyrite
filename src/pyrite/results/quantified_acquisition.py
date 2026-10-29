"""Detached quantity view for measured-energy acquisition counts (#755)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import numpy as np

from .quantified import _check_energy_unit

if TYPE_CHECKING:
    from pint import Quantity

    from ..instrument.acquisition import AcquisitionBatch


@dataclass(frozen=True, slots=True, eq=False)
class QuantifiedAcquisitionBatch:
    """Exposure-normalized detector counts on measured-energy bins.

    Axes are (pixel, measured energy bin), with pixel indices (row, column)
    and N+1 measured-energy edges. Count magnitudes are dimensionless, but
    represent detector events over the exposure rather than source photons
    per incident electron. Underflow, overflow and below-threshold events
    are distinct from the reporting bins; no event is silently re-binned.

    The backing CPU NumPy arrays are detached snapshots. Properties create
    fresh quantities, and converting the energy axis does not re-score a
    detector, multiply by exposure, or re-draw Poisson counts.
    """

    _coordinates: np.ndarray = field(repr=False)
    _edges_eV: np.ndarray = field(repr=False)
    _expected: np.ndarray = field(repr=False)
    _realized: np.ndarray | None = field(repr=False)
    _underflow_expected: np.ndarray = field(repr=False)
    _overflow_expected: np.ndarray = field(repr=False)
    _below_cut_expected: np.ndarray = field(repr=False)
    _underflow_realized: np.ndarray | None = field(repr=False)
    _overflow_realized: np.ndarray | None = field(repr=False)
    _below_cut_realized: np.ndarray | None = field(repr=False)
    components: tuple[str, ...]
    observation_digest: str
    _energy_unit: str = field(default="eV", repr=False)

    axis_names: ClassVar[tuple[str, str]] = ("pixel", "measured_photon_energy")
    sampling: ClassVar[Literal["bin_counts"]] = "bin_counts"
    measure: ClassVar[Literal["per_bin"]] = "per_bin"
    normalization: ClassVar[Literal["acquisition_exposure"]] = "acquisition_exposure"
    count_basis: ClassVar[Literal["detector_events_per_exposure"]] = "detector_events_per_exposure"

    @classmethod
    def from_batch(cls, batch: AcquisitionBatch) -> Self:
        """Snapshot a validated numeric acquisition without altering identity."""
        return cls(
            batch.coordinates,
            batch.measured_edges_eV,
            batch.expected,
            batch.realized,
            batch.underflow_expected,
            batch.overflow_expected,
            batch.below_cut_expected,
            batch.underflow_realized,
            batch.overflow_realized,
            batch.below_cut_realized,
            batch.components,
            batch.observation_digest,
        )

    def __post_init__(self) -> None:
        _check_energy_unit(self._energy_unit)
        self._snapshot_arrays()
        coords, edges = self._coordinates, self._edges_eV
        if coords.shape[1] != 2 or np.any(coords < 0):
            raise ValueError("pixel coordinates must be nonnegative (n_pixel, 2)")
        if edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("measured energy edges must be strictly increasing")
        self._validate_count_channels(coords.shape[0], edges.size - 1)
        if (
            not isinstance(self.components, tuple)
            or not self.components
            or any(not isinstance(item, str) or not item for item in self.components)
        ):
            raise ValueError("components must contain named source components")
        digest = self.observation_digest
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("observation_digest must be a SHA-256 hex digest")

    def _snapshot_arrays(self) -> None:
        """Validate CPU storage types and detach every array before axis checks."""
        specs = (
            ("_coordinates", "iu", 2),
            ("_edges_eV", "f", 1),
            ("_expected", "f", 2),
            ("_realized", "iu", 2),
            ("_underflow_expected", "f", 1),
            ("_overflow_expected", "f", 1),
            ("_below_cut_expected", "f", 1),
            ("_underflow_realized", "iu", 1),
            ("_overflow_realized", "iu", 1),
            ("_below_cut_realized", "iu", 1),
        )
        for name, kinds, ndim in specs:
            value = getattr(self, name)
            if value is None and "realized" in name:
                continue
            if (
                not isinstance(value, np.ndarray)
                or value.ndim != ndim
                or value.dtype.kind not in kinds
            ):
                raise TypeError(f"{name} must be a {ndim}D NumPy array with {kinds} dtype")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            snapshot = value.copy()
            snapshot.setflags(write=False)
            object.__setattr__(self, name, snapshot)

    def _validate_count_channels(self, n_pixel: int, n_bin: int) -> None:
        """Check aligned nonnegative channels and all-or-none realization."""
        names = (
            "_realized",
            "_underflow_realized",
            "_overflow_realized",
            "_below_cut_realized",
        )
        present = [getattr(self, name) is not None for name in names]
        if any(present) and not all(present):
            raise ValueError("realized channels must be all present or absent")
        for name in (
            "_expected",
            "_realized",
            "_underflow_expected",
            "_overflow_expected",
            "_below_cut_expected",
            "_underflow_realized",
            "_overflow_realized",
            "_below_cut_realized",
        ):
            value = getattr(self, name)
            shape = (n_pixel, n_bin) if name in {"_expected", "_realized"} else (n_pixel,)
            if value is not None and (value.shape != shape or np.any(value < 0)):
                raise ValueError(f"{name} must be nonnegative with shape {shape}")

    @property
    def pixel_coordinates(self) -> np.ndarray:
        """New integer (row, column) pixel index array, in acquisition order."""
        return self._coordinates.copy()

    @property
    def measured_energy_edges(self) -> Quantity:
        """N+1 boundaries of measured energy bins, never source nodes."""
        from ..units import Q_

        return Q_(self._edges_eV.copy(), "eV").to(self._energy_unit)

    @property
    def bin_widths(self) -> Quantity:
        """Widths in displayed energy units; N values."""
        from ..units import Q_

        return Q_(np.diff(self._edges_eV), "eV").to(self._energy_unit)

    @staticmethod
    def _quantity(value: np.ndarray) -> Quantity:
        from ..units import Q_

        return Q_(value.copy(), "dimensionless")

    @property
    def expected_counts(self) -> Quantity:
        """Expected response-weighted events over the full exposure."""
        return self._quantity(self._expected)

    @property
    def realized_counts(self) -> Quantity | None:
        """Original integer draws, or None when acquired in expected mode."""
        return None if self._realized is None else self._quantity(self._realized)

    @property
    def counts(self) -> Quantity:
        """Use realized counts if available, else deterministic expected counts."""
        return self._quantity(self._expected if self._realized is None else self._realized)

    @property
    def total_counts(self) -> Quantity:
        """Sum only reporting bins per pixel, excluding all auxiliary channels."""
        from ..units import Q_

        return Q_(np.sum(self.counts.magnitude, axis=-1), "dimensionless")

    def _auxiliary(self, category: str, *, realized: bool | None = None) -> Quantity:
        if realized is None:
            realized = self._realized is not None
        value = getattr(self, f"_{category}_{'realized' if realized else 'expected'}")
        if value is None:
            # Validated batches always contain expected auxiliary channels and
            # either all realized channels or none. A missing selected channel
            # indicates invalid internal state, not an optional read-back.
            raise RuntimeError(f"missing {category} auxiliary count channel")
        return self._quantity(value)

    @property
    def underflow_counts(self) -> Quantity:
        """Events below the reporting energy range, not reported-bin counts."""
        return self._auxiliary("underflow")

    @property
    def overflow_counts(self) -> Quantity:
        """Events above the reporting energy range, not reported-bin counts."""
        return self._auxiliary("overflow")

    @property
    def below_cut_counts(self) -> Quantity:
        """Events inside reporting range but below the acquisition threshold."""
        return self._auxiliary("below_cut")

    @property
    def underflow_expected(self) -> Quantity:
        return self._auxiliary("underflow", realized=False)

    @property
    def overflow_expected(self) -> Quantity:
        return self._auxiliary("overflow", realized=False)

    @property
    def below_cut_expected(self) -> Quantity:
        return self._auxiliary("below_cut", realized=False)

    def to_energy(self, unit: str) -> Self:
        """Convert measured edges and widths; counts and identity do not change."""
        return replace(self, _energy_unit=unit)
