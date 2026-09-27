"""Native measured-bin scoring and coordinate-stable counting acquisitions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.constants import elementary_charge

from ..detectors import Detector, NativeSpectrum
from .observation import REALIZATION_RNG, Acquisition


def _readonly(value: object, *, dtype=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


def _canonical_digest(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def electron_count(
    acquisition: Acquisition,
    *,
    rep_rate_hz: float,
    bunch_charge_pc: float,
) -> float:
    """Return physical incident electrons during an acquisition exposure.

    Validation: pixel-acquisition-counting
    """
    if not isinstance(acquisition, Acquisition):
        raise TypeError("acquisition must be an Acquisition")
    cadence = _nonnegative_real("rep_rate_hz", rep_rate_hz)
    charge = _nonnegative_real("bunch_charge_pc", bunch_charge_pc)
    if cadence == 0.0 or charge == 0.0:
        return 0.0
    return acquisition.exposure_s * cadence * charge * 1.0e-12 / elementary_charge


@dataclass(frozen=True)
class AcquisitionBatch:
    """Expected and optional realized registered counts for selected pixels."""

    coordinates: np.ndarray
    measured_edges_eV: np.ndarray
    expected: np.ndarray
    underflow_expected: np.ndarray
    overflow_expected: np.ndarray
    below_cut_expected: np.ndarray
    realized: np.ndarray | None
    underflow_realized: np.ndarray | None
    overflow_realized: np.ndarray | None
    below_cut_realized: np.ndarray | None
    components: tuple[str, ...]
    observation_digest: str

    def __post_init__(self) -> None:
        coordinates = _readonly(self.coordinates, dtype=np.int64)
        edges = _readonly(self.measured_edges_eV, dtype=float)
        expected = _readonly(self.expected, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
            raise ValueError("AcquisitionBatch.coordinates must have shape (n_pixel, 2)")
        if np.any(coordinates < 0):
            raise ValueError("AcquisitionBatch.coordinates must be nonnegative")
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
            raise ValueError("AcquisitionBatch.measured_edges_eV must be increasing")
        if expected.shape != (coordinates.shape[0], edges.size - 1):
            raise ValueError("AcquisitionBatch.expected must have shape (n_pixel, n_bin)")
        if not np.all(np.isfinite(expected)) or np.any(expected < 0.0):
            raise ValueError("AcquisitionBatch.expected must be finite and nonnegative")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "measured_edges_eV", edges)
        object.__setattr__(self, "expected", expected)
        for name in ("underflow_expected", "overflow_expected", "below_cut_expected"):
            values = _readonly(getattr(self, name), dtype=float)
            if values.shape != (coordinates.shape[0],):
                raise ValueError(f"AcquisitionBatch.{name} must have shape (n_pixel,)")
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"AcquisitionBatch.{name} must be finite and nonnegative")
            object.__setattr__(self, name, values)
        realized_names = (
            "realized",
            "underflow_realized",
            "overflow_realized",
            "below_cut_realized",
        )
        present = tuple(getattr(self, name) is not None for name in realized_names)
        if any(present) and not all(present):
            raise ValueError("AcquisitionBatch realized channels must be all present or all absent")
        if all(present):
            realized = _readonly(self.realized, dtype=np.int64)
            if realized.shape != expected.shape or np.any(realized < 0):
                raise ValueError("AcquisitionBatch.realized must be nonnegative and match expected")
            object.__setattr__(self, "realized", realized)
            for name in realized_names[1:]:
                values = _readonly(getattr(self, name), dtype=np.int64)
                if values.shape != (coordinates.shape[0],) or np.any(values < 0):
                    raise ValueError(f"AcquisitionBatch.{name} must be nonnegative per pixel")
                object.__setattr__(self, name, values)
        components = tuple(sorted(self.components))
        if not components or any(not component for component in components):
            raise ValueError("AcquisitionBatch.components must contain nonempty names")
        if len(set(components)) != len(components):
            raise ValueError("AcquisitionBatch.components must be unique")
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self,
            "observation_digest",
            _canonical_digest("observation_digest", self.observation_digest),
        )

    @property
    def counts(self) -> np.ndarray:
        """Registered realized counts, or deterministic expectations in expected mode."""
        return self.expected if self.realized is None else self.realized

    @property
    def total_counts(self) -> np.ndarray:
        """Sum registered reporting bins; never draw an independent total."""
        return np.sum(self.counts, axis=-1)

    def window_counts(self, energy_range_eV: tuple[float, float]) -> np.ndarray:
        """Sum complete half-open reporting bins between two configured edges."""
        low, high = map(float, energy_range_eV)
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("energy_range_eV must be a finite increasing pair")
        low_matches = np.flatnonzero(np.isclose(self.measured_edges_eV, low, rtol=0.0, atol=1e-12))
        high_matches = np.flatnonzero(
            np.isclose(self.measured_edges_eV, high, rtol=0.0, atol=1e-12)
        )
        if low_matches.size != 1 or high_matches.size != 1:
            raise ValueError("energy window bounds must match acquisition reporting edges")
        start, stop = int(low_matches[0]), int(high_matches[0])
        if start >= stop:
            raise ValueError("energy_range_eV must select at least one reporting bin")
        return np.sum(self.counts[:, start:stop], axis=-1)


def _partition_native(
    native: NativeSpectrum,
    acquisition: Acquisition,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_left = native.edges_eV[:-1]
    source_right = native.edges_eV[1:]
    widths = source_right - source_left
    target = np.asarray(acquisition.measured_edges_eV)
    threshold = acquisition.hit_threshold_eV

    registered_left = np.maximum(target[:-1], threshold)
    overlap = np.maximum(
        0.0,
        np.minimum(source_right[:, None], target[1:])
        - np.maximum(source_left[:, None], registered_left),
    )
    registered_fraction = overlap / widths[:, None]
    underflow_fraction = (
        np.maximum(
            0.0,
            np.minimum(source_right, target[0]) - source_left,
        )
        / widths
    )
    overflow_fraction = (
        np.maximum(
            0.0,
            source_right - np.maximum(source_left, target[-1]),
        )
        / widths
    )
    below_cut_fraction = (
        np.maximum(
            0.0,
            np.minimum(source_right, min(threshold, target[-1]))
            - np.maximum(source_left, target[0]),
        )
        / widths
    )

    registered = native.events @ registered_fraction
    underflow = native.events @ underflow_fraction
    overflow = native.events @ overflow_fraction
    below_cut = native.events @ below_cut_fraction
    return registered, underflow, overflow, below_cut


def _coordinate_rng(
    observation_digest: str,
    seed: int,
    row: int,
    column: int,
    component: str,
) -> np.random.Generator:
    payload = json.dumps(
        [REALIZATION_RNG, observation_digest, int(seed), int(row), int(column), component],
        separators=(",", ":"),
    ).encode()
    stream_seed = int.from_bytes(hashlib.sha256(payload).digest(), "little")
    return np.random.Generator(np.random.Philox(stream_seed))


def score_acquisition(
    detector: Detector,
    energy_eV: np.ndarray,
    accepted_density: np.ndarray,
    acquisition: Acquisition,
    *,
    rep_rate_hz: float,
    bunch_charge_pc: float,
    observation_digest: str,
    coordinates: np.ndarray,
    component: str,
) -> AcquisitionBatch:
    """Score a selected pixel batch into expected or coordinate-stable counts."""
    if not isinstance(detector, Detector):
        raise TypeError("detector must be a Detector")
    if not isinstance(acquisition, Acquisition):
        raise TypeError("acquisition must be an Acquisition")
    digest = _canonical_digest("observation_digest", observation_digest)
    if not isinstance(component, str) or not component:
        raise ValueError("component must be a nonempty string")
    raw_coordinates = np.asarray(coordinates)
    if (
        raw_coordinates.ndim != 2
        or raw_coordinates.shape[1:] != (2,)
        or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in raw_coordinates.ravel()
        )
    ):
        raise ValueError("coordinates must have shape (n_pixel, 2) with integer values")
    resolved_coordinates = raw_coordinates.astype(np.int64)
    if np.any(resolved_coordinates < 0):
        raise ValueError("coordinates must be nonnegative")
    density = np.asarray(accepted_density, dtype=float)
    if density.ndim != 2 or density.shape[0] != resolved_coordinates.shape[0]:
        raise ValueError("accepted_density must have shape (n_pixel, n_energy)")

    native = detector.native_score(np.asarray(energy_eV), density)
    registered, underflow, overflow, below_cut = _partition_native(native, acquisition)
    scale = electron_count(
        acquisition,
        rep_rate_hz=rep_rate_hz,
        bunch_charge_pc=bunch_charge_pc,
    )
    expected = np.clip(registered * scale, 0.0, None)
    underflow_expected = np.clip(underflow * scale, 0.0, None)
    overflow_expected = np.clip(overflow * scale, 0.0, None)
    below_cut_expected = np.clip(below_cut * scale, 0.0, None)

    realized = underflow_realized = overflow_realized = below_cut_realized = None
    if acquisition.mode == "poisson":
        assert acquisition.seed is not None
        realized = np.empty(expected.shape, dtype=np.int64)
        underflow_realized = np.empty(expected.shape[0], dtype=np.int64)
        overflow_realized = np.empty(expected.shape[0], dtype=np.int64)
        below_cut_realized = np.empty(expected.shape[0], dtype=np.int64)
        for index, (row, column) in enumerate(resolved_coordinates.tolist()):
            means = np.concatenate(
                (
                    [
                        underflow_expected[index],
                        overflow_expected[index],
                        below_cut_expected[index],
                    ],
                    expected[index],
                )
            )
            draw = _coordinate_rng(digest, acquisition.seed, row, column, component).poisson(means)
            underflow_realized[index] = draw[0]
            overflow_realized[index] = draw[1]
            below_cut_realized[index] = draw[2]
            realized[index] = draw[3:]

    return AcquisitionBatch(
        coordinates=resolved_coordinates,
        measured_edges_eV=np.asarray(acquisition.measured_edges_eV),
        expected=expected,
        underflow_expected=underflow_expected,
        overflow_expected=overflow_expected,
        below_cut_expected=below_cut_expected,
        realized=realized,
        underflow_realized=underflow_realized,
        overflow_realized=overflow_realized,
        below_cut_realized=below_cut_realized,
        components=(component,),
        observation_digest=digest,
    )


def combine_acquisitions(batches: Sequence[AcquisitionBatch]) -> AcquisitionBatch:
    """Add independently scored component batches without redrawing totals."""
    resolved = tuple(batches)
    if not resolved:
        raise ValueError("combine_acquisitions requires at least one batch")
    first = resolved[0]
    if any(not isinstance(batch, AcquisitionBatch) for batch in resolved):
        raise TypeError("combine_acquisitions accepts AcquisitionBatch objects")
    for batch in resolved[1:]:
        if not np.array_equal(batch.coordinates, first.coordinates):
            raise ValueError("acquisition coordinates must match")
        if not np.array_equal(batch.measured_edges_eV, first.measured_edges_eV):
            raise ValueError("acquisition reporting edges must match")
        if batch.observation_digest != first.observation_digest:
            raise ValueError("acquisition observation digests must match")
        if (batch.realized is None) != (first.realized is None):
            raise ValueError("acquisition realization modes must match")
    components = tuple(component for batch in resolved for component in batch.components)
    if len(set(components)) != len(components):
        raise ValueError("acquisition components must not overlap")

    def summed(name: str):
        values = [getattr(batch, name) for batch in resolved]
        if values[0] is None:
            return None
        return np.sum(np.stack(values), axis=0)

    return AcquisitionBatch(
        coordinates=first.coordinates,
        measured_edges_eV=first.measured_edges_eV,
        expected=summed("expected"),
        underflow_expected=summed("underflow_expected"),
        overflow_expected=summed("overflow_expected"),
        below_cut_expected=summed("below_cut_expected"),
        realized=summed("realized"),
        underflow_realized=summed("underflow_realized"),
        overflow_realized=summed("overflow_realized"),
        below_cut_realized=summed("below_cut_realized"),
        components=components,
        observation_digest=first.observation_digest,
    )


__all__ = [
    "AcquisitionBatch",
    "REALIZATION_RNG",
    "combine_acquisitions",
    "electron_count",
    "score_acquisition",
]
