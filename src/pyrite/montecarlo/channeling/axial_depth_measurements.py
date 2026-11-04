"""Compile photon-source comparisons into research depth envelopes (#884).

Input arrays contain photon yields *per incident* in matched, integrated
observable bins, not electron survival probabilities or spectral densities.
Every uncertainty and between-depth allowance supplied by the caller must be
a defensible simultaneous bound. This module validates and combines numbers;
it cannot independently certify measurements or permit production admission.

Validation: cr-axial-depth-budget (unverified; numerical research contract).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np

from .axial_depth_budget import AxialPhotonBoundInterval, AxialPhotonDepthEvidence


@dataclass(frozen=True)
class AxialPhotonPairedProbe:
    """One source perturbation measured against the SAME reference bins.

    ``candidate_yield_bins`` is the perturbed mean photon yield per incident.
    ``difference_stat_error_upper_bins`` bounds the error of the *paired
    difference* (not the error on each mean, nor its SEM). The additional
    ``difference_depth_drift_upper_bins`` must bound any change in the true
    difference away from the measured checkpoint over the entire interval.
    Both bounds must hold simultaneously for every bin. Zero drift is only
    appropriate when a genuine uniform estimate is available.
    """

    candidate_yield_bins: Sequence[float]
    difference_stat_error_upper_bins: Sequence[float]
    difference_depth_drift_upper_bins: Sequence[float]
    evidence_id: str


@dataclass(frozen=True)
class AxialPhotonSourceIntervalMeasurement:
    """Source-probe inputs at one checkpoint, covering a certified depth interval.

    ``reference_stat_error_upper_bins`` and
    ``reference_depth_drift_upper_bins`` are one-sided *lower* allowances for
    the reference yield, each bin-wise and uniform over ``(start, stop]``.

    All five perturbations are required, even when the certified effect is
    zero. ``matching_surface`` is kept separate here but is combined with the
    grid/supercell term in :class:`AxialPhotonBoundInterval` so the existing
    source-depth guard can consume the result unchanged.

    ``simultaneous_confidence`` MUST cover every provided statistical bound,
    bin and component jointly, throughout the interval, with provenance.
    The code cannot convert a bare SEM into such a confidence guarantee.
    """

    start_depth_ang: float
    stop_depth_ang: float
    reference_yield_bins: Sequence[float]
    reference_stat_error_upper_bins: Sequence[float]
    reference_depth_drift_upper_bins: Sequence[float]
    return_flux: AxialPhotonPairedProbe
    unresolved: AxialPhotonPairedProbe
    matching_surface: AxialPhotonPairedProbe
    grid_supercell: AxialPhotonPairedProbe
    source_model: AxialPhotonPairedProbe
    evidence_id: str
    simultaneous_confidence: float


_PROBES = ("return_flux", "unresolved", "matching_surface", "grid_supercell", "source_model")


def _bins(label: str, values: Sequence[float], size: int | None = None) -> np.ndarray:
    """Accept only finite, nonnegative 1D integrated-yield bins."""
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{label}: expected nonempty real numeric 1D photon bins")
    result = raw.astype(float)
    if (size is not None and result.size != size) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label}: photon bin shape mismatch or nonfinite values")
    if np.any(result < 0):
        raise ValueError(f"{label}: photon yields and errors must be nonnegative")
    return result


def _id(label: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} requires nonempty provenance ID")
    return value


def _sum_positive(label: str, values: np.ndarray) -> float:
    result = fsum(float(v) for v in values)
    if not isfinite(result):
        raise ValueError(f"{label}: sum over photon bins overflowed")
    return result


def _probe_upper(probe: AxialPhotonPairedProbe, reference: np.ndarray, label: str) -> float:
    if not isinstance(probe, AxialPhotonPairedProbe):
        raise TypeError(f"{label} requires an AxialPhotonPairedProbe")
    candidate = _bins(f"{label}.candidate_yield_bins", probe.candidate_yield_bins, reference.size)
    stat = _bins(
        f"{label}.difference_stat_error_upper_bins",
        probe.difference_stat_error_upper_bins,
        reference.size,
    )
    drift = _bins(
        f"{label}.difference_depth_drift_upper_bins",
        probe.difference_depth_drift_upper_bins,
        reference.size,
    )
    return _sum_positive(label, np.abs(candidate - reference) + stat + drift)


def source_interval_bounds(
    measurement: AxialPhotonSourceIntervalMeasurement,
) -> AxialPhotonBoundInterval:
    """Compile one interval with conservative per-bin L1 photon error bounds.

    The resulting relative bound concerns the *integrated observable* named
    by the caller. It does not certify each fine bin's relative accuracy; to
    guard a fine bin use a separate observable key and a one-bin measurement.
    No cancellation across spectral/angle bins or perturbed channels is used.
    """
    if not isinstance(measurement, AxialPhotonSourceIntervalMeasurement):
        raise TypeError("expected AxialPhotonSourceIntervalMeasurement")
    # A cumulative photon yield normally tends to zero at the entrance. A
    # positive *uniform* lower bound on (0, z] would then be impossible.
    if not isfinite(measurement.start_depth_ang) or measurement.start_depth_ang <= 0:
        raise ValueError("first source interval must start at positive depth")
    reference = _bins("reference_yield_bins", measurement.reference_yield_bins)
    ref_stat = _bins(
        "reference_stat_error_upper_bins",
        measurement.reference_stat_error_upper_bins,
        reference.size,
    )
    ref_drift = _bins(
        "reference_depth_drift_upper_bins",
        measurement.reference_depth_drift_upper_bins,
        reference.size,
    )
    lower = _sum_positive("reference", reference) - _sum_positive(
        "reference lower allowances", ref_stat + ref_drift
    )
    if not isfinite(lower) or lower <= 0:
        raise ValueError("reference source has no positive lower photon-yield bound")
    _id("interval", measurement.evidence_id)
    ids = [measurement.evidence_id]
    errors = {}
    for name in _PROBES:
        probe = getattr(measurement, name)
        errors[name] = _probe_upper(probe, reference, name)
        ids.append(_id(name, probe.evidence_id))
    if len(set(ids)) != len(ids):
        raise ValueError("probe/interval evidence IDs must be distinct")
    return AxialPhotonBoundInterval(
        start_depth_ang=measurement.start_depth_ang,
        stop_depth_ang=measurement.stop_depth_ang,
        reference_lower=lower,
        return_upper=errors["return_flux"],
        unresolved_upper=errors["unresolved"],
        grid_supercell_upper=errors["matching_surface"] + errors["grid_supercell"],
        source_model_upper=errors["source_model"],
        evidence_ids=tuple(ids),
    )


def compile_axial_photon_depth_evidence(
    *,
    case_key: str,
    observable_key: str,
    model_id: str,
    evidence_id: str,
    measurements: Sequence[AxialPhotonSourceIntervalMeasurement],
) -> AxialPhotonDepthEvidence:
    """Assemble a source-depth envelope with a conservative JOINT confidence.

    For each interval the supplier has already established simultaneous
    confidence over all its bins and components. The union bound combines
    potentially dependent intervals: ``1 - sum(1 - confidence_i)``. This is
    *not* the minimum interval confidence. Intervals must be nonoverlapping,
    ordered and individually uniform; gaps remain gaps. The output is suited
    to :func:`require_axial_photon_depth_budget` and the guarded research
    sampler, but constructing it is NOT validation or native admission.
    """
    if not measurements:
        raise ValueError("no source-depth measurements; axial zone refused")
    intervals = []
    risk = 0.0
    for row in measurements:
        if not isinstance(row, AxialPhotonSourceIntervalMeasurement):
            raise TypeError("measurements require AxialPhotonSourceIntervalMeasurement")
        level = row.simultaneous_confidence
        if not isfinite(level) or not 0 < level < 1:
            raise ValueError("simultaneous_confidence must be in (0,1)")
        risk += 1.0 - level
        intervals.append(source_interval_bounds(row))
    confidence = 1.0 - risk
    if confidence <= 0:
        raise ValueError("joint confidence is not positive across source intervals")
    return AxialPhotonDepthEvidence(
        case_key=case_key,
        observable_key=observable_key,
        model_id=model_id,
        evidence_id=_id("evidence", evidence_id),
        simultaneous_confidence=confidence,
        intervals=tuple(intervals),
    )
