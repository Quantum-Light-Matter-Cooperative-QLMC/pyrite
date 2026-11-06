"""Research-only source-error envelopes from the #883 physical outgoing fates (#884).

A bound on a *fraction of electrons* is not a bound on photon yield. This
module converts the #883 terminal fates to incident-normalized photon-source
bounds ONLY when the caller supplies a uniform per-fate, per-observable
response cap. A cap must cover the **difference** between the modeled and
reference photon source caused by that fate, including any return into the
entrance zone. It is not inferred from the outgoing instrument.

The terminal-event CDF is monotone in depth. A one-sided Hoeffding bound at
an interval's STOP therefore bounds the corresponding omitted-fate probability
at *every* depth in (start, stop], without interpolating checkpoint errors.

Validation: cr-axial-depth-budget
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import fsum, isfinite, log, sqrt

import numpy as np

from .axial_depth_budget import AxialPhotonDepthEvidence
from .axial_outgoing_launch import AxialHandoffRecord


@dataclass(frozen=True)
class AxialFatePhotonResponseCaps:
    """Certified externally, worst-case *absolute* photon error per fate.

    Every cap is a vector of **integrated** photon yields per electron in the
    same spectral/angle/polarization observable as the matching source budget.
    It applies uniformly over all covered depths and all relevant exit/return
    histories, not merely the measured sample or an average outgoing electron.

    ``outgoing_return_error_upper_bins`` must include all *additional* photons
    from physical-outgoing particles that return (or have not been accounted
    for) after leaving the quantum subspace. A return probability of unity is
    implicitly allowed: no unmeasured "small backscatter" factor is applied.

    ``unresolved_error_upper_bins`` covers all omitted source effects of an
    unresolved fate. No artificial upper limit of one photon is imposed.

    ``failure_probability_upper`` is a supplied bound on the probability that
    EITHER cap fails, jointly over bins/fates/depths. Zero asserts a rigorous
    deterministic envelope; its proof is external, not established here.
    """

    case_key: str
    observable_key: str
    model_id: str
    evidence_id: str
    outgoing_return_error_upper_bins: Sequence[float]
    unresolved_error_upper_bins: Sequence[float]
    failure_probability_upper: float


@dataclass(frozen=True)
class AxialFateIntervalSourceUpper:
    """Estimated incident fractions and uniform source upper bounds for (a,b]."""

    start_depth_ang: float
    stop_depth_ang: float
    observed_outgoing_fraction: float
    observed_unresolved_fraction: float
    fate_fraction_allowance: float
    return_source_error_upper: float
    unresolved_source_error_upper: float


def _id(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires a nonempty provenance identifier")
    return value


def _total_cap(name: str, values: Sequence[float]) -> float:
    a = np.asarray(values)
    if a.ndim != 1 or a.size == 0 or a.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be nonempty real integrated photon bins")
    v = a.astype(float)
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = fsum(float(x) for x in v)
    if not isfinite(total):
        raise ValueError(f"{name} total overflowed")
    return total


def _terminal_fate_depths(
    records: Sequence[AxialHandoffRecord], sampled_max_depth_ang: float
) -> tuple[list[float], list[float]]:
    """Validate unit-weight records before estimating CDFs; omitted incidents are undetectable."""
    if not records:
        raise ValueError("no independently sampled incident records")
    seen: set[int] = set()
    outgoing, unresolved = [], []
    for record in records:
        if not isinstance(record, AxialHandoffRecord):
            raise TypeError("records must be unweighted AxialHandoffRecord entries")
        if (
            isinstance(record.incident_id, bool)
            or not isinstance(record.incident_id, (int, np.integer))
            or record.incident_id < 0
            or record.incident_id in seen
        ):
            raise ValueError("sampled incident IDs must be unique nonnegative integers")
        seen.add(int(record.incident_id))
        if not isfinite(record.depth_ang) or not 0 <= record.depth_ang <= sampled_max_depth_ang:
            raise ValueError("record depth is outside the sampled zone")
        # A retained record is censored at its observation horizon, unlike
        # an outgoing/unresolved event whose terminal fate is already known.
        if record.fate == "retained" and record.depth_ang != sampled_max_depth_ang:
            raise ValueError("retained records must reach the sampled maximum depth")
        if record.fate == "physical_outgoing":
            outgoing.append(record.depth_ang)
        elif record.fate == "unresolved":
            unresolved.append(record.depth_ang)
        elif record.fate != "retained":
            raise ValueError(f"invalid physical outgoing fate: {record.fate!r}")
    return outgoing, unresolved


def physical_fate_source_uppers(
    records: Sequence[AxialHandoffRecord],
    *,
    intervals: Sequence[tuple[float, float]],
    sampled_max_depth_ang: float,
    caps: AxialFatePhotonResponseCaps,
    sampling_failure_probability: float,
) -> tuple[AxialFateIntervalSourceUpper, ...]:
    """Compute *simultaneous* conservative upper bounds on photon error.

    Assumes independently sampled, identically distributed incident fates at
    a common maximum depth using the #883 physical instrument. The caller
    must retain independent seeds/primary histories and justify any weighting
    or heterogeneity *before* using this unit-weight estimator. Each record
    represents exactly ONE incident; importance-weighted records are refused.

    Hoeffding + Bonferroni covers the two terminal-fate CDFs at every interval
    stop with joint failure probability at most ``sampling_failure_probability``.
    For each interval the CDF upper bound at its stop is uniform over the
    whole interval by monotonicity. This does not give a physical return-flux
    calculation: the cap supplies the worst-case response of outgoing fates. The multiplier
    also assumes no error at depth z for a history without a class event by z.
    This holds for yield at crystal thickness z, not for depth-resolved use
    where a later exit returns above z.

    Validation: cr-axial-depth-budget
    """
    if not isinstance(caps, AxialFatePhotonResponseCaps):
        raise TypeError("explicit AxialFatePhotonResponseCaps are required")
    for name in ("case_key", "observable_key", "model_id", "evidence_id"):
        _id(name, getattr(caps, name))
    if not isfinite(caps.failure_probability_upper) or not 0 <= caps.failure_probability_upper < 1:
        raise ValueError("cap failure_probability_upper must be in [0,1)")
    if not isfinite(sampling_failure_probability) or not 0 < sampling_failure_probability < 1:
        raise ValueError("sampling_failure_probability must be in (0,1)")
    if not isfinite(sampled_max_depth_ang) or sampled_max_depth_ang <= 0:
        raise ValueError("sampled_max_depth_ang must be finite and positive")
    if not intervals:
        raise ValueError("at least one depth interval is required")
    return_cap = _total_cap("outgoing return cap", caps.outgoing_return_error_upper_bins)
    unresolved_cap = _total_cap("unresolved cap", caps.unresolved_error_upper_bins)
    if len(caps.outgoing_return_error_upper_bins) != len(caps.unresolved_error_upper_bins):
        raise ValueError("response caps must have matching observable bins")

    starts_stops = []
    last_stop = 0.0
    for start, stop in intervals:
        if not all(isfinite(x) for x in (start, stop)) or not 0 < start < stop:
            raise ValueError("source intervals require finite 0 < start < stop")
        if start < last_stop or stop > sampled_max_depth_ang:
            raise ValueError("source intervals overlap or exceed sampled depth")
        last_stop = stop
        starts_stops.append((start, stop))

    outgoing, unresolved = _terminal_fate_depths(records, sampled_max_depth_ang)
    outgoing_depths = np.asarray(outgoing)
    unresolved_depths = np.asarray(unresolved)
    # Exactly 2*m one-sided Bernoulli CDF bounds, simultaneously. The same
    # incidents may be reused across depths; the union bound needs no
    # independence between these CDF estimates, only between incidents.
    allowance = sqrt(log(2 * len(intervals) / sampling_failure_probability) / (2 * len(records)))
    result = []
    for start, stop in starts_stops:
        f_out = float(np.count_nonzero(outgoing_depths <= stop)) / len(records)
        f_unresolved = float(np.count_nonzero(unresolved_depths <= stop)) / len(records)
        # Caps are per *missing electron*, not populations. Multiplication
        # is safe only if their uniform response bound holds for EVERY fate.
        return_upper = min(1.0, f_out + allowance) * return_cap
        unresolved_upper = min(1.0, f_unresolved + allowance) * unresolved_cap
        if not isfinite(return_upper) or not isfinite(unresolved_upper):
            raise ValueError("source error upper bound overflowed")
        result.append(
            AxialFateIntervalSourceUpper(
                start, stop, f_out, f_unresolved, allowance, return_upper, unresolved_upper
            )
        )
    return tuple(result)


def add_physical_fate_source_bounds(
    source_evidence: AxialPhotonDepthEvidence,
    records: Sequence[AxialHandoffRecord],
    *,
    sampled_max_depth_ang: float,
    caps: AxialFatePhotonResponseCaps,
    sample_evidence_id: str,
    sampling_failure_probability: float = 0.01,
) -> AxialPhotonDepthEvidence:
    """Add *separate* #883 fate bounds to a supplied #884 source budget.

    Existing paired return/unresolved terms are retained: this is an additive
    conservative supplement, not a replacement or proof that the channels
    are nonoverlapping. To avoid double-counting in a tight budget, upstream
    comparisons must instead define and justify a disjoint decomposition.

    All provided source intervals must match the sampled region. Joint
    confidence is reduced by both the Hoeffding risk and any supplied response
    cap risk. The resulting envelope is uncalibrated and can only be used with
    the research preflight, never as a production admission token.

    Validation: cr-axial-depth-budget
    """
    if not isinstance(source_evidence, AxialPhotonDepthEvidence):
        raise TypeError("source_evidence must be AxialPhotonDepthEvidence")
    if not isinstance(caps, AxialFatePhotonResponseCaps) or (
        source_evidence.case_key != caps.case_key
        or source_evidence.observable_key != caps.observable_key
        or source_evidence.model_id != caps.model_id
    ):
        raise ValueError("outgoing response caps do not match source case/observable/model")
    sample_id = _id("sample_evidence_id", sample_evidence_id)
    if sample_id == caps.evidence_id or sample_id == source_evidence.evidence_id:
        raise ValueError("sample provenance must be distinct from source and cap evidence")
    risk = sampling_failure_probability + caps.failure_probability_upper
    confidence = source_evidence.simultaneous_confidence - risk
    if not 0 < confidence < 1:
        raise ValueError("combined source and outgoing joint confidence is insufficient")
    bounds = physical_fate_source_uppers(
        records,
        intervals=[(r.start_depth_ang, r.stop_depth_ang) for r in source_evidence.intervals],
        sampled_max_depth_ang=sampled_max_depth_ang,
        caps=caps,
        sampling_failure_probability=sampling_failure_probability,
    )
    combined = []
    for interval, extra in zip(source_evidence.intervals, bounds, strict=True):
        new_ids = (*interval.evidence_ids, caps.evidence_id, sample_id)
        if len(set(new_ids)) != len(new_ids):
            raise ValueError("fate provenance must be distinct from each source interval's IDs")
        combined.append(
            replace(
                interval,
                return_upper=interval.return_upper + extra.return_source_error_upper,
                unresolved_upper=interval.unresolved_upper + extra.unresolved_source_error_upper,
                evidence_ids=new_ids,
            )
        )
    return replace(
        source_evidence,
        evidence_id=f"{source_evidence.evidence_id}+{sample_id}+{caps.evidence_id}",
        simultaneous_confidence=confidence,
        intervals=tuple(combined),
    )
