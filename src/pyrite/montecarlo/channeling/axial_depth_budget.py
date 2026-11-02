"""Research-only source-normalized depth budgets for axial handoff (#884).

These are *supplied* uniform confidence bounds, not a method for deriving
physical return flux, photon source, or grid errors from missing particles.
No constructor here certifies a calibration or enables production axial runs.

Photon yields and absolute error bounds are photons / incident electron in one
fully specified spectral, angular and polarization observable. Depth is Å.
The bounds must hold simultaneously over each declared depth interval.

Validation: cr-axial-depth-budget (unverified; contract tests only).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from .axial_outgoing_launch import (
        AxialHandoffRecord,
        AxialIncident,
        AxialMatchingDeclaration,
        AxialZoneKinematics,
    )
    from .axial_fiber_unraveling import FiberChannels


def _nonnegative_finite(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class AxialPhotonBoundInterval:
    """Simultaneous *uniform* bounds on a depth interval ``(start, stop]``.

    ``reference_lower`` lower-bounds the incident-normalized reference photon
    yield. Each ``*_upper`` upper-bounds the *absolute* error in that same
    observable in photons per incident, not a probability of a lost electron.
    The four budgets must cover disjoint error sources; overlapping effects
    cannot be made independent merely by assigning separate labels.

    Every bound must be valid at **every depth** in this interval (not only
    at sampled endpoints). An observed mean or SEM is not a confidence bound.
    """

    start_depth_ang: float
    stop_depth_ang: float
    reference_lower: float
    return_upper: float
    unresolved_upper: float
    grid_supercell_upper: float
    source_model_upper: float
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "start_depth_ang",
            "stop_depth_ang",
            "reference_lower",
            "return_upper",
            "unresolved_upper",
            "grid_supercell_upper",
            "source_model_upper",
        ):
            _nonnegative_finite(name, getattr(self, name))
        if self.stop_depth_ang <= self.start_depth_ang:
            raise ValueError("stop_depth_ang must be greater than start_depth_ang")
        if self.reference_lower <= 0:
            raise ValueError("positive reference_lower required for a relative source error")
        if not self.evidence_ids or any(
            not isinstance(ref, str) or not ref.strip() for ref in self.evidence_ids
        ):
            raise ValueError("every bound interval needs nonempty evidence_ids")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("duplicate evidence_ids")

    @property
    def absolute_error_upper(self) -> float:
        """Conservative bound, adding all source-error channels linearly."""
        return (
            self.return_upper
            + self.unresolved_upper
            + self.grid_supercell_upper
            + self.source_model_upper
        )

    @property
    def relative_error_upper(self) -> float:
        """Error bound relative to the *lower* reference source yield."""
        return self.absolute_error_upper / self.reference_lower


@dataclass(frozen=True)
class AxialPhotonDepthEvidence:
    """Research-only interval envelope for one resolved case and observable.

    ``case_key`` includes material/axis, energy, incident distribution,
    matching cut, grid, supercell, scattering and solver identity. The
    ``observable_key`` fixes photon energy/angle/polarization bins and the
    radiative partition. ``model_id`` identifies the source implementation.

    Confidence is a **joint** coverage level over every depth and component,
    not a per-measurement confidence level. The producer is responsible for
    independent source-model closure and for retaining the underlying data.
    A constructed instance is never an independent physics sign-off.
    """

    case_key: str
    observable_key: str
    model_id: str
    evidence_id: str
    simultaneous_confidence: float
    intervals: tuple[AxialPhotonBoundInterval, ...]

    def __post_init__(self) -> None:
        for name in ("case_key", "observable_key", "model_id", "evidence_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty")
        if not isfinite(self.simultaneous_confidence) or not 0 < self.simultaneous_confidence < 1:
            raise ValueError("simultaneous_confidence must be in (0,1)")
        if not self.intervals:
            raise ValueError("at least one certified depth interval is required")
        last_stop = -1.0
        for interval in self.intervals:
            if not isinstance(interval, AxialPhotonBoundInterval):
                raise TypeError("intervals must contain AxialPhotonBoundInterval")
            if interval.start_depth_ang < last_stop:
                raise ValueError("depth intervals must be sorted and nonoverlapping")
            last_stop = interval.stop_depth_ang


def require_axial_photon_depth_budget(
    *,
    case_key: str,
    observable_key: str,
    model_id: str,
    requested_depth_ang: float,
    evidence: AxialPhotonDepthEvidence | None,
    max_relative_error: float = 0.01,
    min_confidence: float = 0.95,
) -> AxialPhotonBoundInterval:
    """Refuse missing, mismatched, extrapolated, or source-inaccurate requests.

    This checks a supplied *uniform* envelope; it does not infer continuity,
    monotonicity or bounds between measured depths. Intervals use ``(a,b]``:
    zero-depth photon yield has no useful relative normalization. Returns the
    matched interval for audit, without granting production eligibility.

    Validation: cr-axial-depth-budget (unverified).
    """
    if not isfinite(requested_depth_ang) or requested_depth_ang <= 0:
        raise ValueError("requested_depth_ang must be finite and positive")
    if not isfinite(max_relative_error) or not 0 < max_relative_error < 1:
        raise ValueError("max_relative_error must be in (0,1)")
    if not isfinite(min_confidence) or not 0 < min_confidence < 1:
        raise ValueError("min_confidence must be in (0,1)")
    if evidence is None or (
        evidence.case_key != case_key
        or evidence.observable_key != observable_key
        or evidence.model_id != model_id
    ):
        raise ValueError("axial zone refused: no matching source-depth calibration")
    if evidence.simultaneous_confidence < min_confidence:
        raise ValueError("axial zone refused: insufficient joint confidence")
    for interval in evidence.intervals:
        if interval.start_depth_ang < requested_depth_ang <= interval.stop_depth_ang:
            if interval.relative_error_upper > max_relative_error:
                raise ValueError("axial zone refused: photon-source error exceeds budget")
            return interval
    raise ValueError("axial zone refused: depth is outside certified source intervals")


def sample_budgeted_physical_handoff(
    channels: FiberChannels,
    entry_orders: np.ndarray,
    *,
    evidence: AxialPhotonDepthEvidence | None,
    case_key: str,
    observable_key: str,
    model_id: str,
    hbar_v_eV_ang: float,
    window: np.ndarray,
    max_depth_ang: float,
    incidents: Sequence[AxialIncident],
    zone: AxialZoneKinematics,
    matching: AxialMatchingDeclaration,
    max_relative_error: float = 0.01,
    min_confidence: float = 0.95,
    carrier_inv_ang: tuple[float, float] = (0.0, 0.0),
    seed: int = 0,
    search_step_ang: float = 100.0,
    depth_tolerance_ang: float = 1e-6,
) -> list[AxialHandoffRecord]:
    """Research handoff with depth/source preflight *before* solver creation.

    This opt-in adapter does not modify or admit the unbudgeted research
    instrument and must not be called a production transport handoff. Native
    Scene/Sweep/CLI runs remain refused under #624/#886.
    """
    require_axial_photon_depth_budget(
        case_key=case_key,
        observable_key=observable_key,
        model_id=model_id,
        requested_depth_ang=max_depth_ang,
        evidence=evidence,
        max_relative_error=max_relative_error,
        min_confidence=min_confidence,
    )
    from .axial_outgoing_launch import sample_physical_handoff

    return sample_physical_handoff(
        channels,
        entry_orders,
        carrier_inv_ang=carrier_inv_ang,
        hbar_v_eV_ang=hbar_v_eV_ang,
        window=window,
        max_depth_ang=max_depth_ang,
        incidents=incidents,
        zone=zone,
        matching=matching,
        seed=seed,
        search_step_ang=search_step_ang,
        depth_tolerance_ang=depth_tolerance_ang,
    )
