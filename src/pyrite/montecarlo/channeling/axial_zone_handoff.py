"""Guarded axial entrance-zone accounting and correlated *candidate* handoffs.

This is research plumbing for issue #623, NOT a dechanneling model or a
production native transport adapter. In particular, a finite-grid sink is
never counted as physical outgoing flux. Physical matching, return-flux
convergence and source partition must be validated before native admission.

Units: lengths and transport clocks in angstrom (c=1); energy in eV except
candidate native kinetic energy (keV); transverse momenta in inverse angstrom.
Validation scope: cr-axial-zone-handoff-accounting (unverified).
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

_HBARC_EV_ANG = 1973.269804
_M_E_EV = 510998.95


@dataclass(frozen=True)
class AxialFluxAccount:
    """Disjoint, incident-normalized weights at a *physical* matching surface.

    ``unresolved_numerical`` explicitly includes finite-grid sinks, unmeasured
    high-momentum transfers, unmatched cross-cut coherence, etc. It must not be
    labelled physical escape. No component is survivor-normalized.
    """

    retained: float
    physical_outgoing: float
    unresolved_numerical: float
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        values = (self.retained, self.physical_outgoing, self.unresolved_numerical)
        if not (np.isfinite(self.tolerance) and 0 <= self.tolerance <= 1e-4):
            raise ValueError("tolerance must be finite and between zero and 1e-4")
        if any(not np.isfinite(v) or v < 0 for v in values):
            raise ValueError("flux components must be finite and nonnegative")
        if abs(sum(values) - 1.0) > self.tolerance:
            raise ValueError(
                "retained + physical outgoing + unresolved must equal incident weight 1"
            )


@dataclass(frozen=True)
class AxialDepthEvidence:
    """Measured admission envelope, specific to one fully resolved case key.

    A populated record is not itself independent validation. The producer of
    this record must retain the measurement and error-estimation provenance.
    Do not plug this research guard into standard runs until the matching
    surface and source have been independently anchored.
    """

    case_key: str
    evidence_id: str
    max_depth_ang: float
    upper_return_fraction: float
    upper_unresolved_fraction: float
    upper_source_error_fraction: float

    def __post_init__(self) -> None:
        if not isinstance(self.case_key, str) or not self.case_key.strip():
            raise ValueError("nonempty case_key required")
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise ValueError("nonempty evidence_id required")
        if not np.isfinite(self.max_depth_ang) or self.max_depth_ang <= 0:
            raise ValueError("max_depth_ang must be finite and positive")
        for name in (
            "upper_return_fraction",
            "upper_unresolved_fraction",
            "upper_source_error_fraction",
        ):
            v = getattr(self, name)
            if not np.isfinite(v) or not 0 <= v <= 1:
                raise ValueError(f"{name} must be a finite fraction in [0,1]")


def require_axial_depth_budget(
    *,
    case_key: str,
    requested_depth_ang: float,
    evidence: AxialDepthEvidence | None,
    max_error_fraction: float = 0.01,
) -> None:
    """Fail closed before running/saving work unless matching evidence covers it.

    The three error bounds are added conservatively (union/triangle bound),
    not root-sum-squared, and must be based on confidence bounds, not means.
    A missing or unrelated calibration cannot grant admission.
    """
    if not np.isfinite(requested_depth_ang) or requested_depth_ang < 0:
        raise ValueError("requested_depth_ang must be finite and nonnegative")
    if not np.isfinite(max_error_fraction) or not 0 < max_error_fraction < 1:
        raise ValueError("max_error_fraction must be in (0,1)")
    if evidence is None or evidence.case_key != case_key:
        raise ValueError("axial zone refused: no matching calibrated depth evidence")
    if requested_depth_ang > evidence.max_depth_ang:
        raise ValueError("axial zone refused: requested depth exceeds calibrated budget")
    bound = (
        evidence.upper_return_fraction
        + evidence.upper_unresolved_fraction
        + evidence.upper_source_error_fraction
    )
    if bound > max_error_fraction:
        raise ValueError("axial zone refused: combined upper error bound exceeds budget")


@dataclass(frozen=True)
class AxialOutgoingCandidate:
    """Research-only proposed native launch, conditional on physical outgoing.

    This is deliberately NOT a ``transport.secondaries.LaunchState``. No code
    here establishes escape physics, post-jump capture or native eligibility.
    ``stream_key`` is the *unchanged* native primary key supplied by caller.
    ``t_ang`` and ``t0_ang`` use the native transport convention, c=1.
    """

    incident_id: int
    depth_ang: float
    r_ang: np.ndarray
    v_hat: np.ndarray
    E_keV: float
    t_ang: float
    t0_ang: float
    stream_key: int
    state_weight: float
    position_index: tuple[int, int]
    momentum_index: tuple[int, int]


def _finite_vector(name: str, value, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} needs {length} finite components")
    return array


def _positive_integer(name: str, value: int, *, max_value: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} exceeds its native integer range")
    return int(value)


def crystal_to_lab_frame(frame_lab) -> np.ndarray:
    """Validate an orthonormal right-handed crystal-to-lab matrix (columns: crystal x, y, z).

    Validation: cr-axial-zone-handoff-physical
    """
    frame = np.asarray(frame_lab, dtype=float)
    if frame.shape != (3, 3) or not np.all(np.isfinite(frame)):
        raise ValueError("frame_lab needs a finite 3x3 matrix")
    if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-10, rtol=0) or not np.isclose(
        np.linalg.det(frame), 1.0, atol=1e-10, rtol=0
    ):
        raise ValueError("frame_lab must be an orthonormal right-handed crystal-to-lab frame")
    return frame


def momentum_inv_ang(kinetic_energy_eV: float) -> float:
    """On-shell electron momentum ``sqrt(E(E + 2 m c^2)) / (hbar c)`` [Å^-1]."""
    return float(np.sqrt(kinetic_energy_eV * (kinetic_energy_eV + 2 * _M_E_EV)) / _HBARC_EV_ANG)


def on_shell_direction(k_perp_inv_ang, kinetic_energy_eV: float, max_angle_rad: float):
    r"""Crystal-frame unit direction $(k_x/p, k_y/p, \sqrt{1-k_\perp^2/p^2})$.

    Refuses rather than clips a transverse wavevector at or beyond
    ``p sin(max_angle_rad)`` (paraxial/on-shell budget).
    Validation: cr-axial-zone-handoff-physical
    """
    kx, ky = (float(k) for k in k_perp_inv_ang)
    p = momentum_inv_ang(kinetic_energy_eV)
    perpendicular2 = (kx * kx + ky * ky) / (p * p)
    if not np.isfinite(perpendicular2) or perpendicular2 >= np.sin(max_angle_rad) ** 2:
        raise ValueError("sampled transverse momentum violates on-shell/paraxial angle budget")
    return np.array([kx / p, ky / p, np.sqrt(1.0 - perpendicular2)])


def propose_correlated_outgoing(
    state_per_ang: np.ndarray,
    window: np.ndarray,
    *,
    grid,
    seed: int,
    incident_id: int,
    account: AxialFluxAccount,
    depth_ang: float,
    entrance_origin_lab_ang,
    frame_lab,
    transverse_origin_ang=(0.0, 0.0),
    carrier_inv_ang=(0.0, 0.0),
    kinetic_energy_eV: float,
    clock_at_surface_ang: float,
    t0_ang: float,
    native_stream_key: int,
    max_angle_rad: float = 0.1,
) -> AxialOutgoingCandidate:
    """Measure joint (x,y,kx,ky) and map a *provided* outgoing state to lab.

    The caller must independently establish the physical outgoing projector,
    that this state is its output and that ``depth_ang``/clock correspond to
    the matching event. No physical escaped population is inferred from a sink
    or from high-energy occupation. ``grid`` is the existing ``RealSpaceGrid``:
    physical wavefunctions carry Å^-1 units, whereas the periodic Weyl POVM
    uses discrete amplitudes ``state_per_ang * sqrt(grid.cell_area_ang2)``.

    ``frame_lab[:,0:3]`` comprises lab unit vectors for crystal x,y,z. The
    periodic pixel refers to a *chosen unwrapped* transverse origin; no
    periodic-image ambiguity is resolved here. The source remains research-
    only even if this mapping succeeds. Angles beyond ``max_angle_rad`` refuse
    rather than silently using an invalid paraxial continuation.
    """
    from .axial_phase_space import phase_space_stream, sample_phase_space

    _positive_integer("seed", seed)
    particle = _positive_integer("incident_id", incident_id)
    stream = _positive_integer("native_stream_key", native_stream_key, max_value=2**64 - 1)
    numbers = (depth_ang, kinetic_energy_eV, clock_at_surface_ang, t0_ang, max_angle_rad)
    if any(not np.isfinite(n) for n in numbers):
        raise ValueError("depth, energy, clocks and angle limit must be finite")
    if depth_ang < 0 or kinetic_energy_eV <= 0 or not 0 < max_angle_rad < np.pi / 2:
        raise ValueError("depth/energy/maximum angle outside supported range")
    if clock_at_surface_ang < t0_ang:
        raise ValueError("surface clock predates incident clock")
    origin = _finite_vector("entrance_origin_lab_ang", entrance_origin_lab_ang, 3)
    offset = _finite_vector("transverse_origin_ang", transverse_origin_ang, 2)
    carrier = _finite_vector("carrier_inv_ang", carrier_inv_ang, 2)
    frame = crystal_to_lab_frame(frame_lab)
    state = np.asarray(state_per_ang, dtype=complex)
    if state.shape != grid.shape or not np.all(np.isfinite(state)):
        raise ValueError("state_per_ang must be finite and match grid.shape")
    weight = float(np.vdot(state, state).real * grid.cell_area_ang2)
    if weight <= 0 or not np.isclose(weight, account.physical_outgoing, atol=1e-9, rtol=1e-9):
        raise ValueError("outgoing state norm must match the accounted physical outgoing weight")
    # Fail before the random draw for impossible/paraxially invalid directions.
    # Index-dependent checks below remain necessary for a sampled momentum.
    outcome = sample_phase_space(
        state * np.sqrt(grid.cell_area_ang2), window, phase_space_stream(seed, particle)
    )
    ix, iy = outcome.position_index
    jx, jy = outcome.momentum_index
    k_perp = (float(grid.kx[jx] + carrier[0]), float(grid.ky[jy] + carrier[1]))
    direction_crystal = on_shell_direction(k_perp, kinetic_energy_eV, max_angle_rad)
    point_crystal = np.array(
        [float(grid.x[ix]) + offset[0], float(grid.y[iy]) + offset[1], depth_ang]
    )
    return AxialOutgoingCandidate(
        incident_id=particle,
        depth_ang=float(depth_ang),
        r_ang=origin + frame @ point_crystal,
        v_hat=frame @ direction_crystal,
        E_keV=float(kinetic_energy_eV / 1000),
        t_ang=float(clock_at_surface_ang),
        t0_ang=float(t0_ang),
        stream_key=stream,
        state_weight=weight,
        position_index=outcome.position_index,
        momentum_index=outcome.momentum_index,
    )
