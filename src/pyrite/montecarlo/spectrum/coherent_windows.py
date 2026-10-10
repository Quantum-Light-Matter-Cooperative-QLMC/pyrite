"""Coherent line-axis windows: where coherent power lives, at the fringe step (#350).

Automatic line-grid resolution refuses ``coherent_emission`` on a uniform axis
(#117): the coherent route's fringes follow the total retardation span of a
row, which over a keV band costs ~1e7 points. This module seeds piecewise
windows instead, one set per ``(reflection, orientation)`` row, so the fine
step is spent only where the row's coherent power is.

Three rules, derived in
``docs/validation/beam-transport/coherent-line-grid-windowed-resolution.md``:

The envelope bounds the captured frozen-carrier field, including affine
absorption, and a dispersive envelope bounds the production field with the
material phase: at fixed energy that phase is affine along each piece, so the
field is exactly its endpoint terms on shifted carriers. A window widens
until both bounds meet the leak share. Physical finite-band normalization and
certified sampling remain open.

envelope
    A row's coherent window is its resonance band ``[min E_res, max E_res]``
    widened on each side until the row's power beyond the window edge is at
    most ``leak_limit`` of the row's per-electron power. Along one electron the
    field's phase ``omega d - g.r`` is continuous across segment joints, so the
    out-of-band spectrum is that of endpoint terms at joints and track
    ends, with each piece's signed attenuation slope retained in its complex
    denominator, not of every segment's own edges separately. Amplitudes
    are the reducer's own per-piece coefficients, captured from the case's
    coherent reduction. Jumps closer in ``tau`` than ``M hbar c / u`` are
    bounded together by the triangle inequality; separated clusters add with
    a cross-term allowance. The inter-electron term adds at most
    ``F (N_e - 1)`` times the floor's tail (Cauchy-Schwarz).

step
    Inside a window the Nyquist step of a row is ``pi hbar c / D`` over the
    time support of every radiating piece of that row (a cross term between a
    strong in-window field and any far tail is bounded only by the square root
    of the tail's power, so no piece is dropped from the span). The
    per-electron span ``D_e`` bounds the ``sum_e |S_e|^2`` floor and the
    all-electron span the ``|sum_e S_e|^2`` term.

decoherence
    The inter-electron term is ``F |sum_e S_e|^2 <= F N_e sum_e |S_e|^2``
    pointwise, so a 100 eV bin whose largest ``F`` satisfies
    ``F N_e <= decoherence_limit`` takes the per-electron step; any other bin,
    and every bin whose ``F`` has no closed-form bound (empirical
    characteristic function, or no offsets at all, ``F = 1``), takes the
    all-electron step.

Validation: coherent-line-grid-windowed-resolution
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import numpy as np

from ..._line_grid_policy import (
    COHERENT_JUMP_CLUSTER_SEPARATION,
    COHERENT_NYQUIST_OVERSAMPLING,
    COHERENT_WINDOW_BIN_EV,
    DEFAULT_COHERENT_DECOHERENCE_LIMIT,
    DEFAULT_COHERENT_LEAK_LIMIT,
)
from ..._line_windows import FeatureSeed
from ...materials.crystal import HBARC_EV_ANG
from ..transport.kinematics import C_ANG_PER_FS
from .coherent_dispersion import CoherentDispersionLaw, DispersionBands
from .coherent_population import pair_scale
from .diagnostics import _offset_population

__all__ = [
    "COHERENT_SOURCE",
    "CoherentRowCollector",
    "CoherentRowField",
    "coherent_case_seeds",
    "coherent_edge_leak",
    "coherent_window_seeds",
    "decoherence_bound",
]

COHERENT_SOURCE = "pxr-coherent"
#: Bisection steps for each window edge; the edge is then fixed to well below
#: one eV over any production bandwidth.
_EDGE_BISECTIONS = 32
#: Carrier-distance ratio between candidate near/far splits of an omitted band.
_SPLIT_RATIO = 2.0**0.25


def decoherence_bound(
    segments: Mapping[str, Any], *, electron_limit: int | None, longitudinal_rms_fs: float | None
) -> Callable[[np.ndarray], np.ndarray] | None:
    """Upper bound on the reducer's inter-electron factor ``F(E)``, or ``None``.

    ``None`` means no bound below one: either no offsets (decoherence inactive,
    the reducer squares the all-electron sum, ``F = 1``) or an infinite slab,
    whose empirical characteristic function is not bounded in closed form. A
    finite footprint uses the analytic ``F_z = exp(-(omega sigma_z)^2)`` the
    reducer applies, decreasing in ``E``; without ``longitudinal_rms_fs`` the
    reducer refuses that case, so no bound is claimed here either.

    Validation: coherent-line-grid-windowed-resolution
    """
    if _offset_population(segments, electron_limit) is None:
        return None
    finite_footprint = (
        segments.get("crystal_width_ang") is not None
        and segments.get("crystal_height_ang") is not None
    )
    if not finite_footprint or longitudinal_rms_fs is None:
        return None
    sigma_z = float(longitudinal_rms_fs) * C_ANG_PER_FS

    def bound(energy_eV):
        omega = np.asarray(energy_eV, dtype=float) / HBARC_EV_ANG
        return np.exp(-((omega * sigma_z) ** 2))

    return bound


@dataclass(frozen=True, slots=True)
class CoherentRowField:
    """One ``(reflection, orientation)`` row's pieces as the reducer phases them.

    ``amplitude`` is the time-domain field amplitude ``coef / dd`` per
    polarization (``coef = sqrt(alpha omega/(4 pi^2 hbar c)) t_L (A_PXR +
    A_CBS)``, the reducer's own coefficient), ``duration_ang`` the support
    ``dd = 2 hbar c a_vac`` and ``centre_ang`` the reducer-geometry ``d``.
    ``start_transmission``/``end_transmission`` are ``exp(-tau/2)`` at the piece
    ends and ``mean_transmission`` the segment mean of ``exp(-tau)``.
    ``attenuation_slope_ang`` is the signed slope of amplitude optical depth,
    ``(tau_end - tau_start)/(2 dd)``, retained before endpoint underflow.
    ``escape_mid_ang`` and ``escape_change_ang`` retain the midpoint distance
    and signed endpoint change for the full dispersive formation phase.
    ``phase_rad`` is the captured midpoint susceptibility phase ``g.r``.
    Without it this representation supplies phase-continuous track bounds,
    but cannot reproduce an absolute complex row field.
    ``mosaic_weight`` is the nonnegative production intensity weight for
    this orientation; it is not absorbed into the captured field amplitudes.
    ``transverse_envelope`` is the row's nonincreasing bound on the analytic
    transverse form factor (finite footprint with a recorded spot), or
    ``None`` (factor one). ``centre_scale_ang`` is each piece's roundoff
    scale: the magnitude of the absolute transport time and position that
    ``centre_ang`` was derived from (``None`` uses ``|centre_ang|``). Joint
    tests scale float64 roundoff by it, so subtracting a per-electron offset
    cannot shrink the tolerance below the error already in the operands.

    Validation: coherent-line-grid-windowed-resolution
    """

    label: str
    energy_eV: np.ndarray
    amplitude: np.ndarray
    duration_ang: np.ndarray
    centre_ang: np.ndarray
    electron: np.ndarray
    start_transmission: np.ndarray
    end_transmission: np.ndarray
    mean_transmission: np.ndarray
    attenuation_slope_ang: np.ndarray | None = None
    escape_mid_ang: np.ndarray | None = None
    escape_change_ang: np.ndarray | None = None
    phase_rad: np.ndarray | None = None
    mosaic_weight: float = 1.0
    transverse_envelope: Callable[[np.ndarray], np.ndarray] | None = None
    centre_scale_ang: np.ndarray | None = None

    @property
    def slope_ang(self) -> np.ndarray:
        """Signed amplitude attenuation slope; capture retains it before underflow."""
        if self.attenuation_slope_ang is not None:
            return self.attenuation_slope_ang
        if np.any(self.start_transmission <= 0) or np.any(self.end_transmission <= 0):
            raise ValueError("zero endpoint transmission requires an explicit attenuation slope")
        return (np.log(self.start_transmission) - np.log(self.end_transmission)) / self.duration_ang

    @property
    def power(self) -> float:
        """Per-electron power over ``2 pi``, by Parseval (disjoint supports)."""
        weight = np.sum(np.abs(self.amplitude) ** 2, axis=0)
        return float(np.sum(weight * self.duration_ang * self.mean_transmission))


class CoherentRowCollector:
    """``coefficient_capture`` hook of ``mc_spectrum``: records every row.

    Validation: coherent-line-grid-windowed-resolution
    """

    def __init__(self):
        self.rows: list[CoherentRowField] = []

    def __call__(self, st, idx, coefs, good, lines):
        sel = _host(good).astype(bool)
        if not sel.any():
            return
        rows = _host(idx)[sel]
        E_vac, a_vac, half_dL, apb, _bma, q = (
            np.asarray(_host(array), dtype=float)[sel] for array in lines
        )
        duration = 2.0 * HBARC_EV_ANG * a_vac
        amplitude = np.stack([np.asarray(_host(c), dtype=complex)[sel] / duration for c in coefs])
        # Recover each endpoint without subtracting nearly equal apb/bma.
        # q = (tau_end - tau_start)/4; the amplitude ratio is exp(-2q).
        ratio = np.exp(-2.0 * np.abs(q))
        bright = apb / (1.0 + ratio)
        start = np.where(q >= 0.0, bright, bright * ratio)
        end = np.where(q >= 0.0, bright * ratio, bright)
        magnitude = np.abs(q)
        safe = np.where(magnitude == 0.0, 1.0, magnitude)
        mean = bright**2 * np.where(
            magnitude == 0.0, 1.0, -np.expm1(-4.0 * magnitude) / (4.0 * safe)
        )
        phase = getattr(st, "capture_phase_rad", None)
        self.rows.append(
            CoherentRowField(
                label=f"row {len(self.rows)}",
                energy_eV=E_vac,
                amplitude=amplitude,
                duration_ang=duration,
                centre_ang=np.asarray(_host(st.d_all_geom), dtype=float)[rows],
                electron=np.asarray(_host(st.seg_elec_id))[rows],
                start_transmission=start,
                end_transmission=end,
                mean_transmission=mean,
                attenuation_slope_ang=2.0 * q / duration,
                escape_mid_ang=0.5
                * sum(np.asarray(_host(a), dtype=float)[rows] for a in st.escape_ends),
                escape_change_ang=2.0 * half_dL,
                phase_rad=None if phase is None else np.asarray(_host(phase), dtype=float)[sel],
                mosaic_weight=float(getattr(st, "capture_mosaic_weight", 1.0)),
                transverse_envelope=getattr(st, "capture_transverse_envelope", None),
                centre_scale_ang=_centre_scale(st, rows),
            )
        )


def _centre_scale(st, rows):
    """``|d| + 2 |r|`` of the complete transport phase, bounding ``|t| + |n_hat . r|``.

    ``d = t - n_hat . r`` (with ``|n_hat| = 1``) gives ``|t| <= |d| + |r|``.
    Validation: coherent-line-grid-windowed-resolution
    """
    d_all = getattr(st, "d_all", None)
    if d_all is None:
        return None
    d = np.abs(np.asarray(_host(d_all), dtype=float)[rows])
    r = np.linalg.norm(np.asarray(_host(st.seg_r), dtype=float)[rows], axis=1)
    return d + 2.0 * r


def _joint_scale(field):
    if field.centre_scale_ang is None:
        return np.abs(field.centre_ang)
    return np.maximum(field.centre_scale_ang, np.abs(field.centre_ang))


def _joint_tolerance(field, order):
    """Float64 roundoff allowed between abutting pieces sorted by ``order``."""
    scale = _joint_scale(field)[order]
    duration = field.duration_ang[order]
    return (
        8.0 * np.finfo(float).eps * (scale[1:] + scale[:-1] + 0.5 * (duration[1:] + duration[:-1]))
    )


def _affine_dispersion_row(
    field: CoherentRowField, slope: float, intercept: float = 0.0
) -> CoherentRowField:
    """Exact Fourier row for ``delta_omega(E) = slope * E + intercept``.

    The variable is ``y = d - hbar c slope L(d)``. Its Jacobian changes the
    duration, carrier, field amplitude, and attenuation slope together.
    ``slope`` has units 1/(Ang eV), ``intercept`` 1/Ang. The map must preserve
    order and disjoint support within each electron. This is an analytic
    building block, not a fit or a certificate for a tabulated material law.

    Validation: coherent-line-grid-windowed-resolution
    """
    if field.escape_mid_ang is None or field.escape_change_ang is None:
        raise ValueError("affine dispersion requires captured escape-path geometry")
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError("affine dispersion coefficients must be finite")
    if np.any(~np.isfinite(field.escape_mid_ang)) or np.any(~np.isfinite(field.escape_change_ang)):
        raise ValueError("affine dispersion requires finite escape-path geometry")
    gradient = field.escape_change_ang / field.duration_ang
    jacobian = 1.0 - HBARC_EV_ANG * slope * gradient
    if np.any(~np.isfinite(jacobian)) or np.any(jacobian <= 0.0):
        raise ValueError("affine dispersion must preserve positive piece durations")
    mapped = replace(
        field,
        energy_eV=(field.energy_eV + HBARC_EV_ANG * intercept * gradient) / jacobian,
        amplitude=field.amplitude / jacobian,
        duration_ang=field.duration_ang * jacobian,
        centre_ang=field.centre_ang - HBARC_EV_ANG * slope * field.escape_mid_ang,
        centre_scale_ang=None
        if field.centre_scale_ang is None
        else field.centre_scale_ang + np.abs(HBARC_EV_ANG * slope * field.escape_mid_ang),
        attenuation_slope_ang=field.slope_ang / jacobian,
        phase_rad=None
        if field.phase_rad is None
        else field.phase_rad + intercept * field.escape_mid_ang,
    )
    lo = mapped.centre_ang - 0.5 * mapped.duration_ang
    hi = mapped.centre_ang + 0.5 * mapped.duration_ang
    original_lo = field.centre_ang - 0.5 * field.duration_ang
    original_hi = field.centre_ang + 0.5 * field.duration_ang
    order = np.lexsort((original_lo, field.electron))
    same = mapped.electron[order][1:] == mapped.electron[order][:-1]
    tolerance = _joint_tolerance(mapped, order)
    if np.any(same & (lo[order][1:] < hi[order][:-1] - tolerance)):
        raise ValueError("affine dispersion maps one electron's pieces onto overlapping support")
    original_tolerance = _joint_tolerance(field, order)
    gap = original_lo[order][1:] > original_hi[order][:-1] + original_tolerance
    if np.any(same & gap & (lo[order][1:] <= hi[order][:-1] + tolerance)):
        raise ValueError("affine dispersion must not close a genuine gap into a cancelling joint")
    return mapped


def _piece_l1_amplitudes(field: CoherentRowField) -> np.ndarray:
    """Per-piece integral of the absolute field, stable under strong damping."""
    q_abs = 0.5 * np.abs(field.slope_ang) * field.duration_ang
    safe = np.where(q_abs == 0.0, 1.0, q_abs)
    bright = np.maximum(field.start_transmission, field.end_transmission)
    mean = bright * np.where(q_abs == 0.0, 1.0, -np.expm1(-2.0 * q_abs) / (2.0 * safe))
    return np.abs(field.amplitude) * (field.duration_ang * mean)


def _electron_l1_power(piece: np.ndarray, electron: np.ndarray) -> float:
    """Triangle inequality inside each electron, independent sum across electrons."""
    if electron.size == 0:
        return 0.0
    _, inverse = np.unique(electron, return_inverse=True)
    grouped = np.zeros((piece.shape[0], int(inverse.max()) + 1))
    for polarization in range(piece.shape[0]):
        np.add.at(grouped[polarization], inverse, piece[polarization])
    return float(np.sum(grouped**2))


def _dispersion_residual_power_bound(
    field: CoherentRowField, residual_max_inv_ang: float, bandwidth_eV: float
) -> float:
    """Bound ``integral sum_e,p |S_actual - S_affine|**2 dE`` on a finite band.

    The caller must supply a certified uniform bound on
    ``|delta_omega(E) - (slope * E + intercept)|`` over the entire band.
    Endpoint samples or a fitted residual do not supply that certificate.
    This absolute bound uses original piece geometry and field amplitudes;
    it includes both midpoint and intra-piece escape phases. Divide by
    ``2 pi hbar c * mapped.power`` for a relative affine Parseval charge.

    Validation: coherent-line-grid-windowed-resolution
    """
    if field.escape_mid_ang is None or field.escape_change_ang is None:
        raise ValueError("a dispersion residual bound requires captured escape-path geometry")
    residual, width = float(residual_max_inv_ang), float(bandwidth_eV)
    if not np.isfinite(residual) or residual < 0.0:
        raise ValueError("the uniform dispersion residual must be finite and nonnegative")
    if not np.isfinite(width) or width < 0.0:
        raise ValueError("the dispersion error bandwidth must be finite and nonnegative")
    if np.any(~np.isfinite(field.escape_mid_ang)) or np.any(~np.isfinite(field.escape_change_ang)):
        raise ValueError("a dispersion residual bound requires finite escape-path geometry")
    if residual == 0.0 or width == 0.0 or field.energy_eV.size == 0:
        return 0.0
    # |exp(-i r L) - 1| <= min(2, |r| |L|). The endpoint maximum
    # bounds |L| everywhere on each affine escape piece.
    escape_max = np.abs(field.escape_mid_ang) + 0.5 * np.abs(field.escape_change_ang)
    phase_error = np.minimum(2.0, residual * escape_max)
    return width * _electron_l1_power(_piece_l1_amplitudes(field) * phase_error, field.electron)


def _host(array):
    get = getattr(array, "get", None)
    return np.asarray(get() if get is not None else array)


class _RowJumps:
    """Amplitude jumps of one row's per-electron fields, ordered in ``tau``.

    Each jump carries the field value just before it (left piece, attenuated
    to its end) and just after it (right piece, attenuated at its start), each
    on its own carrier; a track end or gap leaves one side zero.

    ``dispersive=True`` also keeps each side's escape gradient ``g = dL/dd``
    and escape distance ``L`` at the jump (the right side's for a joint, with
    ``escape_mismatch = |L_left - L_right|``), and drops a jump as identically
    zero only when those agree too: the dispersive carrier is
    ``E_j + hbar c g delta_omega(E)`` and the endpoint phase
    ``-delta_omega(E) L``.
    """

    def __init__(self, row: CoherentRowField, *, dispersive: bool = False):
        lo = row.centre_ang - 0.5 * row.duration_ang
        hi = row.centre_ang + 0.5 * row.duration_ang
        order = np.lexsort((lo, row.electron))
        pieces = order
        energy, electron = row.energy_eV[order], row.electron[order]
        slope = HBARC_EV_ANG * row.slope_ang[order]
        lo, hi = lo[order], hi[order]
        start = row.amplitude[:, order] * row.start_transmission[order]
        end = row.amplitude[:, order] * row.end_transmission[order]
        same = electron[1:] == electron[:-1]
        # Only allow float64 roundoff when reconstructing the two endpoints.
        # A relative physical-length tolerance can erase real gaps after a
        # common time translation, falsely cancelling their endpoint fields.
        # The roundoff scale is that of the absolute transport operands, so an
        # offset-free (translated) centre keeps the operands' own error budget.
        tolerance = _joint_tolerance(row, order)
        joint = same & (np.abs(lo[1:] - hi[:-1]) <= tolerance)
        n, n_pol = energy.size, row.amplitude.shape[0]
        # Jump k sits before piece k (its start) or, where piece k-1 joins it,
        # at the joint; jump n + k is the free end after piece k.
        joined_after = np.concatenate((joint, [False]))
        left_A = np.zeros((n_pol, n), dtype=complex)
        left_A[:, 1:][:, joint] = end[:, :-1][:, joint]
        left_E = energy.copy()
        left_E[1:][joint] = energy[:-1][joint]
        left_slope = slope.copy()
        left_slope[1:][joint] = slope[:-1][joint]
        free_end = ~joined_after
        self.left_A = np.concatenate([left_A, end[:, free_end]], axis=1)
        self.left_E = np.concatenate([left_E, energy[free_end]])
        self.right_A = np.concatenate(
            [start, np.zeros((n_pol, int(free_end.sum())), complex)], axis=1
        )
        self.right_E = np.concatenate([energy, energy[free_end]])
        self.left_slope = np.concatenate([left_slope, slope[free_end]])
        self.right_slope = np.concatenate([slope, slope[free_end]])
        tau = np.concatenate([lo, hi[free_end]])
        owner = np.concatenate([electron, electron[free_end]])
        order = np.lexsort((tau, owner))
        # An identically zero endpoint term must not bridge clusters or count
        # towards their cross-term allowance. Test the captured values exactly:
        # equal amplitudes cancel only on equal complex carrier denominators.
        # Never drop a small nonzero jump because its squared norm underflows.
        equal_amplitude = np.all(self.left_A == self.right_A, axis=0)
        equal_denominator = (self.left_E == self.right_E) & (self.left_slope == self.right_slope)
        if dispersive:
            if row.escape_mid_ang is None or row.escape_change_ang is None:
                raise ValueError("dispersive endpoint terms require captured escape geometry")
            gradient = (row.escape_change_ang / row.duration_ang)[pieces]
            start_L = (row.escape_mid_ang - 0.5 * row.escape_change_ang)[pieces]
            end_L = (row.escape_mid_ang + 0.5 * row.escape_change_ang)[pieces]
            left_g = gradient.copy()
            left_g[1:][joint] = gradient[:-1][joint]
            left_L = start_L.copy()
            left_L[1:][joint] = end_L[:-1][joint]
            self.left_g = np.concatenate([left_g, gradient[free_end]])
            self.right_g = np.concatenate([gradient, gradient[free_end]])
            left_L = np.concatenate([left_L, end_L[free_end]])
            self.escape = np.concatenate([start_L, end_L[free_end]])
            self.escape_mismatch = np.abs(left_L - self.escape)
            equal_denominator &= (self.left_g == self.right_g) & (self.escape_mismatch == 0.0)
        zero_amplitude = np.all(self.left_A == 0.0, axis=0)
        zero_jump = equal_amplitude & (equal_denominator | zero_amplitude)
        order = order[~zero_jump[order]]
        self.left_A, self.right_A = self.left_A[:, order], self.right_A[:, order]
        self.left_E, self.right_E = self.left_E[order], self.right_E[order]
        self.left_slope = self.left_slope[order]
        self.right_slope = self.right_slope[order]
        if dispersive:
            self.left_g, self.right_g = self.left_g[order], self.right_g[order]
            self.escape = self.escape[order]
            self.escape_mismatch = self.escape_mismatch[order]
        tau, owner = tau[order], owner[order]
        self.owner = owner
        gap = np.diff(tau)
        self.gap = np.where(owner[1:] == owner[:-1], gap, np.inf)
        self.joints = int(joint.sum())
        self.jumps = int(tau.size)

    def terms(self, edge_eV, upper):
        """Positive upper bound on each damped jump's squared tail integral.

        For z_i = (E-E_i) + i H lambda_i, decompose the jump as
        (a1-a2)/z1 + a2 (z2-z1)/(z1 z2). Since |z_i| >= u_i+s,
        its L2 norm is at most |a1-a2|/sqrt(u1) +
        |a2| |z2-z1|/sqrt(3 min(u1,u2)^3), by Minkowski. This
        avoids subtracting almost equal tail integrals at a smooth joint.
        Free ends retain their single-denominator bound.

        Validation: coherent-line-grid-windowed-resolution
        """
        sign = 1.0 if upper else -1.0
        u1 = sign * (edge_eV - self.left_E)
        u2 = sign * (edge_eV - self.right_E)
        a1, a2 = self.left_A, self.right_A
        difference = np.hypot(self.left_E - self.right_E, self.left_slope - self.right_slope)
        second = difference / (np.sqrt(3.0) * np.minimum(u1, u2) ** 1.5)
        norm = np.minimum(
            np.abs(a1 - a2) / np.sqrt(u1) + np.abs(a2) * second,
            np.abs(a1 - a2) / np.sqrt(u2) + np.abs(a1) * second,
        )
        return np.sum(norm**2, axis=0)

    def edge_moduli(self, edge_eV, upper):
        """Each jump's decreasing-envelope modulus ``B_J`` at ``edge_eV``.

        For complex damped denominators z_i, decompose the jump as
        (a1-a2)/z1 + a2 (z2-z1)/(z1 z2). The two moduli and their
        derivative norms are bounded by the decreasing real envelopes
        |a1-a2|/u1 and |a2| |z2-z1|/(u1 u2). Either swapped
        decomposition also bounds the jump; choose the tighter at the edge
        and hold that decomposition fixed in the cross-term proof.
        Polarizations add in root-sum-square.
        """
        sign = 1.0 if upper else -1.0
        u1 = sign * (edge_eV - self.left_E)
        u2 = sign * (edge_eV - self.right_E)
        a1, a2 = self.left_A, self.right_A
        difference = np.hypot(self.left_E - self.right_E, self.left_slope - self.right_slope)
        modulus = np.minimum(
            np.abs(a1 - a2) / u1 + np.abs(a2) * difference / (u1 * u2),
            np.abs(a1 - a2) / u2 + np.abs(a1) * difference / (u1 * u2),
        )
        return np.sqrt(np.sum(modulus**2, axis=0))


def coherent_edge_leak(
    jumps: _RowJumps,
    power: float,
    edge_eV: float,
    *,
    upper: bool,
    separation: float = COHERENT_JUMP_CLUSTER_SEPARATION,
) -> float:
    """Bound on a row's per-electron power beyond ``edge_eV``, as a fraction.

    Jumps closer in ``tau`` than ``M hbar c / u_min`` (``M = separation``,
    ``u_min`` the edge's distance to the nearest carrier) form a cluster whose
    tail is bounded by the triangle inequality, ``(sum_J sqrt T_J)^2`` with
    ``T_J`` each damped jump term's positive tail bound. Jumps in clusters
    ``k`` ranks apart are at least ``k M hbar c / u_min`` apart in ``tau``.
    A pair's cross term
    has modulus bounded by a decreasing real envelope; its derivative norm
    is bounded by the negative derivative of that envelope. Integrating by
    parts therefore bounds it by ``2 hbar c B_J B_K /
    dtau``, ``B`` the jumps' edge moduli (:meth:`_RowJumps.edge_moduli`), so
    by ``2 u_min B_J B_K / (k M)``. Summed over rank pairs the cross terms
    are at most ``4 (1 + ln n_C) u_min / M`` times ``sum_C (sum_J B_J)^2``
    over ``n_C`` clusters. The fraction is ``hbar c`` times the bound
    over the row's Parseval power ``2 pi sum |a|^2 dd <exp(-tau)>``.

    Identically zero captured jumps are removed before clustering, so an
    exact collinear subdivision has the whole flight's bound at every edge.

    Validation: coherent-line-grid-windowed-resolution
    """
    if jumps.jumps == 0:
        return 0.0
    terms = jumps.terms(edge_eV, upper)
    sign = 1.0 if upper else -1.0
    u_min = float(np.min(sign * (edge_eV - np.concatenate((jumps.left_E, jumps.right_E)))))
    threshold = float(separation) * HBARC_EV_ANG / u_min
    starts = np.concatenate(([0], np.flatnonzero(jumps.gap > threshold) + 1))
    bound = float((np.add.reduceat(np.sqrt(np.maximum(terms, 0.0)), starts) ** 2).sum())
    count = starts.size
    if count > 1:
        moduli = np.add.reduceat(jumps.edge_moduli(edge_eV, upper), starts)
        bound += 4.0 * (1.0 + np.log(count)) * u_min / float(separation) * float(np.sum(moduli**2))
    return HBARC_EV_ANG * bound / (2.0 * np.pi * power)


def _dispersion_blocks(bands: DispersionBands, lo_eV, hi_eV, band_edge, shift_scale, *, upper):
    """Partition ``[lo_eV, hi_eV]`` into blocks of whole smooth intervals.

    Each block carries the union of its intervals' phase enclosures. A block
    grows outward from the window edge while its width stays within a quarter
    of its near end's distance ``u`` from the resonance band edge, so the
    ``1/u`` tail profile is resolved with few blocks, and while the union's
    carrier shift ``shift_scale * (phase_max - phase_min)`` (``shift_scale =
    hbar c max |g|``) stays within a tenth of it. Any partition is valid;
    this one only bounds the work.
    """
    keep = (bands.hi_eV > lo_eV) & (bands.lo_eV < hi_eV)
    lo = np.maximum(bands.lo_eV[keep], lo_eV)
    hi = np.minimum(bands.hi_eV[keep], hi_eV)
    phase_min, phase_max = bands.phase_min[keep], bands.phase_max[keep]
    if not upper:
        lo, hi, phase_min, phase_max = hi[::-1], lo[::-1], phase_min[::-1], phase_max[::-1]
    blocks = []
    start = 0
    while start < lo.size:
        near = lo[start]
        reach = 0.25 * max(abs(near - band_edge), 1e-300)
        stop = start + 1
        while (
            stop < lo.size
            and abs(hi[stop] - near) <= reach
            and shift_scale * (max(phase_max[start : stop + 1]) - min(phase_min[start : stop + 1]))
            <= 0.1 * reach
        ):
            stop += 1
        blocks.append(
            (
                min(near, hi[stop - 1]),
                max(near, hi[stop - 1]),
                float(phase_min[start:stop].min()),
                float(phase_max[start:stop].max()),
            )
        )
        start = stop
    return np.array(blocks, dtype=float).reshape(-1, 4)


def _dispersive_region_leak(
    jumps: _RowJumps,
    power: float,
    bands: DispersionBands,
    edge_eV: float,
    axis_eV: float,
    *,
    upper: bool,
    separation: float = COHERENT_JUMP_CLUSTER_SEPARATION,
) -> float:
    """Bound on the dispersive row power between ``edge_eV`` and ``axis_eV``, as a fraction.

    Source: at fixed ``E`` the reducer's in-medium phase ``-delta_omega(E)
    L(d)`` is affine in ``d`` on each piece, so a piece's production field is
    exactly its endpoint terms with denominator ``z_i(E) = E - E_i - hbar c
    g_i delta_omega(E) + i hbar c lambda_i`` (``g = dL/dd``) and endpoint
    phase ``E tau/hbar c - delta_omega(E) L``, continuous at a joint. On each
    block of smooth material intervals the certified enclosure of
    ``delta_omega`` bounds ``|z_i|`` below by the distance to the shifted
    carrier, and each jump's ``L^2`` norm follows in closed form. Clusters
    take the triangle inequality. Across clusters the cross-term phase slope
    is ``dtau/hbar c - delta_omega' dL`` with ``|dL| <= r_e |dtau|`` (``r_e``
    the electron's largest consecutive ``|dL|/dtau``). With ``eps = hbar c
    M1 r_e < 1``, clusters split at gaps above ``M hbar c/(u_min (1 -
    eps))`` keep it at least ``k M/u_min``, and integration by parts with
    the slope's variation ``TV`` multiplies the frozen allowance by ``1 +
    hbar c r_e TV/(2 (1 - eps))`` and each derivative envelope by ``1 +
    hbar c |g| M1``; an electron with ``eps >= 1`` is one cluster. Returns
    ``inf`` when a shifted carrier can reach the omitted band. With
    ``delta_omega = 0`` this is at most :func:`coherent_edge_leak` with its
    diagonal truncated at ``axis_eV``.

    Validation: coherent-line-grid-windowed-resolution
    """
    sign = 1.0 if upper else -1.0
    if jumps.jumps == 0 or (axis_eV - edge_eV) * sign <= 0.0:
        return 0.0
    H = HBARC_EV_ANG
    region = (edge_eV, axis_eV) if upper else (axis_eV, edge_eV)
    if region[0] < bands.lo_eV[0] or region[1] > bands.hi_eV[-1]:
        raise ValueError("the material bands must cover the whole omitted region")
    carriers = np.concatenate((jumps.left_E, jumps.right_E))
    band_edge = float(carriers.max() if upper else carriers.min())
    shift_scale = H * float(np.abs(np.concatenate((jumps.left_g, jumps.right_g))).max())
    blocks = _dispersion_blocks(bands, *region, band_edge, shift_scale, upper=upper)
    b_lo, b_hi, p_min, p_max = (blocks[:, k : k + 1] for k in range(4))
    near, far = (b_lo, b_hi) if upper else (b_hi, b_lo)

    def distance(energy, gradient, at, phase_lo, phase_hi):
        """Lower bound of ``sign (E - E_i - H g delta)`` over the phase enclosure."""
        shift = H * np.where(sign * gradient >= 0.0, gradient * phase_hi, gradient * phase_lo)
        return sign * (at - energy - shift)

    d_near_L = distance(jumps.left_E, jumps.left_g, near, p_min, p_max)
    d_near_R = distance(jumps.right_E, jumps.right_g, near, p_min, p_max)
    if np.any(d_near_L <= 0.0) or np.any(d_near_R <= 0.0):
        return float("inf")
    d_far_L = distance(jumps.left_E, jumps.left_g, far, p_min, p_max)
    d_far_R = distance(jumps.right_E, jumps.right_g, far, p_min, p_max)

    def inverse_norm(d_near, d_far):
        return np.sqrt((d_far - d_near) / (d_near * d_far))

    m_near, m_far = np.minimum(d_near_L, d_near_R), np.minimum(d_far_L, d_far_R)
    product_norm = np.sqrt((m_far**3 - m_near**3) / (3.0 * m_near**3 * m_far**3))
    phase_abs = np.maximum(np.abs(p_min), np.abs(p_max))
    gradient_change = np.abs(jumps.left_g - jumps.right_g)
    carrier_change = np.hypot(
        np.abs(jumps.left_E - jumps.right_E) + H * gradient_change * phase_abs,
        jumps.left_slope - jumps.right_slope,
    )
    # A joint's endpoint escapes may differ by roundoff; that phase
    # exp(-i delta_omega dL) rides on the left amplitude.
    escape_phase = np.minimum(2.0, phase_abs * jumps.escape_mismatch)
    a1, a2 = jumps.left_A[:, None, :], jumps.right_A[:, None, :]
    numerator = np.abs(a1 - a2) + np.maximum(np.abs(a1), np.abs(a2)) * escape_phase
    norm = np.minimum(
        numerator * inverse_norm(d_near_L, d_far_L) + np.abs(a2) * carrier_change * product_norm,
        numerator * inverse_norm(d_near_R, d_far_R) + np.abs(a1) * carrier_change * product_norm,
    )
    terms = np.sum(norm**2, axis=(0, 1))

    # Cross terms: one envelope over the whole omitted region.
    keep = (bands.hi_eV > region[0]) & (bands.lo_eV < region[1])
    phase_lo, phase_hi = float(bands.phase_min[keep].min()), float(bands.phase_max[keep].max())
    slope_max = float(bands.slope_max[keep].max())
    variation = float(bands.variation[keep].sum())
    phase_abs = max(abs(phase_lo), abs(phase_hi))
    u1 = distance(jumps.left_E, jumps.left_g, edge_eV, phase_lo, phase_hi)
    u2 = distance(jumps.right_E, jumps.right_g, edge_eV, phase_lo, phase_hi)
    owner = jumps.owner
    starts_e = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
    # Lipschitz constant of the endpoint escape L along each electron's
    # sorted jumps: the largest consecutive |dL|/dtau bounds every pair.
    same = np.isfinite(jumps.gap)
    escape_step = np.abs(np.diff(jumps.escape))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            same & (escape_step > 0.0), escape_step / np.where(same, jumps.gap, 1.0), 0.0
        )
    ratio = np.nan_to_num(ratio, nan=np.inf)
    rate = np.zeros(owner.size)
    np.maximum.at(rate, np.arange(1, owner.size), ratio)
    lipschitz = np.maximum.reduceat(rate, starts_e)
    detuning = H * slope_max * lipschitz
    u_min = float(min(u1.min(), u2.min()))
    if u_min > 0.0 and np.isfinite(variation):
        separable = detuning < 1.0
        electron_threshold = np.where(
            separable,
            float(separation) * H / (u_min * np.where(separable, 1.0 - detuning, 1.0)),
            np.inf,
        )
        threshold = np.repeat(electron_threshold, np.diff(np.r_[starts_e, owner.size]))[1:]
        # Electrons never share a cluster, even when no gap separates within one.
        split = (jumps.gap > threshold) | ~same
        starts = np.concatenate(([0], np.flatnonzero(split) + 1))
    else:
        separable = np.zeros(starts_e.size, dtype=bool)
        starts = starts_e
    bound = float((np.add.reduceat(np.sqrt(np.maximum(terms, 0.0)), starts) ** 2).sum())
    count = starts.size
    if count > starts_e.size:
        u1_far = distance(jumps.left_E, jumps.left_g, axis_eV, phase_lo, phase_hi)
        u2_far = distance(jumps.right_E, jumps.right_g, axis_eV, phase_lo, phase_hi)
        stretch = 1.0 + H * slope_max * np.maximum(np.abs(jumps.left_g), np.abs(jumps.right_g))
        change = np.hypot(
            np.abs(jumps.left_E - jumps.right_E) + H * gradient_change * phase_abs,
            jumps.left_slope - jumps.right_slope,
        )
        escape_phase = np.minimum(2.0, phase_abs * jumps.escape_mismatch)
        escape_rate = jumps.escape_mismatch * slope_max
        reach = np.minimum(u1_far, u2_far)
        larger = np.maximum(np.abs(jumps.left_A), np.abs(jumps.right_A))
        numerator = np.abs(jumps.left_A - jumps.right_A) + larger * escape_phase
        moduli = []
        # Decompose a1 e^{i theta}/z1 - a2/z2 about either carrier; derivative
        # envelopes carry |z_i'| <= stretch and the E-dependence of
        # z1 - z2 and of the mismatch phase.
        for u_near, u_reach, a_other in ((u1, u1_far, jumps.right_A), (u2, u2_far, jumps.left_A)):
            c1 = numerator * stretch + larger * escape_rate * u_reach
            c2 = (
                np.abs(a_other) * (change * stretch + H * gradient_change * slope_max * reach)
                + larger * change * escape_rate * reach
            )
            moduli.append(c1 / u_near + c2 / (u1 * u2))
        modulus = np.sqrt(np.sum(np.minimum(*moduli) ** 2, axis=0))
        cluster_moduli = np.add.reduceat(modulus, starts)
        cluster_owner = owner[starts]
        owner_index = np.searchsorted(owner[starts_e], cluster_owner)
        # Only clusters of one electron interfere; a one-cluster electron has
        # no cross term.
        clusters_per_electron = np.bincount(owner_index, minlength=starts_e.size)
        paired = separable & (clusters_per_electron > 1)
        kappa = np.where(
            paired,
            1.0 + H * lipschitz * variation / (2.0 * np.where(paired, 1.0 - detuning, 1.0)),
            0.0,
        )[owner_index]
        bound += (
            4.0
            * (1.0 + np.log(count))
            * u_min
            / float(separation)
            * float(np.sum(kappa * cluster_moduli**2))
        )
    return H * bound / (2.0 * np.pi * power)


def dispersive_edge_leak(
    jumps: _RowJumps,
    power: float,
    bands: DispersionBands,
    edge_eV: float,
    axis_eV: float,
    *,
    upper: bool,
    separation: float = COHERENT_JUMP_CLUSTER_SEPARATION,
) -> float:
    """Certified dispersive row power between ``edge_eV`` and ``axis_eV``, as a fraction.

    The omitted region is the union of a near part ``[edge, Y]`` and a far
    part ``[Y, axis]``, so the sum of their region bounds
    (:func:`_dispersive_region_leak`) bounds it for any split ``Y``. A large
    material slope far from the window (an absorption edge, the plasmon
    region) then only forces the far part into one cluster per electron.
    Splits are tried at carrier distances growing by ``2**0.25`` from the
    edge's; the least bound, with no split among the candidates, is returned.

    Validation: coherent-line-grid-windowed-resolution
    """
    leak = partial(_dispersive_region_leak, jumps, power, bands, upper=upper, separation=separation)
    best = leak(edge_eV, axis_eV)
    sign = 1.0 if upper else -1.0
    if jumps.jumps == 0 or (axis_eV - edge_eV) * sign <= 0.0:
        return best
    carriers = np.concatenate((jumps.left_E, jumps.right_E))
    band_edge = float(carriers.max() if upper else carriers.min())
    distance = sign * (edge_eV - band_edge)
    if not distance > 0.0:
        return best
    while (axis_eV - (split := band_edge + sign * _SPLIT_RATIO * distance)) * sign > 0.0:
        best = min(best, leak(edge_eV, split) + leak(split, axis_eV))
        distance *= _SPLIT_RATIO
    return best


def _edge(leak, band_edge, axis_edge, factor, limit, *, upper):
    """Smallest halo edge in ``[band_edge, axis_edge]`` meeting ``limit``."""
    if (axis_edge - band_edge) * (1.0 if upper else -1.0) <= 0.0:
        return float(axis_edge), 0.0
    at_axis = factor(axis_edge) * leak(axis_edge)
    if at_axis > limit:
        return float(axis_edge), float(at_axis)
    near, far = float(band_edge), float(axis_edge)
    for _ in range(_EDGE_BISECTIONS):
        middle = 0.5 * (near + far)
        if middle in (near, far):
            break
        if factor(middle) * leak(middle) > limit:
            near = middle
        else:
            far = middle
    return far, float(factor(far) * leak(far))


def _dispersive_leak(jumps, power, bands, axis_eV, edge_eV, *, upper):
    return dispersive_edge_leak(jumps, power, bands, edge_eV, axis_eV, upper=upper)


def _widen(leak, edge, axis_edge, factor, limit, *, upper):
    """Keep ``edge`` when its dispersive bound meets ``limit``, else widen it.

    Windows only grow: the frozen-carrier edge stays the inner bound.
    """
    at_edge = factor(edge) * leak(edge)
    if at_edge <= limit:
        return float(edge), float(at_edge), False
    widened, bound = _edge(leak, edge, axis_edge, factor, limit, upper=upper)
    return widened, bound, True


def _support_span(lo, hi, electron=None):
    """Span of the union support ``[lo, hi]``, overall or widest per electron."""
    if electron is None:
        return float(hi.max() - lo.min())
    order = np.argsort(electron, kind="stable")
    electron, lo, hi = electron[order], lo[order], hi[order]
    starts = np.flatnonzero(np.concatenate(([True], electron[1:] != electron[:-1])))
    return float((np.maximum.reduceat(hi, starts) - np.minimum.reduceat(lo, starts)).max())


def _dispersion_window_audit(rows, summary, law, decoherence, electron_count):
    """Certified dispersive row power the windows exclude from the finite axis.

    Per row and side, :func:`dispersive_edge_leak` between the window edge and
    the axis edge, times ``1 + F (N_e - 1)`` (``F`` at the upper edge, at the
    axis start for the lower side), relative to the frozen Parseval
    reference ``2 pi hbar c sum |a|^2 dd <exp(-tau)>``; the absolute bound is
    in the units of ``integral sum_e,p |S|^2 dE``. It is not relative to the
    production row power and not a sampling-error certificate.
    ``decoherence`` must be the production nonincreasing analytic upper bound
    from :func:`decoherence_bound`, or None (F <= 1), not an arbitrary callback.

    Validation: coherent-line-grid-windowed-resolution
    """
    bands = law.bands()
    output = {
        "material_fingerprint": law.fingerprint,
        "intervals": int(bands.lo_eV.size),
        "normalization": "integral of float64 row intensity dE with E in eV; "
        "fractions are of the frozen Parseval reference",
        "scope": "float64 material phase and endpoint reconstruction convention",
        "relative_production_bound": False,
        "rows": [],
    }
    start, stop = float(bands.lo_eV[0]), float(bands.hi_eV[-1])
    count = max(float(electron_count), 1.0)
    for field, record in zip(rows, summary["rows"], strict=True):
        if field.power <= 0.0 or "window_eV" not in record:
            continue
        lower, upper = record["window_eV"]
        row_decoherence = _row_decoherence(decoherence, field)

        def factor(energy, row_decoherence=row_decoherence):
            if row_decoherence is None:
                return count
            return 1.0 + float(row_decoherence(np.asarray(energy, dtype=float))) * (count - 1)

        jumps = _RowJumps(field, dispersive=True)
        leak = {
            "lower": float(
                factor(start)
                * dispersive_edge_leak(jumps, field.power, bands, lower, start, upper=False)
            ),
            "upper": float(
                factor(upper)
                * dispersive_edge_leak(jumps, field.power, bands, upper, stop, upper=True)
            ),
        }
        fraction = leak["lower"] + leak["upper"]
        reference = 2.0 * np.pi * HBARC_EV_ANG * field.power
        d_lo = field.centre_ang - 0.5 * field.duration_ang
        d_hi = field.centre_ang + 0.5 * field.duration_ang
        L_lo = field.escape_mid_ang - 0.5 * field.escape_change_ang
        L_hi = field.escape_mid_ang + 0.5 * field.escape_change_ang
        L_span = float(np.maximum(L_lo, L_hi).max() - np.minimum(L_lo, L_hi).min())
        inside = (bands.hi_eV > lower) & (bands.lo_eV < upper)
        slope = float(bands.slope_max[inside].max()) if inside.any() else 0.0
        span = _support_span(d_lo, d_hi) + HBARC_EV_ANG * slope * L_span
        output["rows"].append(
            {
                "row": field.label,
                "leak_bound": leak,
                "reference_eV": float(reference),
                "excluded_power_bound_eV": float(fraction * reference),
                "frozen_reference_fraction": float(fraction),
                "phase_slope_step_all_eV": float(
                    np.pi * HBARC_EV_ANG / span / summary["oversampling"]
                ),
            }
        )
    return output


def _row_decoherence(decoherence, field):
    """Row bound ``F_z(E) sup_{E' >= E} F_perp(E')``: nonincreasing, >= the reducer's F.

    Validation: transverse-bunch-form-factor
    """
    envelope = getattr(field, "transverse_envelope", None)
    if decoherence is None or envelope is None:
        return decoherence

    def bound(energy_eV):
        return decoherence(energy_eV) * envelope(energy_eV)

    return bound


def _decoherence_label(decoherence, rows):
    if decoherence is None:
        return "none (F = 1)"
    if any(getattr(row, "transverse_envelope", None) is not None for row in rows):
        return "analytic F_z x transverse envelope"
    return "analytic F_z"


def coherent_window_seeds(
    rows,
    *,
    start_eV: float,
    stop_eV: float,
    electron_count: float,
    decoherence: Callable[[np.ndarray], np.ndarray] | None,
    leak_limit: float = DEFAULT_COHERENT_LEAK_LIMIT,
    decoherence_limit: float = DEFAULT_COHERENT_DECOHERENCE_LIMIT,
    bin_eV: float = COHERENT_WINDOW_BIN_EV,
    oversampling: float = COHERENT_NYQUIST_OVERSAMPLING,
    dispersion: DispersionBands | None = None,
) -> tuple[list[FeatureSeed], dict[str, Any]]:
    """Coherent windows of every ``(reflection, orientation)`` row.

    Source equation: the coherent reducer's own row field
    (:class:`CoherentRowField`, captured from ``mc_spectrum``): per piece the
    time-domain carrier ``a_j exp(-i omega_j tau)`` on ``|tau - d_j| <= dd_j/2``
    with ``a_j = c_j / dd_j`` from the reducer's complex coefficient ``c_j``
    per polarization, damped by ``exp(-tau_abs/2)``, phase continuous along a
    track.

    Each row's window spans its resonance band, widened by the smallest halo
    whose tail bound (:func:`coherent_edge_leak`) times ``1 + F (N_e - 1)``
    (``F`` at the edge, or at the axis start for the lower tail) is at most
    ``leak_limit``; a halo that would leave the axis stops at it and reports
    the bound there. Bins of ``bin_eV`` then take ``pi hbar c / D`` over the
    per-electron or all-electron support span, divided by ``oversampling``
    nodes per Nyquist step. Limiting cases: one undamped segment gives the
    ``sinc**2`` tail bound ``w/(pi^2 u)`` within ``M hbar c / dd`` of its edge
    and the Nyquist step ``pi hbar c / dd = w / 2``; ``F -> 0`` takes the
    per-electron step everywhere.

    Returns ``(seeds, summary)``.

    Validation: coherent-line-grid-windowed-resolution
    """
    if not 0.0 < float(leak_limit) < 1.0:
        raise ValueError("leak_limit must lie in (0, 1)")
    if not 0.0 < float(decoherence_limit) < 1.0:
        raise ValueError("decoherence_limit must lie in (0, 1)")
    start, stop = float(start_eV), float(stop_eV)
    count = max(float(electron_count), 1.0)

    def factor_at(energy, decoherence):
        if decoherence is None:
            return float(count)
        return 1.0 + float(decoherence(np.asarray(energy, dtype=float))) * (count - 1)

    seeds: list[FeatureSeed] = []
    summary: dict[str, Any] = {
        "leak_limit": float(leak_limit),
        "decoherence_limit": float(decoherence_limit),
        "bin_eV": float(bin_eV),
        "oversampling": float(oversampling),
        "cluster_separation": COHERENT_JUMP_CLUSTER_SEPARATION,
        "electron_count": count,
        "decoherence_bound": _decoherence_label(decoherence, rows),
        "rows": [],
    }
    for field in rows:
        keep = (
            np.isfinite(field.energy_eV)
            & np.isfinite(field.centre_ang)
            & (field.duration_ang > 0.0)
            & np.all(np.isfinite(field.amplitude), axis=0)
        )
        if not keep.any():
            summary["rows"].append({"row": field.label, "pieces": 0})
            continue
        field = CoherentRowField(
            label=field.label,
            energy_eV=field.energy_eV[keep],
            amplitude=field.amplitude[:, keep],
            duration_ang=field.duration_ang[keep],
            centre_ang=field.centre_ang[keep],
            electron=field.electron[keep],
            start_transmission=field.start_transmission[keep],
            end_transmission=field.end_transmission[keep],
            mean_transmission=field.mean_transmission[keep],
            attenuation_slope_ang=field.slope_ang[keep],
            escape_mid_ang=None if field.escape_mid_ang is None else field.escape_mid_ang[keep],
            escape_change_ang=None
            if field.escape_change_ang is None
            else field.escape_change_ang[keep],
            phase_rad=None if field.phase_rad is None else field.phase_rad[keep],
            transverse_envelope=field.transverse_envelope,
            centre_scale_ang=None
            if field.centre_scale_ang is None
            else field.centre_scale_ang[keep],
        )
        row_decoherence = _row_decoherence(decoherence, field)
        power = field.power
        lo_t = field.centre_ang - 0.5 * field.duration_ang
        hi_t = field.centre_ang + 0.5 * field.duration_ang
        span_all = _support_span(lo_t, hi_t)
        span_electron = _support_span(lo_t, hi_t, field.electron)
        nyquist_all = np.pi * HBARC_EV_ANG / span_all
        nyquist_electron = np.pi * HBARC_EV_ANG / span_electron
        step_all = nyquist_all / float(oversampling)
        step_electron = nyquist_electron / float(oversampling)
        band_lo, band_hi = float(field.energy_eV.min()), float(field.energy_eV.max())
        row = {
            "row": field.label,
            "pieces": int(field.energy_eV.size),
            "band_eV": [band_lo, band_hi],
            "span_all_ang": span_all,
            "span_electron_ang": span_electron,
            "nyquist_all_eV": nyquist_all,
            "nyquist_electron_eV": nyquist_electron,
            "step_all_eV": step_all,
            "step_electron_eV": step_electron,
            "points": 0,
        }
        if power <= 0.0:
            summary["rows"].append(row)
            continue
        jumps = _RowJumps(field)
        lower_value = factor_at(start, row_decoherence)
        factors = {
            True: partial(factor_at, decoherence=row_decoherence),
            False: lambda _edge_eV, value=lower_value: value,
        }
        axes = {True: stop, False: start}
        edges, leaks = {}, {}
        for side, band_edge in ((True, band_hi), (False, band_lo)):
            edges[side], leaks[side] = _edge(
                partial(coherent_edge_leak, jumps, power, upper=side),
                band_edge,
                axes[side],
                factors[side],
                float(leak_limit),
                upper=side,
            )
        if dispersion is not None:
            dispersive = _RowJumps(field, dispersive=True)
            dispersion_leak, widened = {}, {}
            for side in (True, False):
                edges[side], dispersion_leak[side], widened[side] = _widen(
                    partial(
                        _dispersive_leak, dispersive, power, dispersion, axes[side], upper=side
                    ),
                    edges[side],
                    axes[side],
                    factors[side],
                    float(leak_limit),
                    upper=side,
                )
            row["dispersion_leak_bound"] = {
                "lower": dispersion_leak[False],
                "upper": dispersion_leak[True],
            }
            row["dispersion_widened"] = {"lower": widened[False], "upper": widened[True]}
            for side in (True, False):
                if widened[side]:
                    # The frozen bound reported is the frozen tail at the edge used.
                    leaks[side] = factors[side](edges[side]) * coherent_edge_leak(
                        jumps, power, edges[side], upper=side
                    )
        upper, upper_leak = edges[True], leaks[True]
        lower, lower_leak = edges[False], leaks[False]
        lower, upper = max(lower, start), min(upper, stop)
        row.update(
            {
                "joints": jumps.joints,
                "jumps": jumps.jumps,
                "window_eV": [lower, upper],
                # A bound at an axis edge is the tail beyond the axis itself,
                # bandwidth truncation rather than window leakage.
                "leak_bound": {"lower": lower_leak, "upper": upper_leak},
                "at_axis": {"lower": lower <= start, "upper": upper >= stop},
            }
        )
        if upper <= lower:
            summary["rows"].append(row)
            continue
        edges = np.arange(lower, upper, float(bin_eV))
        edges = np.append(edges, upper)
        bins_lo, bins_hi = edges[:-1], edges[1:]
        if row_decoherence is None:
            grouped = np.zeros(bins_lo.size, dtype=bool)
        else:
            grouped = row_decoherence(bins_lo) * count <= float(decoherence_limit)
        steps = np.where(grouped, step_electron, step_all)
        row["electron_step_from_eV"] = float(bins_lo[grouped].min()) if grouped.any() else None
        index = 0
        while index < bins_lo.size:
            run = index
            while run < bins_lo.size and steps[run] == steps[index]:
                run += 1
            lo, hi = float(bins_lo[index]), float(bins_hi[run - 1])
            seeds.append(
                FeatureSeed(
                    source=COHERENT_SOURCE,
                    label=f"{field.label} {lo:.0f}-{hi:.0f} eV",
                    centre_eV=lo,
                    below_eV=0.0,
                    above_eV=hi - lo,
                    spacing_eV=float(steps[index]),
                )
            )
            row["points"] += int(np.ceil((hi - lo) / float(steps[index])))
            index = run
        summary["rows"].append(row)
    summary["seeds"] = len(seeds)
    return seeds, summary


def coherent_case_seeds(
    rows,
    segments: Mapping[str, Any],
    *,
    electron_limit: int | None,
    start_eV: float,
    stop_eV: float,
    longitudinal_rms_fs: float | None,
    dispersion_law: CoherentDispersionLaw | None = None,
    physical_electrons: float | None = None,
) -> tuple[list[FeatureSeed], dict[str, Any]]:
    """Coherent windows of one direction's captured rows.

    ``rows`` come from a :class:`CoherentRowCollector` passed to the case's
    own coherent line reduction; ``segments`` supply the offset population
    for :func:`decoherence_bound`.

    Validation: coherent-line-grid-windowed-resolution
    """
    decoherence = decoherence_bound(
        segments, electron_limit=electron_limit, longitudinal_rms_fs=longitudinal_rms_fs
    )
    electrons = [row.electron for row in rows]
    # N_e of the Cauchy-Schwarz bound: the line electrons that left pieces.
    electron_count = int(np.unique(np.concatenate(electrons)).size) if electrons else 1
    samples = int(segments["Ne"] if electron_limit is None else electron_limit)
    bound_count = electron_count
    if physical_electrons is not None:
        # |P-G| <= (H-1) G for H emitting incident electrons.
        # Including missed entries in M is essential; H only bounds support.
        # Validation: coherent-physical-bunch-population
        scale = pair_scale(physical_electrons, samples)
        bound_count = 1.0 + scale * max(electron_count - 1, 0)
        if scale == 0.0 or electron_count <= 1:

            def decoherence(energy):
                return np.zeros_like(np.asarray(energy, dtype=float))

    seeds, summary = coherent_window_seeds(
        rows,
        start_eV=start_eV,
        stop_eV=stop_eV,
        electron_count=bound_count,
        decoherence=decoherence,
        dispersion=None if dispersion_law is None else dispersion_law.bands(),
    )
    if physical_electrons is not None:
        summary.update(
            physical_electrons=float(physical_electrons),
            incident_samples=samples,
            emitting_samples=electron_count,
            tail_population_bound=bound_count,
        )
    if physical_electrons is not None and (scale == 0.0 or electron_count <= 1):
        summary["decoherence_bound"] = "self-only (no sampled cross-electron excess)"
    if dispersion_law is not None:
        summary["dispersion"] = _dispersion_window_audit(
            rows, summary, dispersion_law, decoherence, bound_count
        )
    return seeds, summary
