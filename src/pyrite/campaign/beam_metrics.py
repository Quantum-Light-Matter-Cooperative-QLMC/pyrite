"""Pure sampled-beam phase-space diagnostics.

All second moments are population moments of the supplied macro-particle sample.
Coordinates are centered internally, so diagnostics describe spread rather than
beam centroid or steering.  Twiss parameters are undefined for a zero-emittance
plane and are reported as ``nan``; the emittance itself remains exactly zero.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.constants import c, physical_constants

_ELECTRON_REST_KEV = physical_constants["electron mass energy equivalent in MeV"][0] * 1.0e3
_ANG_PER_FS = c * 1.0e-5
_ANG_PER_MM = 1.0e7


class BeamLike(Protocol):
    """BeamSpec-shaped source properties required for current diagnostics."""

    bunch_charge_pc: float
    rep_rate_hz: float


@dataclass(frozen=True)
class PlaneMetrics:
    """One transverse plane's RMS, emittance, and Twiss diagnostics.

    ``position_sigma_mm`` has mm units, ``slope_sigma_rad`` is dimensionless,
    and geometric/normalized emittance values have mm rad units.
    """

    position_sigma_mm: float
    slope_sigma_rad: float
    geometric_emittance_mm_rad: float
    normalized_emittance_mm_rad: float
    alpha: float
    beta_mm_per_rad: float
    gamma_rad_per_mm: float


@dataclass(frozen=True)
class BeamMetrics:
    """RMS, emittance, and pulsed-current diagnostics for one sampled bunch."""

    x: PlaneMetrics
    y: PlaneMetrics
    sigma_t_fs: float
    sigma_z_mm: float
    sigma_delta: float
    longitudinal_emittance_fs: float
    bunch_charge_pc: float
    gaussian_equivalent_peak_current_a: float
    average_current_a: float


def _array(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _moments(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    if a.shape != b.shape:
        raise ValueError("phase-space arrays must have equal lengths")
    da = a - a.mean()
    db = b - b.mean()
    return float(np.mean(da * da)), float(np.mean(db * db)), float(np.mean(da * db))


def _plane_metrics(position_mm: np.ndarray, slope: np.ndarray, beta_gamma: float) -> PlaneMetrics:
    variance, slope_variance, covariance = _moments(position_mm, slope)
    determinant = variance * slope_variance - covariance**2
    # Roundoff can make a mathematically zero determinant faintly negative.
    emittance = float(np.sqrt(max(determinant, 0.0)))
    if emittance == 0.0:
        alpha = beta = gamma = float("nan")
    else:
        alpha = -covariance / emittance
        beta = variance / emittance
        gamma = slope_variance / emittance
    return PlaneMetrics(
        position_sigma_mm=float(np.sqrt(variance)),
        slope_sigma_rad=float(np.sqrt(slope_variance)),
        geometric_emittance_mm_rad=emittance,
        normalized_emittance_mm_rad=beta_gamma * emittance,
        alpha=alpha,
        beta_mm_per_rad=beta,
        gamma_rad_per_mm=gamma,
    )


def sampled_beam_metrics(
    x_mm: object,
    x_prime_rad: object,
    y_mm: object,
    y_prime_rad: object,
    t_fs: object,
    delta: object,
    *,
    energy_keV: float,
    bunch_charge_pc: float,
    rep_rate_hz: float,
) -> BeamMetrics:
    """Compute diagnostics from initial sampled ``(x, x', y, y', t, delta)``.

    ``delta`` is relative energy deviation, ``(E - <E>) / <E>``.  Twiss uses
    ``alpha = -<xx'>/eps``, ``beta = <x^2>/eps``, and
    ``gamma = <x'^2>/eps``.  The longitudinal emittance is
    ``sqrt(<t^2><delta^2> - <t delta>^2)`` in fs.  A Gaussian bunch has
    ``I_peak = Q / (sqrt(2 pi) sigma_t)``; a point bunch therefore reports
    infinite Gaussian-equivalent peak current (or zero when its charge is
    zero). This descriptor does not claim the sampled bunch is Gaussian:
    uniform and explicit-offset samples need their known density or a density
    estimator for an actual peak current.

    Limiting cases: uncorrelated zero spread gives zero emittance; a
    monoenergetic point bunch gives zero longitudinal emittance; zero charge or
    repetition rate gives zero average current.

    Validation: beam-phase-space-metrics
    """
    arrays = tuple(
        _array(value, name=name)
        for name, value in (
            ("x_mm", x_mm),
            ("x_prime_rad", x_prime_rad),
            ("y_mm", y_mm),
            ("y_prime_rad", y_prime_rad),
            ("t_fs", t_fs),
            ("delta", delta),
        )
    )
    if len({array.size for array in arrays}) != 1:
        raise ValueError("phase-space arrays must have equal lengths")
    if not np.isfinite(energy_keV) or energy_keV < 0.0:
        raise ValueError("energy_keV must be finite and non-negative")
    if not np.isfinite(bunch_charge_pc) or bunch_charge_pc < 0.0:
        raise ValueError("bunch_charge_pc must be finite and non-negative")
    if not np.isfinite(rep_rate_hz) or rep_rate_hz < 0.0:
        raise ValueError("rep_rate_hz must be finite and non-negative")

    x, xp, y, yp, time, energy_delta = arrays
    gamma_rel = 1.0 + float(energy_keV) / _ELECTRON_REST_KEV
    beta_gamma = float(np.sqrt(gamma_rel**2 - 1.0))
    time_variance, delta_variance, time_delta_covariance = _moments(time, energy_delta)
    longitudinal_emittance = float(
        np.sqrt(max(time_variance * delta_variance - time_delta_covariance**2, 0.0))
    )
    sigma_t = float(np.sqrt(time_variance))
    charge_c = float(bunch_charge_pc) * 1.0e-12
    peak_current = 0.0 if charge_c == 0.0 else np.inf
    if sigma_t > 0.0:
        peak_current = charge_c / (np.sqrt(2.0 * np.pi) * sigma_t * 1.0e-15)
    return BeamMetrics(
        x=_plane_metrics(x, xp, beta_gamma),
        y=_plane_metrics(y, yp, beta_gamma),
        sigma_t_fs=sigma_t,
        sigma_z_mm=sigma_t * _ANG_PER_FS / _ANG_PER_MM,
        sigma_delta=float(np.sqrt(delta_variance)),
        longitudinal_emittance_fs=longitudinal_emittance,
        bunch_charge_pc=float(bunch_charge_pc),
        gaussian_equivalent_peak_current_a=float(peak_current),
        average_current_a=charge_c * float(rep_rate_hz),
    )


def initial_state_metrics(
    position_ang: object,
    direction: object,
    arrival_offset_ang: object,
    energy_keV: object,
    beam: BeamLike,
) -> BeamMetrics:
    """Compute diagnostics from transport's initial sampled phase-space arrays.

    Positions are transport-frame Angstrom coordinates, directions are unit
    vectors, and arrival offsets use transport's ``c=1`` Angstrom clock.  Slopes
    are ``v_x/v_z`` and ``v_y/v_z``; directions parallel to the entrance plane
    are rejected because those slopes are undefined.
    """
    position = np.asarray(position_ang, dtype=float)
    directions = np.asarray(direction, dtype=float)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("position_ang must have shape (N, 3)")
    if directions.shape != position.shape:
        raise ValueError("direction must have shape (N, 3) matching position_ang")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(directions)):
        raise ValueError("initial position and direction arrays must be finite")
    if np.any(np.isclose(directions[:, 2], 0.0)):
        raise ValueError("direction z components must be nonzero to form slopes")
    energy = _array(energy_keV, name="energy_keV")
    if energy.size != len(position):
        raise ValueError("energy_keV must have one value per initial particle")
    mean_energy = float(energy.mean())
    if mean_energy <= 0.0:
        raise ValueError("initial mean energy must be positive")
    return sampled_beam_metrics(
        position[:, 0] / _ANG_PER_MM,
        directions[:, 0] / directions[:, 2],
        position[:, 1] / _ANG_PER_MM,
        directions[:, 1] / directions[:, 2],
        _array(arrival_offset_ang, name="arrival_offset_ang") / _ANG_PER_FS,
        energy / mean_energy - 1.0,
        energy_keV=mean_energy,
        bunch_charge_pc=beam.bunch_charge_pc,
        rep_rate_hz=beam.rep_rate_hz,
    )
