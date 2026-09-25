"""Host-side bulk dielectric loss spectrum for an unbounded material.

This candidate covers only the tabulated valence response. It does not
include deep-core losses, surfaces, recoil products, or stopping calibration.
"""

from dataclasses import dataclass

import numpy as np
from scipy.constants import physical_constants

_BOHR_RADIUS_ANG = 1e10 * physical_constants["Bohr radius"][0]


@dataclass(frozen=True)
class BulkValencePartition:
    """Piecewise-linear valence loss spectrum on a caller-supplied energy grid."""

    energy_ev: float
    threshold_ev: float
    soft_stopping_ev_per_ang: float
    hard_stopping_ev_per_ang: float
    hard_rate_per_ang: float
    total_rate_per_ang: float
    total_stopping_ev_per_ang: float
    lower_ev: np.ndarray
    upper_ev: np.ndarray
    density_lower_per_ang_ev: np.ndarray
    density_upper_per_ang_ev: np.ndarray
    cumulative_hard_per_ang: np.ndarray


def _linear_loss_moments(lower, upper, y0, y1):
    width = upper - lower
    slope = y1 - y0
    zeroth = width * (y0 + 0.5 * slope)
    first = width * (lower * (y0 + 0.5 * slope) + width * (0.5 * y0 + slope / 3.0))
    return zeroth, first


def _elf(loss_ev: np.ndarray, recoil_ev: np.ndarray, oscillators: np.ndarray) -> np.ndarray:
    """Evaluate the fitted loss function on broadcast loss/recoil arrays."""
    center, strength, width, dispersion = oscillators.T
    resonance = center + dispersion * recoil_ev[..., None]
    loss = loss_ev[..., None]
    return np.sum(
        strength * width * loss / ((resonance**2 - loss**2) ** 2 + (width * loss) ** 2),
        axis=-1,
    )


def bulk_dielectric_diimfp(
    energy_ev: float,
    loss_ev: np.ndarray,
    oscillators: np.ndarray,
    gap_ev: float,
    *,
    quadrature_order: int = 96,
) -> np.ndarray:
    """Return the valence bulk DIIMFP in Å⁻¹ eV⁻¹.

    ``oscillators`` has rows (center eV, strength eV², width eV, dispersion).
    Source: Tougaard and Yubero, Surf. Interface Anal. 54 (2022), Eqs. 9,
    10, 12. Assumptions: isotropic homogeneous bulk, nonrelativistic electron,
    loss small compared with incident energy, and supplied oscillator fit.
    The integration variable is recoil energy Q=ħ²k²/(2m); dk/k=dQ/(2Q).
    Limit: losses below the gap or at either kinematic endpoint have zero rate.
    Validation: dielectric-bulk-loss
    """
    losses = np.asarray(loss_ev, dtype=np.float64)
    osc = np.asarray(oscillators, dtype=np.float64)
    if losses.ndim != 1:
        raise ValueError("loss energies must be a one-dimensional array")
    if osc.ndim != 2 or osc.shape[1] != 4 or osc.shape[0] == 0:
        raise ValueError("oscillators must have four columns and at least one row")
    if not np.isfinite(energy_ev) or energy_ev <= 0.0:
        raise ValueError("incident energy must be finite and positive")
    if not np.isfinite(gap_ev) or gap_ev < 0.0:
        raise ValueError("band gap must be finite and non-negative")
    if not np.all(np.isfinite(losses)) or np.any((losses < 0.0) | (losses > energy_ev)):
        raise ValueError("loss energies must be finite and within [0, E]")
    if (
        not np.all(np.isfinite(osc))
        or np.any(osc[:, 0] <= 0.0)
        or np.any(osc[:, 1] < 0.0)
        or np.any(osc[:, 2] <= 0.0)
        or np.any(osc[:, 3] < 0.0)
    ):
        raise ValueError("oscillator parameters must be finite and physical")
    if quadrature_order < 8:
        raise ValueError("quadrature order must be at least eight")

    result = np.zeros_like(losses)
    active = np.flatnonzero((losses > gap_ev) & (losses < energy_ev))
    if active.size == 0:
        return result
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    sqrt_energy = np.sqrt(energy_ev)
    for start in range(0, active.size, 256):
        selected = active[start : start + 256]
        loss = losses[selected]
        sqrt_remaining = np.sqrt(energy_ev - loss)
        q_min = (loss / (sqrt_energy + sqrt_remaining)) ** 2
        q_max = (sqrt_energy + sqrt_remaining) ** 2
        log_min = np.log(q_min)
        log_span = np.log(q_max) - log_min
        q = np.exp(log_min[:, None] + 0.5 * log_span[:, None] * (nodes[None, :] + 1.0))
        elf = _elf(loss[:, None], q, osc)
        result[selected] = (
            log_span
            * np.sum(elf * weights[None, :], axis=1)
            / (4.0 * np.pi * energy_ev * _BOHR_RADIUS_ANG)
        )
    return result


def build_bulk_valence_partition(
    energy_ev: float,
    loss_grid_ev: np.ndarray,
    oscillators: np.ndarray,
    gap_ev: float,
    threshold_ev: float,
) -> BulkValencePartition:
    """Split one fixed, linearly interpolated valence DIIMFP at Wc.

    Source: Tougaard and Yubero (2022), Eqs. 9–12 for the differential
    inverse mean free path; exact polynomial moments of its linear bins.
    Assumptions: homogeneous bulk, supplied valence oscillator fit and grid;
    the caller resolves the high-loss cutoff and grid convergence. No core
    tail or corrected-stopping calibration is implied.
    Limit: Wc at or above the grid maximum leaves no hard valence events.
    Validation: dielectric-bulk-loss
    """
    grid = np.asarray(loss_grid_ev, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("loss grid must be a finite vector with at least two points")
    if np.any(np.diff(grid) <= 0.0) or grid[0] != 0.0 or grid[-1] >= energy_ev:
        raise ValueError("loss grid must start at zero and increase within [0, E)")
    if not np.isfinite(threshold_ev) or threshold_ev <= 0.0:
        raise ValueError("inelastic transfer threshold must be finite and positive")
    # Preserve the zero-rate band-gap boundary under linear interpolation.
    if 0.0 < gap_ev < grid[-1]:
        grid = np.union1d(grid, gap_ev)
    rate = bulk_dielectric_diimfp(energy_ev, grid, oscillators, gap_ev)
    lower, upper = grid[:-1], grid[1:]
    y0, y1 = rate[:-1], rate[1:]
    total0, total1 = _linear_loss_moments(lower, upper, y0, y1)
    keep = upper > threshold_ev
    hard_lower = np.maximum(lower[keep], threshold_ev)
    hard_upper = upper[keep]
    hard_y0 = y0[keep] + (y1[keep] - y0[keep]) * (
        (hard_lower - lower[keep]) / (upper[keep] - lower[keep])
    )
    hard_y1 = y1[keep]
    hard0, hard1 = _linear_loss_moments(hard_lower, hard_upper, hard_y0, hard_y1)
    positive = hard0 > 0.0
    cumulative = np.cumsum(hard0[positive])
    total_stopping = float(np.sum(total1))
    hard_stopping = float(np.sum(hard1))
    return BulkValencePartition(
        energy_ev=float(energy_ev),
        threshold_ev=float(threshold_ev),
        soft_stopping_ev_per_ang=total_stopping - hard_stopping,
        hard_stopping_ev_per_ang=hard_stopping,
        hard_rate_per_ang=float(cumulative[-1]) if cumulative.size else 0.0,
        total_rate_per_ang=float(np.sum(total0)),
        total_stopping_ev_per_ang=total_stopping,
        lower_ev=hard_lower[positive],
        upper_ev=hard_upper[positive],
        density_lower_per_ang_ev=hard_y0[positive],
        density_upper_per_ang_ev=hard_y1[positive],
        cumulative_hard_per_ang=cumulative,
    )


def sample_bulk_valence_loss(partition: BulkValencePartition, uniform: float) -> float:
    """Invert the hard-event CDF of a fixed linear valence spectrum.

    Source: exact integral of a linear DIIMFP bin. Assumes one uniform variate
    for an event drawn at the partition's hard rate. At a zero hard rate there
    is no event to sample. Limit: a flat bin reduces to uniform loss sampling.
    Validation: dielectric-bulk-loss
    """
    if partition.hard_rate_per_ang <= 0.0:
        raise ValueError("partition has no hard valence losses")
    if not np.isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise ValueError("loss uniform must be finite and in [0, 1)")
    target = uniform * partition.hard_rate_per_ang
    index = min(
        int(np.searchsorted(partition.cumulative_hard_per_ang, target, side="right")),
        partition.lower_ev.size - 1,
    )
    preceding = partition.cumulative_hard_per_ang[index - 1] if index else 0.0
    lower = partition.lower_ev[index]
    width = partition.upper_ev[index] - lower
    y0 = partition.density_lower_per_ang_ev[index]
    dy = partition.density_upper_per_ang_ev[index] - y0
    area_fraction = (target - preceding) / width
    discriminant = max(y0 * y0 + 2.0 * dy * area_fraction, 0.0)
    denominator = y0 + np.sqrt(discriminant)
    fraction = 2.0 * area_fraction / denominator if denominator > 0.0 else 0.0
    return float(lower + width * np.clip(fraction, 0.0, 1.0))


def sample_bulk_dielectric_recoil(
    energy_ev: float,
    loss_ev: float,
    oscillators: np.ndarray,
    gap_ev: float,
    u_recoil: float,
    *,
    grid_size: int = 2049,
) -> tuple[float, float]:
    """Sample recoil energy Q and primary polar cosine conditional on loss W.

    The finite-momentum ELF is the conditional density in log Q. The polar
    angle follows the nonrelativistic momentum triangle. No azimuth or
    secondary-particle state is assigned here.
    Validation: dielectric-bulk-loss
    """
    if not np.isfinite(u_recoil) or not 0.0 <= u_recoil < 1.0:
        raise ValueError("recoil uniform must be finite and in [0, 1)")
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 33:
        raise ValueError("recoil grid size must be an integer of at least 33")
    rate = bulk_dielectric_diimfp(energy_ev, np.asarray([loss_ev]), oscillators, gap_ev)[0]
    if rate <= 0.0:
        raise ValueError("loss has no positive dielectric recoil rate")

    remaining = energy_ev - loss_ev
    root_e = np.sqrt(energy_ev)
    root_remaining = np.sqrt(remaining)
    q_min = (loss_ev / (root_e + root_remaining)) ** 2
    q_max = (root_e + root_remaining) ** 2
    log_q = np.linspace(np.log(q_min), np.log(q_max), grid_size)
    density = _elf(np.asarray(loss_ev), np.exp(log_q), np.asarray(oscillators))
    areas = 0.5 * (density[:-1] + density[1:]) * np.diff(log_q)
    cumulative = np.concatenate(([0.0], np.cumsum(areas)))
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0.0:
        raise ValueError("loss has no finite dielectric recoil distribution")
    recoil_ev = float(np.exp(np.interp(u_recoil * cumulative[-1], cumulative, log_q)))
    cos_primary = (energy_ev + remaining - recoil_ev) / (2.0 * np.sqrt(energy_ev * remaining))
    return recoil_ev, float(np.clip(cos_primary, -1.0, 1.0))
