"""Shared scalar inputs and sample-frame rotations for photon references."""

from collections.abc import Iterable, Iterator

import numpy as np

from ...materials.photon_cross_sections import EPDL_E_MAX_EV, EPDL_E_MIN_EV

MAX_REJECTIONS = 100_000


class UniformDraws(Iterator[float]):
    """Count consumed draws without creating or owning a random stream."""

    def __init__(self, uniforms: Iterable[float]) -> None:
        self._iterator = iter(uniforms)
        self.count = 0

    def __next__(self) -> float:
        try:
            draw = float(next(self._iterator))
        except (StopIteration, TypeError, ValueError) as exc:
            raise ValueError("photon uniform sequence exhausted or contains a non-scalar") from exc
        if not np.isfinite(draw) or not 0 <= draw < 1:
            raise ValueError("photon uniforms must be finite and in [0, 1)")
        self.count += 1
        return draw


def incident_ray(energy_eV: float, direction: object) -> np.ndarray:
    """Require a supported scalar energy [eV] and finite unit direction."""
    if not np.isfinite(energy_eV) or not EPDL_E_MIN_EV <= energy_eV <= EPDL_E_MAX_EV:
        raise ValueError("photon energy is outside the EPDL domain")
    ray = np.asarray(direction, dtype=float)
    if ray.shape != (3,) or not np.isfinite(ray).all():
        raise ValueError("incident direction must have shape (3,) and be finite")
    if not np.isclose(np.linalg.norm(ray), 1, rtol=0, atol=1e-12):
        raise ValueError("incident direction must be a unit vector")
    return ray


def cumulative_weights(weights: np.ndarray) -> np.ndarray:
    """Finite nonnegative categorical weights, with a positive finite total."""
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim != 1
        or not weights.size
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("interaction weights must be finite and non-negative")
    cumulative = np.cumsum(weights)
    if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0:
        raise ValueError("interaction weights must have a positive finite total")
    return cumulative


def select_interval(cumulative: np.ndarray, draw: float) -> int:
    """Strict inverse CDF, including zero weights and the last representable draw."""
    threshold = min(draw * cumulative[-1], np.nextafter(cumulative[-1], 0.0))
    return int(np.searchsorted(cumulative, threshold, side="right"))


def scatter_direction(
    ray: np.ndarray, delta: float, azimuth: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate by ``delta=1-cos(theta)`` and azimuth [rad] about a unit ray.

    A least-aligned Cartesian axis gives a nondegenerate transverse basis.
    Returns the rotated direction and its transverse unit azimuth vector.
    Source: orthonormal spherical-coordinate rotation, n'=n cosθ+eφ sinθ.
    Assumes unpolarized photons; delta=0 preserves the incident direction.
    Validation: photon-compton-scattering, photon-rayleigh-scattering,
        photon-photoelectric-sampling
    """
    axis = np.eye(3)[np.argmin(np.abs(ray))]
    tangent = np.cross(ray, axis)
    tangent /= np.linalg.norm(tangent)
    transverse = np.cos(azimuth) * tangent + np.sin(azimuth) * np.cross(ray, tangent)
    sine = np.sqrt(delta * (2 - delta))
    direction = ray * (1 - delta) + transverse * sine
    direction.flags.writeable = False
    return direction, transverse


def clipped_factor_table(
    table: tuple[np.ndarray, np.ndarray], limit: float
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict a lin-lin table to [0, limit], retaining its final partial interval."""
    x, values = table
    stop = np.searchsorted(x, limit, side="right")
    grid, factor = x[:stop], values[:stop]
    if grid[-1] < limit:
        grid = np.append(grid, limit)
        factor = np.append(factor, np.interp(limit, x, values))
    if not np.isfinite(factor).all() or np.any(factor < 0):
        raise ValueError("scattering factors must be finite and non-negative")
    return grid, factor
