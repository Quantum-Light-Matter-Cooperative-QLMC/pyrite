"""Scalar direction and boundary helpers shared by the transport cores.

Split from :mod:`pyrite.montecarlo.transport.cores` to keep that module within
the line budget; the CPU cores and the shell collision sampler import them
from here, and the CUDA kernel inlines the same arithmetic.
"""

import numpy as np
from numba import njit

from ..geometry import X_MAX, X_MIN, Y_MAX, Y_MIN, Z_MAX, Z_MIN


@njit(cache=True)
def _rotate_direction_scalar(dx, dy, dz, cos_t, phi):
    sin_t = np.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    if abs(dx) < 0.9:
        refx, refy = 1.0, 0.0
    else:
        refx, refy = 0.0, 1.0
    ux, uy, uz = -dz * refy, dz * refx, dx * refy - dy * refx
    u_mag = np.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / u_mag, uy / u_mag, uz / u_mag
    wx, wy, wz = dy * uz - dz * uy, dz * ux - dx * uz, dx * uy - dy * ux
    a, b = sin_t * cos_phi, sin_t * sin_phi
    outx, outy, outz = (
        cos_t * dx + a * ux + b * wx,
        cos_t * dy + a * uy + b * wy,
        cos_t * dz + a * uz + b * wz,
    )
    mag = np.sqrt(outx * outx + outy * outy + outz * outz)
    return outx / mag, outy / mag, outz / mag


@njit(cache=True)
def _rotate_directions(d, cos_t, phi):
    """Rotate unit vectors by polar angle ``cos_t`` and azimuth ``phi``."""
    out = np.empty_like(d)
    for i in range(d.shape[0]):
        out[i] = _rotate_direction_scalar(d[i, 0], d[i, 1], d[i, 2], cos_t[i], phi[i])
    return out


@njit(cache=True)
def _first_prism_exit_scalar(px, py, pz, dx, dy, dz, z_min, z_max, width, height):
    """Nearest positive ray/prism intersection, matching face-order ties."""
    best_t, best_face = np.inf, -1
    half_w, half_h = 0.5 * width, 0.5 * height
    if dx < 0.0:
        t = (-half_w - px) / dx
        if 0.0 < t < best_t:
            best_t, best_face = t, X_MIN
    if dx > 0.0:
        t = (half_w - px) / dx
        if 0.0 < t < best_t:
            best_t, best_face = t, X_MAX
    if dy < 0.0:
        t = (-half_h - py) / dy
        if 0.0 < t < best_t:
            best_t, best_face = t, Y_MIN
    if dy > 0.0:
        t = (half_h - py) / dy
        if 0.0 < t < best_t:
            best_t, best_face = t, Y_MAX
    if dz < 0.0:
        t = (z_min - pz) / dz
        if 0.0 < t < best_t:
            best_t, best_face = t, Z_MIN
    if dz > 0.0:
        t = (z_max - pz) / dz
        if 0.0 < t < best_t:
            best_t, best_face = t, Z_MAX
    return best_t, best_face


@njit(cache=True)
def _searchsorted_right_scalar(bounds, x, n):
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if bounds[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo
