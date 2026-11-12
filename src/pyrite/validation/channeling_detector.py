"""Finite square-aperture quadrature for intrinsic channeling diagnostics.

The face points at the source; its horizontal axis lies in the beam/centre-ray
plane. This is geometric acceptance only, without sensor response.
"""

from __future__ import annotations

import numpy as np


def square_aperture_rule(beam, center_ray, solid_angle_sr, distance_mm, order=3):
    """Return directions, solid-angle weights and side length [mm].

    Tensor Gauss–Legendre quadrature integrates the exact planar-face Jacobian
    ``dΩ = d dx dy / (d² + x² + y²)^(3/2)``. The square's side is obtained
    by inverting the exact on-axis rectangular solid angle. ``order`` is a
    numerical integration order, independent of the detector pixel count.
    For a collinear beam and centre ray, roll is defined by the least-aligned
    Cartesian axis; those two vectors alone do not define a physical roll.

    Source: inverse of :func:`pyrite._planar_geometry.solid_angle_sr` and
    the planar projected-area Jacobian; no radiation model is introduced.
    """
    omega, distance = float(solid_angle_sr), float(distance_mm)
    if not np.isfinite(omega) or not 0 < omega < 2 * np.pi:
        raise ValueError("solid_angle_sr must be finite and between 0 and 2*pi")
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("distance_mm must be finite and positive")
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    beam, normal = np.asarray(beam, dtype=float), np.asarray(center_ray, dtype=float)
    for vector in (beam, normal):
        if vector.shape != (3,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) == 0:
            raise ValueError("beam and center_ray must be finite nonzero 3-vectors")
    beam, normal = beam / np.linalg.norm(beam), normal / np.linalg.norm(normal)
    horizontal = beam - (beam @ normal) * normal
    if np.linalg.norm(horizontal) < 1e-12:
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        horizontal = axis - (axis @ normal) * normal
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    tangent = np.tan(omega / 4)
    half = distance * np.sqrt(tangent**2 + tangent * np.hypot(1, tangent))
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, y = np.meshgrid(half * nodes, half * nodes)
    centers = distance * normal + x[..., None] * horizontal + y[..., None] * vertical
    lengths = np.linalg.norm(centers, axis=-1)
    directions = centers / lengths[..., None]
    domega = np.outer(weights, weights) * half**2 * distance / lengths**3
    return directions.reshape(-1, 3), domega.ravel(), float(2 * half)
