"""Primary-beam attenuation through bounded downstream filter plates."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ..materials.attenuation import linear_attenuation_inv_mm

if TYPE_CHECKING:
    from .model import FilterPlate


def attenuation_matrix(filters: Sequence[FilterPlate], energy_eV: np.ndarray) -> np.ndarray:
    """Stack each plate's linear attenuation coefficient in inverse mm.

    Returns shape ``(n_filter, n_energy)``, the ``mu_by_filter_inv_mm`` input of
    :func:`primary_transmission`; no filters gives shape ``(0, n_energy)``. A
    zero photon energy -- the lower node of a grid that starts at 0 eV -- takes
    the photoabsorption limit ``mu -> +inf`` as ``E -> 0+``, so the plate is
    opaque there; tabulated coefficients exist only for positive energies.
    """
    energy = np.asarray(energy_eV, dtype=float)
    if energy.ndim != 1 or not np.all(np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError("energy_eV must be a one-dimensional array of finite non-negative values")
    coefficient = np.full((len(filters), energy.size), np.inf)
    positive = energy > 0.0
    for row, plate in enumerate(filters):
        coefficient[row, positive] = linear_attenuation_inv_mm(plate.material, energy[positive])
    return coefficient


def primary_transmission(path_length_mm: object, mu_by_filter_inv_mm: object) -> np.ndarray:
    """Apply stacked-filter Beer--Lambert attenuation.

    For pixel ``p``, energy ``E``, and filter ``j``, the primary-beam factor is
    ``T_p(E) = exp(-sum_j mu_j(E) * ell_pj)``. Path lengths and attenuation
    coefficients are respectively in millimetres and inverse millimetres, so
    optical depth is dimensionless. Filters act independently; scattering,
    fluorescence, diffraction, and secondary production are outside this
    model. No filters returns an exact multiplicative identity.

    An infinite coefficient marks an opaque energy (see
    :func:`attenuation_matrix`): ``T = 0`` wherever the ray crosses that filter,
    and a zero path length still contributes exactly nothing, so an uncovered
    pixel keeps ``T = 1``. Finite coefficients take the unchanged closed form.

    ``path_length_mm`` has shape ``(..., n_filter)`` and
    ``mu_by_filter_inv_mm`` has shape ``(n_filter, n_energy)``.

    Validation: positioned-filter-attenuation
    """
    try:
        paths = np.asarray(path_length_mm, dtype=float)
        coefficient = np.asarray(mu_by_filter_inv_mm, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("paths and attenuation coefficients must be real arrays") from exc
    if paths.ndim < 1:
        raise ValueError("path_length_mm must have at least one dimension")
    if coefficient.ndim != 2:
        raise ValueError("mu_by_filter_inv_mm must be two-dimensional")
    if paths.shape[-1] != coefficient.shape[0]:
        raise ValueError("path and attenuation arrays must have the same filter count")
    if coefficient.shape[1] == 0:
        raise ValueError("mu_by_filter_inv_mm must contain at least one energy")
    if not np.all(np.isfinite(paths)) or np.any(paths < 0.0):
        raise ValueError("path_length_mm must contain only finite non-negative values")
    if np.any(np.isnan(coefficient)) or np.any(coefficient < 0.0):
        raise ValueError(
            "mu_by_filter_inv_mm must contain only non-negative values (+inf marks an opaque energy)"
        )
    if paths.shape[-1] == 0:
        return np.ones((*paths.shape[:-1], coefficient.shape[1]), dtype=float)
    opaque = np.isinf(coefficient)
    if not np.any(opaque):
        return np.exp(-np.tensordot(paths, coefficient, axes=([-1], [0])))
    # 0 * inf is undefined, so the opaque entries are split out: they block a
    # ray exactly when its path through that filter is positive.
    depth = np.tensordot(paths, np.where(opaque, 0.0, coefficient), axes=([-1], [0]))
    blocked = np.tensordot(paths > 0.0, opaque, axes=([-1], [0])) > 0
    return np.where(blocked, 0.0, np.exp(-depth))


__all__ = ["primary_transmission"]
