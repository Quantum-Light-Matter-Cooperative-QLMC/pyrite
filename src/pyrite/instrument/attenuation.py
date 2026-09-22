"""Primary-beam attenuation through bounded downstream filter plates."""

import numpy as np


def primary_transmission(path_length_mm: object, mu_by_filter_inv_mm: object) -> np.ndarray:
    """Apply stacked-filter Beer--Lambert attenuation.

    For pixel ``p``, energy ``E``, and filter ``j``, the primary-beam factor is
    ``T_p(E) = exp(-sum_j mu_j(E) * ell_pj)``. Path lengths and attenuation
    coefficients are respectively in millimetres and inverse millimetres, so
    optical depth is dimensionless. Filters act independently; scattering,
    fluorescence, diffraction, and secondary production are outside this
    model. No filters returns an exact multiplicative identity.

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
    if not np.all(np.isfinite(coefficient)) or np.any(coefficient < 0.0):
        raise ValueError("mu_by_filter_inv_mm must contain only finite non-negative values")
    if paths.shape[-1] == 0:
        return np.ones((*paths.shape[:-1], coefficient.shape[1]), dtype=float)
    return np.exp(-np.tensordot(paths, coefficient, axes=([-1], [0])))


__all__ = ["primary_transmission"]
