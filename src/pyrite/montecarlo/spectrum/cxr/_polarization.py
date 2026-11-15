"""Vectorized line polarization bases (#1200).

Separate from :mod:`._kernels` so the batched table builder can form every
reflection row's basis in whole-array operations.
"""

import numpy as np

from ._kernels import _polarization_pair


def _row_norms(rows):
    """``np.linalg.norm`` of each row, by its own 1-D ``sqrt(x.dot(x))``."""
    return np.sqrt(np.array([row.dot(row) for row in rows], dtype=float))


def _polarization_pairs(k_hat, g_rows):
    """:func:`_polarization_pair` for every row of ``g_rows`` (``(N, 3)``).

    The cross products and quotients are the same elementwise operations, and
    the norms are each row's own 1-D norm, so every row is bit-identical to
    the scalar call (#1200). Rows with ``g`` parallel to ``k_hat`` take the
    scalar fallback.
    """
    g_rows = np.asarray(g_rows, dtype=float).reshape(-1, 3)
    k_hat = np.asarray(k_hat, dtype=float)
    n_plane = np.cross(k_hat[None, :], g_rows)
    npl = _row_norms(n_plane)
    degenerate = npl < 1e-12
    npl[degenerate] = 1.0
    e_s = n_plane / npl[:, None]
    e_p = np.cross(e_s, k_hat[None, :])
    epl = _row_norms(e_p)
    epl[degenerate] = 1.0  # zero rows, replaced below; avoids a 0/0 warning
    e_p /= epl[:, None]
    for i in np.flatnonzero(degenerate):
        e_s[i], e_p[i] = _polarization_pair(k_hat, g_rows[i])
    return e_s, e_p
