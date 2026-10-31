"""Fused NumPy-backend line reduction: ``sum_j w_j sinc^2`` without temporaries.

The CPU fallback used to build the dense ``(n_line, n_E)`` ``_sincsq_lineshape``
matrix eagerly (five full-size temporaries per chunk) and contract it with a
BLAS matvec per target. Profiling a Si 300 keV CPU run put ~85% of the line
phase in that elementwise pass. This kernel evaluates each ``(line, energy)``
profile value once, accumulates every target from it, and parallelises over
energies, so no two iterations write the same output element.

Per-element profile arithmetic matches ``_sincsq_lineshape`` (same ``x``, same
``x == 0`` guard); only the summation order over lines differs from the BLAS
contraction, a rounding-level reassociation.

Validation: line-hkl-batch
"""

import numpy as np
from numba import njit, prange

_PI = np.pi


@njit(parallel=True, cache=True)
def _sincsq_reduce(E_r, a_width, weights, E_grid, out):  # pragma: no cover - numba
    # Three scalar accumulators (unused weight rows are zero) keep the inner
    # loop free of output stores; ``sin`` dominates, the spare FMAs are free.
    n_l = E_r.shape[0]
    for e in prange(E_grid.shape[0]):
        E = E_grid[e]
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for j in range(n_l):
            x = a_width[j] * (E - E_r[j]) / _PI
            if x == 0.0:
                x = 1.0e-20
            y = _PI * x
            s = np.sin(y) / y
            v = s * s
            s0 += weights[0, j] * v
            s1 += weights[1, j] * v
            s2 += weights[2, j] * v
        out[0, e] += s0
        out[1, e] += s1
        out[2, e] += s2


def numpy_backend() -> bool:
    """True when the spectrum array namespace is NumPy (CPU scoring)."""
    from ...._backend import xp

    return getattr(xp, "__name__", "") == "numpy"


def accumulate_sincsq(E_r, a_width, weights, E_grid, targets):
    """Add ``weights[t] @ sinc^2(a_width (E_grid - E_r))`` into ``targets[t]``.

    ``E_r``, ``a_width`` are 1-D line arrays; ``weights`` is a sequence of
    1-D per-line weights, one per 1-D ``targets`` spectrum on ``E_grid``.
    NumPy arrays only; callers keep the array-API path for other backends.
    """
    if E_r.size == 0 or E_grid.size == 0:
        return
    if not 1 <= len(targets) <= 3 or len(weights) != len(targets):
        raise ValueError("accumulate_sincsq takes one to three weight/target pairs")
    dtype = np.result_type(E_r, a_width, E_grid, *weights)
    w = np.zeros((3, E_r.size), dtype=dtype)
    for t, row in enumerate(weights):
        w[t] = row
    out = np.zeros((3, E_grid.size), dtype=dtype)
    _sincsq_reduce(
        np.ascontiguousarray(E_r, dtype=dtype),
        np.ascontiguousarray(a_width, dtype=dtype),
        w,
        np.ascontiguousarray(E_grid, dtype=dtype),
        out,
    )
    for t, tgt in enumerate(targets):
        tgt += out[t]
