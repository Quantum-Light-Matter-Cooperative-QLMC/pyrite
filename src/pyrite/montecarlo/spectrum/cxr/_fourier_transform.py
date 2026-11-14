"""Experimental transform callbacks; FINUFFT is an optional runtime import.

Validation: cxr-line-fourier-reduction
"""

from importlib import import_module

import numpy as np

_DIRECT_ELEMENTS = 1 << 18


def direct_mode_transform(phase, strengths, j0, j1, *, max_elements=_DIRECT_ELEMENTS):
    """Batched ``sum_l c_l exp(-i*j*phase_l)`` with bounded exponential blocks.

    Callback for ``partitioned_positive_modes``. Both source and mode axes
    are blocked; ``max_elements`` limits each complex exponential matrix,
    not the returned output or all process memory. Inputs follow the planner
    contract (validated real phases, two moment rows, half-open mode range).

    Validation: cxr-line-fourier-reduction
    """
    if max_elements < 1:
        raise ValueError("max_elements must be positive")
    out = np.zeros((strengths.shape[0], j1 - j0), dtype=np.complex128)
    source_block = min(1024, max_elements)
    mode_block = max(1, max_elements // source_block)
    for start in range(j0, j1, mode_block):
        stop = min(j1, start + mode_block)
        j = np.arange(start, stop)
        for lo in range(0, phase.size, source_block):
            hi = min(phase.size, lo + source_block)
            waves = np.exp(-1j * phase[lo:hi, None] * j)
            out[:, start - j0 : stop - j0] += strengths[:, lo:hi] @ waves
    return out


def finufft_mode_transform(phase, strengths, j0, j1, *, eps=1e-12, nthreads=1):
    """Batched FINUFFT type-1 callback in increasing requested integer modes.

    FINUFFT's centred output uses k=-floor(m/2)..ceil(m/2)-1. Modulating
    sources by exp(-i*shift*phase), shift=j0+floor(m/2), maps those k to
    j0..j1-1. Explicit negative sign, CMCL ordering, complex128, and thread
    count prevent reliance on library defaults. ``eps`` is the library's
    requested transform tolerance, not a bound on final spectral error.

    Validation: cxr-line-fourier-reduction
    """
    finufft = import_module("finufft")
    modes = j1 - j0
    shift = j0 + modes // 2
    sources = np.ascontiguousarray(strengths * np.exp(-1j * shift * phase), dtype=np.complex128)
    return finufft.nufft1d1(
        np.ascontiguousarray(phase, dtype=np.float64),
        sources,
        modes,
        eps=eps,
        isign=-1,
        modeord=0,
        nthreads=nthreads,
    )


def hybrid_mode_transform(phase, strengths, j0, j1, *, eps=1e-12, nthreads=1):
    """Provisional cost rule: direct small rectangles, batched FINUFFT otherwise.

    Thresholds match the #1192 frozen-chunk cost audit (12521). They are not
    tuned crossover measurements. No FINUFFT import occurs for small blocks.

    Validation: cxr-line-fourier-reduction
    """
    modes = j1 - j0
    if phase.size >= 64 and modes >= 32 and phase.size * modes >= 262144:
        return finufft_mode_transform(phase, strengths, j0, j1, eps=eps, nthreads=nthreads)
    return direct_mode_transform(phase, strengths, j0, j1)
