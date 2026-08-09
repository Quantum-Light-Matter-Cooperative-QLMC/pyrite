"""Spectrum chunk sizing and device-memory admission."""

import numpy as np

from ..._compat import env_value
from .. import spectrum as _spectrum_mod
from .._backend import _GPU, BACKEND
from .._resources import admitted_chunk, resolve_resource_policy


def _env_chunk(name, default):
    """Chunk-size default, overridable via env for the A1 sweep-acceleration spike
    (docs/acceleration-technique-evaluation.md, A1: sweep spec/brem chunk on the lab
    box and read the GPU spectrum-phase time). Read once at import so it applies in
    the main GPU process; an explicit per-case ``spec_chunk``/``brem_chunk`` still
    wins. Unset / blank / non-positive / non-integer -> the memory-safe default."""
    try:
        v = int(env_value(name, ""))
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


# Segments per spectrum matmul. 0 (the default) = size adaptively per call from
# the energy-grid width via _adaptive_chunk; a positive CXR_MC_SPEC_CHUNK /
# CXR_MC_BREM_CHUNK env value pins a fixed chunk (A1 sweep-accel spike), and an
# explicit per-case spec_chunk/brem_chunk still wins over both.
_RESOURCE_POLICY = resolve_resource_policy(BACKEND)
_SPEC_CHUNK = _env_chunk("CXR_MC_SPEC_CHUNK", 0)
_BREM_CHUNK = _env_chunk("CXR_MC_BREM_CHUNK", 0)
# Transient-memory budget [MB] for one spectrum matmul. The chunk loops in
# mc_spectrum / mc_brem_spectrum hold ~3 float64 (chunk, nbins) arrays at peak
# (x, S = sinc(x)**2, and sinc's internal temporary), so transient bytes
# ~= 3 * chunk * nbins * 8. Default 1920 MB reproduces the old fixed
# chunk=40000 exactly on the pre-2026-07 ~2000-bin line grid.
_POLICY_DEVICE_MB = (
    _RESOURCE_POLICY.device_budget_bytes // (1 << 20)
    if _RESOURCE_POLICY.device_budget_bytes is not None
    else 1920
)
_SPEC_BUDGET_MB = _env_chunk("CXR_MC_SPEC_BUDGET_MB", min(1920, _POLICY_DEVICE_MB))


def _adaptive_chunk(nbins):
    """Segments per spectrum matmul sized so the ~3 concurrent (chunk, nbins)
    intermediates in the mc_spectrum / mc_brem_spectrum chunk loops fit in
    _SPEC_BUDGET_MB.

    Replaces the fixed defaults after the 2026-07-18 qlmc OOMs: widening the
    line grid to 30000 eV grew nbins ~3x and silently tripled the per-matmul
    transient (the fixed chunk had been tuned on the old narrow grid). Holding
    the byte product constant instead means the chunk shrinks as the grid
    widens and grows as it narrows. Chunking is mathematically exact (it only
    partitions a sum over segments), so this changes memory/speed, not physics.

    The transients (x, sinc(x), sinc(x)**2) are REAL-dtype, so the byte budget
    uses _REAL_BYTES: 4 on the GPU (fp32) -> ~2x the chunk the old hardcoded 8
    allowed, 8 on the CPU (fp64) -> the original size bit-for-bit.
    """
    itemsize = _real_itemsize()
    worker_intermediate_arrays = 3
    per_row_bytes = worker_intermediate_arrays * nbins * itemsize

    requested = max(
        1000,
        min(
            1_000_000 * _SPEC_BUDGET_MB // per_row_bytes,
            100_000,
        ),
    )

    return admitted_chunk(
        requested_chunk=requested,
        bins=nbins,
        itemsize=itemsize,
        budget_bytes=(_RESOURCE_POLICY.device_budget_bytes if _GPU else None),
    )


def _admit_chunk(chunk, bins):
    itemsize = _real_itemsize()

    return admitted_chunk(
        requested_chunk=int(chunk),
        bins=int(bins),
        itemsize=itemsize,
        budget_bytes=(_RESOURCE_POLICY.device_budget_bytes if _GPU else None),
    )


def _real_itemsize() -> int:
    return np.dtype(_spectrum_mod.REAL).itemsize
