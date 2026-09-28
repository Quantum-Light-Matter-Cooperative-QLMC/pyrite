"""Spectrum chunk sizing and shared runner resource policy."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import psutil

from ..._backend import _GPU, BACKEND
from ..._env import env_value
from .. import spectrum as _spectrum_mod
from .._resources import admitted_chunk, resolve_resource_policy


def _env_chunk(name, default):
    """Chunk-size default, overridable via env for the A1 sweep-acceleration spike
    (docs/repo-design/compute/compute-performance-optimization.md: sweep spec/brem chunk on the lab
    box and read the GPU spectrum-phase time). Read once at import so it applies in
    the main GPU process; an explicit per-case ``spec_chunk``/``brem_chunk`` still
    wins. Unset / blank / non-positive / non-integer -> the memory-safe default."""
    try:
        v = int(env_value(name, ""))
    except TypeError, ValueError:
        return default
    return v if v > 0 else default


def _available_mem_mb() -> int:
    return psutil.virtual_memory().available // 1_000_000


@dataclass
class _RunnerResourcePolicy:
    """Mutable runner resource decisions and test seams, resolved once at import."""

    requested: str
    name: str
    device_budget_bytes: int | None
    device_reserve_bytes: int | None
    host_fraction: float
    host_reserve_bytes: int
    oom_retries: int
    release_every: int
    gpu: bool
    nsys: bool
    n_cpus: int | None
    total_mem_mb: int
    available_mem_mb: Callable[[], int]
    spec_chunk: int
    brem_chunk: int
    spec_budget_mb: int
    free_every: int
    worker_mem_mb: int
    pipeline_worker_mem_mb: int
    pipeline_prefetch_ahead: int
    free_watermark_mb: int
    gpu_pool_fraction: float
    gpu_pool_share: int
    gpu_oom_retries: int
    gpu_oom: tuple[type[BaseException], ...]
    cases_since_free: int = 0
    pool_peak_bytes: int = 0
    pool_limit_set: bool = False


def _resolve_runner_resource_policy() -> _RunnerResourcePolicy:
    resolved = resolve_resource_policy(BACKEND)
    policy_device_mb = (
        resolved.device_budget_bytes // (1 << 20)
        if resolved.device_budget_bytes is not None
        else 1920
    )
    gpu_pool_fraction = float(
        env_value(
            "PYRITE_MC_GPU_POOL_FRAC",
            (
                str(resolved.device_budget_bytes / BACKEND.device.total_memory_bytes)
                if _GPU
                and resolved.device_budget_bytes is not None
                and BACKEND.device.total_memory_bytes
                else "0"
            ),
        )
    )
    return _RunnerResourcePolicy(
        requested=resolved.requested,
        name=resolved.name,
        device_budget_bytes=resolved.device_budget_bytes,
        device_reserve_bytes=resolved.device_reserve_bytes,
        host_fraction=resolved.host_fraction,
        host_reserve_bytes=resolved.host_reserve_bytes,
        oom_retries=resolved.oom_retries,
        release_every=resolved.release_every,
        gpu=_GPU,
        nsys=env_value("PYRITE_MC_NSYS", "") not in ("", "0"),
        n_cpus=None,
        total_mem_mb=psutil.virtual_memory().total // 1_000_000,
        available_mem_mb=_available_mem_mb,
        spec_chunk=_env_chunk("PYRITE_MC_SPEC_CHUNK", 0),
        brem_chunk=_env_chunk("PYRITE_MC_BREM_CHUNK", 0),
        spec_budget_mb=_env_chunk("PYRITE_MC_SPEC_BUDGET_MB", min(1920, policy_device_mb)),
        free_every=_env_chunk("PYRITE_MC_FREE_EVERY", resolved.release_every),
        worker_mem_mb=_env_chunk("PYRITE_MC_WORKER_MEM_MB", 6144),
        pipeline_worker_mem_mb=_env_chunk("PYRITE_MC_PIPELINE_WORKER_MEM_MB", 1536),
        pipeline_prefetch_ahead=2,
        free_watermark_mb=_env_chunk("PYRITE_MC_FREE_WATERMARK_MB", 0),
        gpu_pool_fraction=gpu_pool_fraction,
        gpu_pool_share=max(1, _env_chunk("PYRITE_MC_GPU_SHARE", 1)),
        gpu_oom_retries=_env_chunk("PYRITE_MC_GPU_OOM_RETRIES", resolved.oom_retries),
        gpu_oom=BACKEND.oom_exceptions,
    )


# Segments per spectrum matmul. Explicit per-case values still win over policy.
_RESOURCE_POLICY = _resolve_runner_resource_policy()
_EEDL_BREM_DENSE_INTERMEDIATES = 8


def _adaptive_chunk(nbins, *, intermediates=3):
    """Segments per spectrum matmul sized so the ~3 concurrent (chunk, nbins)
    intermediates in the mc_spectrum / mc_brem_spectrum chunk loops fit in
    the policy spectrum budget.

    Replaces the fixed defaults after the 2026-07-18 remote-host OOMs: widening the
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
    per_row_bytes = int(intermediates) * nbins * itemsize

    requested = max(
        1000,
        min(
            1_000_000 * _RESOURCE_POLICY.spec_budget_mb // per_row_bytes,
            100_000,
        ),
    )

    return admitted_chunk(
        requested_chunk=requested,
        bins=nbins,
        itemsize=itemsize,
        budget_bytes=(_RESOURCE_POLICY.device_budget_bytes if _RESOURCE_POLICY.gpu else None),
        intermediates=int(intermediates),
    )


def _admit_chunk(chunk, bins, *, intermediates=3):
    itemsize = _real_itemsize()

    return admitted_chunk(
        requested_chunk=int(chunk),
        bins=int(bins),
        itemsize=itemsize,
        budget_bytes=(_RESOURCE_POLICY.device_budget_bytes if _RESOURCE_POLICY.gpu else None),
        intermediates=int(intermediates),
    )


def _real_itemsize() -> int:
    return np.dtype(_spectrum_mod.REAL).itemsize
