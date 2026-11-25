"""Spectrum chunk sizing and shared runner resource policy."""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module

import numpy as np
import psutil

from ..._backend import _GPU, BACKEND, NumPyBackend
from ..._util.env import env_value
from .. import spectrum as _spectrum_mod
from .._resources import admitted_chunk, resolve_resource_policy
from ..spectrum.brem_blocks import eedl_staged_panel_count
from ..transport import host_memory


def _env_chunk(name, default):
    """Chunk-size default, overridable via env for the A1 sweep-acceleration spike
    (docs/repo-design/compute/compute-performance-optimization.md: sweep spec/brem chunk on the lab
    box and read the GPU spectrum-phase time). Read once at import so it applies in
    the main GPU process; an explicit per-case ``spec_chunk``/``brem_chunk`` still
    wins. Unset / blank / non-positive / non-integer -> the memory-safe default."""
    try:
        v = int(env_value(name, ""))
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


def _available_mem_mb() -> int:
    """MemAvailable, bounded by the tightest cgroup v2 headroom (#1460)."""
    available = int(psutil.virtual_memory().available)
    headroom = host_memory._cgroup_headroom_bytes()
    return (available if headroom is None else min(available, headroom)) // 1_000_000


def _total_mem_mb() -> int:
    """MemTotal, bounded by the tightest cgroup v2 ``memory.max`` (#1460).

    Inside a batch job the cgroup limit, not the node's RAM, is what the
    kernel OOM-kills against."""
    total = int(psutil.virtual_memory().total)
    limit = host_memory._cgroup_limit_bytes()
    return (total if limit is None else min(total, limit)) // 1_000_000


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
    #: Pinned characteristic segment chunk (``PYRITE_MC_CHARACTERISTIC_CHUNK``);
    #: 0 selects :func:`_characteristic_auto_chunk` (#1455).
    characteristic_chunk: int
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
    #: Allow the CUDA transport pool ceiling above the budget (#1315): only
    #: under the auto resource policy without an explicit pool fraction.
    transport_ceiling: bool = False
    #: Pool cap this process last set and its phase ("transport"/"budget").
    pool_limit_bytes: int | None = None
    pool_limit_phase: str | None = None


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
        total_mem_mb=_total_mem_mb(),
        available_mem_mb=_available_mem_mb,
        spec_chunk=_env_chunk("PYRITE_MC_CXR_CHUNK", 0),
        brem_chunk=_env_chunk("PYRITE_MC_BREM_CHUNK", 0),
        characteristic_chunk=_env_chunk("PYRITE_MC_CHARACTERISTIC_CHUNK", 0),
        spec_budget_mb=_env_chunk("PYRITE_MC_CXR_BUDGET_MB", min(1920, policy_device_mb)),
        free_every=_env_chunk("PYRITE_MC_FREE_EVERY", resolved.release_every),
        worker_mem_mb=_env_chunk("PYRITE_MC_WORKER_MEM_MB", 6144),
        pipeline_worker_mem_mb=_env_chunk("PYRITE_MC_PIPELINE_WORKER_MEM_MB", 1536),
        pipeline_prefetch_ahead=2,
        free_watermark_mb=_env_chunk("PYRITE_MC_FREE_WATERMARK_MB", 0),
        gpu_pool_fraction=gpu_pool_fraction,
        gpu_pool_share=max(1, _env_chunk("PYRITE_MC_GPU_SHARE", 1)),
        gpu_oom_retries=_env_chunk("PYRITE_MC_GPU_OOM_RETRIES", resolved.oom_retries),
        gpu_oom=BACKEND.oom_exceptions,
        transport_ceiling=(
            resolved.requested == "auto" and env_value("PYRITE_MC_GPU_POOL_FRAC") is None
        ),
    )


# Segments per spectrum matmul. Explicit per-case values still win over policy.
_RESOURCE_POLICY = _resolve_runner_resource_policy()
_EEDL_BREM_DENSE_INTERMEDIATES = 8


def _adaptive_chunk(nbins, *, intermediates=3):
    """Segments per spectrum matmul sized so the ~3 concurrent (chunk, nbins)
    intermediates in the mc_cxr_spectrum / mc_brem_spectrum chunk loops fit in
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
        1,
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


def _brem_photon_block(nbins, composition, model, *, n_layers=1):
    """Photon nodes per ``mc_brem_spectrum`` pass so its grid staging fits the budget.

    Per photon node one pass stages each EEDL element's unit-base panel row
    (binary64 on the host, plus a REAL copy on a device; the BremsLib route
    stages EEDL too as its fallback), one ``mu(E)`` per absorbing layer and
    the grid/output vectors; Bethe--Heitler stages no panels. This
    grid-invariant staging gets its own ``spec_budget_mb``: the
    ``(chunk, node)`` transients are already bounded by ``_adaptive_chunk``
    and ``_admit_chunk`` on the whole grid and only shrink inside a block.

    A grid that fits returns ``nbins``, so its path is unchanged. Blocking is
    exact (#819), like segment chunking, so the width is a run diagnostic, not
    identity.
    """
    nbins = int(nbins)
    panels = (
        0
        if model == "bethe-heitler"
        else sum(eedl_staged_panel_count(element) for element, _ in composition)
    )
    device_copy = _real_itemsize() if _RESOURCE_POLICY.gpu else 0
    per_node = panels * (8 + device_copy) + (int(n_layers) + 4) * 8
    return max(1, min(nbins, 1_000_000 * _RESOURCE_POLICY.spec_budget_mb // per_node))


def _admit_chunk(chunk, bins, *, intermediates=3):
    itemsize = _real_itemsize()

    return admitted_chunk(
        requested_chunk=int(chunk),
        bins=int(bins),
        itemsize=itemsize,
        budget_bytes=(_RESOURCE_POLICY.device_budget_bytes if _RESOURCE_POLICY.gpu else None),
        intermediates=int(intermediates),
    )


#: Bytes per characteristic chunk element: the line-group arrays are binary64.
_CHARACTERISTIC_ITEMSIZE = 8


def _characteristic_auto_chunk(n_shell):
    """Device characteristic chunk from its own intermediates (#1455).

    Holds :func:`~pyrite.montecarlo.spectrum.characteristic._chunk_row_elements`
    per row plus the fixed line-group working set within the policy spectrum
    budget, capped at ``_AUTO_CHUNK_ROWS``: wider chunks would save under one
    launch set per million rows while holding more device scratch. Unlike
    :func:`_adaptive_chunk`, no characteristic intermediate spans the
    line-energy grid, so its size does not enter. Chunking only regroups the
    FP64 segment sum (#1313).
    """
    from ..spectrum import characteristic

    per_row = characteristic._chunk_row_elements(n_shell) * _CHARACTERISTIC_ITEMSIZE
    fixed = characteristic._chunk_fixed_elements() * _CHARACTERISTIC_ITEMSIZE
    budget = 1_000_000 * _RESOURCE_POLICY.spec_budget_mb - fixed
    return max(1, min(characteristic._AUTO_CHUNK_ROWS, budget // per_row))


def _admit_characteristic_chunk(chunk, n_shell):
    """Cap a device characteristic chunk to the device budget before allocation."""
    from ..spectrum import characteristic

    budget = _RESOURCE_POLICY.device_budget_bytes if _RESOURCE_POLICY.gpu else None
    if budget is None:
        return int(chunk)
    fixed = characteristic._chunk_fixed_elements() * _CHARACTERISTIC_ITEMSIZE
    return admitted_chunk(
        requested_chunk=int(chunk),
        bins=characteristic._chunk_row_elements(n_shell),
        itemsize=_CHARACTERISTIC_ITEMSIZE,
        budget_bytes=max(0, budget - fixed),
        intermediates=1,
    )


def _real_itemsize() -> int:
    return np.dtype(_spectrum_mod.REAL).itemsize


_SPECTRUM_BACKEND_MODULES = (
    "_brem_cpu_fused",
    "_characteristic_chunk_kernel",
    "_characteristic_fused",
    "_device_sizing",
    "brem",
    "brem_bremslib",
    "brem_events",
    "brem_fallbacks",
    "brem_prepared",
    "channeling_flux",
    "characteristic",
    "coherent_population",
    "cxr",
    "cxr._attribution",
    "cxr._band_mass",
    "cxr._batched",
    "cxr._batched_finalize",
    "cxr._bin_quadrature",
    "cxr._coverage",
    "cxr._cpu_reduce",
    "cxr._escape_mu",
    "cxr._formation",
    "cxr._fused",
    "cxr._grouped",
    "cxr._kernels",
    "cxr._line_amplitude",
    "cxr._line_tables",
    "cxr._mosaic_sampled",
    "cxr._per_hkl",
    "cxr._position_twins",
    "cxr._recoil",
    "cxr._setup",
    "cxr._spectrum",
    "cxr._temporal",
    "diffracted_brem",
    "diffracted_characteristic",
    "optical_finite_screen",
    "segment_escape",
)


@contextmanager
def _cpu_spectrum_backend():
    """Switch every spectrum binding, including lazy modules, in the serial driver.

    The static import guard keeps this list complete as modules are added.
    Kernels remain lazy; CPU dispatch never
    enters their device caches. Transport keeps its original backend.
    """
    modules = [_spectrum_mod]
    modules.extend(
        import_module(_spectrum_mod.__name__ + "." + name) for name in _SPECTRUM_BACKEND_MODULES
    )
    replacements = {"xp": np, "REAL": np.float64, "BACKEND": NumPyBackend()}
    previous = []
    gpu = _RESOURCE_POLICY.gpu
    try:
        _RESOURCE_POLICY.gpu = False
        for module in modules:
            for name, value in replacements.items():
                if name in vars(module):
                    previous.append((module, name, getattr(module, name)))
                    setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(previous):
            setattr(module, name, value)
        _RESOURCE_POLICY.gpu = gpu
