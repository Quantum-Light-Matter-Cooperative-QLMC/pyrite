import cupy as xp
import numpy as np
from cupyx import jit

F32_ZERO = np.float32(0.0)
F32_TINY = np.float32(1.0e-20)

U32_ZERO = np.uint32(0)
U32_ONE = np.uint32(1)
U32_TWO = np.uint32(2)
U32_THREE = np.uint32(3)
U32_FOUR = np.uint32(4)

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectrumKernelConfig:
    nthreads: int
    energies_per_block: int


DEFAULT_SPECTRUM_KERNEL_CONFIG = SpectrumKernelConfig(
    nthreads=512,
    energies_per_block=3,
)


@jit.rawkernel()
def _kernel_1e(
    E_r,
    aw,
    w,
    E_grid,
    spec,
    n_lines,
    n_E,
):
    base = jit.blockIdx.x
    k0 = base
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x

    # k0 is always valid for every launched block.
    E0 = E_grid[k0]

    # One private accumulator per energy.
    acc0 = F32_ZERO

    # Each thread gets:
    # tid, tid+nthreads, tid+2*nthreads, ...
    line = tid

    while line < n_lines:
        wt = w[line]
        if wt != F32_ZERO:
            Er = E_r[line]
            a = aw[line]

            # Energy 0
            x0 = a * (E0 - Er)
            if x0 == F32_ZERO:
                x0 = F32_TINY

            s0 = xp.sin(x0) / x0
            acc0 += wt * s0 * s0

        line += nthreads

    shared = jit.shared_memory(xp.float32, None)

    off0 = U32_ZERO
    shared[off0 + tid] = acc0

    jit.syncthreads()

    # Tree-reduce
    stride = nthreads // U32_TWO

    while stride > U32_ZERO:
        if tid < stride:
            shared[off0 + tid] += shared[off0 + tid + stride]

        jit.syncthreads()
        stride //= U32_TWO

    # Thread zero owns the final reduced values.
    if tid == U32_ZERO:
        spec[k0] += shared[off0]


@jit.rawkernel()
def _kernel_2e(
    E_r,
    aw,
    w,
    E_grid,
    spec,
    n_lines,
    n_E,
):
    base = jit.blockIdx.x * U32_TWO

    k0 = base
    k1 = base + U32_ONE

    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x

    # k0 is always valid for every launched block.
    E0 = E_grid[k0]
    has_k1 = k1 < n_E

    E1 = F32_ZERO

    if has_k1:
        E1 = E_grid[k1]

    # One private accumulator per energy.
    acc0 = F32_ZERO
    acc1 = F32_ZERO

    # Each thread gets:
    # tid, tid+nthreads, tid+2*nthreads, ...
    line = tid

    while line < n_lines:
        wt = w[line]
        if wt != F32_ZERO:
            # Load line geometry only for a live pair.  The fused prologue
            # leaves rejected pairs in deterministic order with zero weight.
            Er = E_r[line]
            a = aw[line]

            # Energy 0
            x0 = a * (E0 - Er)
            if x0 == F32_ZERO:
                x0 = F32_TINY

            s0 = xp.sin(x0) / x0
            acc0 += wt * s0 * s0

            # Energy 1
            if has_k1:
                x1 = a * (E1 - Er)
                if x1 == F32_ZERO:
                    x1 = F32_TINY

                s1 = xp.sin(x1) / x1
                acc1 += wt * s1 * s1

        line += nthreads

    # Two shared-memory reduction regions:
    #
    # [ acc0 partials ]
    # [ acc1 partials ]
    shared = jit.shared_memory(xp.float32, None)

    off0 = U32_ZERO
    off1 = nthreads

    shared[off0 + tid] = acc0
    shared[off1 + tid] = acc1

    jit.syncthreads()

    # Tree-reduce both energies simultaneously.
    stride = nthreads // U32_TWO

    while stride > U32_ZERO:
        if tid < stride:
            shared[off0 + tid] += shared[off0 + tid + stride]
            shared[off1 + tid] += shared[off1 + tid + stride]

        jit.syncthreads()
        stride //= U32_TWO

    # Thread zero owns the final reduced values.
    if tid == U32_ZERO:
        spec[k0] += shared[off0]

        if has_k1:
            spec[k1] += shared[off1]


@jit.rawkernel()
def _kernel_3e(
    E_r,
    aw,
    w,
    E_grid,
    spec,
    n_lines,
    n_E,
):
    base = jit.blockIdx.x * U32_THREE

    k0 = base
    k1 = base + U32_ONE
    k2 = base + U32_TWO

    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x

    # k0 is always valid for every launched block.
    E0 = E_grid[k0]

    # The final block may contain fewer than three valid energies.
    has_k1 = k1 < n_E
    has_k2 = k2 < n_E

    E1 = F32_ZERO
    E2 = F32_ZERO

    if has_k1:
        E1 = E_grid[k1]

    if has_k2:
        E2 = E_grid[k2]

    # One private accumulator per energy.
    acc0 = F32_ZERO
    acc1 = F32_ZERO
    acc2 = F32_ZERO

    # Each thread gets:
    # tid, tid+nthreads, tid+2*nthreads, ...
    line = tid

    while line < n_lines:
        wt = w[line]
        if wt != F32_ZERO:
            # Load once and reuse for all three energies.
            Er = E_r[line]
            a = aw[line]

            # Energy 0
            x0 = a * (E0 - Er)
            if x0 == F32_ZERO:
                x0 = F32_TINY

            s0 = xp.sin(x0) / x0
            acc0 += wt * s0 * s0

            # Energy 1
            if has_k1:
                x1 = a * (E1 - Er)
                if x1 == F32_ZERO:
                    x1 = F32_TINY

                s1 = xp.sin(x1) / x1
                acc1 += wt * s1 * s1

            # Energy 2
            if has_k2:
                x2 = a * (E2 - Er)
                if x2 == F32_ZERO:
                    x2 = F32_TINY

                s2 = xp.sin(x2) / x2
                acc2 += wt * s2 * s2

        line += nthreads

    # Three shared-memory reduction regions:
    #
    # [ acc0 partials ]
    # [ acc1 partials ]
    # [ acc2 partials ]
    shared = jit.shared_memory(xp.float32, None)

    off0 = U32_ZERO
    off1 = nthreads
    off2 = U32_TWO * nthreads

    shared[off0 + tid] = acc0
    shared[off1 + tid] = acc1
    shared[off2 + tid] = acc2

    jit.syncthreads()

    # Tree-reduce all three energies simultaneously.
    stride = nthreads // U32_TWO

    while stride > U32_ZERO:
        if tid < stride:
            shared[off0 + tid] += shared[off0 + tid + stride]
            shared[off1 + tid] += shared[off1 + tid + stride]
            shared[off2 + tid] += shared[off2 + tid + stride]

        jit.syncthreads()
        stride //= U32_TWO

    # Thread zero owns the final reduced values.
    if tid == U32_ZERO:
        spec[k0] += shared[off0]

        if has_k1:
            spec[k1] += shared[off1]

        if has_k2:
            spec[k2] += shared[off2]


@jit.rawkernel()
def _kernel_4e(
    E_r,
    aw,
    w,
    E_grid,
    spec,
    n_lines,
    n_E,
):
    base = jit.blockIdx.x * U32_FOUR

    k0 = base
    k1 = base + U32_ONE
    k2 = base + U32_TWO
    k3 = base + U32_THREE

    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x

    # k0 is always valid for every launched block.
    E0 = E_grid[k0]

    # The final block may contain fewer than four valid energies.
    has_k1 = k1 < n_E
    has_k2 = k2 < n_E
    has_k3 = k3 < n_E

    E1 = F32_ZERO
    E2 = F32_ZERO
    E3 = F32_ZERO

    if has_k1:
        E1 = E_grid[k1]

    if has_k2:
        E2 = E_grid[k2]

    if has_k3:
        E3 = E_grid[k3]

    # One private accumulator per energy.
    acc0 = F32_ZERO
    acc1 = F32_ZERO
    acc2 = F32_ZERO
    acc3 = F32_ZERO

    # Each thread gets:
    # tid, tid+nthreads, tid+2*nthreads, ...
    line = tid

    while line < n_lines:
        wt = w[line]
        if wt != F32_ZERO:
            # Load once and reuse for all four energies.
            Er = E_r[line]
            a = aw[line]

            # Energy 0
            x0 = a * (E0 - Er)
            if x0 == F32_ZERO:
                x0 = F32_TINY

            s0 = xp.sin(x0) / x0
            acc0 += wt * s0 * s0

            # Energy 1
            if has_k1:
                x1 = a * (E1 - Er)
                if x1 == F32_ZERO:
                    x1 = F32_TINY

                s1 = xp.sin(x1) / x1
                acc1 += wt * s1 * s1

            # Energy 2
            if has_k2:
                x2 = a * (E2 - Er)
                if x2 == F32_ZERO:
                    x2 = F32_TINY

                s2 = xp.sin(x2) / x2
                acc2 += wt * s2 * s2

            # Energy 3
            if has_k3:
                x3 = a * (E3 - Er)
                if x3 == F32_ZERO:
                    x3 = F32_TINY

                s3 = xp.sin(x3) / x3
                acc3 += wt * s3 * s3

        line += nthreads

    # ---------------------------------------------------------
    # BLOCK REDUCTION
    # ---------------------------------------------------------
    shared = jit.shared_memory(xp.float32, None)

    off0 = U32_ZERO
    off1 = nthreads
    off2 = U32_TWO * nthreads
    off3 = U32_THREE * nthreads

    shared[off0 + tid] = acc0
    shared[off1 + tid] = acc1
    shared[off2 + tid] = acc2
    shared[off3 + tid] = acc3

    jit.syncthreads()

    # Tree-reduce all four energies simultaneously.
    stride = nthreads // U32_TWO

    while stride > U32_ZERO:
        if tid < stride:
            shared[off0 + tid] += shared[off0 + tid + stride]
            shared[off1 + tid] += shared[off1 + tid + stride]
            shared[off2 + tid] += shared[off2 + tid + stride]
            shared[off3 + tid] += shared[off3 + tid + stride]

        jit.syncthreads()
        stride //= U32_TWO

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    # Thread zero owns the final reduced values.
    if tid == U32_ZERO:
        spec[k0] += shared[off0]

        if has_k1:
            spec[k1] += shared[off1]

        if has_k2:
            spec[k2] += shared[off2]

        if has_k3:
            spec[k3] += shared[off3]


# * ---------------------------------------------------------
# * Define kernel runner
# * ---------------------------------------------------------

_REDUCTION_KERNELS = {
    1: _kernel_1e,
    2: _kernel_2e,
    3: _kernel_3e,
    4: _kernel_4e,
}


def run_reduction_kernel(
    E_r,
    aw,
    w,
    E_grid,
    *,
    out=None,
    config,
):
    nthreads = config.nthreads
    energies_per_block = config.energies_per_block

    if nthreads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")

    try:
        kernel = _REDUCTION_KERNELS[energies_per_block]
    except KeyError:
        raise ValueError(f"energies_per_block must be one of {tuple(_REDUCTION_KERNELS)}") from None

    n_E = int(E_grid.size)

    # Kernels accumulate so callers can stream deterministic fixed-order
    # prologue blocks directly into the final spectrum without a CuPy add
    # launch.  A fresh call retains the old return-new-array contract.
    if out is None:
        out = xp.zeros(n_E, dtype=xp.float32)
    if n_E == 0:
        return out

    # ceil(n_E / energies_per_block)
    nblocks = (n_E + energies_per_block - 1) // energies_per_block

    grid = (nblocks,)
    block = (int(nthreads),)

    shared_bytes = energies_per_block * nthreads * np.dtype(np.float32).itemsize

    kernel(
        grid,
        block,
        (
            E_r,
            aw,
            w,
            E_grid,
            out,
            np.uint32(E_r.size),
            np.uint32(n_E),
        ),
        shared_mem=shared_bytes,
    )

    return out
