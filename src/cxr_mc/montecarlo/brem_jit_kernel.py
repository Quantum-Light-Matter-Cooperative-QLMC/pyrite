"""CUDA JIT reduction for the bremsstrahlung spectrum.

One CUDA block owns one or more photon-energy bins. Threads stride over
transport segments, evaluate Bethe-Heitler + Elwert and Beer-Lambert
attenuation directly, then reduce the segment sum in shared memory.

The kernel consumes per-segment absorber path lengths for each layer rather
than a dense (segment, energy) attenuation matrix, so peak scratch memory is
O(Nseg * Nlayer + Nlayer * NE) rather than O(Nseg * NE).
"""

from dataclasses import dataclass

import cupy as xp
import numpy as np
from cupyx import jit

F32_ZERO = np.float32(0.0)
F32_ONE = np.float32(1.0)
F32_1E_M3 = np.float32(1.0e-3)
F32_1E_M6 = np.float32(1.0e-6)
F32_1E_M8 = np.float32(1.0e-8)
F32_1E_M30 = np.float32(1.0e-30)
F32_TWO = np.float32(2.0)
F32_16_OVER_3 = np.float32(16.0 / 3.0)
F32_MC2_KEV = np.float32(510.99895)
F32_ALPHA = np.float32(7.2973525693e-3)
F32_R_E_CM2 = np.float32(7.9407877e-26)
F32_TWO_PI_ALPHA = np.float32(2.0 * np.pi * 7.2973525693e-3)

U32_ZERO = np.uint32(0)
U32_ONE = np.uint32(1)
U32_TWO = np.uint32(2)
U32_THREE = np.uint32(3)
U32_FOUR = np.uint32(4)


@dataclass(frozen=True)
class BremKernelConfig:
    nthreads: int
    energies_per_block: int


# Starter configuration only; retune independently on each GPU generation.
DEFAULT_BREM_KERNEL_CONFIG = BremKernelConfig(nthreads=256, energies_per_block=2)


@jit.rawkernel(device=True)
def _dsigma_scalar(T_i, k_eV, Z, p_i, beta_i):
    k = k_eV * F32_1E_M3
    T_f = T_i - k
    if T_f <= F32_1E_M6 or k <= F32_ZERO:
        return F32_ZERO

    p_f = xp.sqrt(T_f * (T_f + F32_TWO * F32_MC2_KEV)) / F32_MC2_KEV
    beta_f = p_f / (F32_ONE + T_f / F32_MC2_KEV)

    dp = p_i - p_f
    if dp < F32_1E_M30:
        dp = F32_1E_M30
    born_log = xp.log((p_i + p_f) / dp)

    zi = Z * F32_TWO_PI_ALPHA
    den_i = F32_ONE - xp.exp(-zi / beta_i)
    den_f = F32_ONE - xp.exp(-zi / beta_f)
    elwert = beta_i / beta_f * den_i / den_f

    k_eV_safe = k_eV
    if k_eV_safe < F32_1E_M30:
        k_eV_safe = F32_1E_M30

    return (
        F32_16_OVER_3
        * F32_ALPHA
        * F32_R_E_CM2
        * Z
        * Z
        / k_eV_safe
        / (p_i * p_i)
        * born_log
        * elwert
    )


@jit.rawkernel(device=True)
def _tau_scalar(path_flat, mu_flat, line, k, n_layers, n_E):
    tau = F32_ZERO
    layer = U32_ZERO
    while layer < n_layers:
        tau += path_flat[line * n_layers + layer] * mu_flat[layer * n_E + k]
        layer += U32_ONE
    return tau


@jit.rawkernel()
def _kernel_1e(T, L_ang, path_flat, mu_flat, E_grid, spec, Z, density_cm3, n_seg, n_E, n_layers):
    k0 = jit.blockIdx.x
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x
    E0 = E_grid[k0]
    acc0 = F32_ZERO

    line = tid
    while line < n_seg:
        Ti = T[line]
        p_i = xp.sqrt(Ti * (Ti + F32_TWO * F32_MC2_KEV)) / F32_MC2_KEV
        beta_i = p_i / (F32_ONE + Ti / F32_MC2_KEV)
        path_weight = density_cm3 * L_ang[line] * F32_1E_M8

        tau0 = _tau_scalar(path_flat, mu_flat, line, k0, n_layers, n_E)
        ds0 = _dsigma_scalar(Ti, E0, Z, p_i, beta_i)
        acc0 += path_weight * ds0 * xp.exp(-tau0)
        line += nthreads

    shared = jit.shared_memory(xp.float32, None)
    shared[tid] = acc0
    jit.syncthreads()
    stride = nthreads // U32_TWO
    while stride > U32_ZERO:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if tid == U32_ZERO:
        spec[k0] += shared[U32_ZERO]


@jit.rawkernel()
def _kernel_2e(T, L_ang, path_flat, mu_flat, E_grid, spec, Z, density_cm3, n_seg, n_E, n_layers):
    base = jit.blockIdx.x * U32_TWO
    k0 = base
    k1 = base + U32_ONE
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x
    E0 = E_grid[k0]
    has1 = k1 < n_E
    E1 = F32_ZERO
    if has1:
        E1 = E_grid[k1]
    acc0 = F32_ZERO
    acc1 = F32_ZERO

    line = tid
    while line < n_seg:
        Ti = T[line]
        p_i = xp.sqrt(Ti * (Ti + F32_TWO * F32_MC2_KEV)) / F32_MC2_KEV
        beta_i = p_i / (F32_ONE + Ti / F32_MC2_KEV)
        path_weight = density_cm3 * L_ang[line] * F32_1E_M8

        tau0 = _tau_scalar(path_flat, mu_flat, line, k0, n_layers, n_E)
        acc0 += path_weight * _dsigma_scalar(Ti, E0, Z, p_i, beta_i) * xp.exp(-tau0)
        if has1:
            tau1 = _tau_scalar(path_flat, mu_flat, line, k1, n_layers, n_E)
            acc1 += path_weight * _dsigma_scalar(Ti, E1, Z, p_i, beta_i) * xp.exp(-tau1)
        line += nthreads

    shared = jit.shared_memory(xp.float32, None)
    off0 = U32_ZERO
    off1 = nthreads
    shared[off0 + tid] = acc0
    shared[off1 + tid] = acc1
    jit.syncthreads()
    stride = nthreads // U32_TWO
    while stride > U32_ZERO:
        if tid < stride:
            shared[off0 + tid] += shared[off0 + tid + stride]
            shared[off1 + tid] += shared[off1 + tid + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if tid == U32_ZERO:
        spec[k0] += shared[off0]
        if has1:
            spec[k1] += shared[off1]


@jit.rawkernel()
def _kernel_3e(T, L_ang, path_flat, mu_flat, E_grid, spec, Z, density_cm3, n_seg, n_E, n_layers):
    base = jit.blockIdx.x * U32_THREE
    k0 = base
    k1 = base + U32_ONE
    k2 = base + U32_TWO
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x
    E0 = E_grid[k0]
    has1 = k1 < n_E
    has2 = k2 < n_E
    E1 = F32_ZERO
    E2 = F32_ZERO
    if has1:
        E1 = E_grid[k1]
    if has2:
        E2 = E_grid[k2]
    acc0 = F32_ZERO
    acc1 = F32_ZERO
    acc2 = F32_ZERO

    line = tid
    while line < n_seg:
        Ti = T[line]
        p_i = xp.sqrt(Ti * (Ti + F32_TWO * F32_MC2_KEV)) / F32_MC2_KEV
        beta_i = p_i / (F32_ONE + Ti / F32_MC2_KEV)
        path_weight = density_cm3 * L_ang[line] * F32_1E_M8

        tau0 = _tau_scalar(path_flat, mu_flat, line, k0, n_layers, n_E)
        acc0 += path_weight * _dsigma_scalar(Ti, E0, Z, p_i, beta_i) * xp.exp(-tau0)
        if has1:
            tau1 = _tau_scalar(path_flat, mu_flat, line, k1, n_layers, n_E)
            acc1 += path_weight * _dsigma_scalar(Ti, E1, Z, p_i, beta_i) * xp.exp(-tau1)
        if has2:
            tau2 = _tau_scalar(path_flat, mu_flat, line, k2, n_layers, n_E)
            acc2 += path_weight * _dsigma_scalar(Ti, E2, Z, p_i, beta_i) * xp.exp(-tau2)
        line += nthreads

    shared = jit.shared_memory(xp.float32, None)
    off0 = U32_ZERO
    off1 = nthreads
    off2 = U32_TWO * nthreads
    shared[off0 + tid] = acc0
    shared[off1 + tid] = acc1
    shared[off2 + tid] = acc2
    jit.syncthreads()
    stride = nthreads // U32_TWO
    while stride > U32_ZERO:
        if tid < stride:
            shared[off0 + tid] += shared[off0 + tid + stride]
            shared[off1 + tid] += shared[off1 + tid + stride]
            shared[off2 + tid] += shared[off2 + tid + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if tid == U32_ZERO:
        spec[k0] += shared[off0]
        if has1:
            spec[k1] += shared[off1]
        if has2:
            spec[k2] += shared[off2]


@jit.rawkernel()
def _kernel_4e(T, L_ang, path_flat, mu_flat, E_grid, spec, Z, density_cm3, n_seg, n_E, n_layers):
    base = jit.blockIdx.x * U32_FOUR
    k0 = base
    k1 = base + U32_ONE
    k2 = base + U32_TWO
    k3 = base + U32_THREE
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x
    E0 = E_grid[k0]
    has1 = k1 < n_E
    has2 = k2 < n_E
    has3 = k3 < n_E
    E1 = F32_ZERO
    E2 = F32_ZERO
    E3 = F32_ZERO
    if has1:
        E1 = E_grid[k1]
    if has2:
        E2 = E_grid[k2]
    if has3:
        E3 = E_grid[k3]
    acc0 = F32_ZERO
    acc1 = F32_ZERO
    acc2 = F32_ZERO
    acc3 = F32_ZERO

    line = tid
    while line < n_seg:
        Ti = T[line]
        p_i = xp.sqrt(Ti * (Ti + F32_TWO * F32_MC2_KEV)) / F32_MC2_KEV
        beta_i = p_i / (F32_ONE + Ti / F32_MC2_KEV)
        path_weight = density_cm3 * L_ang[line] * F32_1E_M8

        tau0 = _tau_scalar(path_flat, mu_flat, line, k0, n_layers, n_E)
        acc0 += path_weight * _dsigma_scalar(Ti, E0, Z, p_i, beta_i) * xp.exp(-tau0)
        if has1:
            tau1 = _tau_scalar(path_flat, mu_flat, line, k1, n_layers, n_E)
            acc1 += path_weight * _dsigma_scalar(Ti, E1, Z, p_i, beta_i) * xp.exp(-tau1)
        if has2:
            tau2 = _tau_scalar(path_flat, mu_flat, line, k2, n_layers, n_E)
            acc2 += path_weight * _dsigma_scalar(Ti, E2, Z, p_i, beta_i) * xp.exp(-tau2)
        if has3:
            tau3 = _tau_scalar(path_flat, mu_flat, line, k3, n_layers, n_E)
            acc3 += path_weight * _dsigma_scalar(Ti, E3, Z, p_i, beta_i) * xp.exp(-tau3)
        line += nthreads

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
    stride = nthreads // U32_TWO
    while stride > U32_ZERO:
        if tid < stride:
            shared[off0 + tid] += shared[off0 + tid + stride]
            shared[off1 + tid] += shared[off1 + tid + stride]
            shared[off2 + tid] += shared[off2 + tid + stride]
            shared[off3 + tid] += shared[off3 + tid + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if tid == U32_ZERO:
        spec[k0] += shared[off0]
        if has1:
            spec[k1] += shared[off1]
        if has2:
            spec[k2] += shared[off2]
        if has3:
            spec[k3] += shared[off3]


_REDUCTION_KERNELS = {1: _kernel_1e, 2: _kernel_2e, 3: _kernel_3e, 4: _kernel_4e}


def run_brem_reduction_kernel(
    T_keV,
    L_ang,
    path_flat,
    mu_flat,
    E_grid,
    *,
    Z,
    density_cm3,
    n_layers,
    out=None,
    config=DEFAULT_BREM_KERNEL_CONFIG,
):
    """Accumulate one composition element's brem contribution into ``out``.

    All array arguments must be contiguous float32 CuPy arrays. ``path_flat`` is
    C-order ``(n_seg, n_layers)`` and ``mu_flat`` is C-order
    ``(n_layers, n_E)``. If ``out`` is supplied it is incremented in place;
    otherwise a zeroed float32 output is allocated.
    """
    nthreads = int(config.nthreads)
    epb = int(config.energies_per_block)
    if nthreads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    try:
        kernel = _REDUCTION_KERNELS[epb]
    except KeyError:
        raise ValueError(f"energies_per_block must be one of {tuple(_REDUCTION_KERNELS)}") from None

    n_E = int(E_grid.size)
    if out is None:
        out = xp.zeros(n_E, dtype=xp.float32)
    if n_E == 0 or T_keV.size == 0:
        return out

    nblocks = (n_E + epb - 1) // epb
    shared_bytes = epb * nthreads * np.dtype(np.float32).itemsize
    kernel(
        (nblocks,),
        (nthreads,),
        (
            T_keV,
            L_ang,
            path_flat,
            mu_flat,
            E_grid,
            out,
            np.float32(Z),
            np.float32(density_cm3),
            np.uint32(T_keV.size),
            np.uint32(n_E),
            np.uint32(n_layers),
        ),
        shared_mem=shared_bytes,
    )
    return out
