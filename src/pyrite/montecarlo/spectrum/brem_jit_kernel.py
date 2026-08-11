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
def _dsigma_weighted_scalar(T_i, k_eV, Z, p_i, incident_prefactor):
    """Weighted Bethe-Heitler + Elwert cell with incident work hoisted.

    ``incident_prefactor`` already contains density, segment length, the common
    Bethe-Heitler constant, ``Z**2``, ``1/p_i**2``, and the incident-side
    ``beta_i * (1-exp(-2*pi*Z*alpha/beta_i))`` factor. Only final-state work
    remains energy dependent here.
    """
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
    den_f = F32_ONE - xp.exp(-zi / beta_f)

    k_eV_safe = k_eV
    if k_eV_safe < F32_1E_M30:
        k_eV_safe = F32_1E_M30

    return incident_prefactor * born_log / (k_eV_safe * beta_f * den_f)


@jit.rawkernel(device=True)
def _tau_scalar(path_flat, mu_flat, line, k, n_layers, n_E):
    # Dominant single-slab case: avoid the tiny layer loop and its index math.
    if n_layers == U32_ONE:
        return path_flat[line] * mu_flat[k]
    tau = F32_ZERO
    layer = U32_ZERO
    while layer < n_layers:
        tau += path_flat[line * n_layers + layer] * mu_flat[layer * n_E + k]
        layer += U32_ONE
    return tau


@jit.rawkernel()
def _kernel_1e(T, p_i_arr, incident_prefactor, path_flat, mu_flat, E_grid, spec, Z, n_seg, n_E, n_layers):
    k0 = jit.blockIdx.x
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x
    E0 = E_grid[k0]
    acc0 = F32_ZERO

    line = tid
    while line < n_seg:
        Ti = T[line]
        p_i = p_i_arr[line]
        pref_i = incident_prefactor[line]

        tau0 = _tau_scalar(path_flat, mu_flat, line, k0, n_layers, n_E)
        ds0 = _dsigma_weighted_scalar(Ti, E0, Z, p_i, pref_i)
        acc0 += ds0 * xp.exp(-tau0)
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
def _kernel_2e(T, p_i_arr, incident_prefactor, path_flat, mu_flat, E_grid, spec, Z, n_seg, n_E, n_layers):
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
        p_i = p_i_arr[line]
        pref_i = incident_prefactor[line]

        tau0 = _tau_scalar(path_flat, mu_flat, line, k0, n_layers, n_E)
        acc0 += _dsigma_weighted_scalar(Ti, E0, Z, p_i, pref_i) * xp.exp(-tau0)
        if has1:
            tau1 = _tau_scalar(path_flat, mu_flat, line, k1, n_layers, n_E)
            acc1 += _dsigma_weighted_scalar(Ti, E1, Z, p_i, pref_i) * xp.exp(-tau1)
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
def _kernel_3e(T, p_i_arr, incident_prefactor, path_flat, mu_flat, E_grid, spec, Z, n_seg, n_E, n_layers):
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
        p_i = p_i_arr[line]
        pref_i = incident_prefactor[line]

        tau0 = _tau_scalar(path_flat, mu_flat, line, k0, n_layers, n_E)
        acc0 += _dsigma_weighted_scalar(Ti, E0, Z, p_i, pref_i) * xp.exp(-tau0)
        if has1:
            tau1 = _tau_scalar(path_flat, mu_flat, line, k1, n_layers, n_E)
            acc1 += _dsigma_weighted_scalar(Ti, E1, Z, p_i, pref_i) * xp.exp(-tau1)
        if has2:
            tau2 = _tau_scalar(path_flat, mu_flat, line, k2, n_layers, n_E)
            acc2 += _dsigma_weighted_scalar(Ti, E2, Z, p_i, pref_i) * xp.exp(-tau2)
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
def _kernel_4e(T, p_i_arr, incident_prefactor, path_flat, mu_flat, E_grid, spec, Z, n_seg, n_E, n_layers):
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
        p_i = p_i_arr[line]
        pref_i = incident_prefactor[line]

        tau0 = _tau_scalar(path_flat, mu_flat, line, k0, n_layers, n_E)
        acc0 += _dsigma_weighted_scalar(Ti, E0, Z, p_i, pref_i) * xp.exp(-tau0)
        if has1:
            tau1 = _tau_scalar(path_flat, mu_flat, line, k1, n_layers, n_E)
            acc1 += _dsigma_weighted_scalar(Ti, E1, Z, p_i, pref_i) * xp.exp(-tau1)
        if has2:
            tau2 = _tau_scalar(path_flat, mu_flat, line, k2, n_layers, n_E)
            acc2 += _dsigma_weighted_scalar(Ti, E2, Z, p_i, pref_i) * xp.exp(-tau2)
        if has3:
            tau3 = _tau_scalar(path_flat, mu_flat, line, k3, n_layers, n_E)
            acc3 += _dsigma_weighted_scalar(Ti, E3, Z, p_i, pref_i) * xp.exp(-tau3)
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
    p_i=None,
    incident_prefactor=None,
    out=None,
    config=DEFAULT_BREM_KERNEL_CONFIG,
):
    """Accumulate one composition element's brem contribution into ``out``.

    The historical call API is retained. ``p_i`` and ``incident_prefactor`` may
    be supplied by the caller to share the incident-state work across composition
    elements; otherwise they are computed once here, still hoisting that work out
    of the photon-energy block loop.

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

    Z32 = np.float32(Z)
    if p_i is None:
        p_i = xp.sqrt(T_keV * (T_keV + F32_TWO * F32_MC2_KEV)) / F32_MC2_KEV
        p_i = xp.ascontiguousarray(p_i, dtype=xp.float32)
    if incident_prefactor is None:
        beta_i = p_i / (F32_ONE + T_keV / F32_MC2_KEV)
        zi = Z32 * F32_TWO_PI_ALPHA
        den_i = F32_ONE - xp.exp(-zi / beta_i)
        incident_prefactor = (
            np.float32(density_cm3)
            * L_ang
            * F32_1E_M8
            * F32_16_OVER_3
            * F32_ALPHA
            * F32_R_E_CM2
            * Z32
            * Z32
            * beta_i
            * den_i
            / (p_i * p_i)
        )
        incident_prefactor = xp.ascontiguousarray(incident_prefactor, dtype=xp.float32)

    nblocks = (n_E + epb - 1) // epb
    shared_bytes = epb * nthreads * np.dtype(np.float32).itemsize
    kernel(
        (nblocks,),
        (nthreads,),
        (
            T_keV,
            p_i,
            incident_prefactor,
            path_flat,
            mu_flat,
            E_grid,
            out,
            Z32,
            np.uint32(T_keV.size),
            np.uint32(n_E),
            np.uint32(n_layers),
        ),
        shared_mem=shared_bytes,
    )
    return out
