"""CUDA JIT reductions for analytic, staged-EEDL, and BremsLib bremsstrahlung.

One CUDA block owns one or more photon-energy bins. Threads stride over
transport segments, evaluate either staged EEDL interpolation or
Bethe-Heitler + Elwert and Beer-Lambert attenuation directly, then reduce the
segment sum in shared memory.

The kernel consumes per-segment absorber path lengths for each layer rather
than a dense (segment, energy) attenuation matrix, so peak scratch memory is
O(Nseg * Nlayer + Nlayer * NE) rather than O(Nseg * NE).
"""

from dataclasses import dataclass

import cupy as xp
import numpy as np

from .._cupy_jit import jit

F32_ZERO = np.float32(0.0)
F32_ONE = np.float32(1.0)
F32_1E_M3 = np.float32(1.0e-3)
F32_1E_M6 = np.float32(1.0e-6)
F32_1E_M8 = np.float32(1.0e-8)
F32_1E_M30 = np.float32(1.0e-30)
F32_1E3 = np.float32(1.0e3)
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
U32_ELEVEN = np.uint32(11)
U32_THIRTEEN = np.uint32(13)


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


@jit.rawkernel(device=True)
def _eedl_or_bh_weighted_scalar(
    T_i,
    k_eV,
    k_index,
    p_i,
    bethe_heitler_prefactor,
    eedl_weight,
    lower_panel,
    panel_fraction,
    available,
    panel_probability,
    n_E,
    Z,
):
    """Weighted EEDL cell, with Bethe--Heitler for uncovered segments."""
    if available > F32_ZERO:
        if k_eV <= F32_ZERO or k_eV > T_i * F32_1E3:
            return F32_ZERO
        lower_offset = lower_panel * n_E + k_index
        upper_offset = lower_offset + n_E
        lower_pdf = panel_probability[lower_offset]
        upper_pdf = panel_probability[upper_offset]
        mixed_pdf = lower_pdf + panel_fraction * (upper_pdf - lower_pdf)
        return eedl_weight * mixed_pdf
    return _dsigma_weighted_scalar(T_i, k_eV, Z, p_i, bethe_heitler_prefactor)


@jit.rawkernel(device=True)
def _bremslib_scaled_scalar(reduced, top, line, values, nominal_ratio):
    """Interpolate one staged 13-node BremsLib row in ``k/T``."""
    lower = U32_ZERO
    while lower < U32_ELEVEN and reduced >= nominal_ratio[lower + U32_ONE]:
        lower += U32_ONE
    x0 = nominal_ratio[lower]
    if lower == U32_ELEVEN:
        x1 = top
    else:
        x1 = nominal_ratio[lower + U32_ONE]
    fraction = (reduced - x0) / (x1 - x0)
    if fraction < F32_ZERO:
        fraction = F32_ZERO
    elif fraction > F32_ONE:
        fraction = F32_ONE
    offset = line * U32_THIRTEEN + lower
    v0 = values[offset]
    v1 = values[offset + U32_ONE]
    return v0 + fraction * (v1 - v0)


@jit.rawkernel(device=True)
def _bremslib_or_fallback_weighted_scalar(
    T_i,
    k_eV,
    k_index,
    line,
    p_i,
    bethe_heitler_prefactor,
    eedl_weight,
    eedl_lower_panel,
    eedl_panel_fraction,
    eedl_available,
    panel_probability,
    bremslib_weight,
    bremslib_lower_top,
    bremslib_upper_top,
    bremslib_energy_fraction,
    bremslib_available,
    bremslib_lower_values,
    bremslib_upper_values,
    nominal_ratio,
    n_E,
    Z,
):
    """Weighted directional BremsLib cell, with the normal EEDL/BH fallback."""
    if bremslib_available > F32_ZERO:
        if k_eV <= F32_ZERO or k_eV > T_i * F32_1E3:
            return F32_ZERO
        reduced = k_eV / (T_i * F32_1E3)
        lower = _bremslib_scaled_scalar(
            reduced, bremslib_lower_top, line, bremslib_lower_values, nominal_ratio
        )
        upper = _bremslib_scaled_scalar(
            reduced, bremslib_upper_top, line, bremslib_upper_values, nominal_ratio
        )
        scaled = lower + bremslib_energy_fraction * (upper - lower)
        return bremslib_weight * scaled / k_eV
    return _eedl_or_bh_weighted_scalar(
        T_i,
        k_eV,
        k_index,
        p_i,
        bethe_heitler_prefactor,
        eedl_weight,
        eedl_lower_panel,
        eedl_panel_fraction,
        eedl_available,
        panel_probability,
        n_E,
        Z,
    )


@jit.rawkernel()
def _eedl_kernel_1e(
    T,
    p_i_arr,
    bethe_heitler_prefactor,
    eedl_weight,
    lower_panel,
    panel_fraction,
    available,
    path_flat,
    mu_flat,
    E_grid,
    panel_probability,
    spec,
    Z,
    n_seg,
    n_E,
    n_layers,
):
    """Fuse EEDL interpolation, fallback, attenuation, and segment reduction."""
    k = jit.blockIdx.x
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x
    photon_eV = E_grid[k]
    acc = F32_ZERO

    line = tid
    while line < n_seg:
        weighted = _eedl_or_bh_weighted_scalar(
            T[line],
            photon_eV,
            k,
            p_i_arr[line],
            bethe_heitler_prefactor[line],
            eedl_weight[line],
            lower_panel[line],
            panel_fraction[line],
            available[line],
            panel_probability,
            n_E,
            Z,
        )
        tau = _tau_scalar(path_flat, mu_flat, line, k, n_layers, n_E)
        acc += weighted * xp.exp(-tau)
        line += nthreads

    shared = jit.shared_memory(xp.float32, None)
    shared[tid] = acc
    jit.syncthreads()
    stride = nthreads // U32_TWO
    while stride > U32_ZERO:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if tid == U32_ZERO:
        spec[k] += shared[U32_ZERO]


@jit.rawkernel()
def _bremslib_kernel_1e(
    T,
    p_i_arr,
    bethe_heitler_prefactor,
    eedl_weight,
    eedl_lower_panel,
    eedl_panel_fraction,
    eedl_available,
    path_flat,
    mu_flat,
    E_grid,
    panel_probability,
    bremslib_weight,
    bremslib_lower_top,
    bremslib_upper_top,
    bremslib_energy_fraction,
    bremslib_available,
    bremslib_lower_values,
    bremslib_upper_values,
    nominal_ratio,
    spec,
    Z,
    n_seg,
    n_E,
    n_layers,
):
    """Fuse BremsLib interpolation, fallback, attenuation, and reduction."""
    k = jit.blockIdx.x
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x
    photon_eV = E_grid[k]
    acc = F32_ZERO

    line = tid
    while line < n_seg:
        weighted = _bremslib_or_fallback_weighted_scalar(
            T[line],
            photon_eV,
            k,
            line,
            p_i_arr[line],
            bethe_heitler_prefactor[line],
            eedl_weight[line],
            eedl_lower_panel[line],
            eedl_panel_fraction[line],
            eedl_available[line],
            panel_probability,
            bremslib_weight[line],
            bremslib_lower_top[line],
            bremslib_upper_top[line],
            bremslib_energy_fraction[line],
            bremslib_available[line],
            bremslib_lower_values,
            bremslib_upper_values,
            nominal_ratio,
            n_E,
            Z,
        )
        tau = _tau_scalar(path_flat, mu_flat, line, k, n_layers, n_E)
        acc += weighted * xp.exp(-tau)
        line += nthreads

    shared = jit.shared_memory(xp.float32, None)
    shared[tid] = acc
    jit.syncthreads()
    stride = nthreads // U32_TWO
    while stride > U32_ZERO:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if tid == U32_ZERO:
        spec[k] += shared[U32_ZERO]


@jit.rawkernel()
def _kernel_1e(
    T, p_i_arr, incident_prefactor, path_flat, mu_flat, E_grid, spec, Z, n_seg, n_E, n_layers
):
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
def _kernel_2e(
    T, p_i_arr, incident_prefactor, path_flat, mu_flat, E_grid, spec, Z, n_seg, n_E, n_layers
):
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
def _kernel_3e(
    T, p_i_arr, incident_prefactor, path_flat, mu_flat, E_grid, spec, Z, n_seg, n_E, n_layers
):
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
def _kernel_4e(
    T, p_i_arr, incident_prefactor, path_flat, mu_flat, E_grid, spec, Z, n_seg, n_E, n_layers
):
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


def run_eedl_brem_reduction_kernel(
    T_keV,
    p_i,
    bethe_heitler_prefactor,
    eedl_weight,
    lower_panel,
    panel_fraction,
    available,
    path_flat,
    mu_flat,
    E_grid,
    panel_probability,
    *,
    Z,
    n_layers,
    out=None,
    config=DEFAULT_BREM_KERNEL_CONFIG,
):
    """Accumulate staged EEDL bremsstrahlung without dense segment-grid arrays.

    ``panel_probability`` is the C-order ``(n_panel, n_E)`` probability table
    already evaluated on ``E_grid``. The per-segment arrays contain the two
    adjacent-panel selector, interpolation weight, and
    ``density * length * sigma / normalization``. Rows with ``available == 0``
    evaluate the retained Bethe--Heitler expression from the supplied fallback
    prefactor. All arrays must be contiguous float32 CuPy arrays except
    ``lower_panel``, which is uint32.
    """
    nthreads = int(config.nthreads)
    if nthreads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    n_E = int(E_grid.size)
    if out is None:
        out = xp.zeros(n_E, dtype=xp.float32)
    if n_E == 0 or T_keV.size == 0:
        return out

    _eedl_kernel_1e(
        (n_E,),
        (nthreads,),
        (
            T_keV,
            p_i,
            bethe_heitler_prefactor,
            eedl_weight,
            lower_panel,
            panel_fraction,
            available,
            path_flat,
            mu_flat,
            E_grid,
            panel_probability,
            out,
            np.float32(Z),
            np.uint32(T_keV.size),
            np.uint32(n_E),
            np.uint32(n_layers),
        ),
        shared_mem=nthreads * np.dtype(np.float32).itemsize,
    )
    return out


def run_bremslib_brem_reduction_kernel(
    T_keV,
    p_i,
    bethe_heitler_prefactor,
    eedl_weight,
    eedl_lower_panel,
    eedl_panel_fraction,
    eedl_available,
    path_flat,
    mu_flat,
    E_grid,
    panel_probability,
    bremslib_weight,
    bremslib_lower_top,
    bremslib_upper_top,
    bremslib_energy_fraction,
    bremslib_available,
    bremslib_lower_values,
    bremslib_upper_values,
    nominal_ratio,
    *,
    Z,
    n_layers,
    out=None,
    config=DEFAULT_BREM_KERNEL_CONFIG,
):
    """Accumulate staged direction-resolved BremsLib bremsstrahlung.

    The BremsLib arrays hold the two incident-energy rows already interpolated
    to each segment's emission angle. The raw kernel performs the remaining
    ``k/T`` and incident-energy interpolation, then falls back per segment to
    the staged EEDL/Bethe--Heitler inputs when BremsLib has no coverage.
    ``bremslib_weight`` includes density, path length, ``4*pi``, the mb-to-cm2
    conversion, and ``Z**2`` so the caller can retain the common isotropic
    ``1/(4*pi)`` normalization after reduction.
    """
    nthreads = int(config.nthreads)
    if nthreads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    n_E = int(E_grid.size)
    if out is None:
        out = xp.zeros(n_E, dtype=xp.float32)
    if n_E == 0 or T_keV.size == 0:
        return out

    _bremslib_kernel_1e(
        (n_E,),
        (nthreads,),
        (
            T_keV,
            p_i,
            bethe_heitler_prefactor,
            eedl_weight,
            eedl_lower_panel,
            eedl_panel_fraction,
            eedl_available,
            path_flat,
            mu_flat,
            E_grid,
            panel_probability,
            bremslib_weight,
            bremslib_lower_top,
            bremslib_upper_top,
            bremslib_energy_fraction,
            bremslib_available,
            bremslib_lower_values,
            bremslib_upper_values,
            nominal_ratio,
            out,
            np.float32(Z),
            np.uint32(T_keV.size),
            np.uint32(n_E),
            np.uint32(n_layers),
        ),
        shared_mem=nthreads * np.dtype(np.float32).itemsize,
    )
    return out


def run_bremslib_element_reduction(
    T_keV,
    L_ang,
    p_i,
    bethe_heitler_prefactor,
    path_flat,
    mu_flat,
    E_grid,
    staged,
    cos_theta,
    eedl_context,
    *,
    Z,
    number_density_ang3,
    n_layers,
    out,
):
    """Stage one element's segment state and launch the BremsLib reducer."""
    from .brem_bremslib import bremslib_segment_state

    state = bremslib_segment_state(staged, T_keV, cos_theta)
    if eedl_context is None:
        eedl_weight = xp.zeros_like(T_keV)
        eedl_lower_panel = xp.zeros(T_keV.size, dtype=np.uint32)
        eedl_panel_fraction = xp.zeros_like(T_keV)
        eedl_available = xp.zeros_like(T_keV)
        panel_probability = xp.zeros(max(2 * int(E_grid.size), 1), dtype=xp.float32)
    else:
        eedl_state = eedl_context.state
        eedl_weight = xp.ascontiguousarray(
            np.float32(number_density_ang3 * 1.0e24)
            * L_ang
            * F32_1E_M8
            * eedl_state.differential_scale_cm2,
            dtype=xp.float32,
        )
        eedl_lower_panel = xp.ascontiguousarray(eedl_state.lower_panel, dtype=np.uint32)
        eedl_panel_fraction = xp.ascontiguousarray(
            eedl_state.panel_fraction, dtype=xp.float32
        )
        eedl_available = xp.ascontiguousarray(eedl_state.available, dtype=xp.float32)
        panel_probability = xp.ascontiguousarray(
            eedl_context.prepared.photon_probability_on_grid_per_eV.reshape(-1),
            dtype=xp.float32,
        )
    weight = np.float32(
        number_density_ang3 * 1.0e24 * 1.0e-8 * 4.0 * np.pi * 1.0e-27 * Z * Z
    )
    lower_row = state.lower_row
    return run_bremslib_brem_reduction_kernel(
        T_keV,
        p_i,
        bethe_heitler_prefactor,
        eedl_weight,
        eedl_lower_panel,
        eedl_panel_fraction,
        eedl_available,
        path_flat,
        mu_flat,
        E_grid,
        panel_probability,
        xp.ascontiguousarray(L_ang * weight, dtype=xp.float32),
        xp.ascontiguousarray(staged.top_reduced_energy[lower_row], dtype=xp.float32),
        xp.ascontiguousarray(staged.top_reduced_energy[lower_row + 1], dtype=xp.float32),
        xp.ascontiguousarray(state.energy_fraction, dtype=xp.float32),
        xp.ascontiguousarray(state.available, dtype=xp.float32),
        xp.ascontiguousarray(state.lower_values.reshape(-1), dtype=xp.float32),
        xp.ascontiguousarray(state.upper_values.reshape(-1), dtype=xp.float32),
        xp.ascontiguousarray(staged.table.nominal_reduced_energy, dtype=xp.float32),
        Z=Z,
        n_layers=n_layers,
        out=out,
    )
