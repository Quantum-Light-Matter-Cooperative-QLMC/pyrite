"""CUDA JIT reduction for coherent CXR line fields.

For each photon-energy bin the kernel sums the complex field over emitting
segments for both orthogonal polarizations, then accumulates
``(|F_s|^2 + |F_p|^2) * mosaic_weight`` directly into the caller's spectrum.
This replaces construction of the dense complex ``SP[segment, energy]`` matrix
and the two complex GEMV reductions used by the CuPy fallback.
"""

from dataclasses import dataclass

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
U32_FIVE = np.uint32(5)
U32_SIX = np.uint32(6)
U32_SEVEN = np.uint32(7)
U32_EIGHT = np.uint32(8)
U32_NINE = np.uint32(9)
U32_TEN = np.uint32(10)
U32_ELEVEN = np.uint32(11)


@dataclass(frozen=True)
class CoherentKernelConfig:
    nthreads: int
    energies_per_block: int


# Starter configuration only; coherent work has much higher register and
# transcendental pressure than the incoherent line kernel. Retune per GPU.
DEFAULT_COHERENT_KERNEL_CONFIG = CoherentKernelConfig(nthreads=256, energies_per_block=2)


_DUMMY_F32 = None


def _dummy():
    """One cached length-1 float32 array to bind the unused in-medium pointers.

    Allocated lazily so importing this module does not touch the device.
    """
    global _DUMMY_F32
    if _DUMMY_F32 is None:
        _DUMMY_F32 = xp.zeros(1, dtype=xp.float32)
    return _DUMMY_F32


@jit.rawkernel(device=True)
def _sinc_unscaled(x):
    if x == F32_ZERO:
        x = F32_TINY
    return xp.sin(x) / x


@jit.rawkernel()
def _kernel_1e(
    E_r,
    aw,
    phase_slope,
    g_phase,
    cs_re,
    cs_im,
    cp_re,
    cp_im,
    E_grid,
    L_esc,
    delta_omega,
    spec,
    wm,
    use_medium,
    n_lines,
    n_E,
):
    base = jit.blockIdx.x
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x
    k0 = base
    E0 = E_grid[k0]
    dw0 = F32_ZERO
    if use_medium:
        dw0 = delta_omega[k0]
    sr0 = F32_ZERO
    si0 = F32_ZERO
    pr0 = F32_ZERO
    pi0 = F32_ZERO

    line = tid
    while line < n_lines:
        Er = E_r[line]
        aa = aw[line]
        ps = phase_slope[line]
        gp = g_phase[line]
        Lj = F32_ZERO
        if use_medium:
            Lj = L_esc[line]
        csr = cs_re[line]
        csi = cs_im[line]
        cpr = cp_re[line]
        cpi = cp_im[line]
        x = aa * (E0 - Er)
        s = _sinc_unscaled(x)
        phase = ps * E0 - gp
        if use_medium:
            phase = phase - Lj * dw0
        cph = xp.cos(phase)
        sph = xp.sin(phase)
        sr0 += s * (csr * cph - csi * sph)
        si0 += s * (csr * sph + csi * cph)
        pr0 += s * (cpr * cph - cpi * sph)
        pi0 += s * (cpr * sph + cpi * cph)
        line += nthreads

    shared = jit.shared_memory(xp.float32, None)
    o0 = U32_ZERO
    shared[o0 + tid] = sr0
    o1 = U32_ONE * nthreads
    shared[o1 + tid] = si0
    o2 = U32_TWO * nthreads
    shared[o2 + tid] = pr0
    o3 = U32_THREE * nthreads
    shared[o3 + tid] = pi0
    jit.syncthreads()
    stride = nthreads // U32_TWO
    while stride > U32_ZERO:
        if tid < stride:
            shared[o0 + tid] += shared[o0 + tid + stride]
            shared[o1 + tid] += shared[o1 + tid + stride]
            shared[o2 + tid] += shared[o2 + tid + stride]
            shared[o3 + tid] += shared[o3 + tid + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if tid == U32_ZERO:
        sr = shared[o0]
        si = shared[o1]
        pr = shared[o2]
        pi = shared[o3]
        spec[k0] += wm * (sr * sr + si * si + pr * pr + pi * pi)


@jit.rawkernel()
def _kernel_2e(
    E_r,
    aw,
    phase_slope,
    g_phase,
    cs_re,
    cs_im,
    cp_re,
    cp_im,
    E_grid,
    L_esc,
    delta_omega,
    spec,
    wm,
    use_medium,
    n_lines,
    n_E,
):
    base = jit.blockIdx.x * U32_TWO
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x
    k0 = base
    E0 = E_grid[k0]
    dw0 = F32_ZERO
    if use_medium:
        dw0 = delta_omega[k0]
    sr0 = F32_ZERO
    si0 = F32_ZERO
    pr0 = F32_ZERO
    pi0 = F32_ZERO
    k1 = base + U32_ONE
    has1 = k1 < n_E
    E1 = F32_ZERO
    dw1 = F32_ZERO
    if has1:
        E1 = E_grid[k1]
        if use_medium:
            dw1 = delta_omega[k1]
    sr1 = F32_ZERO
    si1 = F32_ZERO
    pr1 = F32_ZERO
    pi1 = F32_ZERO

    line = tid
    while line < n_lines:
        Er = E_r[line]
        aa = aw[line]
        ps = phase_slope[line]
        gp = g_phase[line]
        Lj = F32_ZERO
        if use_medium:
            Lj = L_esc[line]
        csr = cs_re[line]
        csi = cs_im[line]
        cpr = cp_re[line]
        cpi = cp_im[line]
        x = aa * (E0 - Er)
        s = _sinc_unscaled(x)
        phase = ps * E0 - gp
        if use_medium:
            phase = phase - Lj * dw0
        cph = xp.cos(phase)
        sph = xp.sin(phase)
        sr0 += s * (csr * cph - csi * sph)
        si0 += s * (csr * sph + csi * cph)
        pr0 += s * (cpr * cph - cpi * sph)
        pi0 += s * (cpr * sph + cpi * cph)
        if has1:
            x = aa * (E1 - Er)
            s = _sinc_unscaled(x)
            phase = ps * E1 - gp
            if use_medium:
                phase = phase - Lj * dw1
            cph = xp.cos(phase)
            sph = xp.sin(phase)
            sr1 += s * (csr * cph - csi * sph)
            si1 += s * (csr * sph + csi * cph)
            pr1 += s * (cpr * cph - cpi * sph)
            pi1 += s * (cpr * sph + cpi * cph)
        line += nthreads

    shared = jit.shared_memory(xp.float32, None)
    o0 = U32_ZERO
    shared[o0 + tid] = sr0
    o1 = U32_ONE * nthreads
    shared[o1 + tid] = si0
    o2 = U32_TWO * nthreads
    shared[o2 + tid] = pr0
    o3 = U32_THREE * nthreads
    shared[o3 + tid] = pi0
    o4 = U32_FOUR * nthreads
    shared[o4 + tid] = sr1
    o5 = U32_FIVE * nthreads
    shared[o5 + tid] = si1
    o6 = U32_SIX * nthreads
    shared[o6 + tid] = pr1
    o7 = U32_SEVEN * nthreads
    shared[o7 + tid] = pi1
    jit.syncthreads()
    stride = nthreads // U32_TWO
    while stride > U32_ZERO:
        if tid < stride:
            shared[o0 + tid] += shared[o0 + tid + stride]
            shared[o1 + tid] += shared[o1 + tid + stride]
            shared[o2 + tid] += shared[o2 + tid + stride]
            shared[o3 + tid] += shared[o3 + tid + stride]
            shared[o4 + tid] += shared[o4 + tid + stride]
            shared[o5 + tid] += shared[o5 + tid + stride]
            shared[o6 + tid] += shared[o6 + tid + stride]
            shared[o7 + tid] += shared[o7 + tid + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if tid == U32_ZERO:
        sr = shared[o0]
        si = shared[o1]
        pr = shared[o2]
        pi = shared[o3]
        spec[k0] += wm * (sr * sr + si * si + pr * pr + pi * pi)
        if has1:
            sr = shared[o4]
            si = shared[o5]
            pr = shared[o6]
            pi = shared[o7]
            spec[k1] += wm * (sr * sr + si * si + pr * pr + pi * pi)


@jit.rawkernel()
def _kernel_3e(
    E_r,
    aw,
    phase_slope,
    g_phase,
    cs_re,
    cs_im,
    cp_re,
    cp_im,
    E_grid,
    L_esc,
    delta_omega,
    spec,
    wm,
    use_medium,
    n_lines,
    n_E,
):
    base = jit.blockIdx.x * U32_THREE
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x
    k0 = base
    E0 = E_grid[k0]
    dw0 = F32_ZERO
    if use_medium:
        dw0 = delta_omega[k0]
    sr0 = F32_ZERO
    si0 = F32_ZERO
    pr0 = F32_ZERO
    pi0 = F32_ZERO
    k1 = base + U32_ONE
    has1 = k1 < n_E
    E1 = F32_ZERO
    dw1 = F32_ZERO
    if has1:
        E1 = E_grid[k1]
        if use_medium:
            dw1 = delta_omega[k1]
    sr1 = F32_ZERO
    si1 = F32_ZERO
    pr1 = F32_ZERO
    pi1 = F32_ZERO
    k2 = base + U32_TWO
    has2 = k2 < n_E
    E2 = F32_ZERO
    dw2 = F32_ZERO
    if has2:
        E2 = E_grid[k2]
        if use_medium:
            dw2 = delta_omega[k2]
    sr2 = F32_ZERO
    si2 = F32_ZERO
    pr2 = F32_ZERO
    pi2 = F32_ZERO

    line = tid
    while line < n_lines:
        Er = E_r[line]
        aa = aw[line]
        ps = phase_slope[line]
        gp = g_phase[line]
        Lj = F32_ZERO
        if use_medium:
            Lj = L_esc[line]
        csr = cs_re[line]
        csi = cs_im[line]
        cpr = cp_re[line]
        cpi = cp_im[line]
        x = aa * (E0 - Er)
        s = _sinc_unscaled(x)
        phase = ps * E0 - gp
        if use_medium:
            phase = phase - Lj * dw0
        cph = xp.cos(phase)
        sph = xp.sin(phase)
        sr0 += s * (csr * cph - csi * sph)
        si0 += s * (csr * sph + csi * cph)
        pr0 += s * (cpr * cph - cpi * sph)
        pi0 += s * (cpr * sph + cpi * cph)
        if has1:
            x = aa * (E1 - Er)
            s = _sinc_unscaled(x)
            phase = ps * E1 - gp
            if use_medium:
                phase = phase - Lj * dw1
            cph = xp.cos(phase)
            sph = xp.sin(phase)
            sr1 += s * (csr * cph - csi * sph)
            si1 += s * (csr * sph + csi * cph)
            pr1 += s * (cpr * cph - cpi * sph)
            pi1 += s * (cpr * sph + cpi * cph)
        if has2:
            x = aa * (E2 - Er)
            s = _sinc_unscaled(x)
            phase = ps * E2 - gp
            if use_medium:
                phase = phase - Lj * dw2
            cph = xp.cos(phase)
            sph = xp.sin(phase)
            sr2 += s * (csr * cph - csi * sph)
            si2 += s * (csr * sph + csi * cph)
            pr2 += s * (cpr * cph - cpi * sph)
            pi2 += s * (cpr * sph + cpi * cph)
        line += nthreads

    shared = jit.shared_memory(xp.float32, None)
    o0 = U32_ZERO
    shared[o0 + tid] = sr0
    o1 = U32_ONE * nthreads
    shared[o1 + tid] = si0
    o2 = U32_TWO * nthreads
    shared[o2 + tid] = pr0
    o3 = U32_THREE * nthreads
    shared[o3 + tid] = pi0
    o4 = U32_FOUR * nthreads
    shared[o4 + tid] = sr1
    o5 = U32_FIVE * nthreads
    shared[o5 + tid] = si1
    o6 = U32_SIX * nthreads
    shared[o6 + tid] = pr1
    o7 = U32_SEVEN * nthreads
    shared[o7 + tid] = pi1
    o8 = U32_EIGHT * nthreads
    shared[o8 + tid] = sr2
    o9 = U32_NINE * nthreads
    shared[o9 + tid] = si2
    o10 = U32_TEN * nthreads
    shared[o10 + tid] = pr2
    o11 = U32_ELEVEN * nthreads
    shared[o11 + tid] = pi2
    jit.syncthreads()
    stride = nthreads // U32_TWO
    while stride > U32_ZERO:
        if tid < stride:
            shared[o0 + tid] += shared[o0 + tid + stride]
            shared[o1 + tid] += shared[o1 + tid + stride]
            shared[o2 + tid] += shared[o2 + tid + stride]
            shared[o3 + tid] += shared[o3 + tid + stride]
            shared[o4 + tid] += shared[o4 + tid + stride]
            shared[o5 + tid] += shared[o5 + tid + stride]
            shared[o6 + tid] += shared[o6 + tid + stride]
            shared[o7 + tid] += shared[o7 + tid + stride]
            shared[o8 + tid] += shared[o8 + tid + stride]
            shared[o9 + tid] += shared[o9 + tid + stride]
            shared[o10 + tid] += shared[o10 + tid + stride]
            shared[o11 + tid] += shared[o11 + tid + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if tid == U32_ZERO:
        sr = shared[o0]
        si = shared[o1]
        pr = shared[o2]
        pi = shared[o3]
        spec[k0] += wm * (sr * sr + si * si + pr * pr + pi * pi)
        if has1:
            sr = shared[o4]
            si = shared[o5]
            pr = shared[o6]
            pi = shared[o7]
            spec[k1] += wm * (sr * sr + si * si + pr * pr + pi * pi)
        if has2:
            sr = shared[o8]
            si = shared[o9]
            pr = shared[o10]
            pi = shared[o11]
            spec[k2] += wm * (sr * sr + si * si + pr * pr + pi * pi)


_REDUCTION_KERNELS = {1: _kernel_1e, 2: _kernel_2e, 3: _kernel_3e}


def run_coherent_reduction_kernel(
    E_r,
    aw,
    phase_slope,
    g_phase,
    c_s_re,
    c_s_im,
    c_p_re,
    c_p_im,
    E_grid,
    *,
    out,
    mosaic_weight=1.0,
    L_esc=None,
    delta_omega=None,
    config=DEFAULT_COHERENT_KERNEL_CONFIG,
):
    """Accumulate one reflection/orientation's coherent intensity into ``out``.

    Inputs are contiguous float32 CuPy arrays. ``phase_slope`` is
    ``(t_abs - n_hat.r) / HBARC_EV_ANG`` so the per-cell phase is
    ``phase_slope[j] * E_grid[k] - g_phase[j]``.

    Under ``xray_dispersion="refractive"`` the caller also supplies the
    per-line escape distance ``L_esc`` (Angstrom) and the per-energy table
    ``delta_omega[k] = (1 - Re n(E_k)) * omega(E_k)``, which add the in-medium
    term ``- L_esc[j] * delta_omega[k]`` to that phase. This is a second
    (per-line scalar) x (per-energy table) product, so it cannot be folded into
    ``phase_slope``. Both must be given together or both omitted; when omitted
    the kernel evaluates the vacuum phase expression unchanged (the in-medium
    term sits behind a launch-uniform branch, so vacuum stays bit-for-bit).
    """
    nthreads = int(config.nthreads)
    epb = int(config.energies_per_block)
    if nthreads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    try:
        kernel = _REDUCTION_KERNELS[epb]
    except KeyError:
        raise ValueError(f"energies_per_block must be one of {tuple(_REDUCTION_KERNELS)}") from None

    if (L_esc is None) != (delta_omega is None):
        raise ValueError("L_esc and delta_omega must be given together")

    n_E = int(E_grid.size)
    if n_E == 0 or E_r.size == 0:
        return out
    use_medium = L_esc is not None
    if use_medium:
        if int(L_esc.size) != int(E_r.size):
            raise ValueError("L_esc must have one entry per line")
        if int(delta_omega.size) != n_E:
            raise ValueError("delta_omega must have one entry per energy bin")
    else:
        L_esc = delta_omega = _dummy()
    nblocks = (n_E + epb - 1) // epb
    shared_bytes = 4 * epb * nthreads * np.dtype(np.float32).itemsize
    kernel(
        (nblocks,),
        (nthreads,),
        (
            E_r,
            aw,
            phase_slope,
            g_phase,
            c_s_re,
            c_s_im,
            c_p_re,
            c_p_im,
            E_grid,
            L_esc,
            delta_omega,
            out,
            np.float32(mosaic_weight),
            np.uint32(1 if use_medium else 0),
            np.uint32(E_r.size),
            np.uint32(n_E),
        ),
        shared_mem=shared_bytes,
    )
    return out
