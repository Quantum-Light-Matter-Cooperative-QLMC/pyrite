"""Fused CUDA prologue for incoherent CXR line emission.

One CUDA thread owns one ``(segment, reciprocal-vector)`` pair and evaluates
resonance kinematics, all five table interpolations, both polarizations, and
the absorption-weighted line coefficient.  Invalid pairs are represented by a
zero weight in fixed pair order.  That avoids the nondeterministic output order
of atomic compaction and lets :mod:`spectrum_jit_kernel` skip rejected pairs
before evaluating the expensive sinc.

The caller deliberately retains the eager CuPy path for coherent emission,
component spectra, non-float32 backends, and sinc-cutoff windowing.
"""

from dataclasses import dataclass

import cupy as xp
import numpy as np
from cupyx import jit

F32_ZERO = np.float32(0.0)
F32_ONE = np.float32(1.0)
F32_TWO = np.float32(2.0)
F32_TEN = np.float32(10.0)
F32_MAX = np.float32(np.finfo(np.float32).max)

U32_ZERO = np.uint32(0)
U32_ONE = np.uint32(1)
U32_TWO = np.uint32(2)
U32_THREE = np.uint32(3)


@dataclass(frozen=True)
class LinePrologueKernelConfig:
    nthreads: int = 256


DEFAULT_LINE_PROLOGUE_KERNEL_CONFIG = LinePrologueKernelConfig()


@jit.rawkernel(device=True)
def _interp_row(table, row, idx, frac, below, above, n_tab):
    base = row * n_tab
    if below:
        return table[base]
    if above:
        return table[base + n_tab - U32_ONE]
    f0 = table[base + idx - U32_ONE]
    return f0 + frac * (table[base + idx] - f0)


@jit.rawkernel(device=True)
def _interp_shared(table, idx, frac, below, above, n_tab):
    if below:
        return table[U32_ZERO]
    if above:
        return table[n_tab - U32_ONE]
    f0 = table[idx - U32_ONE]
    return f0 + frac * (table[idx] - f0)


@jit.rawkernel(device=True)
def _bracket_index(grid, x, n_tab):
    """Equivalent to ``clip(searchsorted(grid, x), 1, n_tab - 1)``."""
    lo = U32_ZERO
    hi = n_tab
    while lo < hi:
        mid = (lo + hi) // U32_TWO
        if grid[mid] < x:
            lo = mid + U32_ONE
        else:
            hi = mid
    if lo < U32_ONE:
        return U32_ONE
    if lo >= n_tab:
        return n_tab - U32_ONE
    return lo


@jit.rawkernel(device=True)
def _polarization_amplitude_sq(
    ex,
    ey,
    ez,
    gx,
    gy,
    gz,
    vx,
    vy,
    vz,
    chi_re,
    chi_im,
    u_re,
    u_im,
    v_dot_kg,
    omega,
    v_dot_g,
    k_dot_g,
    k_dot_v,
    gamma,
    detuning,
):
    g_dot_e = gx * ex + gy * ey + gz * ez
    v_dot_e = vx * ex + vy * ey + vz * ez
    f_pxr = (v_dot_kg * g_dot_e - omega * omega * v_dot_e) / detuning
    braced_ge = g_dot_e - v_dot_g * v_dot_e
    braced_kg = k_dot_g - k_dot_v * v_dot_g
    f_cbs = -(braced_ge + v_dot_e * braced_kg / v_dot_g) / (gamma * v_dot_g)
    re = chi_re * f_pxr + u_re * f_cbs
    im = chi_im * f_pxr + u_im * f_cbs
    return re * re + im * im


@jit.rawkernel()
def _line_prologue_kernel(
    v_flat,
    denom,
    gamma,
    t_L,
    L_esc,
    line_electron,
    g_flat,
    es_flat,
    ep_flat,
    mosaic_weight,
    E_tab,
    chi_re_tab,
    chi_im_tab,
    u_re_tab,
    u_im_tab,
    mu_tab,
    E_r_out,
    aw_out,
    weight_out,
    lo_keep,
    hi_keep,
    hbarc,
    electron_mass_eV,
    alpha_fs,
    pref_c1,
    nx,
    ny,
    nz,
    n_pairs,
    n_g,
    n_tab,
):
    pair = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if pair >= n_pairs:
        return

    # Every slot is initialized, so rejected pairs can be consumed safely by
    # the reduction kernel without a separate mask-compaction pass.
    E_r_out[pair] = F32_ZERO
    aw_out[pair] = F32_ZERO
    weight_out[pair] = F32_ZERO

    seg = pair // n_g
    g = pair - seg * n_g
    if not line_electron[seg]:
        return

    vbase = seg * U32_THREE
    gbase = g * U32_THREE
    vx = v_flat[vbase]
    vy = v_flat[vbase + U32_ONE]
    vz = v_flat[vbase + U32_TWO]
    gx = g_flat[gbase]
    gy = g_flat[gbase + U32_ONE]
    gz = g_flat[gbase + U32_TWO]

    dnm = denom[seg]
    v_dot_g = vx * gx + vy * gy + vz * gz
    omega = v_dot_g / dnm
    E_res = hbarc * omega
    if E_res <= lo_keep or E_res <= F32_TEN or E_res >= hi_keep:
        return

    below = E_res <= E_tab[U32_ZERO]
    above = E_res >= E_tab[n_tab - U32_ONE]
    idx = _bracket_index(E_tab, E_res, n_tab)
    x0 = E_tab[idx - U32_ONE]
    frac = (E_res - x0) / (E_tab[idx] - x0)

    chi_re = _interp_row(chi_re_tab, g, idx, frac, below, above, n_tab)
    chi_im = _interp_row(chi_im_tab, g, idx, frac, below, above, n_tab)
    u_re = _interp_row(u_re_tab, g, idx, frac, below, above, n_tab) / electron_mass_eV
    u_im = _interp_row(u_im_tab, g, idx, frac, below, above, n_tab) / electron_mass_eV
    mu = _interp_shared(mu_tab, idx, frac, below, above, n_tab)

    esx = es_flat[gbase]
    esy = es_flat[gbase + U32_ONE]
    esz = es_flat[gbase + U32_TWO]
    epx = ep_flat[gbase]
    epy = ep_flat[gbase + U32_ONE]
    epz = ep_flat[gbase + U32_TWO]
    kx = omega * nx
    ky = omega * ny
    kz = omega * nz
    kgx = kx + gx
    kgy = ky + gy
    kgz = kz + gz
    detuning = kgx * kgx + kgy * kgy + kgz * kgz - omega * omega
    k_dot_g = kx * gx + ky * gy + kz * gz
    v_dot_kg = vx * kgx + vy * kgy + vz * kgz
    k_dot_v = omega * (F32_ONE - dnm)

    gm = gamma[seg]
    a2 = _polarization_amplitude_sq(
        esx,
        esy,
        esz,
        gx,
        gy,
        gz,
        vx,
        vy,
        vz,
        chi_re,
        chi_im,
        u_re,
        u_im,
        v_dot_kg,
        omega,
        v_dot_g,
        k_dot_g,
        k_dot_v,
        gm,
        detuning,
    )
    a2 += _polarization_amplitude_sq(
        epx,
        epy,
        epz,
        gx,
        gy,
        gz,
        vx,
        vy,
        vz,
        chi_re,
        chi_im,
        u_re,
        u_im,
        v_dot_kg,
        omega,
        v_dot_g,
        k_dot_g,
        k_dot_v,
        gm,
        detuning,
    )

    duration = t_L[seg]
    pref = alpha_fs * omega / pref_c1 * (duration * duration) * xp.exp(-(L_esc[seg] * mu))
    weight = pref * a2 * mosaic_weight[g]
    if weight > F32_ZERO and weight < F32_MAX and duration > F32_ZERO:
        E_r_out[pair] = E_res
        aw_out[pair] = dnm * duration / (F32_TWO * hbarc)
        weight_out[pair] = weight


def run_line_prologue_kernel(
    v_flat,
    denom,
    gamma,
    t_L,
    L_esc,
    line_electron,
    g_flat,
    es_flat,
    ep_flat,
    mosaic_weight,
    E_tab,
    chi_re_tab,
    chi_im_tab,
    u_re_tab,
    u_im_tab,
    mu_tab,
    *,
    lo_keep,
    hi_keep,
    hbarc,
    electron_mass_eV,
    alpha_fs,
    pref_c1,
    n_hat,
    config=DEFAULT_LINE_PROLOGUE_KERNEL_CONFIG,
):
    """Return fixed-order ``(E_res, a_width, weight)`` float32 buffers.

    Array inputs must be contiguous.  Geometry arrays are flattened C-order;
    the per-g tables have shape ``(n_g, n_tab)`` before flattening.
    """
    nthreads = int(config.nthreads)
    if nthreads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    n_g = int(mosaic_weight.size)
    n_tab = int(E_tab.size)
    if n_g == 0 or n_tab < 2:
        raise ValueError("line prologue requires at least one g row and two tabulation points")
    n_seg = int(denom.size)
    n_pairs = n_seg * n_g
    E_r = xp.empty(n_pairs, dtype=xp.float32)
    aw = xp.empty(n_pairs, dtype=xp.float32)
    weight = xp.empty(n_pairs, dtype=xp.float32)
    if n_pairs == 0:
        return E_r, aw, weight

    nblocks = (n_pairs + nthreads - 1) // nthreads
    _line_prologue_kernel(
        (nblocks,),
        (nthreads,),
        (
            v_flat,
            denom,
            gamma,
            t_L,
            L_esc,
            line_electron,
            g_flat,
            es_flat,
            ep_flat,
            mosaic_weight,
            E_tab,
            chi_re_tab,
            chi_im_tab,
            u_re_tab,
            u_im_tab,
            mu_tab,
            E_r,
            aw,
            weight,
            np.float32(lo_keep),
            np.float32(hi_keep),
            np.float32(hbarc),
            np.float32(electron_mass_eV),
            np.float32(alpha_fs),
            np.float32(pref_c1),
            np.float32(n_hat[0]),
            np.float32(n_hat[1]),
            np.float32(n_hat[2]),
            np.uint32(n_pairs),
            np.uint32(n_g),
            np.uint32(n_tab),
        ),
    )
    return E_r, aw, weight
