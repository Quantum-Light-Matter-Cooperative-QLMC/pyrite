"""Streaming CUDA JIT pipeline for coherent CXR line emission.

The coherent spectrum cannot reduce independent segment blocks as intensities:
``|sum_j F_j|^2`` requires the complex field from every segment to be added
before squaring.  This module therefore splits the GPU work into three stages:

1. ``run_coherent_prologue_kernel`` evaluates the energy-independent line data
   once per ``(g, segment)`` pair for the current segment block.  Outputs use
   g-major fixed order; rejected pairs keep zero field coefficients and need no
   mask compaction or row-count synchronization.
2. ``run_coherent_field_accumulation_kernel`` owns one ``(g, energy-group)``
   CUDA block, reduces the current segment block into four persistent field
   planes (sigma/pi real/imag), and adds those partial fields in place.  Each
   output cell has exactly one writer per launch, so no atomics are required.
3. ``finalize_coherent_fields`` squares the completed fields and sums the
   mosaic-weighted reflection/orientation rows into the spectrum.  Different
   g rows therefore remain incoherent by construction.

Peak pair scratch is O(6 * segment_block * N_g) plus two O(segment_block)
segment-only geometry arrays. ``aw`` and ``phase_slope`` are no longer duplicated
for every g row. No device->host line-count transfer is needed between the
prologue and reduction stages.
"""

from dataclasses import dataclass

import cupy as xp
import numpy as np

from .._cupy_jit import jit

F32_ZERO = np.float32(0.0)
F32_ONE = np.float32(1.0)
F32_TWO = np.float32(2.0)
F32_TEN = np.float32(10.0)
F32_TINY = np.float32(1.0e-20)
F32_MAX = np.float32(np.finfo(np.float32).max)

U32_ZERO = np.uint32(0)
U32_ONE = np.uint32(1)
U32_TWO = np.uint32(2)
U32_THREE = np.uint32(3)
U32_FOUR = np.uint32(4)
U32_FIVE = np.uint32(5)
U32_SIX = np.uint32(6)
U32_SEVEN = np.uint32(7)


@dataclass(frozen=True)
class CoherentStreamKernelConfig:
    prologue_nthreads: int = 256
    reduction_nthreads: int = 256
    energies_per_block: int = 2
    finalize_nthreads: int = 256


DEFAULT_COHERENT_STREAM_KERNEL_CONFIG = CoherentStreamKernelConfig()


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
def _interp_elemental_mu(log_mu_table, idx, log_frac, below, above, n_mu, n_tab):
    """Sum per-element log-log attenuation interpolation on device.

    Validation: line-absorption-tabulation
    """
    mu = F32_ZERO
    element = U32_ZERO
    while element < n_mu:
        base = element * n_tab
        if below:
            log_mu = log_mu_table[base]
        elif above:
            log_mu = log_mu_table[base + n_tab - U32_ONE]
        else:
            log0 = log_mu_table[base + idx - U32_ONE]
            log_mu = log0 + log_frac * (log_mu_table[base + idx] - log0)
        mu += xp.exp(log_mu)
        element += U32_ONE
    return mu


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


@jit.rawkernel()
def _coherent_prologue_kernel(
    v_flat,
    denom,
    v_dot_n,
    gamma,
    t_L,
    L_esc,
    line_electron,
    r_flat,
    g_flat,
    es_flat,
    ep_flat,
    g2,
    n_dot_g,
    g_dot_es,
    g_dot_ep,
    E_tab,
    chi_re_tab,
    chi_im_tab,
    u_re_tab,
    u_im_tab,
    log_mu_tab,
    n_re_tab,
    E_r_out,
    aw_out,
    g_phase_out,
    cs_re_out,
    cs_im_out,
    cp_re_out,
    cp_im_out,
    lo_keep,
    hi_keep,
    hbarc,
    alpha_fs,
    pref_c1,
    use_medium,
    n_pairs,
    n_seg,
    n_g,
    n_mu,
    n_tab,
):
    # g-major fixed order: pair = g*n_seg + seg.  The reduction kernel then
    # reads each g row contiguously while its threads stride over segments.
    pair = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if pair >= n_pairs:
        return

    g = pair // n_seg
    seg = pair - g * n_seg

    # A zero coefficient is a complete rejected-pair marker. Geometry buffers
    # may remain untouched on rejection because the reducer reads coefficients
    # first and only loads E/phase data for a live field contribution.
    cs_re_out[pair] = F32_ZERO
    cs_im_out[pair] = F32_ZERO
    cp_re_out[pair] = F32_ZERO
    cp_im_out[pair] = F32_ZERO

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

    # In-medium resonance root. The Maxwell dispersion relation k = n(omega) omega
    # makes the vacuum root implicit,
    #     omega_res = v.g / (1 - Re n(omega_res) (v.n_hat)),
    # solved here by the same 3-pass fixed point the CPU core uses: the map
    # contracts at rate ~delta ~1e-5, so two passes already sit past float32
    # rounding. Only Re n enters -- Im n is the absorption the Beer-Lambert
    # factor below already applies. Validation: xray-in-medium-resonance
    n_re = F32_ONE
    if use_medium:
        vdn = v_dot_n[seg]
        dnm = F32_ONE - vdn
        for _it in range(3):
            E_it = hbarc * (v_dot_g / dnm)
            below_it = E_it <= E_tab[U32_ZERO]
            above_it = E_it >= E_tab[n_tab - U32_ONE]
            idx_it = _bracket_index(E_tab, E_it, n_tab)
            x0_it = E_tab[idx_it - U32_ONE]
            frac_it = (E_it - x0_it) / (E_tab[idx_it] - x0_it)
            n_re = _interp_shared(n_re_tab, idx_it, frac_it, below_it, above_it, n_tab)
            dnm = F32_ONE - n_re * vdn

    omega = v_dot_g / dnm
    E_res = hbarc * omega
    if E_res <= lo_keep or E_res <= F32_TEN or E_res >= hi_keep:
        return

    below = E_res <= E_tab[U32_ZERO]
    above = E_res >= E_tab[n_tab - U32_ONE]
    idx = _bracket_index(E_tab, E_res, n_tab)
    x0 = E_tab[idx - U32_ONE]
    frac = (E_res - x0) / (E_tab[idx] - x0)
    bounded_energy = min(max(E_res, x0), E_tab[idx])
    log_frac = xp.log1p((bounded_energy - x0) / x0) / xp.log1p((E_tab[idx] - x0) / x0)

    chi_re = _interp_row(chi_re_tab, g, idx, frac, below, above, n_tab)
    chi_im = _interp_row(chi_im_tab, g, idx, frac, below, above, n_tab)
    u_re = _interp_row(u_re_tab, g, idx, frac, below, above, n_tab)
    u_im = _interp_row(u_im_tab, g, idx, frac, below, above, n_tab)
    mu = _interp_elemental_mu(log_mu_tab, idx, log_frac, below, above, n_mu, n_tab)

    # k.v = omega (1 - denom) survives the substitution exactly (denom absorbed
    # the index), k.g takes one power of Re n through k_mag = |k|, and the PXR
    # numerator's k^2 takes two. |k+g|^2 - k^2 = g^2 + 2 k.g keeps its form.
    k_mag = omega
    if use_medium:
        k_mag = omega * n_re
    k_dot_v = omega * (F32_ONE - dnm)
    k_dot_g = k_mag * n_dot_g[g]
    v_dot_kg = v_dot_g + k_dot_v
    detuning = g2[g] + F32_TWO * k_dot_g

    duration = t_L[seg]
    gm = gamma[seg]
    transmission = xp.exp(-(L_esc[seg] * mu))
    amp = xp.sqrt(alpha_fs * omega / pref_c1 * transmission)
    if amp != amp or amp <= F32_ZERO or amp >= F32_MAX or duration <= F32_ZERO:
        return
    coef_scale = amp * duration

    esx = es_flat[gbase]
    esy = es_flat[gbase + U32_ONE]
    esz = es_flat[gbase + U32_TWO]
    epx = ep_flat[gbase]
    epy = ep_flat[gbase + U32_ONE]
    epz = ep_flat[gbase + U32_TWO]

    # sigma polarization: A = chi*f_pxr + (U/m)*f_cbs, with both scalar
    # kinematic factors REAL. This is the real/imag expansion of the existing
    # coherent complex expression, preserving its phase information.
    g_dot_e = g_dot_es[g]
    v_dot_e = vx * esx + vy * esy + vz * esz
    numerator = v_dot_kg * g_dot_e - k_mag * k_mag * v_dot_e
    braced_ge = g_dot_e - v_dot_g * v_dot_e
    braced_kg = k_dot_g - k_dot_v * v_dot_g
    bracket = braced_ge + v_dot_e * braced_kg / v_dot_g
    cbs_den = gm * v_dot_g
    # Preserve the coherent eager expression order as closely as scalar real/imag
    # arithmetic allows: (chi/detuning)*numerator and (-u/cbs_den)*bracket.
    a_re = (chi_re / detuning) * numerator + (-u_re / cbs_den) * bracket
    a_im = (chi_im / detuning) * numerator + (-u_im / cbs_den) * bracket
    csr = coef_scale * a_re
    csi = coef_scale * a_im

    # pi polarization.
    g_dot_e = g_dot_ep[g]
    v_dot_e = vx * epx + vy * epy + vz * epz
    numerator = v_dot_kg * g_dot_e - k_mag * k_mag * v_dot_e
    braced_ge = g_dot_e - v_dot_g * v_dot_e
    braced_kg = k_dot_g - k_dot_v * v_dot_g
    bracket = braced_ge + v_dot_e * braced_kg / v_dot_g
    a_re = (chi_re / detuning) * numerator + (-u_re / cbs_den) * bracket
    a_im = (chi_im / detuning) * numerator + (-u_im / cbs_den) * bracket
    cpr = coef_scale * a_re
    cpi = coef_scale * a_im

    E_r_out[pair] = E_res
    # The sinc half-width rides on the resonance denominator, so under the
    # refractive model it stops being a per-segment hoist and is emitted here in
    # the pair layout the field reducer already understands.
    if use_medium:
        aw_out[pair] = dnm * duration / (F32_TWO * hbarc)
    rbase = seg * U32_THREE
    g_phase_out[pair] = (
        r_flat[rbase] * gx + r_flat[rbase + U32_ONE] * gy + r_flat[rbase + U32_TWO] * gz
    )
    cs_re_out[pair] = csr
    cs_im_out[pair] = csi
    cp_re_out[pair] = cpr
    cp_im_out[pair] = cpi


@jit.rawkernel(device=True)
def _sinc_unscaled(x):
    if x == F32_ZERO:
        x = F32_TINY
    return xp.sin(x) / x


@jit.rawkernel()
def _field_kernel_1e(
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
    fs_re,
    fs_im,
    fp_re,
    fp_im,
    aw_pair,
    slope_pair,
    use_medium,
    n_seg,
    n_g,
    n_E,
    e_blocks,
):
    block = jit.blockIdx.x
    g = block // e_blocks
    eb = block - g * e_blocks
    if g >= n_g:
        return
    k0 = eb
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x
    E0 = E_grid[k0]
    dw0 = F32_ZERO
    if use_medium:
        dw0 = delta_omega[k0]
    sr0 = F32_ZERO
    si0 = F32_ZERO
    pr0 = F32_ZERO
    pi0 = F32_ZERO

    seg = tid
    base = g * n_seg
    while seg < n_seg:
        line = base + seg
        csr = cs_re[line]
        csi = cs_im[line]
        cpr = cp_re[line]
        cpi = cp_im[line]
        if csr != F32_ZERO or csi != F32_ZERO or cpr != F32_ZERO or cpi != F32_ZERO:
            Er = E_r[line]
            aw_i = seg
            if aw_pair:
                aw_i = line
            slope_i = seg
            if slope_pair:
                slope_i = line
            aa = aw[aw_i]
            ps = phase_slope[slope_i]
            gp = g_phase[line]
            Lj = F32_ZERO
            if use_medium:
                Lj = L_esc[seg]
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
        seg += nthreads

    shared = jit.shared_memory(xp.float32, None)
    o0 = U32_ZERO
    o1 = U32_ONE * nthreads
    o2 = U32_TWO * nthreads
    o3 = U32_THREE * nthreads
    shared[o0 + tid] = sr0
    shared[o1 + tid] = si0
    shared[o2 + tid] = pr0
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
        out = g * n_E + k0
        fs_re[out] += shared[o0]
        fs_im[out] += shared[o1]
        fp_re[out] += shared[o2]
        fp_im[out] += shared[o3]


@jit.rawkernel()
def _field_kernel_2e(
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
    fs_re,
    fs_im,
    fp_re,
    fp_im,
    aw_pair,
    slope_pair,
    use_medium,
    n_seg,
    n_g,
    n_E,
    e_blocks,
):
    block = jit.blockIdx.x
    g = block // e_blocks
    eb = block - g * e_blocks
    if g >= n_g:
        return
    k0 = eb * U32_TWO
    k1 = k0 + U32_ONE
    has1 = k1 < n_E
    tid = jit.threadIdx.x
    nthreads = jit.blockDim.x
    E0 = E_grid[k0]
    dw0 = F32_ZERO
    if use_medium:
        dw0 = delta_omega[k0]
    E1 = F32_ZERO
    dw1 = F32_ZERO
    if has1:
        E1 = E_grid[k1]
        if use_medium:
            dw1 = delta_omega[k1]

    sr0 = F32_ZERO
    si0 = F32_ZERO
    pr0 = F32_ZERO
    pi0 = F32_ZERO
    sr1 = F32_ZERO
    si1 = F32_ZERO
    pr1 = F32_ZERO
    pi1 = F32_ZERO

    seg = tid
    base = g * n_seg
    while seg < n_seg:
        line = base + seg
        csr = cs_re[line]
        csi = cs_im[line]
        cpr = cp_re[line]
        cpi = cp_im[line]
        if csr != F32_ZERO or csi != F32_ZERO or cpr != F32_ZERO or cpi != F32_ZERO:
            Er = E_r[line]
            aw_i = seg
            if aw_pair:
                aw_i = line
            slope_i = seg
            if slope_pair:
                slope_i = line
            aa = aw[aw_i]
            ps = phase_slope[slope_i]
            gp = g_phase[line]
            Lj = F32_ZERO
            if use_medium:
                Lj = L_esc[seg]

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
        seg += nthreads

    shared = jit.shared_memory(xp.float32, None)
    o0 = U32_ZERO
    o1 = U32_ONE * nthreads
    o2 = U32_TWO * nthreads
    o3 = U32_THREE * nthreads
    o4 = U32_FOUR * nthreads
    o5 = U32_FIVE * nthreads
    o6 = U32_SIX * nthreads
    o7 = U32_SEVEN * nthreads
    shared[o0 + tid] = sr0
    shared[o1 + tid] = si0
    shared[o2 + tid] = pr0
    shared[o3 + tid] = pi0
    shared[o4 + tid] = sr1
    shared[o5 + tid] = si1
    shared[o6 + tid] = pr1
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
        out = g * n_E + k0
        fs_re[out] += shared[o0]
        fs_im[out] += shared[o1]
        fp_re[out] += shared[o2]
        fp_im[out] += shared[o3]
        if has1:
            out = g * n_E + k1
            fs_re[out] += shared[o4]
            fs_im[out] += shared[o5]
            fp_re[out] += shared[o6]
            fp_im[out] += shared[o7]


_FIELD_KERNELS = {1: _field_kernel_1e, 2: _field_kernel_2e}


@jit.rawkernel()
def _finalize_fields_kernel(fs_re, fs_im, fp_re, fp_im, wm, spec, n_g, n_E):
    k = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if k >= n_E:
        return
    acc = F32_ZERO
    g = U32_ZERO
    while g < n_g:
        i = g * n_E + k
        sr = fs_re[i]
        si = fs_im[i]
        pr = fp_re[i]
        pi = fp_im[i]
        acc += wm[g] * (sr * sr + si * si + pr * pr + pi * pi)
        g += U32_ONE
    spec[k] += acc


_DUMMY_F32 = None


def _dummy():
    """One cached length-1 float32 array to bind the unused in-medium pointers.

    Allocated lazily so importing this module does not touch the device.
    """
    global _DUMMY_F32
    if _DUMMY_F32 is None:
        _DUMMY_F32 = xp.zeros(1, dtype=xp.float32)
    return _DUMMY_F32


def _validate_threads(nthreads, name):
    if nthreads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError(f"{name} must be one of 32, 64, 128, 256, 512, 1024")


def allocate_coherent_fields(n_g, n_E):
    """Return four flattened ``(n_g, n_E)`` float32 persistent field planes."""
    size = int(n_g) * int(n_E)
    storage = xp.zeros(4 * size, dtype=xp.float32)
    return tuple(storage[i * size : (i + 1) * size] for i in range(4))


def run_coherent_prologue_kernel(
    v_flat,
    denom,
    gamma,
    t_L,
    L_esc,
    line_electron,
    r_flat,
    phase_slope_seg,
    g_flat,
    es_flat,
    ep_flat,
    E_tab,
    chi_re_tab,
    chi_im_tab,
    u_re_tab,
    u_im_tab,
    log_mu_tab,
    *,
    lo_keep,
    hi_keep,
    hbarc,
    electron_mass_eV,
    alpha_fs,
    pref_c1,
    n_hat,
    n_g,
    g2=None,
    n_dot_g=None,
    g_dot_es=None,
    g_dot_ep=None,
    aw_seg=None,
    v_dot_n=None,
    n_re_tab=None,
    config=DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
):
    """Build g-major coherent line data for one contiguous segment block.

    The public/internal call shape is retained for compatibility. The returned
    tuple also retains its historical eight entries, but ``aw`` and
    ``phase_slope`` are now segment-sized rather than duplicated across g rows.
    The field reducer accepts both layouts, so callers that construct pair-sized
    geometry directly remain valid.

    ``g2``/``n_dot_g``/``g_dot_e*`` and ``aw_seg`` are optional precomputed
    hoists. When omitted they are constructed once here, preserving older direct
    callers while the main line path avoids rebuilding them per segment block.

    Under ``xray_dispersion="refractive"`` the caller supplies the per-segment
    ``v_dot_n = v.n_hat`` together with ``n_re_tab``, the real refractive index
    on the ``E_tab`` grid. The kernel then solves the implicit in-medium
    resonance per (segment, g) instead of reading the hoisted vacuum ``denom``,
    and returns ``aw`` in the pair layout because the sinc half-width follows
    that denominator. Both must be given together or both omitted; when omitted
    the kernel evaluates the vacuum kinematics unchanged.
    """
    nthreads = int(config.prologue_nthreads)
    _validate_threads(nthreads, "prologue_nthreads")
    n_g = int(n_g)
    n_tab = int(E_tab.size)
    n_mu = int(log_mu_tab.shape[0])
    n_seg = int(denom.size)
    if n_g <= 0 or n_tab < 2:
        raise ValueError("coherent prologue requires at least one g row and two tabulation points")
    n_pairs = n_g * n_seg

    if (v_dot_n is None) != (n_re_tab is None):
        raise ValueError("v_dot_n and n_re_tab must be given together")
    use_medium = n_re_tab is not None
    if use_medium:
        if int(v_dot_n.size) != n_seg:
            raise ValueError("v_dot_n must have one entry per segment")
        if int(n_re_tab.size) != n_tab:
            raise ValueError("n_re_tab must use the E_tab grid")
        if aw_seg is not None:
            raise ValueError("aw_seg cannot be hoisted per segment under the refractive model")

    G = g_flat.reshape(n_g, 3)
    ES = es_flat.reshape(n_g, 3)
    EP = ep_flat.reshape(n_g, 3)
    if g2 is None:
        g2 = G[:, 0] * G[:, 0] + G[:, 1] * G[:, 1] + G[:, 2] * G[:, 2]
    if n_dot_g is None:
        nx, ny, nz = np.float32(n_hat[0]), np.float32(n_hat[1]), np.float32(n_hat[2])
        n_dot_g = G[:, 0] * nx + G[:, 1] * ny + G[:, 2] * nz
    if g_dot_es is None:
        g_dot_es = G[:, 0] * ES[:, 0] + G[:, 1] * ES[:, 1] + G[:, 2] * ES[:, 2]
    if g_dot_ep is None:
        g_dot_ep = G[:, 0] * EP[:, 0] + G[:, 1] * EP[:, 1] + G[:, 2] * EP[:, 2]
    if aw_seg is None and not use_medium:
        aw_seg = denom * t_L / np.float32(2.0 * float(hbarc))

    # Backward-compatible direct callers may still supply raw U_g tables. The
    # optimized main path supplies pre-scaled U_g/m_e tables and sets mass=1.
    if float(electron_mass_eV) != 1.0:
        inv_m = np.float32(1.0 / float(electron_mass_eV))
        u_re_kernel = u_re_tab * inv_m
        u_im_kernel = u_im_tab * inv_m
    else:
        u_re_kernel = u_re_tab
        u_im_kernel = u_im_tab

    # Only g-dependent quantities live in pair scratch; under the refractive
    # model the half-width joins them.
    n_slots = 7 if use_medium else 6
    storage = xp.empty(n_slots * n_pairs, dtype=xp.float32)
    E_r = storage[0 * n_pairs : 1 * n_pairs]
    g_phase = storage[1 * n_pairs : 2 * n_pairs]
    cs_re = storage[2 * n_pairs : 3 * n_pairs]
    cs_im = storage[3 * n_pairs : 4 * n_pairs]
    cp_re = storage[4 * n_pairs : 5 * n_pairs]
    cp_im = storage[5 * n_pairs : 6 * n_pairs]
    if use_medium:
        aw_seg = storage[6 * n_pairs : 7 * n_pairs]
    if n_pairs == 0:
        return E_r, aw_seg, phase_slope_seg, g_phase, cs_re, cs_im, cp_re, cp_im

    if not use_medium:
        v_dot_n = n_re_tab = _dummy()

    nblocks = (n_pairs + nthreads - 1) // nthreads
    _coherent_prologue_kernel(
        (nblocks,),
        (nthreads,),
        (
            v_flat,
            denom,
            v_dot_n,
            gamma,
            t_L,
            L_esc,
            line_electron,
            r_flat,
            g_flat,
            es_flat,
            ep_flat,
            g2,
            n_dot_g,
            g_dot_es,
            g_dot_ep,
            E_tab,
            chi_re_tab,
            chi_im_tab,
            u_re_kernel,
            u_im_kernel,
            log_mu_tab,
            n_re_tab,
            E_r,
            aw_seg,
            g_phase,
            cs_re,
            cs_im,
            cp_re,
            cp_im,
            np.float32(lo_keep),
            np.float32(hi_keep),
            np.float32(hbarc),
            np.float32(alpha_fs),
            np.float32(pref_c1),
            np.uint32(1 if use_medium else 0),
            np.uint32(n_pairs),
            np.uint32(n_seg),
            np.uint32(n_g),
            np.uint32(n_mu),
            np.uint32(n_tab),
        ),
    )
    return E_r, aw_seg, phase_slope_seg, g_phase, cs_re, cs_im, cp_re, cp_im


def run_coherent_field_accumulation_kernel(
    E_r,
    aw,
    phase_slope,
    g_phase,
    cs_re,
    cs_im,
    cp_re,
    cp_im,
    E_grid,
    *,
    fields,
    n_g,
    n_seg,
    L_esc=None,
    delta_omega=None,
    config=DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
):
    """Add one segment block's complex fields into persistent g-by-energy planes.

    ``aw`` and ``phase_slope`` may each be either the optimized segment-sized
    array (length ``n_seg``) or the pair-sized layout (length ``n_g*n_seg``), and
    they are selected independently: the refractive prologue emits a pair-sized
    ``aw`` (its half-width follows the per-(segment, g) in-medium denominator)
    while ``phase_slope`` stays segment-sized. This also keeps direct kernel
    tests/callers, which pass both in one layout, source-compatible.

    Under ``xray_dispersion="refractive"`` the caller supplies the block's
    per-segment escape distance ``L_esc`` (Angstrom, length ``n_seg``; it is
    g-independent, so it never takes the pair layout) and the per-energy table
    ``delta_omega[k] = (1 - Re n(E_k)) * omega(E_k)``. Together they add
    ``- L_esc[j] * delta_omega[k]`` to the propagation phase. Both must be given
    together or both omitted; when omitted the kernel evaluates the vacuum
    phase expression unchanged.
    """
    nthreads = int(config.reduction_nthreads)
    _validate_threads(nthreads, "reduction_nthreads")
    epb = int(config.energies_per_block)
    try:
        kernel = _FIELD_KERNELS[epb]
    except KeyError:
        raise ValueError(f"energies_per_block must be one of {tuple(_FIELD_KERNELS)}") from None

    n_g = int(n_g)
    n_seg = int(n_seg)
    n_E = int(E_grid.size)
    if n_g == 0 or n_seg == 0 or n_E == 0:
        return fields
    aw_pair = int(aw.size) != n_seg
    if aw_pair and int(aw.size) != n_g * n_seg:
        raise ValueError("aw must have length n_seg or n_g*n_seg")
    slope_pair = int(phase_slope.size) != n_seg
    if slope_pair and int(phase_slope.size) != n_g * n_seg:
        raise ValueError("phase_slope must have length n_seg or n_g*n_seg")
    if (L_esc is None) != (delta_omega is None):
        raise ValueError("L_esc and delta_omega must be given together")
    use_medium = L_esc is not None
    if use_medium:
        if int(L_esc.size) != n_seg:
            raise ValueError("L_esc must have one entry per segment")
        if int(delta_omega.size) != n_E:
            raise ValueError("delta_omega must have one entry per energy bin")
    else:
        L_esc = delta_omega = _dummy()

    e_blocks = (n_E + epb - 1) // epb
    nblocks = n_g * e_blocks
    shared_bytes = 4 * epb * nthreads * np.dtype(np.float32).itemsize
    fs_re, fs_im, fp_re, fp_im = fields
    kernel(
        (nblocks,),
        (nthreads,),
        (
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
            fs_re,
            fs_im,
            fp_re,
            fp_im,
            np.uint32(1 if aw_pair else 0),
            np.uint32(1 if slope_pair else 0),
            np.uint32(1 if use_medium else 0),
            np.uint32(n_seg),
            np.uint32(n_g),
            np.uint32(n_E),
            np.uint32(e_blocks),
        ),
        shared_mem=shared_bytes,
    )
    return fields


def finalize_coherent_fields(
    fields,
    mosaic_weight,
    *,
    out,
    n_g,
    config=DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
):
    """Square complete per-g fields and add their incoherent mosaic sum to ``out``."""
    nthreads = int(config.finalize_nthreads)
    _validate_threads(nthreads, "finalize_nthreads")
    n_g = int(n_g)
    n_E = int(out.size)
    if n_g == 0 or n_E == 0:
        return out
    fs_re, fs_im, fp_re, fp_im = fields
    nblocks = (n_E + nthreads - 1) // nthreads
    _finalize_fields_kernel(
        (nblocks,),
        (nthreads,),
        (
            fs_re,
            fs_im,
            fp_re,
            fp_im,
            mosaic_weight,
            out,
            np.uint32(n_g),
            np.uint32(n_E),
        ),
    )
    return out
