"""CUDA JIT segmented reduction: per-electron coherent intensities.

The decoherence blend's ``sum_e |sum_{j in e} field_j|^2`` term of the
streaming coherent route (``coherent_stream_jit_kernel``), split out of that
module for size. It evaluates the same per-line field -- legacy sinc or the
formation factor -- as the flat field accumulator, so both blend terms share
their support. Validation: coherent-inter-electron-decoherence,
coherent-formation-absorption
"""

import cupy as xp
import numpy as np

from .._cupy_jit import jit
from .coherent_jit_kernel import _formation_im, _formation_re
from .coherent_stream_jit_kernel import (
    _FIELD_KERNELS,
    DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
    F32_ZERO,
    U32_ONE,
    U32_TWO,
    U32_ZERO,
    _dummy,
    _formation_args,
    _sinc_windowed,
    _validate_threads,
)


@jit.rawkernel()
def _grouped_intensity_kernel(
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
    half_dL,
    apb,
    bma,
    q,
    group_starts,
    grouped,
    aw_pair,
    slope_pair,
    use_medium,
    use_formation,
    sinc_cutoff,
    use_sinc_cutoff,
    n_seg,
    n_groups,
    n_g,
    n_E,
    e_blocks,
    energies_per_block,
):
    """Accumulate ``sum_e |sum_{j in e} field_j|^2`` without per-e launches."""
    block = jit.blockIdx.x
    g = block // e_blocks
    eb = block - g * e_blocks
    if g >= n_g:
        return
    k0 = eb * energies_per_block
    k1 = k0 + U32_ONE
    has1 = energies_per_block == U32_TWO and k1 < n_E
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

    mag0 = F32_ZERO
    mag1 = F32_ZERO
    group = tid
    base = g * n_seg
    while group < n_groups:
        sr0 = F32_ZERO
        si0 = F32_ZERO
        pr0 = F32_ZERO
        pi0 = F32_ZERO
        sr1 = F32_ZERO
        si1 = F32_ZERO
        pr1 = F32_ZERO
        pi1 = F32_ZERO
        seg = group_starts[group]
        stop = group_starts[group + U32_ONE]
        while seg < stop:
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
                hd = F32_ZERO
                ab = F32_ZERO
                bm = F32_ZERO
                qq = F32_ZERO
                if use_formation:
                    hd = half_dL[seg]
                    ab = apb[line]
                    bm = bma[line]
                    qq = q[line]

                if use_formation:
                    v = aa * (E0 - Er) - hd * dw0
                    fr = F32_ZERO
                    fi = F32_ZERO
                    if use_sinc_cutoff == U32_ZERO or (v >= -sinc_cutoff and v <= sinc_cutoff):
                        sv = xp.sin(v)
                        cv = xp.cos(v)
                        fr = _formation_re(v, sv, cv, ab, bm, qq)
                        fi = _formation_im(v, sv, cv, ab, bm, qq)
                    phase = ps * E0 - gp - Lj * dw0
                    cph = xp.cos(phase)
                    sph = xp.sin(phase)
                    er = fr * cph - fi * sph
                    ei = fr * sph + fi * cph
                    sr0 += csr * er - csi * ei
                    si0 += csr * ei + csi * er
                    pr0 += cpr * er - cpi * ei
                    pi0 += cpr * ei + cpi * er
                else:
                    x = aa * (E0 - Er)
                    s = _sinc_windowed(x, sinc_cutoff, use_sinc_cutoff)
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
                    if use_formation:
                        v = aa * (E1 - Er) - hd * dw1
                        fr = F32_ZERO
                        fi = F32_ZERO
                        if use_sinc_cutoff == U32_ZERO or (v >= -sinc_cutoff and v <= sinc_cutoff):
                            sv = xp.sin(v)
                            cv = xp.cos(v)
                            fr = _formation_re(v, sv, cv, ab, bm, qq)
                            fi = _formation_im(v, sv, cv, ab, bm, qq)
                        phase = ps * E1 - gp - Lj * dw1
                        cph = xp.cos(phase)
                        sph = xp.sin(phase)
                        er = fr * cph - fi * sph
                        ei = fr * sph + fi * cph
                        sr1 += csr * er - csi * ei
                        si1 += csr * ei + csi * er
                        pr1 += cpr * er - cpi * ei
                        pi1 += cpr * ei + cpi * er
                    else:
                        x = aa * (E1 - Er)
                        s = _sinc_windowed(x, sinc_cutoff, use_sinc_cutoff)
                        phase = ps * E1 - gp
                        if use_medium:
                            phase = phase - Lj * dw1
                        cph = xp.cos(phase)
                        sph = xp.sin(phase)
                        sr1 += s * (csr * cph - csi * sph)
                        si1 += s * (csr * sph + csi * cph)
                        pr1 += s * (cpr * cph - cpi * sph)
                        pi1 += s * (cpr * sph + cpi * cph)
            seg += U32_ONE
        mag0 += sr0 * sr0 + si0 * si0 + pr0 * pr0 + pi0 * pi0
        if has1:
            mag1 += sr1 * sr1 + si1 * si1 + pr1 * pr1 + pi1 * pi1
        group += nthreads

    shared = jit.shared_memory(xp.float32, None)
    o0 = U32_ZERO
    o1 = nthreads
    shared[tid] = mag0
    shared[o1 + tid] = mag1
    jit.syncthreads()
    stride = nthreads // U32_TWO
    while stride > U32_ZERO:
        if tid < stride:
            shared[tid] += shared[tid + stride]
            shared[o1 + tid] += shared[o1 + tid + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if tid == U32_ZERO:
        out = g * n_E + k0
        grouped[out] += shared[o0]
        if has1:
            grouped[out + U32_ONE] += shared[o1]


def run_coherent_grouped_intensity_kernel(
    E_r,
    aw,
    phase_slope,
    g_phase,
    cs_re,
    cs_im,
    cp_re,
    cp_im,
    E_grid,
    group_starts,
    *,
    out,
    n_g,
    n_seg,
    L_esc=None,
    delta_omega=None,
    half_dL=None,
    apb=None,
    bma=None,
    q=None,
    sinc_cutoff=None,
    config=DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
):
    """Add grouped field intensities for one compact, whole-electron block.

    ``sinc_cutoff`` uses the same unscaled-argument window as the flat field
    accumulator so both terms of the decoherence blend have identical support,
    and the same formation mode (``half_dL``, ``apb``, ``bma``, ``q``).

    Validation: coherent-formation-absorption
    """
    nthreads = int(config.reduction_nthreads)
    _validate_threads(nthreads, "reduction_nthreads")
    epb = int(config.energies_per_block)
    if epb not in _FIELD_KERNELS:
        raise ValueError(f"energies_per_block must be one of {tuple(_FIELD_KERNELS)}")
    n_g = int(n_g)
    n_seg = int(n_seg)
    n_E = int(E_grid.size)
    n_groups = int(group_starts.size) - 1
    if n_groups < 0:
        raise ValueError("group_starts must contain at least one offset")
    if n_g == 0 or n_seg == 0 or n_E == 0 or n_groups == 0:
        return out
    aw_pair = int(aw.size) != n_seg
    if aw_pair and int(aw.size) != n_g * n_seg:
        raise ValueError("aw must have length n_seg or n_g*n_seg")
    slope_pair = int(phase_slope.size) != n_seg
    if slope_pair and int(phase_slope.size) != n_g * n_seg:
        raise ValueError("phase_slope must have length n_seg or n_g*n_seg")
    if int(out.size) != n_g * n_E:
        raise ValueError("out must have shape (n_g, n_E)")
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
    use_formation, half_dL, apb, bma, q = _formation_args(
        half_dL, apb, bma, q, use_medium=use_medium, n_seg=n_seg, n_pairs=n_g * n_seg
    )
    use_sinc_cutoff = sinc_cutoff is not None
    cutoff = np.float32(0.0 if sinc_cutoff is None else sinc_cutoff)
    e_blocks = (n_E + epb - 1) // epb
    _grouped_intensity_kernel(
        (n_g * e_blocks,),
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
            half_dL,
            apb,
            bma,
            q,
            group_starts,
            out.reshape(-1),
            np.uint32(1 if aw_pair else 0),
            np.uint32(1 if slope_pair else 0),
            np.uint32(1 if use_medium else 0),
            np.uint32(1 if use_formation else 0),
            cutoff,
            np.uint32(1 if use_sinc_cutoff else 0),
            np.uint32(n_seg),
            np.uint32(n_groups),
            np.uint32(n_g),
            np.uint32(n_E),
            np.uint32(e_blocks),
            np.uint32(epb),
        ),
        shared_mem=2 * nthreads * np.dtype(np.float32).itemsize,
    )
    return out
