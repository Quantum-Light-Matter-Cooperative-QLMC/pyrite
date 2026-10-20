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
    resonances_eV,
    sinc_slopes,
    phase_slope,
    g_phase,
    sigma_coeffs_re,
    sigma_coeffs_im,
    pi_coeffs_re,
    pi_coeffs_im,
    energies_eV,
    escape_distances,
    delta_omega,
    half_escape_changes,
    endpoint_sums,
    endpoint_diffs,
    attenuation_q_values,
    group_starts,
    grouped,
    sinc_slopes_per_pair,
    phase_slopes_per_pair,
    use_medium,
    use_formation,
    sinc_cutoff,
    use_sinc_cutoff,
    n_segments,
    n_groups,
    n_reflections,
    n_energies,
    n_energy_blocks,
    energies_per_block,
):
    """Accumulate ``sum_e |sum_{j in e} field_j|^2`` without per-e launches."""
    block = jit.blockIdx.x
    reflection = block // n_energy_blocks
    energy_block = block - reflection * n_energy_blocks
    if reflection >= n_reflections:
        return
    energy_index0 = energy_block * energies_per_block
    energy_index1 = energy_index0 + U32_ONE
    has_energy_1 = energies_per_block == U32_TWO and energy_index1 < n_energies
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x
    photon0_eV = energies_eV[energy_index0]
    medium_shift0 = F32_ZERO
    if use_medium:
        medium_shift0 = delta_omega[energy_index0]
    photon1_eV = F32_ZERO
    medium_shift1 = F32_ZERO
    if has_energy_1:
        photon1_eV = energies_eV[energy_index1]
        if use_medium:
            medium_shift1 = delta_omega[energy_index1]

    intensity_sum_0 = F32_ZERO
    intensity_sum_1 = F32_ZERO
    group_index = thread_idx
    line_offset = reflection * n_segments
    while group_index < n_groups:
        sigma_re_sum0 = F32_ZERO
        sigma_im_sum0 = F32_ZERO
        pi_re_sum0 = F32_ZERO
        pi_im_sum0 = F32_ZERO
        sigma_re_sum1 = F32_ZERO
        sigma_im_sum1 = F32_ZERO
        pi_re_sum1 = F32_ZERO
        pi_im_sum1 = F32_ZERO
        segment = group_starts[group_index]
        group_stop = group_starts[group_index + U32_ONE]
        while segment < group_stop:
            line_id = line_offset + segment
            sigma_re = sigma_coeffs_re[line_id]
            sigma_im = sigma_coeffs_im[line_id]
            pi_re = pi_coeffs_re[line_id]
            pi_im = pi_coeffs_im[line_id]
            if (
                sigma_re != F32_ZERO
                or sigma_im != F32_ZERO
                or pi_re != F32_ZERO
                or pi_im != F32_ZERO
            ):
                resonance_eV = resonances_eV[line_id]
                sinc_slope_index = segment
                if sinc_slopes_per_pair:
                    sinc_slope_index = line_id
                phase_slope_index = segment
                if phase_slopes_per_pair:
                    phase_slope_index = line_id
                sinc_slope = sinc_slopes[sinc_slope_index]
                phase_rate = phase_slope[phase_slope_index]
                lattice_phase = g_phase[line_id]
                escape_distance = F32_ZERO
                if use_medium:
                    escape_distance = escape_distances[segment]
                half_path_change = F32_ZERO
                end_sum = F32_ZERO
                end_diff = F32_ZERO
                absorb_q = F32_ZERO
                if use_formation:
                    half_path_change = half_escape_changes[segment]
                    end_sum = endpoint_sums[line_id]
                    end_diff = endpoint_diffs[line_id]
                    absorb_q = attenuation_q_values[line_id]

                if use_formation:
                    phase_arg = (
                        sinc_slope * (photon0_eV - resonance_eV) - half_path_change * medium_shift0
                    )
                    form_real = F32_ZERO
                    form_imag = F32_ZERO
                    if use_sinc_cutoff == U32_ZERO or (
                        phase_arg >= -sinc_cutoff and phase_arg <= sinc_cutoff
                    ):
                        sin_arg = xp.sin(phase_arg)
                        cos_arg = xp.cos(phase_arg)
                        form_real = _formation_re(
                            phase_arg, sin_arg, cos_arg, end_sum, end_diff, absorb_q
                        )
                        form_imag = _formation_im(
                            phase_arg, sin_arg, cos_arg, end_sum, end_diff, absorb_q
                        )
                    phase = (
                        phase_rate * photon0_eV - lattice_phase - escape_distance * medium_shift0
                    )
                    cos_phase = xp.cos(phase)
                    sin_phase = xp.sin(phase)
                    phased_field_re = form_real * cos_phase - form_imag * sin_phase
                    phased_field_im = form_real * sin_phase + form_imag * cos_phase
                    sigma_re_sum0 += sigma_re * phased_field_re - sigma_im * phased_field_im
                    sigma_im_sum0 += sigma_re * phased_field_im + sigma_im * phased_field_re
                    pi_re_sum0 += pi_re * phased_field_re - pi_im * phased_field_im
                    pi_im_sum0 += pi_re * phased_field_im + pi_im * phased_field_re
                else:
                    sinc_argument = sinc_slope * (photon0_eV - resonance_eV)
                    sinc_value = _sinc_windowed(sinc_argument, sinc_cutoff, use_sinc_cutoff)
                    phase = phase_rate * photon0_eV - lattice_phase
                    if use_medium:
                        phase = phase - escape_distance * medium_shift0
                    cos_phase = xp.cos(phase)
                    sin_phase = xp.sin(phase)
                    sigma_re_sum0 += sinc_value * (sigma_re * cos_phase - sigma_im * sin_phase)
                    sigma_im_sum0 += sinc_value * (sigma_re * sin_phase + sigma_im * cos_phase)
                    pi_re_sum0 += sinc_value * (pi_re * cos_phase - pi_im * sin_phase)
                    pi_im_sum0 += sinc_value * (pi_re * sin_phase + pi_im * cos_phase)

                if has_energy_1:
                    if use_formation:
                        phase_arg = (
                            sinc_slope * (photon1_eV - resonance_eV)
                            - half_path_change * medium_shift1
                        )
                        form_real = F32_ZERO
                        form_imag = F32_ZERO
                        if use_sinc_cutoff == U32_ZERO or (
                            phase_arg >= -sinc_cutoff and phase_arg <= sinc_cutoff
                        ):
                            sin_arg = xp.sin(phase_arg)
                            cos_arg = xp.cos(phase_arg)
                            form_real = _formation_re(
                                phase_arg, sin_arg, cos_arg, end_sum, end_diff, absorb_q
                            )
                            form_imag = _formation_im(
                                phase_arg, sin_arg, cos_arg, end_sum, end_diff, absorb_q
                            )
                        phase = (
                            phase_rate * photon1_eV
                            - lattice_phase
                            - escape_distance * medium_shift1
                        )
                        cos_phase = xp.cos(phase)
                        sin_phase = xp.sin(phase)
                        phased_field_re = form_real * cos_phase - form_imag * sin_phase
                        phased_field_im = form_real * sin_phase + form_imag * cos_phase
                        sigma_re_sum1 += sigma_re * phased_field_re - sigma_im * phased_field_im
                        sigma_im_sum1 += sigma_re * phased_field_im + sigma_im * phased_field_re
                        pi_re_sum1 += pi_re * phased_field_re - pi_im * phased_field_im
                        pi_im_sum1 += pi_re * phased_field_im + pi_im * phased_field_re
                    else:
                        sinc_argument = sinc_slope * (photon1_eV - resonance_eV)
                        sinc_value = _sinc_windowed(sinc_argument, sinc_cutoff, use_sinc_cutoff)
                        phase = phase_rate * photon1_eV - lattice_phase
                        if use_medium:
                            phase = phase - escape_distance * medium_shift1
                        cos_phase = xp.cos(phase)
                        sin_phase = xp.sin(phase)
                        sigma_re_sum1 += sinc_value * (sigma_re * cos_phase - sigma_im * sin_phase)
                        sigma_im_sum1 += sinc_value * (sigma_re * sin_phase + sigma_im * cos_phase)
                        pi_re_sum1 += sinc_value * (pi_re * cos_phase - pi_im * sin_phase)
                        pi_im_sum1 += sinc_value * (pi_re * sin_phase + pi_im * cos_phase)
            segment += U32_ONE
        intensity_sum_0 += (
            sigma_re_sum0 * sigma_re_sum0
            + sigma_im_sum0 * sigma_im_sum0
            + pi_re_sum0 * pi_re_sum0
            + pi_im_sum0 * pi_im_sum0
        )
        if has_energy_1:
            intensity_sum_1 += (
                sigma_re_sum1 * sigma_re_sum1
                + sigma_im_sum1 * sigma_im_sum1
                + pi_re_sum1 * pi_re_sum1
                + pi_im_sum1 * pi_im_sum1
            )
        group_index += n_threads

    shared = jit.shared_memory(xp.float32, None)
    offset0 = U32_ZERO
    offset1 = n_threads
    shared[thread_idx] = intensity_sum_0
    shared[offset1 + thread_idx] = intensity_sum_1
    jit.syncthreads()
    stride = n_threads // U32_TWO
    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[thread_idx] += shared[thread_idx + stride]
            shared[offset1 + thread_idx] += shared[offset1 + thread_idx + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if thread_idx == U32_ZERO:
        field_offset = reflection * n_energies + energy_index0
        grouped[field_offset] += shared[offset0]
        if has_energy_1:
            grouped[field_offset + U32_ONE] += shared[offset1]


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
    n_threads = int(config.reduction_nthreads)
    _validate_threads(n_threads, "reduction_nthreads")
    energies_per_block = int(config.energies_per_block)
    if energies_per_block not in _FIELD_KERNELS:
        raise ValueError(f"energies_per_block must be one of {tuple(_FIELD_KERNELS)}")
    n_g = int(n_g)
    n_seg = int(n_seg)
    n_energies = int(E_grid.size)
    n_groups = int(group_starts.size) - 1
    if n_groups < 0:
        raise ValueError("group_starts must contain at least one offset")
    if n_g == 0 or n_seg == 0 or n_energies == 0 or n_groups == 0:
        return out
    sinc_slopes_per_pair = int(aw.size) != n_seg
    if sinc_slopes_per_pair and int(aw.size) != n_g * n_seg:
        raise ValueError("aw must have length n_seg or n_g*n_seg")
    phase_slopes_per_pair = int(phase_slope.size) != n_seg
    if phase_slopes_per_pair and int(phase_slope.size) != n_g * n_seg:
        raise ValueError("phase_slope must have length n_seg or n_g*n_seg")
    if int(out.size) != n_g * n_energies:
        raise ValueError("out must have shape (n_g, n_E)")
    if (L_esc is None) != (delta_omega is None):
        raise ValueError("L_esc and delta_omega must be given together")
    use_medium = L_esc is not None
    if use_medium:
        if int(L_esc.size) != n_seg:
            raise ValueError("L_esc must have one entry per segment")
        if int(delta_omega.size) != n_energies:
            raise ValueError("delta_omega must have one entry per energy bin")
    else:
        L_esc = delta_omega = _dummy()
    use_formation, half_dL, apb, bma, q = _formation_args(
        half_dL, apb, bma, q, use_medium=use_medium, n_seg=n_seg, n_pairs=n_g * n_seg
    )
    use_sinc_cutoff = sinc_cutoff is not None
    cutoff = np.float32(0.0 if sinc_cutoff is None else sinc_cutoff)
    n_energy_blocks = (n_energies + energies_per_block - 1) // energies_per_block
    _grouped_intensity_kernel(
        (n_g * n_energy_blocks,),
        (n_threads,),
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
            np.uint32(1 if sinc_slopes_per_pair else 0),
            np.uint32(1 if phase_slopes_per_pair else 0),
            np.uint32(1 if use_medium else 0),
            np.uint32(1 if use_formation else 0),
            cutoff,
            np.uint32(1 if use_sinc_cutoff else 0),
            np.uint32(n_seg),
            np.uint32(n_groups),
            np.uint32(n_g),
            np.uint32(n_energies),
            np.uint32(n_energy_blocks),
            np.uint32(energies_per_block),
        ),
        shared_mem=2 * n_threads * np.dtype(np.float32).itemsize,
    )
    return out
