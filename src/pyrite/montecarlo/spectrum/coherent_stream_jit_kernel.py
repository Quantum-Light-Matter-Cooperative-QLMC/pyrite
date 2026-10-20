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
from .coherent_jit_kernel import _formation_im, _formation_re

F32_ZERO = np.float32(0.0)
F32_ONE = np.float32(1.0)
F32_TWO = np.float32(2.0)
F32_TEN = np.float32(10.0)
F32_TINY = np.float32(1.0e-20)
F32_HALF = np.float32(0.5)
F32_QUARTER = np.float32(0.25)
# ``_formation.FORMATION_SMALL_Q``: |q| below which a+b, b-a come from cosh/sinh.
F32_FORMATION_SMALL_Q = np.float32(0.5)
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
def _interp_row(table, row, upper_idx, fraction, below, above, n_samples):
    row_offset = row * n_samples
    if below:
        return table[row_offset]
    if above:
        return table[row_offset + n_samples - U32_ONE]
    lower_value = table[row_offset + upper_idx - U32_ONE]
    return lower_value + fraction * (table[row_offset + upper_idx] - lower_value)


@jit.rawkernel(device=True)
def _interp_shared(table, upper_idx, fraction, below, above, n_samples):
    if below:
        return table[U32_ZERO]
    if above:
        return table[n_samples - U32_ONE]
    lower_value = table[upper_idx - U32_ONE]
    return lower_value + fraction * (table[upper_idx] - lower_value)


@jit.rawkernel(device=True)
def _interp_elemental_mu(
    log_mu_table, upper_idx, log_fraction, below, above, n_elements, n_samples
):
    """Sum per-element log-log attenuation interpolation on device.

    Validation: line-absorption-tabulation
    """
    mu = F32_ZERO
    element = U32_ZERO
    while element < n_elements:
        row_offset = element * n_samples
        if below:
            log_mu = log_mu_table[row_offset]
        elif above:
            log_mu = log_mu_table[row_offset + n_samples - U32_ONE]
        else:
            lower_log_mu = log_mu_table[row_offset + upper_idx - U32_ONE]
            log_mu = lower_log_mu + log_fraction * (
                log_mu_table[row_offset + upper_idx] - lower_log_mu
            )
        mu += xp.exp(log_mu)
        element += U32_ONE
    return mu


@jit.rawkernel(device=True)
def _bracket_index(grid, energy_eV, n_samples):
    """Equivalent to ``clip(searchsorted(grid, energy_eV), 1, n_samples - 1)``."""
    lower_idx = U32_ZERO
    upper_idx = n_samples
    while lower_idx < upper_idx:
        mid_idx = (lower_idx + upper_idx) // U32_TWO
        if grid[mid_idx] < energy_eV:
            lower_idx = mid_idx + U32_ONE
        else:
            upper_idx = mid_idx
    if lower_idx < U32_ONE:
        return U32_ONE
    if lower_idx >= n_samples:
        return n_samples - U32_ONE
    return lower_idx


@jit.rawkernel()
def _coherent_prologue_kernel(
    v_flat,
    denom,
    v_dot_n,
    grad_flat,
    gamma,
    t_L,
    L_esc,
    L_start,
    L_end,
    cxr_electron,
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
    table_row,
    n_re_tab,
    E_r_out,
    aw_out,
    g_phase_out,
    cs_re_out,
    cs_im_out,
    cp_re_out,
    cp_im_out,
    apb_out,
    bma_out,
    q_out,
    lo_keep,
    hi_keep,
    hbarc,
    root_rtol,
    alpha_fs,
    pref_c1,
    use_medium,
    use_formation,
    pair_count,
    n_segments,
    n_reflections,
    n_elements,
    n_samples,
):
    # g-major fixed order: pair_index = reflection*n_segments + segment.  The reduction kernel then
    # reads each g row contiguously while its threads stride over segments.
    """Validation: xray-in-medium-resonance"""
    pair_index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if pair_index >= pair_count:
        return

    reflection = pair_index // n_segments
    segment = pair_index - reflection * n_segments

    # A zero coefficient is a complete rejected-pair marker. Geometry buffers
    # may remain untouched on rejection because the reducer reads coefficients
    # first and only loads E/phase data for a live field contribution.
    cs_re_out[pair_index] = F32_ZERO
    cs_im_out[pair_index] = F32_ZERO
    cp_re_out[pair_index] = F32_ZERO
    cp_im_out[pair_index] = F32_ZERO

    if not cxr_electron[segment]:
        return

    velocity_offset = segment * U32_THREE
    reflection_offset = reflection * U32_THREE
    vx = v_flat[velocity_offset]
    vy = v_flat[velocity_offset + U32_ONE]
    vz = v_flat[velocity_offset + U32_TWO]
    gx = g_flat[reflection_offset]
    gy = g_flat[reflection_offset + U32_ONE]
    gz = g_flat[reflection_offset + U32_TWO]

    resonance_denom = denom[segment]
    vacuum_denom = resonance_denom
    v_dot_g = vx * gx + vy * gy + vz * gz

    # First-order Snell fixed point: D = 1-v.n-delta v.grad L.
    # Retain the CPU twin's three passes and float32 convergence guard;
    # reject critical-angle gradients before producing any amplitude.
    # Validation: xray-in-medium-resonance
    n_re = F32_ONE
    grad_x = grad_flat[velocity_offset]
    grad_y = grad_flat[velocity_offset + U32_ONE]
    grad_z = grad_flat[velocity_offset + U32_TWO]
    gradient_norm_sq = grad_x * grad_x + grad_y * grad_y + grad_z * grad_z
    velocity_dot_grad = vx * grad_x + vy * grad_y + vz * grad_z
    if use_medium:
        velocity_dot_n = v_dot_n[segment]
        resonance_denom = F32_ONE - velocity_dot_n
        previous_denominator = resonance_denom
        for _it in range(3):
            trial_energy_eV = hbarc * (v_dot_g / resonance_denom)
            trial_below = trial_energy_eV <= E_tab[U32_ZERO]
            trial_above = trial_energy_eV >= E_tab[n_samples - U32_ONE]
            trial_upper = _bracket_index(E_tab, trial_energy_eV, n_samples)
            trial_lower_eV = E_tab[trial_upper - U32_ONE]
            trial_fraction = (trial_energy_eV - trial_lower_eV) / (
                E_tab[trial_upper] - trial_lower_eV
            )
            n_re = _interp_shared(
                n_re_tab, trial_upper, trial_fraction, trial_below, trial_above, n_samples
            )
            previous_denominator = resonance_denom
            if velocity_dot_grad == -velocity_dot_n:
                resonance_denom = F32_ONE - n_re * velocity_dot_n
            else:
                resonance_denom = F32_ONE - velocity_dot_n - (F32_ONE - n_re) * velocity_dot_grad
        # `abs` is not a cupyx.jit builtin; `xp.abs` is. Written as `>` rather
        # than `not (<=)` so a NaN root falls through to the existing NaN
        # handling below instead of taking a new exit.
        if (
            xp.abs(resonance_denom - previous_denominator) > root_rtol * xp.abs(resonance_denom)
            or F32_TWO * xp.abs(F32_ONE - n_re) * gradient_norm_sq >= F32_ONE
        ):
            return

    omega = v_dot_g / resonance_denom
    resonance_eV = hbarc * omega
    if resonance_eV <= lo_keep or resonance_eV <= F32_TEN or resonance_eV >= hi_keep:
        return

    below = resonance_eV <= E_tab[U32_ZERO]
    above = resonance_eV >= E_tab[n_samples - U32_ONE]
    upper_idx = _bracket_index(E_tab, resonance_eV, n_samples)
    lower_eV = E_tab[upper_idx - U32_ONE]
    fraction = (resonance_eV - lower_eV) / (E_tab[upper_idx] - lower_eV)
    bounded_energy = min(max(resonance_eV, lower_eV), E_tab[upper_idx])
    log_fraction = xp.log1p((bounded_energy - lower_eV) / lower_eV) / xp.log1p(
        (E_tab[upper_idx] - lower_eV) / lower_eV
    )

    row = table_row[reflection]
    chi_re = _interp_row(chi_re_tab, row, upper_idx, fraction, below, above, n_samples)
    chi_im = _interp_row(chi_im_tab, row, upper_idx, fraction, below, above, n_samples)
    u_re = _interp_row(u_re_tab, row, upper_idx, fraction, below, above, n_samples)
    u_im = _interp_row(u_im_tab, row, upper_idx, fraction, below, above, n_samples)
    mu = _interp_elemental_mu(
        log_mu_tab, upper_idx, log_fraction, below, above, n_elements, n_samples
    )

    # Scalar products of k_eff = omega(n_hat + delta grad L).
    k_mag = omega
    if use_medium:
        delta = F32_ONE - n_re
        if gradient_norm_sq != F32_ZERO:
            k_mag = omega * xp.sqrt(F32_ONE - F32_TWO * delta + delta * delta * gradient_norm_sq)
    k_dot_v = omega * (F32_ONE - resonance_denom)
    k_dot_g = omega * (
        n_dot_g[reflection] + (F32_ONE - n_re) * (grad_x * gx + grad_y * gy + grad_z * gz)
    )
    v_dot_kg = v_dot_g + k_dot_v
    detuning = g2[reflection] + F32_TWO * k_dot_g

    duration = t_L[segment]
    lorentz_factor = gamma[segment]
    # Formation mode (issue #181): the attenuation moves out of the amplitude
    # into the per-piece formation constants of ``_formation.
    # formation_coefficients`` -- tau at the piece ends, ordered along travel --
    # which the field kernels turn into the complex factor F.
    # Validation: coherent-formation-absorption
    transmission = F32_ONE
    end_sum = F32_ZERO
    end_diff = F32_ZERO
    absorb_q = F32_ZERO
    if use_formation:
        start_optical_depth = L_start[segment] * mu
        end_optical_depth = L_end[segment] * mu
        absorb_q = F32_QUARTER * (end_optical_depth - start_optical_depth)
        if xp.abs(absorb_q) < F32_FORMATION_SMALL_Q:
            mid_attenuation = xp.exp(-(F32_QUARTER * (start_optical_depth + end_optical_depth)))
            end_sum = F32_TWO * mid_attenuation * xp.cosh(absorb_q)
            end_diff = -(F32_TWO * mid_attenuation * xp.sinh(absorb_q))
        else:
            start_amplitude = xp.exp(-(F32_HALF * start_optical_depth))
            end_amplitude = xp.exp(-(F32_HALF * end_optical_depth))
            end_sum = start_amplitude + end_amplitude
            end_diff = end_amplitude - start_amplitude
        # Written so a NaN falls through to rejection, like the root guard.
        if not (end_sum > F32_ZERO and end_sum < F32_MAX and end_diff == end_diff):
            return
    else:
        transmission = xp.exp(-(L_esc[segment] * mu))
    amplitude = xp.sqrt(alpha_fs * omega / pref_c1 * transmission)
    if (
        amplitude != amplitude
        or amplitude <= F32_ZERO
        or amplitude >= F32_MAX
        or duration <= F32_ZERO
    ):
        return
    field_scale = amplitude * duration

    esx = es_flat[reflection_offset]
    esy = es_flat[reflection_offset + U32_ONE]
    esz = es_flat[reflection_offset + U32_TWO]
    epx = ep_flat[reflection_offset]
    epy = ep_flat[reflection_offset + U32_ONE]
    epz = ep_flat[reflection_offset + U32_TWO]

    # sigma polarization: A = chi*f_pxr + (U/m)*f_cbs, with both scalar
    # kinematic factors REAL. This is the real/imag expansion of the existing
    # coherent complex expression, preserving its phase information.
    g_dot_e = g_dot_es[reflection]
    v_dot_e = vx * esx + vy * esy + vz * esz
    k_dot_e = omega * (F32_ONE - n_re) * (grad_x * esx + grad_y * esy + grad_z * esz)
    numerator = v_dot_kg * (g_dot_e + k_dot_e) - k_mag * k_mag * v_dot_e
    braced_ge = g_dot_e - v_dot_g * v_dot_e
    braced_kg = k_dot_g - k_dot_v * v_dot_g
    bracket = braced_ge + v_dot_e * braced_kg / v_dot_g
    cbs_denominator = lorentz_factor * v_dot_g
    # Preserve the coherent eager expression order as closely as scalar real/imag
    # arithmetic allows: (chi/detuning)*numerator and (-u/cbs_denominator)*bracket.
    amplitude_real = (chi_re / detuning) * numerator + (-u_re / cbs_denominator) * bracket
    amplitude_imag = (chi_im / detuning) * numerator + (-u_im / cbs_denominator) * bracket
    sigma_re = field_scale * amplitude_real
    sigma_im = field_scale * amplitude_imag

    # pi polarization.
    g_dot_e = g_dot_ep[reflection]
    v_dot_e = vx * epx + vy * epy + vz * epz
    k_dot_e = omega * (F32_ONE - n_re) * (grad_x * epx + grad_y * epy + grad_z * epz)
    numerator = v_dot_kg * (g_dot_e + k_dot_e) - k_mag * k_mag * v_dot_e
    braced_ge = g_dot_e - v_dot_g * v_dot_e
    braced_kg = k_dot_g - k_dot_v * v_dot_g
    bracket = braced_ge + v_dot_e * braced_kg / v_dot_g
    amplitude_real = (chi_re / detuning) * numerator + (-u_re / cbs_denominator) * bracket
    amplitude_imag = (chi_im / detuning) * numerator + (-u_im / cbs_denominator) * bracket
    pi_re = field_scale * amplitude_real
    pi_im = field_scale * amplitude_imag

    E_r_out[pair_index] = resonance_eV
    # The sinc half-width rides on the resonance denominator, so under the
    # refractive model it stops being a per-segment hoist and is emitted here in
    # the pair layout the field reducer already understands.
    if use_medium:
        aw_out[pair_index] = resonance_denom * duration / (F32_TWO * hbarc)
    # Formation mode centres the line on the VACUUM sinc centre and width; the
    # field kernels add the escape-path refractive slope (``_formation``).
    if use_formation:
        E_r_out[pair_index] = hbarc * (v_dot_g / vacuum_denom)
        aw_out[pair_index] = vacuum_denom * duration / (F32_TWO * hbarc)
        apb_out[pair_index] = end_sum
        bma_out[pair_index] = end_diff
        q_out[pair_index] = absorb_q
    position_offset = segment * U32_THREE
    g_phase_out[pair_index] = (
        r_flat[position_offset] * gx
        + r_flat[position_offset + U32_ONE] * gy
        + r_flat[position_offset + U32_TWO] * gz
    )
    cs_re_out[pair_index] = sigma_re
    cs_im_out[pair_index] = sigma_im
    cp_re_out[pair_index] = pi_re
    cp_im_out[pair_index] = pi_im


@jit.rawkernel(device=True)
def _sinc_unscaled(sinc_argument):
    if sinc_argument == F32_ZERO:
        sinc_argument = F32_TINY
    return xp.sin(sinc_argument) / sinc_argument


@jit.rawkernel(device=True)
def _sinc_windowed(sinc_argument, sinc_cutoff, use_sinc_cutoff):
    if use_sinc_cutoff and (sinc_argument < -sinc_cutoff or sinc_argument > sinc_cutoff):
        return F32_ZERO
    return _sinc_unscaled(sinc_argument)


@jit.rawkernel()
def _field_kernel_1e(
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
    fs_re,
    fs_im,
    fp_re,
    fp_im,
    sinc_slopes_per_pair,
    phase_slopes_per_pair,
    use_medium,
    use_formation,
    sinc_cutoff,
    use_sinc_cutoff,
    n_segments,
    n_reflections,
    n_energies,
    n_energy_blocks,
):
    block = jit.blockIdx.x
    reflection = block // n_energy_blocks
    energy_block = block - reflection * n_energy_blocks
    if reflection >= n_reflections:
        return
    energy_index0 = energy_block
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x
    photon0_eV = energies_eV[energy_index0]
    medium_shift0 = F32_ZERO
    if use_medium:
        medium_shift0 = delta_omega[energy_index0]
    sigma_re_sum0 = F32_ZERO
    sigma_im_sum0 = F32_ZERO
    pi_re_sum0 = F32_ZERO
    pi_im_sum0 = F32_ZERO

    segment = thread_idx
    line_offset = reflection * n_segments
    while segment < n_segments:
        line_id = line_offset + segment
        sigma_re = sigma_coeffs_re[line_id]
        sigma_im = sigma_coeffs_im[line_id]
        pi_re = pi_coeffs_re[line_id]
        pi_im = pi_coeffs_im[line_id]
        if sigma_re != F32_ZERO or sigma_im != F32_ZERO or pi_re != F32_ZERO or pi_im != F32_ZERO:
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
                phase = phase_rate * photon0_eV - lattice_phase - escape_distance * medium_shift0
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
        segment += n_threads

    shared = jit.shared_memory(xp.float32, None)
    offset0 = U32_ZERO
    offset1 = U32_ONE * n_threads
    offset2 = U32_TWO * n_threads
    offset3 = U32_THREE * n_threads
    shared[offset0 + thread_idx] = sigma_re_sum0
    shared[offset1 + thread_idx] = sigma_im_sum0
    shared[offset2 + thread_idx] = pi_re_sum0
    shared[offset3 + thread_idx] = pi_im_sum0
    jit.syncthreads()
    stride = n_threads // U32_TWO
    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[offset0 + thread_idx] += shared[offset0 + thread_idx + stride]
            shared[offset1 + thread_idx] += shared[offset1 + thread_idx + stride]
            shared[offset2 + thread_idx] += shared[offset2 + thread_idx + stride]
            shared[offset3 + thread_idx] += shared[offset3 + thread_idx + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if thread_idx == U32_ZERO:
        field_offset = reflection * n_energies + energy_index0
        fs_re[field_offset] += shared[offset0]
        fs_im[field_offset] += shared[offset1]
        fp_re[field_offset] += shared[offset2]
        fp_im[field_offset] += shared[offset3]


@jit.rawkernel()
def _field_kernel_2e(
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
    fs_re,
    fs_im,
    fp_re,
    fp_im,
    sinc_slopes_per_pair,
    phase_slopes_per_pair,
    use_medium,
    use_formation,
    sinc_cutoff,
    use_sinc_cutoff,
    n_segments,
    n_reflections,
    n_energies,
    n_energy_blocks,
):
    block = jit.blockIdx.x
    reflection = block // n_energy_blocks
    energy_block = block - reflection * n_energy_blocks
    if reflection >= n_reflections:
        return
    energy_index0 = energy_block * U32_TWO
    energy_index1 = energy_index0 + U32_ONE
    has_energy_1 = energy_index1 < n_energies
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

    sigma_re_sum0 = F32_ZERO
    sigma_im_sum0 = F32_ZERO
    pi_re_sum0 = F32_ZERO
    pi_im_sum0 = F32_ZERO
    sigma_re_sum1 = F32_ZERO
    sigma_im_sum1 = F32_ZERO
    pi_re_sum1 = F32_ZERO
    pi_im_sum1 = F32_ZERO

    segment = thread_idx
    line_offset = reflection * n_segments
    while segment < n_segments:
        line_id = line_offset + segment
        sigma_re = sigma_coeffs_re[line_id]
        sigma_im = sigma_coeffs_im[line_id]
        pi_re = pi_coeffs_re[line_id]
        pi_im = pi_coeffs_im[line_id]
        if sigma_re != F32_ZERO or sigma_im != F32_ZERO or pi_re != F32_ZERO or pi_im != F32_ZERO:
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
                phase = phase_rate * photon0_eV - lattice_phase - escape_distance * medium_shift0
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
                        sinc_slope * (photon1_eV - resonance_eV) - half_path_change * medium_shift1
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
                        phase_rate * photon1_eV - lattice_phase - escape_distance * medium_shift1
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
        segment += n_threads

    shared = jit.shared_memory(xp.float32, None)
    offset0 = U32_ZERO
    offset1 = U32_ONE * n_threads
    offset2 = U32_TWO * n_threads
    offset3 = U32_THREE * n_threads
    offset4 = U32_FOUR * n_threads
    offset5 = U32_FIVE * n_threads
    offset6 = U32_SIX * n_threads
    offset7 = U32_SEVEN * n_threads
    shared[offset0 + thread_idx] = sigma_re_sum0
    shared[offset1 + thread_idx] = sigma_im_sum0
    shared[offset2 + thread_idx] = pi_re_sum0
    shared[offset3 + thread_idx] = pi_im_sum0
    shared[offset4 + thread_idx] = sigma_re_sum1
    shared[offset5 + thread_idx] = sigma_im_sum1
    shared[offset6 + thread_idx] = pi_re_sum1
    shared[offset7 + thread_idx] = pi_im_sum1
    jit.syncthreads()
    stride = n_threads // U32_TWO
    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[offset0 + thread_idx] += shared[offset0 + thread_idx + stride]
            shared[offset1 + thread_idx] += shared[offset1 + thread_idx + stride]
            shared[offset2 + thread_idx] += shared[offset2 + thread_idx + stride]
            shared[offset3 + thread_idx] += shared[offset3 + thread_idx + stride]
            shared[offset4 + thread_idx] += shared[offset4 + thread_idx + stride]
            shared[offset5 + thread_idx] += shared[offset5 + thread_idx + stride]
            shared[offset6 + thread_idx] += shared[offset6 + thread_idx + stride]
            shared[offset7 + thread_idx] += shared[offset7 + thread_idx + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if thread_idx == U32_ZERO:
        field_offset = reflection * n_energies + energy_index0
        fs_re[field_offset] += shared[offset0]
        fs_im[field_offset] += shared[offset1]
        fp_re[field_offset] += shared[offset2]
        fp_im[field_offset] += shared[offset3]
        if has_energy_1:
            field_offset = reflection * n_energies + energy_index1
            fs_re[field_offset] += shared[offset4]
            fs_im[field_offset] += shared[offset5]
            fp_re[field_offset] += shared[offset6]
            fp_im[field_offset] += shared[offset7]


_FIELD_KERNELS = {1: _field_kernel_1e, 2: _field_kernel_2e}


@jit.rawkernel()
def _finalize_fields_kernel(
    fs_re, fs_im, fp_re, fp_im, mosaic_weight, spectrum, n_reflections, n_energies
):
    energy_index = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    if energy_index >= n_energies:
        return
    intensity_sum = F32_ZERO
    reflection = U32_ZERO
    while reflection < n_reflections:
        field_offset = reflection * n_energies + energy_index
        sigma_real = fs_re[field_offset]
        sigma_imag = fs_im[field_offset]
        pi_real = fp_re[field_offset]
        pi_imag = fp_im[field_offset]
        intensity_sum += mosaic_weight[reflection] * (
            sigma_real * sigma_real
            + sigma_imag * sigma_imag
            + pi_real * pi_real
            + pi_imag * pi_imag
        )
        reflection += U32_ONE
    spectrum[energy_index] += intensity_sum


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


def _formation_args(half_dL, apb, bma, q, *, use_medium, n_seg, n_pairs):
    """Validate the field kernels' formation arguments; bind dummies when off."""
    formation = (half_dL, apb, bma, q)
    use_formation = half_dL is not None
    if any((f is None) == use_formation for f in formation):
        raise ValueError("half_dL, apb, bma and q must be given together")
    if not use_formation:
        dummy = _dummy()
        return False, dummy, dummy, dummy, dummy
    if not use_medium:
        raise ValueError("the formation factor needs L_esc and delta_omega")
    if int(half_dL.size) != n_seg:
        raise ValueError("half_dL must have one entry per segment")
    if any(int(f.size) != n_pairs for f in (apb, bma, q)):
        raise ValueError("apb, bma and q must use the pair layout n_g*n_seg")
    return True, half_dL, apb, bma, q


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
    cxr_electron,
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
    root_rtol,
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
    escape_gradient=None,
    L_start=None,
    L_end=None,
    table_row=None,
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

    Under the in-medium dispersion the caller supplies the per-segment
    ``v_dot_n = v.n_hat`` together with ``n_re_tab``, the real refractive index
    on the ``E_tab`` grid. The kernel then solves the implicit in-medium
    resonance per (segment, g) instead of reading the hoisted vacuum ``denom``,
    and returns ``aw`` in the pair layout because the sinc half-width follows
    that denominator. Both must be given together or both omitted; when omitted
    the kernel evaluates the vacuum kinematics unchanged.

    ``root_rtol`` is the CPU twin's ``_in_medium_kinematics._RESONANCE_ROOT_RTOL``,
    passed in for the same reason ``hbarc`` is: this module cannot import the
    line core. It bounds how far the last fixed-point pass may move ``denom``
    before the pair is rejected as non-converged.

    Formation mode (issue #181) takes the per-segment escape distances
    ``L_start``/``L_end`` at the piece ends, ordered along travel, together and
    only under the refractive model. The coefficients then carry no
    transmission, ``resonances_eV``/``aw`` are the vacuum sinc centre and width, and the
    tuple gains three pair-layout entries ``(apb, bma, q)`` of
    ``_formation.formation_coefficients`` for the field kernels.
    Validation: coherent-formation-absorption

    ``table_row[g]`` names the coupling-table row g row reads, so mosaic
    orientations share their reflection's tables; omitted, row ``g`` reads
    table row ``g``.
    Validation: line-reflection-coupling-tables
    """
    n_threads = int(config.prologue_nthreads)
    _validate_threads(n_threads, "prologue_nthreads")
    n_g = int(n_g)
    n_tab = int(E_tab.size)
    if log_mu_tab.ndim != 2 or int(log_mu_tab.shape[1]) != n_tab:
        raise ValueError("log_mu_tab must have shape (elements, E_tab.size)")
    n_mu = int(log_mu_tab.shape[0])
    log_mu_kernel = log_mu_tab.reshape(-1)
    n_segments = int(denom.size)
    if n_g <= 0 or n_tab < 2:
        raise ValueError("coherent prologue requires at least one g row and two tabulation points")
    n_pairs = n_g * n_segments
    if table_row is None:
        table_row = xp.arange(n_g, dtype=xp.uint32)
    else:
        table_row = xp.ascontiguousarray(table_row, dtype=xp.uint32)
        if int(table_row.size) != n_g:
            raise ValueError("table_row must have one entry per g row")

    if (v_dot_n is None) != (n_re_tab is None):
        raise ValueError("v_dot_n and n_re_tab must be given together")
    use_medium = n_re_tab is not None
    if use_medium:
        if int(v_dot_n.size) != n_segments:
            raise ValueError("v_dot_n must have one entry per segment")
        if int(n_re_tab.size) != n_tab:
            raise ValueError("n_re_tab must use the E_tab grid")
        if aw_seg is not None:
            raise ValueError("aw_seg cannot be hoisted per segment under the refractive model")
    if (L_start is None) != (L_end is None):
        raise ValueError("L_start and L_end must be given together")
    use_formation = L_start is not None
    if use_formation:
        if not use_medium:
            raise ValueError("the formation factor needs the refractive model (v_dot_n, n_re_tab)")
        if int(L_start.size) != n_segments or int(L_end.size) != n_segments:
            raise ValueError("L_start and L_end must have one entry per segment")

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
    n_slots = (7 if use_medium else 6) + (3 if use_formation else 0)
    storage = xp.empty(n_slots * n_pairs, dtype=xp.float32)
    resonances_eV = storage[0 * n_pairs : 1 * n_pairs]
    g_phase = storage[1 * n_pairs : 2 * n_pairs]
    sigma_coeffs_re = storage[2 * n_pairs : 3 * n_pairs]
    sigma_coeffs_im = storage[3 * n_pairs : 4 * n_pairs]
    pi_coeffs_re = storage[4 * n_pairs : 5 * n_pairs]
    pi_coeffs_im = storage[5 * n_pairs : 6 * n_pairs]
    if use_medium:
        aw_seg = storage[6 * n_pairs : 7 * n_pairs]
    formation = ()
    if use_formation:
        formation = tuple(storage[i * n_pairs : (i + 1) * n_pairs] for i in (7, 8, 9))
    line_data = (
        resonances_eV,
        aw_seg,
        phase_slope_seg,
        g_phase,
        sigma_coeffs_re,
        sigma_coeffs_im,
        pi_coeffs_re,
        pi_coeffs_im,
        *formation,
    )
    if n_pairs == 0:
        return line_data

    if not use_medium:
        v_dot_n = n_re_tab = _dummy()
    if use_formation:
        apb_out, bma_out, q_out = formation
    else:
        L_start = L_end = apb_out = bma_out = q_out = _dummy()

    if escape_gradient is None:
        # Compatibility for direct callers without escape geometry.
        escape_gradient = xp.broadcast_to(-xp.asarray(n_hat, dtype=xp.float32), (n_segments, 3))
    grad_flat = xp.ascontiguousarray(escape_gradient, dtype=xp.float32).reshape(-1)
    if int(grad_flat.size) != 3 * n_segments:
        raise ValueError("escape_gradient must have shape (n_seg, 3)")
    n_blocks = (n_pairs + n_threads - 1) // n_threads
    _coherent_prologue_kernel(
        (n_blocks,),
        (n_threads,),
        (
            v_flat,
            denom,
            v_dot_n,
            grad_flat,
            gamma,
            t_L,
            L_esc,
            L_start,
            L_end,
            cxr_electron,
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
            log_mu_kernel,
            table_row,
            n_re_tab,
            resonances_eV,
            aw_seg,
            g_phase,
            sigma_coeffs_re,
            sigma_coeffs_im,
            pi_coeffs_re,
            pi_coeffs_im,
            apb_out,
            bma_out,
            q_out,
            np.float32(lo_keep),
            np.float32(hi_keep),
            np.float32(hbarc),
            np.float32(root_rtol),
            np.float32(alpha_fs),
            np.float32(pref_c1),
            np.uint32(1 if use_medium else 0),
            np.uint32(1 if use_formation else 0),
            np.uint32(n_pairs),
            np.uint32(n_segments),
            np.uint32(n_g),
            np.uint32(n_mu),
            np.uint32(n_tab),
        ),
    )
    return line_data


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
    half_dL=None,
    apb=None,
    bma=None,
    q=None,
    sinc_cutoff=None,
    config=DEFAULT_COHERENT_STREAM_KERNEL_CONFIG,
):
    """Add one segment block's complex fields into persistent g-by-energy planes.

    ``aw`` and ``phase_slope`` may each be either the optimized segment-sized
    array (length ``n_seg``) or the pair-sized layout (length ``n_g*n_seg``), and
    they are selected independently: the refractive prologue emits a pair-sized
    ``aw`` (its half-width follows the per-(segment, g) in-medium denominator)
    while ``phase_slope`` stays segment-sized. This also keeps direct kernel
    tests/callers, which pass both in one layout, source-compatible.

    Under the in-medium dispersion the caller supplies the block's
    per-segment escape distance ``L_esc`` (Angstrom, length ``n_seg``; it is
    g-independent, so it never takes the pair layout) and the per-energy table
    ``delta_omega[k] = (1 - Re n(E_k)) * omega(E_k)``. Together they add
    ``- L_esc[j] * delta_omega[k]`` to the propagation phase. Both must be given
    together or both omitted; when omitted the kernel evaluates the vacuum
    phase expression unchanged. When ``sinc_cutoff`` is given, each line is
    omitted outside that magnitude of the unscaled sinc argument.

    Formation mode (issue #181) takes the segment-layout ``half_dL`` and the
    pair-layout ``(apb, bma, q)`` from the prologue, all four together and only
    with the in-medium pair; each line then contributes the complex formation
    factor ``F(v)`` of ``_formation`` with ``v = aw (E - E_r) - half_dL
    delta_omega`` in place of the sinc, and the cutoff bounds ``|v|``.
    Validation: coherent-formation-absorption
    """
    n_threads = int(config.reduction_nthreads)
    _validate_threads(n_threads, "reduction_nthreads")
    energies_per_block = int(config.energies_per_block)
    try:
        kernel = _FIELD_KERNELS[energies_per_block]
    except KeyError:
        raise ValueError(f"energies_per_block must be one of {tuple(_FIELD_KERNELS)}") from None

    n_g = int(n_g)
    n_seg = int(n_seg)
    n_energies = int(E_grid.size)
    if n_g == 0 or n_seg == 0 or n_energies == 0:
        return fields
    sinc_slopes_per_pair = int(aw.size) != n_seg
    if sinc_slopes_per_pair and int(aw.size) != n_g * n_seg:
        raise ValueError("aw must have length n_seg or n_g*n_seg")
    phase_slopes_per_pair = int(phase_slope.size) != n_seg
    if phase_slopes_per_pair and int(phase_slope.size) != n_g * n_seg:
        raise ValueError("phase_slope must have length n_seg or n_g*n_seg")
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
    n_blocks = n_g * n_energy_blocks
    shared_memory_bytes = 4 * energies_per_block * n_threads * np.dtype(np.float32).itemsize
    fs_re, fs_im, fp_re, fp_im = fields
    kernel(
        (n_blocks,),
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
            fs_re,
            fs_im,
            fp_re,
            fp_im,
            np.uint32(1 if sinc_slopes_per_pair else 0),
            np.uint32(1 if phase_slopes_per_pair else 0),
            np.uint32(1 if use_medium else 0),
            np.uint32(1 if use_formation else 0),
            cutoff,
            np.uint32(1 if use_sinc_cutoff else 0),
            np.uint32(n_seg),
            np.uint32(n_g),
            np.uint32(n_energies),
            np.uint32(n_energy_blocks),
        ),
        shared_mem=shared_memory_bytes,
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
    n_threads = int(config.finalize_nthreads)
    _validate_threads(n_threads, "finalize_nthreads")
    n_g = int(n_g)
    n_energies = int(out.size)
    if n_g == 0 or n_energies == 0:
        return out
    fs_re, fs_im, fp_re, fp_im = fields
    n_blocks = (n_energies + n_threads - 1) // n_threads
    _finalize_fields_kernel(
        (n_blocks,),
        (n_threads,),
        (
            fs_re,
            fs_im,
            fp_re,
            fp_im,
            mosaic_weight,
            out,
            np.uint32(n_g),
            np.uint32(n_energies),
        ),
    )
    return out
