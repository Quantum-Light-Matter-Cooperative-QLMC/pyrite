"""CUDA JIT reduction for coherent CXR line fields.

For each photon-energy bin the kernel sums the complex field over emitting
segments for both orthogonal polarizations, then accumulates
``(|F_s|^2 + |F_p|^2) * mosaic_weight`` directly into the caller's spectrum.
This replaces construction of the dense complex ``SP[segment, energy]`` matrix
and the two complex GEMV reductions used by the CuPy fallback.

Under absorption each line's finite-time factor is the complex formation factor
of ``spectrum.cxr._formation`` (the NumPy reference) instead of the real sinc.
Validation: coherent-formation-absorption
"""

from dataclasses import dataclass

import cupy as xp
import numpy as np

from .._cupy_jit import jit

F32_ZERO = np.float32(0.0)
F32_TINY = np.float32(1.0e-20)
F32_ONE = np.float32(1.0)
F32_TWO = np.float32(2.0)
F32_HALF = np.float32(0.5)
F32_THIRD = np.float32(1.0 / 3.0)
F32_SIXTH = np.float32(1.0 / 6.0)
# ``_formation.FORMATION_SERIES_W2``: |w|^2 below which F takes its series.
F32_FORMATION_SERIES_W2 = np.float32(1.0e-6)
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
def _sinc_unscaled(sinc_argument):
    if sinc_argument == F32_ZERO:
        sinc_argument = F32_TINY
    return xp.sin(sinc_argument) / sinc_argument


@jit.rawkernel(device=True)
def _sinc_windowed(sinc_argument, sinc_cutoff, use_sinc_cutoff):
    if use_sinc_cutoff and (sinc_argument < -sinc_cutoff or sinc_argument > sinc_cutoff):
        return F32_ZERO
    return _sinc_unscaled(sinc_argument)


@jit.rawkernel(device=True)
def _formation_re(phase_arg, sin_phase, cos_phase, end_sum, end_diff, absorb_q):
    # Re F of ``_formation.formation_factor``; ``sin_phase``/``cos_phase`` are sin/cos(phase_arg).
    formation_norm_squared = absorb_q * absorb_q + phase_arg * phase_arg
    if formation_norm_squared < F32_FORMATION_SERIES_W2:
        return (
            F32_HALF
            * end_sum
            * (F32_ONE - absorb_q * absorb_q * F32_THIRD - phase_arg * phase_arg * F32_SIXTH)
        )
    return (phase_arg * end_sum * sin_phase - absorb_q * end_diff * cos_phase) / (
        F32_TWO * formation_norm_squared
    )


@jit.rawkernel(device=True)
def _formation_im(phase_arg, sin_phase, cos_phase, end_sum, end_diff, absorb_q):
    # Im F of ``_formation.formation_factor``; ``sin_phase``/``cos_phase`` are sin/cos(phase_arg).
    formation_norm_squared = absorb_q * absorb_q + phase_arg * phase_arg
    if formation_norm_squared < F32_FORMATION_SERIES_W2:
        return -F32_HALF * end_sum * absorb_q * phase_arg * F32_THIRD
    return -(phase_arg * end_diff * cos_phase + absorb_q * end_sum * sin_phase) / (
        F32_TWO * formation_norm_squared
    )


@jit.rawkernel()
def _kernel_1e(
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
    spectrum,
    mosaic_weight,
    use_medium,
    use_formation,
    sinc_cutoff,
    use_sinc_cutoff,
    n_lines,
    n_energies,
):
    energy_offset = jit.blockIdx.x
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x
    energy_index0 = energy_offset
    photon0_eV = energies_eV[energy_index0]
    medium_shift0 = F32_ZERO
    if use_medium:
        medium_shift0 = delta_omega[energy_index0]
    sigma_re_sum0 = F32_ZERO
    sigma_im_sum0 = F32_ZERO
    pi_re_sum0 = F32_ZERO
    pi_im_sum0 = F32_ZERO

    line_id = thread_idx
    while line_id < n_lines:
        resonance_eV = resonances_eV[line_id]
        sinc_slope = sinc_slopes[line_id]
        phase_rate = phase_slope[line_id]
        lattice_phase = g_phase[line_id]
        escape_distance = F32_ZERO
        if use_medium:
            escape_distance = escape_distances[line_id]
        half_path_change = F32_ZERO
        end_sum = F32_ZERO
        end_diff = F32_ZERO
        absorb_q = F32_ZERO
        if use_formation:
            half_path_change = half_escape_changes[line_id]
            end_sum = endpoint_sums[line_id]
            end_diff = endpoint_diffs[line_id]
            absorb_q = attenuation_q_values[line_id]
        sigma_re = sigma_coeffs_re[line_id]
        sigma_im = sigma_coeffs_im[line_id]
        pi_re = pi_coeffs_re[line_id]
        pi_im = pi_coeffs_im[line_id]
        if use_formation:
            phase_arg = sinc_slope * (photon0_eV - resonance_eV) - half_path_change * medium_shift0
            form_real = F32_ZERO
            form_imag = F32_ZERO
            if use_sinc_cutoff == U32_ZERO or (
                phase_arg >= -sinc_cutoff and phase_arg <= sinc_cutoff
            ):
                sin_arg = xp.sin(phase_arg)
                cos_arg = xp.cos(phase_arg)
                form_real = _formation_re(phase_arg, sin_arg, cos_arg, end_sum, end_diff, absorb_q)
                form_imag = _formation_im(phase_arg, sin_arg, cos_arg, end_sum, end_diff, absorb_q)
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
        line_id += n_threads

    shared = jit.shared_memory(xp.float32, None)
    offset0 = U32_ZERO
    shared[offset0 + thread_idx] = sigma_re_sum0
    offset1 = U32_ONE * n_threads
    shared[offset1 + thread_idx] = sigma_im_sum0
    offset2 = U32_TWO * n_threads
    shared[offset2 + thread_idx] = pi_re_sum0
    offset3 = U32_THREE * n_threads
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
        sigma_real = shared[offset0]
        sigma_imag = shared[offset1]
        pi_real = shared[offset2]
        pi_imag = shared[offset3]
        spectrum[energy_index0] += mosaic_weight * (
            sigma_real * sigma_real
            + sigma_imag * sigma_imag
            + pi_real * pi_real
            + pi_imag * pi_imag
        )


@jit.rawkernel()
def _kernel_2e(
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
    spectrum,
    mosaic_weight,
    use_medium,
    use_formation,
    sinc_cutoff,
    use_sinc_cutoff,
    n_lines,
    n_energies,
):
    energy_offset = jit.blockIdx.x * U32_TWO
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x
    energy_index0 = energy_offset
    photon0_eV = energies_eV[energy_index0]
    medium_shift0 = F32_ZERO
    if use_medium:
        medium_shift0 = delta_omega[energy_index0]
    sigma_re_sum0 = F32_ZERO
    sigma_im_sum0 = F32_ZERO
    pi_re_sum0 = F32_ZERO
    pi_im_sum0 = F32_ZERO
    energy_index1 = energy_offset + U32_ONE
    has_energy_1 = energy_index1 < n_energies
    photon1_eV = F32_ZERO
    medium_shift1 = F32_ZERO
    if has_energy_1:
        photon1_eV = energies_eV[energy_index1]
        if use_medium:
            medium_shift1 = delta_omega[energy_index1]
    sigma_re_sum1 = F32_ZERO
    sigma_im_sum1 = F32_ZERO
    pi_re_sum1 = F32_ZERO
    pi_im_sum1 = F32_ZERO

    line_id = thread_idx
    while line_id < n_lines:
        resonance_eV = resonances_eV[line_id]
        sinc_slope = sinc_slopes[line_id]
        phase_rate = phase_slope[line_id]
        lattice_phase = g_phase[line_id]
        escape_distance = F32_ZERO
        if use_medium:
            escape_distance = escape_distances[line_id]
        half_path_change = F32_ZERO
        end_sum = F32_ZERO
        end_diff = F32_ZERO
        absorb_q = F32_ZERO
        if use_formation:
            half_path_change = half_escape_changes[line_id]
            end_sum = endpoint_sums[line_id]
            end_diff = endpoint_diffs[line_id]
            absorb_q = attenuation_q_values[line_id]
        sigma_re = sigma_coeffs_re[line_id]
        sigma_im = sigma_coeffs_im[line_id]
        pi_re = pi_coeffs_re[line_id]
        pi_im = pi_coeffs_im[line_id]
        if use_formation:
            phase_arg = sinc_slope * (photon0_eV - resonance_eV) - half_path_change * medium_shift0
            form_real = F32_ZERO
            form_imag = F32_ZERO
            if use_sinc_cutoff == U32_ZERO or (
                phase_arg >= -sinc_cutoff and phase_arg <= sinc_cutoff
            ):
                sin_arg = xp.sin(phase_arg)
                cos_arg = xp.cos(phase_arg)
                form_real = _formation_re(phase_arg, sin_arg, cos_arg, end_sum, end_diff, absorb_q)
                form_imag = _formation_im(phase_arg, sin_arg, cos_arg, end_sum, end_diff, absorb_q)
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
                phase = phase_rate * photon1_eV - lattice_phase - escape_distance * medium_shift1
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
        line_id += n_threads

    shared = jit.shared_memory(xp.float32, None)
    offset0 = U32_ZERO
    shared[offset0 + thread_idx] = sigma_re_sum0
    offset1 = U32_ONE * n_threads
    shared[offset1 + thread_idx] = sigma_im_sum0
    offset2 = U32_TWO * n_threads
    shared[offset2 + thread_idx] = pi_re_sum0
    offset3 = U32_THREE * n_threads
    shared[offset3 + thread_idx] = pi_im_sum0
    offset4 = U32_FOUR * n_threads
    shared[offset4 + thread_idx] = sigma_re_sum1
    offset5 = U32_FIVE * n_threads
    shared[offset5 + thread_idx] = sigma_im_sum1
    offset6 = U32_SIX * n_threads
    shared[offset6 + thread_idx] = pi_re_sum1
    offset7 = U32_SEVEN * n_threads
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
        sigma_real = shared[offset0]
        sigma_imag = shared[offset1]
        pi_real = shared[offset2]
        pi_imag = shared[offset3]
        spectrum[energy_index0] += mosaic_weight * (
            sigma_real * sigma_real
            + sigma_imag * sigma_imag
            + pi_real * pi_real
            + pi_imag * pi_imag
        )
        if has_energy_1:
            sigma_real = shared[offset4]
            sigma_imag = shared[offset5]
            pi_real = shared[offset6]
            pi_imag = shared[offset7]
            spectrum[energy_index1] += mosaic_weight * (
                sigma_real * sigma_real
                + sigma_imag * sigma_imag
                + pi_real * pi_real
                + pi_imag * pi_imag
            )


@jit.rawkernel()
def _kernel_3e(
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
    spectrum,
    mosaic_weight,
    use_medium,
    use_formation,
    sinc_cutoff,
    use_sinc_cutoff,
    n_lines,
    n_energies,
):
    energy_offset = jit.blockIdx.x * U32_THREE
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x
    energy_index0 = energy_offset
    photon0_eV = energies_eV[energy_index0]
    medium_shift0 = F32_ZERO
    if use_medium:
        medium_shift0 = delta_omega[energy_index0]
    sigma_re_sum0 = F32_ZERO
    sigma_im_sum0 = F32_ZERO
    pi_re_sum0 = F32_ZERO
    pi_im_sum0 = F32_ZERO
    energy_index1 = energy_offset + U32_ONE
    has_energy_1 = energy_index1 < n_energies
    photon1_eV = F32_ZERO
    medium_shift1 = F32_ZERO
    if has_energy_1:
        photon1_eV = energies_eV[energy_index1]
        if use_medium:
            medium_shift1 = delta_omega[energy_index1]
    sigma_re_sum1 = F32_ZERO
    sigma_im_sum1 = F32_ZERO
    pi_re_sum1 = F32_ZERO
    pi_im_sum1 = F32_ZERO
    energy_index2 = energy_offset + U32_TWO
    has_energy_2 = energy_index2 < n_energies
    photon2_eV = F32_ZERO
    medium_shift2 = F32_ZERO
    if has_energy_2:
        photon2_eV = energies_eV[energy_index2]
        if use_medium:
            medium_shift2 = delta_omega[energy_index2]
    sigma_re_sum2 = F32_ZERO
    sigma_im_sum2 = F32_ZERO
    pi_re_sum2 = F32_ZERO
    pi_im_sum2 = F32_ZERO

    line_id = thread_idx
    while line_id < n_lines:
        resonance_eV = resonances_eV[line_id]
        sinc_slope = sinc_slopes[line_id]
        phase_rate = phase_slope[line_id]
        lattice_phase = g_phase[line_id]
        escape_distance = F32_ZERO
        if use_medium:
            escape_distance = escape_distances[line_id]
        half_path_change = F32_ZERO
        end_sum = F32_ZERO
        end_diff = F32_ZERO
        absorb_q = F32_ZERO
        if use_formation:
            half_path_change = half_escape_changes[line_id]
            end_sum = endpoint_sums[line_id]
            end_diff = endpoint_diffs[line_id]
            absorb_q = attenuation_q_values[line_id]
        sigma_re = sigma_coeffs_re[line_id]
        sigma_im = sigma_coeffs_im[line_id]
        pi_re = pi_coeffs_re[line_id]
        pi_im = pi_coeffs_im[line_id]
        if use_formation:
            phase_arg = sinc_slope * (photon0_eV - resonance_eV) - half_path_change * medium_shift0
            form_real = F32_ZERO
            form_imag = F32_ZERO
            if use_sinc_cutoff == U32_ZERO or (
                phase_arg >= -sinc_cutoff and phase_arg <= sinc_cutoff
            ):
                sin_arg = xp.sin(phase_arg)
                cos_arg = xp.cos(phase_arg)
                form_real = _formation_re(phase_arg, sin_arg, cos_arg, end_sum, end_diff, absorb_q)
                form_imag = _formation_im(phase_arg, sin_arg, cos_arg, end_sum, end_diff, absorb_q)
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
                phase = phase_rate * photon1_eV - lattice_phase - escape_distance * medium_shift1
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
        if has_energy_2:
            if use_formation:
                phase_arg = (
                    sinc_slope * (photon2_eV - resonance_eV) - half_path_change * medium_shift2
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
                phase = phase_rate * photon2_eV - lattice_phase - escape_distance * medium_shift2
                cos_phase = xp.cos(phase)
                sin_phase = xp.sin(phase)
                phased_field_re = form_real * cos_phase - form_imag * sin_phase
                phased_field_im = form_real * sin_phase + form_imag * cos_phase
                sigma_re_sum2 += sigma_re * phased_field_re - sigma_im * phased_field_im
                sigma_im_sum2 += sigma_re * phased_field_im + sigma_im * phased_field_re
                pi_re_sum2 += pi_re * phased_field_re - pi_im * phased_field_im
                pi_im_sum2 += pi_re * phased_field_im + pi_im * phased_field_re
            else:
                sinc_argument = sinc_slope * (photon2_eV - resonance_eV)
                sinc_value = _sinc_windowed(sinc_argument, sinc_cutoff, use_sinc_cutoff)
                phase = phase_rate * photon2_eV - lattice_phase
                if use_medium:
                    phase = phase - escape_distance * medium_shift2
                cos_phase = xp.cos(phase)
                sin_phase = xp.sin(phase)
                sigma_re_sum2 += sinc_value * (sigma_re * cos_phase - sigma_im * sin_phase)
                sigma_im_sum2 += sinc_value * (sigma_re * sin_phase + sigma_im * cos_phase)
                pi_re_sum2 += sinc_value * (pi_re * cos_phase - pi_im * sin_phase)
                pi_im_sum2 += sinc_value * (pi_re * sin_phase + pi_im * cos_phase)
        line_id += n_threads

    shared = jit.shared_memory(xp.float32, None)
    offset0 = U32_ZERO
    shared[offset0 + thread_idx] = sigma_re_sum0
    offset1 = U32_ONE * n_threads
    shared[offset1 + thread_idx] = sigma_im_sum0
    offset2 = U32_TWO * n_threads
    shared[offset2 + thread_idx] = pi_re_sum0
    offset3 = U32_THREE * n_threads
    shared[offset3 + thread_idx] = pi_im_sum0
    offset4 = U32_FOUR * n_threads
    shared[offset4 + thread_idx] = sigma_re_sum1
    offset5 = U32_FIVE * n_threads
    shared[offset5 + thread_idx] = sigma_im_sum1
    offset6 = U32_SIX * n_threads
    shared[offset6 + thread_idx] = pi_re_sum1
    offset7 = U32_SEVEN * n_threads
    shared[offset7 + thread_idx] = pi_im_sum1
    offset8 = U32_EIGHT * n_threads
    shared[offset8 + thread_idx] = sigma_re_sum2
    offset9 = U32_NINE * n_threads
    shared[offset9 + thread_idx] = sigma_im_sum2
    offset10 = U32_TEN * n_threads
    shared[offset10 + thread_idx] = pi_re_sum2
    offset11 = U32_ELEVEN * n_threads
    shared[offset11 + thread_idx] = pi_im_sum2
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
            shared[offset8 + thread_idx] += shared[offset8 + thread_idx + stride]
            shared[offset9 + thread_idx] += shared[offset9 + thread_idx + stride]
            shared[offset10 + thread_idx] += shared[offset10 + thread_idx + stride]
            shared[offset11 + thread_idx] += shared[offset11 + thread_idx + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if thread_idx == U32_ZERO:
        sigma_real = shared[offset0]
        sigma_imag = shared[offset1]
        pi_real = shared[offset2]
        pi_imag = shared[offset3]
        spectrum[energy_index0] += mosaic_weight * (
            sigma_real * sigma_real
            + sigma_imag * sigma_imag
            + pi_real * pi_real
            + pi_imag * pi_imag
        )
        if has_energy_1:
            sigma_real = shared[offset4]
            sigma_imag = shared[offset5]
            pi_real = shared[offset6]
            pi_imag = shared[offset7]
            spectrum[energy_index1] += mosaic_weight * (
                sigma_real * sigma_real
                + sigma_imag * sigma_imag
                + pi_real * pi_real
                + pi_imag * pi_imag
            )
        if has_energy_2:
            sigma_real = shared[offset8]
            sigma_imag = shared[offset9]
            pi_real = shared[offset10]
            pi_imag = shared[offset11]
            spectrum[energy_index2] += mosaic_weight * (
                sigma_real * sigma_real
                + sigma_imag * sigma_imag
                + pi_real * pi_real
                + pi_imag * pi_imag
            )


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
    half_dL=None,
    apb=None,
    bma=None,
    q=None,
    sinc_cutoff=None,
    config=DEFAULT_COHERENT_KERNEL_CONFIG,
):
    """Accumulate one reflection/orientation's coherent intensity into ``out``.

    Inputs are contiguous float32 CuPy arrays. ``phase_slope`` is
    ``(t_abs - n_hat.r) / HBARC_EV_ANG`` so the per-cell phase is
    ``phase_slope[j] * E_grid[k] - g_phase[j]``.

    Under the in-medium dispersion the caller also supplies the
    per-line escape distance ``L_esc`` (Angstrom) and the per-energy table
    ``delta_omega[k] = (1 - Re n(E_k)) * omega(E_k)``, which add the in-medium
    term ``- L_esc[j] * delta_omega[k]`` to that phase. This is a second
    (per-line scalar) x (per-energy table) product, so it cannot be folded into
    ``phase_slope``. Both must be given together or both omitted; when omitted
    the kernel evaluates the vacuum phase expression unchanged (the in-medium
    term sits behind a launch-uniform branch, so vacuum stays bit-for-bit).
    When ``sinc_cutoff`` is given, line contributions whose unscaled sinc
    argument exceeds it in magnitude are omitted.

    Formation mode (issue #181) takes the per-line ``half_dL`` (half the escape
    distance change along the piece) and the ``(apb, bma, q)`` of
    ``_formation.formation_coefficients``, all four together and only with the
    in-medium pair. ``E_r``/``aw`` are then the VACUUM sinc centre and width,
    and each line contributes ``F(v) exp(i phase)`` with
    ``v = aw (E - E_r) - half_dL delta_omega``; the sinc cutoff bounds ``|v|``.
    The attenuation lives in ``F``, so ``c_s``/``c_p`` carry none. Omitted, the
    kernel evaluates the legacy real-sinc expression unchanged.
    Validation: coherent-formation-absorption

    Validation: cross-reflection-coherence
    """
    n_threads = int(config.nthreads)
    energies_per_block = int(config.energies_per_block)
    if n_threads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    try:
        kernel = _REDUCTION_KERNELS[energies_per_block]
    except KeyError:
        raise ValueError(f"energies_per_block must be one of {tuple(_REDUCTION_KERNELS)}") from None

    if (L_esc is None) != (delta_omega is None):
        raise ValueError("L_esc and delta_omega must be given together")

    n_energies = int(E_grid.size)
    if n_energies == 0 or E_r.size == 0:
        return out
    use_medium = L_esc is not None
    if use_medium:
        if int(L_esc.size) != int(E_r.size):
            raise ValueError("L_esc must have one entry per line")
        if int(delta_omega.size) != n_energies:
            raise ValueError("delta_omega must have one entry per energy bin")
    else:
        L_esc = delta_omega = _dummy()
    formation = (half_dL, apb, bma, q)
    use_formation = half_dL is not None
    if any((f is None) != (not use_formation) for f in formation):
        raise ValueError("half_dL, apb, bma and q must be given together")
    if use_formation:
        if not use_medium:
            raise ValueError("the formation factor needs L_esc and delta_omega")
        if any(int(f.size) != int(E_r.size) for f in formation):
            raise ValueError("half_dL, apb, bma and q must have one entry per line")
    else:
        half_dL = apb = bma = q = _dummy()
    use_sinc_cutoff = sinc_cutoff is not None
    cutoff = np.float32(0.0 if sinc_cutoff is None else sinc_cutoff)
    n_blocks = (n_energies + energies_per_block - 1) // energies_per_block
    shared_memory_bytes = 4 * energies_per_block * n_threads * np.dtype(np.float32).itemsize
    kernel(
        (n_blocks,),
        (n_threads,),
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
            half_dL,
            apb,
            bma,
            q,
            out,
            np.float32(mosaic_weight),
            np.uint32(1 if use_medium else 0),
            np.uint32(1 if use_formation else 0),
            cutoff,
            np.uint32(1 if use_sinc_cutoff else 0),
            np.uint32(E_r.size),
            np.uint32(n_energies),
        ),
        shared_mem=shared_memory_bytes,
    )
    return out
