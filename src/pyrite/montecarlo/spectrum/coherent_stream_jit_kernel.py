"""Streaming CUDA JIT pipeline for coherent CXR line emission.

The coherent spectrum cannot reduce independent segment blocks as intensities:
``|sum_j F_j|^2`` requires the complex field from every segment to be added
before squaring.  This module therefore splits the GPU work into three stages:

1. ``coherent_prologue_jit_kernel.run_coherent_prologue_kernel`` evaluates the energy-independent line data
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

Peak pair scratch is O(8 * segment_block * N_g), plus three formation rows.
Exact recoil makes ``aw`` and ``phase_slope`` per (segment, g)
(``line-recoil-in-medium-root``). No device->host line-count transfer is
needed between the prologue and reduction stages.
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
