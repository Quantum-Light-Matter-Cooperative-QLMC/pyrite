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

from ..._bremslib_table import STAGED_RATIO_COUNT
from .._cupy_jit import jit

F32_ZERO = np.float32(0.0)
F32_ONE = np.float32(1.0)
F32_1E_M3 = np.float32(1.0e-3)
F32_1E_M6 = np.float32(1.0e-6)
F32_1E_M8 = np.float32(1.0e-8)
F32_1E_M30 = np.float32(1.0e-30)
F32_1E3 = np.float32(1.0e3)
F32_TWO = np.float32(2.0)
F32_ONE_SIXTH = np.float32(1.0 / 6.0)
F32_16_OVER_3 = np.float32(16.0 / 3.0)
F32_MC2_KEV = np.float32(510.99895)
F32_MC2_EV = np.float32(510998.95)
F32_ALPHA = np.float32(7.2973525693e-3)
F32_R_E_CM2 = np.float32(7.9407877e-26)
F32_TWO_PI_ALPHA = np.float32(2.0 * np.pi * 7.2973525693e-3)

U32_ZERO = np.uint32(0)
U32_ONE = np.uint32(1)
U32_TWO = np.uint32(2)
U32_THREE = np.uint32(3)
U32_FOUR = np.uint32(4)
U32_LAST_INNER_NODE = np.uint32(STAGED_RATIO_COUNT - 2)
U32_RATIO_COUNT = np.uint32(STAGED_RATIO_COUNT)


@dataclass(frozen=True)
class BremKernelConfig:
    nthreads: int
    energies_per_block: int


# Starter configuration only; retune independently on each GPU generation.
DEFAULT_BREM_KERNEL_CONFIG = BremKernelConfig(nthreads=256, energies_per_block=2)


@jit.rawkernel(device=True)
def _dsigma_weighted_scalar(
    incident_energy_keV, photon_energy_eV, atomic_number, incident_momentum, incident_prefactor
):
    """Weighted Bethe-Heitler + Elwert cell with incident work hoisted.

    ``incident_prefactor`` already contains density, segment length, the common
    Bethe-Heitler constant, ``atomic_number**2``, ``1/incident_momentum**2``, and the incident-side
    ``beta_i * (1-exp(-2*pi*atomic_number*alpha/beta_i))`` factor. Only final-state work
    remains energy dependent here.
    """
    photon_energy_keV = photon_energy_eV * F32_1E_M3
    final_energy_keV = incident_energy_keV - photon_energy_keV
    if final_energy_keV <= F32_1E_M6 or photon_energy_keV <= F32_ZERO:
        return F32_ZERO

    final_momentum = (
        xp.sqrt(final_energy_keV * (final_energy_keV + F32_TWO * F32_MC2_KEV)) / F32_MC2_KEV
    )
    final_beta = final_momentum / (F32_ONE + final_energy_keV / F32_MC2_KEV)

    momentum_difference = incident_momentum - final_momentum
    if momentum_difference < F32_1E_M30:
        momentum_difference = F32_1E_M30
    born_log = xp.log((incident_momentum + final_momentum) / momentum_difference)

    coulomb_factor = atomic_number * F32_TWO_PI_ALPHA
    final_elwert_denominator = F32_ONE - xp.exp(-coulomb_factor / final_beta)

    safe_photon_energy_eV = photon_energy_eV
    if safe_photon_energy_eV < F32_1E_M30:
        safe_photon_energy_eV = F32_1E_M30

    return (
        incident_prefactor
        * born_log
        / (safe_photon_energy_eV * final_beta * final_elwert_denominator)
    )


@jit.rawkernel(device=True)
def _transmission_scalar(path_flat, mu_flat, line_id, energy_index, n_layers, n_energies):
    """Segment mean of ``exp(-tau)`` at photon bin ``energy_index`` (issue #176).

    ``path_flat`` holds the escape path per layer from both segment ends,
    C-order ``(n_seg, 2, n_layers)``; ``tau`` is linear between them, so the
    mean is ``exp(-min tau) * (1 - exp(-optical_depth_difference)) / optical_depth_difference`` with ``optical_depth_difference = |tau_1 - tau_0|``,
    as :func:`pyrite.montecarlo.spectrum.segment_escape.mean_transmission`.
    Validation: segment-escape-average
    """
    segment_path_offset = line_id * U32_TWO * n_layers
    start_optical_depth = F32_ZERO
    end_optical_depth = F32_ZERO
    layer = U32_ZERO
    while layer < n_layers:
        attenuation_coefficient = mu_flat[layer * n_energies + energy_index]
        start_optical_depth += path_flat[segment_path_offset + layer] * attenuation_coefficient
        end_optical_depth += (
            path_flat[segment_path_offset + n_layers + layer] * attenuation_coefficient
        )
        layer += U32_ONE
    min_optical_depth = (
        start_optical_depth if start_optical_depth < end_optical_depth else end_optical_depth
    )
    optical_depth_difference = (
        end_optical_depth - start_optical_depth
        if end_optical_depth > start_optical_depth
        else start_optical_depth - end_optical_depth
    )
    mean_transmission_ratio = (
        F32_ONE
        - F32_ONE / F32_TWO * optical_depth_difference
        + F32_ONE_SIXTH * optical_depth_difference * optical_depth_difference
    )
    if optical_depth_difference > F32_1E_M3:
        mean_transmission_ratio = -xp.expm1(-optical_depth_difference) / optical_depth_difference
    return xp.exp(-min_optical_depth) * mean_transmission_ratio


@jit.rawkernel(device=True)
def _eedl_or_bh_weighted_scalar(
    incident_energy_keV,
    photon_energy_eV,
    k_index,
    incident_momentum,
    bethe_heitler_prefactor,
    eedl_weight,
    lower_panel,
    panel_fraction,
    available,
    panel_probability,
    n_energies,
    atomic_number,
):
    """Weighted EEDL cell, with Bethe--Heitler for uncovered segments."""
    if available > F32_ZERO:
        if photon_energy_eV <= F32_ZERO or photon_energy_eV > incident_energy_keV * F32_1E3:
            return F32_ZERO
        lower_offset = lower_panel * n_energies + k_index
        upper_offset = lower_offset + n_energies
        lower_pdf = panel_probability[lower_offset]
        upper_pdf = panel_probability[upper_offset]
        mixed_pdf = lower_pdf + panel_fraction * (upper_pdf - lower_pdf)
        return eedl_weight * mixed_pdf
    return _dsigma_weighted_scalar(
        incident_energy_keV,
        photon_energy_eV,
        atomic_number,
        incident_momentum,
        bethe_heitler_prefactor,
    )


@jit.rawkernel(device=True)
def _bremslib_scaled_scalar(reduced, top, line_id, values, nominal_ratio):
    """Interpolate one staged BremsLib row (``STAGED_RATIO_COUNT`` nodes) in ``k/T``.

    Validation: bremslib-angular-model
    """
    lower = U32_ZERO
    while lower < U32_LAST_INNER_NODE and reduced >= nominal_ratio[lower + U32_ONE]:
        lower += U32_ONE
    lower_reduced_energy = nominal_ratio[lower]
    if lower == U32_LAST_INNER_NODE:
        upper_reduced_energy = top
    else:
        upper_reduced_energy = nominal_ratio[lower + U32_ONE]
    fraction = (reduced - lower_reduced_energy) / (upper_reduced_energy - lower_reduced_energy)
    if fraction < F32_ZERO:
        fraction = F32_ZERO
    elif fraction > F32_ONE:
        fraction = F32_ONE
    offset = line_id * U32_RATIO_COUNT + lower
    lower_scaled_sdcs = values[offset]
    upper_scaled_sdcs = values[offset + U32_ONE]
    return lower_scaled_sdcs + fraction * (upper_scaled_sdcs - lower_scaled_sdcs)


@jit.rawkernel(device=True)
def _bremslib_or_fallback_weighted_scalar(
    incident_energy_keV,
    photon_energy_eV,
    k_index,
    line_id,
    incident_momentum,
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
    n_energies,
    atomic_number,
    plasma_eV,
):
    """Weighted directional BremsLib cell, with the normal EEDL/BH fallback.

    ``plasma_eV > 0`` applies the Ter--Mikaelian ``k**2/(k**2 + (gamma*Ep)**2)``
    to the BremsLib branch, as the portable scorer does; the fallback is never
    reached then (the strict scorer refuses it).

    Validation: bremslib-angular-model, dielectric-suppression
    """
    if bremslib_available > F32_ZERO:
        if photon_energy_eV <= F32_ZERO or photon_energy_eV > incident_energy_keV * F32_1E3:
            return F32_ZERO
        reduced = photon_energy_eV / (incident_energy_keV * F32_1E3)
        lower = _bremslib_scaled_scalar(
            reduced, bremslib_lower_top, line_id, bremslib_lower_values, nominal_ratio
        )
        upper = _bremslib_scaled_scalar(
            reduced, bremslib_upper_top, line_id, bremslib_upper_values, nominal_ratio
        )
        scaled = lower + bremslib_energy_fraction * (upper - lower)
        if plasma_eV > F32_ZERO:
            scale = (F32_ONE + incident_energy_keV * F32_1E3 / F32_MC2_EV) * plasma_eV
            k2 = photon_energy_eV * photon_energy_eV
            scaled = scaled * (k2 / (k2 + scale * scale))
        return bremslib_weight * scaled / photon_energy_eV
    return _eedl_or_bh_weighted_scalar(
        incident_energy_keV,
        photon_energy_eV,
        k_index,
        incident_momentum,
        bethe_heitler_prefactor,
        eedl_weight,
        eedl_lower_panel,
        eedl_panel_fraction,
        eedl_available,
        panel_probability,
        n_energies,
        atomic_number,
    )


@jit.rawkernel()
def _eedl_kernel_1e(
    kinetic_energies_keV,
    incident_momenta,
    bethe_heitler_prefactor,
    eedl_weight,
    lower_panel,
    panel_fraction,
    available,
    path_flat,
    mu_flat,
    energies_eV,
    panel_probability,
    spectrum,
    atomic_number,
    n_segments,
    n_energies,
    n_layers,
):
    """Fuse EEDL interpolation, fallback, attenuation, and segment reduction."""
    energy_index = jit.blockIdx.x
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x
    photon_eV = energies_eV[energy_index]
    spectrum_sum = F32_ZERO

    line_id = thread_idx
    while line_id < n_segments:
        weighted_cross_section = _eedl_or_bh_weighted_scalar(
            kinetic_energies_keV[line_id],
            photon_eV,
            energy_index,
            incident_momenta[line_id],
            bethe_heitler_prefactor[line_id],
            eedl_weight[line_id],
            lower_panel[line_id],
            panel_fraction[line_id],
            available[line_id],
            panel_probability,
            n_energies,
            atomic_number,
        )
        transmission = _transmission_scalar(
            path_flat, mu_flat, line_id, energy_index, n_layers, n_energies
        )
        spectrum_sum += weighted_cross_section * transmission
        line_id += n_threads

    shared = jit.shared_memory(xp.float32, None)
    shared[thread_idx] = spectrum_sum
    jit.syncthreads()
    stride = n_threads // U32_TWO
    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[thread_idx] += shared[thread_idx + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if thread_idx == U32_ZERO:
        spectrum[energy_index] += shared[U32_ZERO]


@jit.rawkernel()
def _bremslib_kernel_1e(
    kinetic_energies_keV,
    incident_momenta,
    bethe_heitler_prefactor,
    eedl_weight,
    eedl_lower_panel,
    eedl_panel_fraction,
    eedl_available,
    path_flat,
    mu_flat,
    energies_eV,
    panel_probability,
    bremslib_weight,
    bremslib_lower_top,
    bremslib_upper_top,
    bremslib_energy_fraction,
    bremslib_available,
    bremslib_lower_values,
    bremslib_upper_values,
    nominal_ratio,
    spectrum,
    atomic_number,
    plasma_eV,
    n_segments,
    n_energies,
    n_layers,
):
    """Fuse BremsLib interpolation, fallback, attenuation, and reduction.

    Validation: bremslib-angular-model
    """
    energy_index = jit.blockIdx.x
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x
    photon_eV = energies_eV[energy_index]
    spectrum_sum = F32_ZERO

    line_id = thread_idx
    while line_id < n_segments:
        weighted_cross_section = _bremslib_or_fallback_weighted_scalar(
            kinetic_energies_keV[line_id],
            photon_eV,
            energy_index,
            line_id,
            incident_momenta[line_id],
            bethe_heitler_prefactor[line_id],
            eedl_weight[line_id],
            eedl_lower_panel[line_id],
            eedl_panel_fraction[line_id],
            eedl_available[line_id],
            panel_probability,
            bremslib_weight[line_id],
            bremslib_lower_top[line_id],
            bremslib_upper_top[line_id],
            bremslib_energy_fraction[line_id],
            bremslib_available[line_id],
            bremslib_lower_values,
            bremslib_upper_values,
            nominal_ratio,
            n_energies,
            atomic_number,
            plasma_eV,
        )
        transmission = _transmission_scalar(
            path_flat, mu_flat, line_id, energy_index, n_layers, n_energies
        )
        spectrum_sum += weighted_cross_section * transmission
        line_id += n_threads

    shared = jit.shared_memory(xp.float32, None)
    shared[thread_idx] = spectrum_sum
    jit.syncthreads()
    stride = n_threads // U32_TWO
    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[thread_idx] += shared[thread_idx + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if thread_idx == U32_ZERO:
        spectrum[energy_index] += shared[U32_ZERO]


@jit.rawkernel()
def _kernel_1e(
    kinetic_energies_keV,
    incident_momenta,
    incident_prefactor,
    path_flat,
    mu_flat,
    energies_eV,
    spectrum,
    atomic_number,
    n_segments,
    n_energies,
    n_layers,
):
    energy_index0 = jit.blockIdx.x
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x
    photon0_eV = energies_eV[energy_index0]
    spectrum_sum_0 = F32_ZERO

    line_id = thread_idx
    while line_id < n_segments:
        incident_energy_keV = kinetic_energies_keV[line_id]
        incident_momentum = incident_momenta[line_id]
        segment_prefactor = incident_prefactor[line_id]

        transmission_0 = _transmission_scalar(
            path_flat, mu_flat, line_id, energy_index0, n_layers, n_energies
        )
        ds0 = _dsigma_weighted_scalar(
            incident_energy_keV, photon0_eV, atomic_number, incident_momentum, segment_prefactor
        )
        spectrum_sum_0 += ds0 * transmission_0
        line_id += n_threads

    shared = jit.shared_memory(xp.float32, None)
    shared[thread_idx] = spectrum_sum_0
    jit.syncthreads()
    stride = n_threads // U32_TWO
    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[thread_idx] += shared[thread_idx + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if thread_idx == U32_ZERO:
        spectrum[energy_index0] += shared[U32_ZERO]


@jit.rawkernel()
def _kernel_2e(
    kinetic_energies_keV,
    incident_momenta,
    incident_prefactor,
    path_flat,
    mu_flat,
    energies_eV,
    spectrum,
    atomic_number,
    n_segments,
    n_energies,
    n_layers,
):
    energy_offset = jit.blockIdx.x * U32_TWO
    energy_index0 = energy_offset
    energy_index1 = energy_offset + U32_ONE
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x
    photon0_eV = energies_eV[energy_index0]
    has_energy_1 = energy_index1 < n_energies
    photon1_eV = F32_ZERO
    if has_energy_1:
        photon1_eV = energies_eV[energy_index1]
    spectrum_sum_0 = F32_ZERO
    spectrum_sum_1 = F32_ZERO

    line_id = thread_idx
    while line_id < n_segments:
        incident_energy_keV = kinetic_energies_keV[line_id]
        incident_momentum = incident_momenta[line_id]
        segment_prefactor = incident_prefactor[line_id]

        transmission_0 = _transmission_scalar(
            path_flat, mu_flat, line_id, energy_index0, n_layers, n_energies
        )
        spectrum_sum_0 += (
            _dsigma_weighted_scalar(
                incident_energy_keV, photon0_eV, atomic_number, incident_momentum, segment_prefactor
            )
            * transmission_0
        )
        if has_energy_1:
            transmission_1 = _transmission_scalar(
                path_flat, mu_flat, line_id, energy_index1, n_layers, n_energies
            )
            spectrum_sum_1 += (
                _dsigma_weighted_scalar(
                    incident_energy_keV,
                    photon1_eV,
                    atomic_number,
                    incident_momentum,
                    segment_prefactor,
                )
                * transmission_1
            )
        line_id += n_threads

    shared = jit.shared_memory(xp.float32, None)
    offset0 = U32_ZERO
    offset1 = n_threads
    shared[offset0 + thread_idx] = spectrum_sum_0
    shared[offset1 + thread_idx] = spectrum_sum_1
    jit.syncthreads()
    stride = n_threads // U32_TWO
    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[offset0 + thread_idx] += shared[offset0 + thread_idx + stride]
            shared[offset1 + thread_idx] += shared[offset1 + thread_idx + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if thread_idx == U32_ZERO:
        spectrum[energy_index0] += shared[offset0]
        if has_energy_1:
            spectrum[energy_index1] += shared[offset1]


@jit.rawkernel()
def _kernel_3e(
    kinetic_energies_keV,
    incident_momenta,
    incident_prefactor,
    path_flat,
    mu_flat,
    energies_eV,
    spectrum,
    atomic_number,
    n_segments,
    n_energies,
    n_layers,
):
    energy_offset = jit.blockIdx.x * U32_THREE
    energy_index0 = energy_offset
    energy_index1 = energy_offset + U32_ONE
    energy_index2 = energy_offset + U32_TWO
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x
    photon0_eV = energies_eV[energy_index0]
    has_energy_1 = energy_index1 < n_energies
    has_energy_2 = energy_index2 < n_energies
    photon1_eV = F32_ZERO
    photon2_eV = F32_ZERO
    if has_energy_1:
        photon1_eV = energies_eV[energy_index1]
    if has_energy_2:
        photon2_eV = energies_eV[energy_index2]
    spectrum_sum_0 = F32_ZERO
    spectrum_sum_1 = F32_ZERO
    spectrum_sum_2 = F32_ZERO

    line_id = thread_idx
    while line_id < n_segments:
        incident_energy_keV = kinetic_energies_keV[line_id]
        incident_momentum = incident_momenta[line_id]
        segment_prefactor = incident_prefactor[line_id]

        transmission_0 = _transmission_scalar(
            path_flat, mu_flat, line_id, energy_index0, n_layers, n_energies
        )
        spectrum_sum_0 += (
            _dsigma_weighted_scalar(
                incident_energy_keV, photon0_eV, atomic_number, incident_momentum, segment_prefactor
            )
            * transmission_0
        )
        if has_energy_1:
            transmission_1 = _transmission_scalar(
                path_flat, mu_flat, line_id, energy_index1, n_layers, n_energies
            )
            spectrum_sum_1 += (
                _dsigma_weighted_scalar(
                    incident_energy_keV,
                    photon1_eV,
                    atomic_number,
                    incident_momentum,
                    segment_prefactor,
                )
                * transmission_1
            )
        if has_energy_2:
            transmission_2 = _transmission_scalar(
                path_flat, mu_flat, line_id, energy_index2, n_layers, n_energies
            )
            spectrum_sum_2 += (
                _dsigma_weighted_scalar(
                    incident_energy_keV,
                    photon2_eV,
                    atomic_number,
                    incident_momentum,
                    segment_prefactor,
                )
                * transmission_2
            )
        line_id += n_threads

    shared = jit.shared_memory(xp.float32, None)
    offset0 = U32_ZERO
    offset1 = n_threads
    offset2 = U32_TWO * n_threads
    shared[offset0 + thread_idx] = spectrum_sum_0
    shared[offset1 + thread_idx] = spectrum_sum_1
    shared[offset2 + thread_idx] = spectrum_sum_2
    jit.syncthreads()
    stride = n_threads // U32_TWO
    while stride > U32_ZERO:
        if thread_idx < stride:
            shared[offset0 + thread_idx] += shared[offset0 + thread_idx + stride]
            shared[offset1 + thread_idx] += shared[offset1 + thread_idx + stride]
            shared[offset2 + thread_idx] += shared[offset2 + thread_idx + stride]
        jit.syncthreads()
        stride //= U32_TWO
    if thread_idx == U32_ZERO:
        spectrum[energy_index0] += shared[offset0]
        if has_energy_1:
            spectrum[energy_index1] += shared[offset1]
        if has_energy_2:
            spectrum[energy_index2] += shared[offset2]


@jit.rawkernel()
def _kernel_4e(
    kinetic_energies_keV,
    incident_momenta,
    incident_prefactor,
    path_flat,
    mu_flat,
    energies_eV,
    spectrum,
    atomic_number,
    n_segments,
    n_energies,
    n_layers,
):
    energy_offset = jit.blockIdx.x * U32_FOUR
    energy_index0 = energy_offset
    energy_index1 = energy_offset + U32_ONE
    energy_index2 = energy_offset + U32_TWO
    energy_index3 = energy_offset + U32_THREE
    thread_idx = jit.threadIdx.x
    n_threads = jit.blockDim.x
    photon0_eV = energies_eV[energy_index0]
    has_energy_1 = energy_index1 < n_energies
    has_energy_2 = energy_index2 < n_energies
    has_energy_3 = energy_index3 < n_energies
    photon1_eV = F32_ZERO
    photon2_eV = F32_ZERO
    photon3_eV = F32_ZERO
    if has_energy_1:
        photon1_eV = energies_eV[energy_index1]
    if has_energy_2:
        photon2_eV = energies_eV[energy_index2]
    if has_energy_3:
        photon3_eV = energies_eV[energy_index3]
    spectrum_sum_0 = F32_ZERO
    spectrum_sum_1 = F32_ZERO
    spectrum_sum_2 = F32_ZERO
    spectrum_sum_3 = F32_ZERO

    line_id = thread_idx
    while line_id < n_segments:
        incident_energy_keV = kinetic_energies_keV[line_id]
        incident_momentum = incident_momenta[line_id]
        segment_prefactor = incident_prefactor[line_id]

        transmission_0 = _transmission_scalar(
            path_flat, mu_flat, line_id, energy_index0, n_layers, n_energies
        )
        spectrum_sum_0 += (
            _dsigma_weighted_scalar(
                incident_energy_keV, photon0_eV, atomic_number, incident_momentum, segment_prefactor
            )
            * transmission_0
        )
        if has_energy_1:
            transmission_1 = _transmission_scalar(
                path_flat, mu_flat, line_id, energy_index1, n_layers, n_energies
            )
            spectrum_sum_1 += (
                _dsigma_weighted_scalar(
                    incident_energy_keV,
                    photon1_eV,
                    atomic_number,
                    incident_momentum,
                    segment_prefactor,
                )
                * transmission_1
            )
        if has_energy_2:
            transmission_2 = _transmission_scalar(
                path_flat, mu_flat, line_id, energy_index2, n_layers, n_energies
            )
            spectrum_sum_2 += (
                _dsigma_weighted_scalar(
                    incident_energy_keV,
                    photon2_eV,
                    atomic_number,
                    incident_momentum,
                    segment_prefactor,
                )
                * transmission_2
            )
        if has_energy_3:
            transmission_3 = _transmission_scalar(
                path_flat, mu_flat, line_id, energy_index3, n_layers, n_energies
            )
            spectrum_sum_3 += (
                _dsigma_weighted_scalar(
                    incident_energy_keV,
                    photon3_eV,
                    atomic_number,
                    incident_momentum,
                    segment_prefactor,
                )
                * transmission_3
            )
        line_id += n_threads

    shared = jit.shared_memory(xp.float32, None)
    offset0 = U32_ZERO
    offset1 = n_threads
    offset2 = U32_TWO * n_threads
    offset3 = U32_THREE * n_threads
    shared[offset0 + thread_idx] = spectrum_sum_0
    shared[offset1 + thread_idx] = spectrum_sum_1
    shared[offset2 + thread_idx] = spectrum_sum_2
    shared[offset3 + thread_idx] = spectrum_sum_3
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
        spectrum[energy_index0] += shared[offset0]
        if has_energy_1:
            spectrum[energy_index1] += shared[offset1]
        if has_energy_2:
            spectrum[energy_index2] += shared[offset2]
        if has_energy_3:
            spectrum[energy_index3] += shared[offset3]


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
    C-order ``(n_seg, 2, n_layers)``: each segment's escape path per layer from
    its start and end point and ``mu_flat`` is C-order
    ``(n_layers, n_energies)``. If ``out`` is supplied it is incremented in place;
    otherwise a zeroed float32 output is allocated.
    """
    n_threads = int(config.nthreads)
    energies_per_block = int(config.energies_per_block)
    if n_threads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    try:
        kernel = _REDUCTION_KERNELS[energies_per_block]
    except KeyError:
        raise ValueError(f"energies_per_block must be one of {tuple(_REDUCTION_KERNELS)}") from None

    n_energies = int(E_grid.size)
    if out is None:
        out = xp.zeros(n_energies, dtype=xp.float32)
    if n_energies == 0 or T_keV.size == 0:
        return out

    atomic_number_f32 = np.float32(Z)
    if p_i is None:
        p_i = xp.sqrt(T_keV * (T_keV + F32_TWO * F32_MC2_KEV)) / F32_MC2_KEV
        p_i = xp.ascontiguousarray(p_i, dtype=xp.float32)
    if incident_prefactor is None:
        incident_beta = p_i / (F32_ONE + T_keV / F32_MC2_KEV)
        coulomb_factor = atomic_number_f32 * F32_TWO_PI_ALPHA
        incident_elwert_denominator = F32_ONE - xp.exp(-coulomb_factor / incident_beta)
        incident_prefactor = (
            np.float32(density_cm3)
            * L_ang
            * F32_1E_M8
            * F32_16_OVER_3
            * F32_ALPHA
            * F32_R_E_CM2
            * atomic_number_f32
            * atomic_number_f32
            * incident_beta
            * incident_elwert_denominator
            / (p_i * p_i)
        )
        incident_prefactor = xp.ascontiguousarray(incident_prefactor, dtype=xp.float32)

    n_blocks = (n_energies + energies_per_block - 1) // energies_per_block
    shared_memory_bytes = energies_per_block * n_threads * np.dtype(np.float32).itemsize
    kernel(
        (n_blocks,),
        (n_threads,),
        (
            T_keV,
            p_i,
            incident_prefactor,
            path_flat,
            mu_flat,
            E_grid,
            out,
            atomic_number_f32,
            np.uint32(T_keV.size),
            np.uint32(n_energies),
            np.uint32(n_layers),
        ),
        shared_mem=shared_memory_bytes,
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

    ``panel_probability`` is the C-order ``(n_panel, n_energies)`` probability table
    already evaluated on ``E_grid``. The per-segment arrays contain the two
    adjacent-panel selector, interpolation weight, and
    ``density * length * sigma / normalization``. Rows with ``available == 0``
    evaluate the retained Bethe--Heitler expression from the supplied fallback
    prefactor. All arrays must be contiguous float32 CuPy arrays except
    ``lower_panel``, which is uint32.

    Validation: brem-spectrum
    """
    n_threads = int(config.nthreads)
    if n_threads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    n_energies = int(E_grid.size)
    if out is None:
        out = xp.zeros(n_energies, dtype=xp.float32)
    if n_energies == 0 or T_keV.size == 0:
        return out

    _eedl_kernel_1e(
        (n_energies,),
        (n_threads,),
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
            np.uint32(n_energies),
            np.uint32(n_layers),
        ),
        shared_mem=n_threads * np.dtype(np.float32).itemsize,
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
    plasma_eV=0.0,
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

    Validation: bremslib-angular-model
    """
    n_threads = int(config.nthreads)
    if n_threads not in (32, 64, 128, 256, 512, 1024):
        raise ValueError("nthreads must be one of 32, 64, 128, 256, 512, 1024")
    n_energies = int(E_grid.size)
    if out is None:
        out = xp.zeros(n_energies, dtype=xp.float32)
    if n_energies == 0 or T_keV.size == 0:
        return out

    _bremslib_kernel_1e(
        (n_energies,),
        (n_threads,),
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
            np.float32(plasma_eV),
            np.uint32(T_keV.size),
            np.uint32(n_energies),
            np.uint32(n_layers),
        ),
        shared_mem=n_threads * np.dtype(np.float32).itemsize,
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
    plasma_eV=0.0,
):
    """Stage one element's segment state and launch the BremsLib reducer.

    Validation: bremslib-angular-model
    """
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
        eedl_panel_fraction = xp.ascontiguousarray(eedl_state.panel_fraction, dtype=xp.float32)
        eedl_available = xp.ascontiguousarray(eedl_state.available, dtype=xp.float32)
        panel_probability = xp.ascontiguousarray(
            eedl_context.prepared.photon_probability_on_grid_per_eV.reshape(-1),
            dtype=xp.float32,
        )
    weight = np.float32(number_density_ang3 * 1.0e24 * 1.0e-8 * 4.0 * np.pi * 1.0e-27 * Z * Z)
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
        xp.asarray(staged.table.nominal_reduced_energy, dtype=xp.float32),
        Z=Z,
        n_layers=n_layers,
        plasma_eV=plasma_eV,
        out=out,
    )
