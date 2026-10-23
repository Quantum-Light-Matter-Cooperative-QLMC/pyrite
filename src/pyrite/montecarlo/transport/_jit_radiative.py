"""Numba scalar twin of the host BremsLib radiative partition.

Only the BremsLib-derived SDCS arrays enter these kernels. The interpolation
and analytic cell integrals are PyRITE's own, shared in form with the host
partition; no BremsLib Fortran code is ported here.

Validation: bremslib-radiative-partition
"""

import numpy as np
from numba import njit

from .dielectric_radiative import dielectric_cell_moments, dielectric_scale_eV
from .kinematics import stream_keys

_MB_CM2 = 1.0e-27
_RADIATIVE_STREAM_SALT = 0xA54FF53A5F1D36F1


def radiative_stream_keys(seed, n_electrons, *, start=0):
    """A per-electron key domain separate from transport and shell collisions."""
    return stream_keys(int(seed) ^ _RADIATIVE_STREAM_SALT, n_electrons, start=start)


@njit(cache=True)
def _incident_row(incident_keV, energy_eV):
    log_energy = np.log(energy_eV * 1e-3)
    if log_energy <= np.log(incident_keV[0]):
        return 0, 0.0
    if log_energy >= np.log(incident_keV[-1]):
        return incident_keV.size - 2, 1.0
    lower_idx, upper_idx = 0, incident_keV.size - 1
    while upper_idx - lower_idx > 1:
        mid_idx = (lower_idx + upper_idx) // 2
        if np.log(incident_keV[mid_idx]) <= log_energy:
            lower_idx = mid_idx
        else:
            upper_idx = mid_idx
    fraction = (log_energy - np.log(incident_keV[lower_idx])) / (
        np.log(incident_keV[lower_idx + 1]) - np.log(incident_keV[lower_idx])
    )
    return lower_idx, fraction


@njit(cache=True)
def _scaled_sdcs_at(reduced, row, fraction, nominal, top, scaled_sdcs):
    values = np.empty(2, dtype=np.float64)
    for side in range(2):
        r = row + side
        if reduced >= top[r]:
            values[side] = scaled_sdcs[r, nominal.size - 1]
            continue
        lower = 0
        while lower + 1 < nominal.size - 1 and reduced >= nominal[lower + 1]:
            lower += 1
        x0 = nominal[lower]
        x1 = top[r] if lower == nominal.size - 2 else nominal[lower + 1]
        weight = (reduced - x0) / (x1 - x0)
        values[side] = scaled_sdcs[r, lower] + weight * (
            scaled_sdcs[r, lower + 1] - scaled_sdcs[r, lower]
        )
    return values[0] + fraction * (values[1] - values[0])


@njit(cache=True)
def _photon_nodes(energy_eV, cutoff_eV, row, nominal, top):
    # Union of both incident-row breakpoints and the cutoff. Their upper
    # reduced-energy tips differ, so both are needed for an exact linear cell.
    nodes = np.empty(nominal.size + 3, dtype=np.float64)
    for node_index in range(nominal.size - 1):
        nodes[node_index] = nominal[node_index]
    start = nominal.size - 1
    nodes[start] = top[row]
    nodes[start + 1] = top[row + 1]
    nodes[start + 2] = cutoff_eV / energy_eV
    nodes[start + 3] = 1.0
    nodes.sort()
    unique = 1
    for node_index in range(1, nodes.size):
        if nodes[node_index] > nodes[unique - 1]:
            nodes[unique] = nodes[node_index]
            unique += 1
    photon = nodes[:unique] * energy_eV
    # (kc / T) * T can round below kc. Restore the exact cutoff so the moment
    # and sampler classify the first hard cell identically.
    for node_index in range(unique):
        if nodes[node_index] == cutoff_eV / energy_eV:
            photon[node_index] = cutoff_eV
    return photon


@njit(cache=True)
def _linear_cell(left, right, chi_left, chi_right):
    if right <= left:
        return 0.0, 0.0
    slope = (chi_right - chi_left) / (right - left)
    intercept = chi_left - slope * left
    rate = intercept * np.log(right / left) + slope * (right - left)
    first = 0.5 * (chi_left + chi_right) * (right - left)
    return rate, first


@njit(cache=True)
def radiative_moments_scalar(
    incident_keV,
    nominal,
    top,
    scaled_sdcs,
    atomic_number,
    energy_eV,
    cutoff_eV,
    plasma_energy_eV=0.0,
):
    """Return (soft first, hard zeroth, hard first) cross sections per atom.

    The caller validates ``0 < cutoff_eV <= energy_eV`` and table coverage.
    All first moments have units eV cm²; the zeroth has units cm².
    ``plasma_energy_eV=0`` keeps the historical unsuppressed partition.

    Validation: bremslib-radiative-partition, dielectric-suppression
    """
    row, fraction = _incident_row(incident_keV, energy_eV)
    nodes = _photon_nodes(energy_eV, cutoff_eV, row, nominal, top)
    scale = _MB_CM2 * atomic_number * atomic_number
    suppression_scale = dielectric_scale_eV(energy_eV, plasma_energy_eV)
    soft, hard_rate, hard_first = 0.0, 0.0, 0.0
    chi_left = _scaled_sdcs_at(nodes[0] / energy_eV, row, fraction, nominal, top, scaled_sdcs)
    for interval_index in range(nodes.size - 1):
        left, right = nodes[interval_index], nodes[interval_index + 1]
        chi_right = _scaled_sdcs_at(right / energy_eV, row, fraction, nominal, top, scaled_sdcs)
        if suppression_scale == 0.0:
            if left == 0.0:
                soft += scale * 0.5 * (chi_left + chi_right) * right
                chi_left = chi_right
                continue
            rate, first = _linear_cell(left, right, chi_left, chi_right)
        else:
            rate, first = dielectric_cell_moments(
                left, right, chi_left, chi_right, suppression_scale
            )
        if right <= cutoff_eV:
            soft += scale * first
        else:
            hard_rate += scale * rate
            hard_first += scale * first
        chi_left = chi_right
    return soft, hard_rate, hard_first


@njit(cache=True)
def sample_hard_photon_energy_scalar(
    incident_keV,
    nominal,
    top,
    scaled_sdcs,
    atomic_number,
    energy_eV,
    cutoff_eV,
    uniform,
    plasma_energy_eV=0.0,
):
    """Invert the same (optionally suppressed) zeroth-moment CDF as the host.

    Validation: bremslib-radiative-partition, dielectric-suppression
    """
    _, hard_rate, _ = radiative_moments_scalar(
        incident_keV,
        nominal,
        top,
        scaled_sdcs,
        atomic_number,
        energy_eV,
        cutoff_eV,
        plasma_energy_eV,
    )
    if hard_rate <= 0.0:
        raise ValueError("partition has no hard photons")
    row, fraction = _incident_row(incident_keV, energy_eV)
    nodes = _photon_nodes(energy_eV, cutoff_eV, row, nominal, top)
    scale = _MB_CM2 * atomic_number * atomic_number
    suppression_scale = dielectric_scale_eV(energy_eV, plasma_energy_eV)
    target = uniform * hard_rate
    cumulative = 0.0
    chi_left = _scaled_sdcs_at(nodes[0] / energy_eV, row, fraction, nominal, top, scaled_sdcs)
    for interval_index in range(nodes.size - 1):
        left, right = nodes[interval_index], nodes[interval_index + 1]
        chi_right = _scaled_sdcs_at(right / energy_eV, row, fraction, nominal, top, scaled_sdcs)
        if right > cutoff_eV:
            if suppression_scale == 0.0:
                mass, _ = _linear_cell(left, right, chi_left, chi_right)
            else:
                mass, _ = dielectric_cell_moments(
                    left, right, chi_left, chi_right, suppression_scale
                )
            cell_mass = scale * mass
            if cumulative + cell_mass >= target or interval_index == nodes.size - 2:
                local = target - cumulative
                low, high = left, right
                for _ in range(52):
                    midpoint_eV = 0.5 * (low + high)
                    chi_mid = chi_left + (chi_right - chi_left) * (midpoint_eV - left) / (
                        right - left
                    )
                    if suppression_scale == 0.0:
                        partial, _ = _linear_cell(left, midpoint_eV, chi_left, chi_mid)
                    else:
                        partial, _ = dielectric_cell_moments(
                            left, midpoint_eV, chi_left, chi_mid, suppression_scale
                        )
                    if scale * partial < local:
                        low = midpoint_eV
                    else:
                        high = midpoint_eV
                return 0.5 * (low + high)
            cumulative += cell_mass
        chi_left = chi_right
    return energy_eV


@njit(cache=True)
def radiative_layer_moments_scalar(
    n_elements,
    n_incident,
    atomic_number,
    density_cm3,
    incident_keV,
    nominal,
    top,
    scaled_sdcs,
    layer,
    energy_eV,
    cutoff_eV,
    element_rates,
    plasma_energy_eV_by_layer=None,
):
    """Return (soft eV/Angstrom, hard 1/Angstrom) and fill element rates."""
    soft, hard = 0.0, 0.0
    plasma_eV = 0.0 if plasma_energy_eV_by_layer is None else plasma_energy_eV_by_layer[layer]
    for element_index in range(n_elements[layer]):
        incident_energy_count = n_incident[layer, element_index]
        soft_moment, rate, _ = radiative_moments_scalar(
            incident_keV[layer, element_index, :incident_energy_count],
            nominal[layer, element_index],
            top[layer, element_index, :incident_energy_count],
            scaled_sdcs[layer, element_index, :incident_energy_count],
            atomic_number[layer, element_index],
            energy_eV,
            cutoff_eV,
            plasma_eV,
        )
        number_per_ang = density_cm3[layer, element_index] * 1e-8
        soft += number_per_ang * soft_moment
        element_rates[element_index] = number_per_ang * rate
        hard += element_rates[element_index]
    return soft, hard
