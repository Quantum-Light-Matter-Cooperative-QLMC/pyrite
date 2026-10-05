"""Numba scalar twin of the host BremsLib radiative partition.

Only the BremsLib-derived SDCS arrays enter these kernels. The interpolation
and analytic cell integrals are PyRITE's own, shared in form with the host
partition; no BremsLib Fortran code is ported here.

Validation: bremslib-radiative-partition
"""

import numpy as np
from numba import njit

from .kinematics import stream_keys

_MB_CM2 = 1.0e-27
_RADIATIVE_STREAM_SALT = 0xA54FF53A5F1D36F1


def radiative_stream_keys(seed, n_electrons):
    """A per-electron key domain separate from transport and shell collisions."""
    return stream_keys(int(seed) ^ _RADIATIVE_STREAM_SALT, n_electrons)


@njit(cache=True)
def _incident_row(incident_keV, energy_eV):
    log_energy = np.log(energy_eV * 1e-3)
    if log_energy <= np.log(incident_keV[0]):
        return 0, 0.0
    if log_energy >= np.log(incident_keV[-1]):
        return incident_keV.size - 2, 1.0
    lo, hi = 0, incident_keV.size - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if np.log(incident_keV[mid]) <= log_energy:
            lo = mid
        else:
            hi = mid
    fraction = (log_energy - np.log(incident_keV[lo])) / (
        np.log(incident_keV[lo + 1]) - np.log(incident_keV[lo])
    )
    return lo, fraction


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
    for i in range(nominal.size - 1):
        nodes[i] = nominal[i]
    start = nominal.size - 1
    nodes[start] = top[row]
    nodes[start + 1] = top[row + 1]
    nodes[start + 2] = cutoff_eV / energy_eV
    nodes[start + 3] = 1.0
    nodes.sort()
    unique = 1
    for i in range(1, nodes.size):
        if nodes[i] > nodes[unique - 1]:
            nodes[unique] = nodes[i]
            unique += 1
    photon = nodes[:unique] * energy_eV
    # (kc / T) * T can round below kc. Restore the exact cutoff so the moment
    # and sampler classify the first hard cell identically.
    for i in range(unique):
        if nodes[i] == cutoff_eV / energy_eV:
            photon[i] = cutoff_eV
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
    incident_keV, nominal, top, scaled_sdcs, atomic_number, energy_eV, cutoff_eV
):
    """Return (soft first, hard zeroth, hard first) cross sections per atom.

    The caller validates ``0 < cutoff_eV <= energy_eV`` and table coverage.
    All first moments have units eV cm²; the zeroth has units cm².

    Validation: bremslib-radiative-partition
    """
    row, fraction = _incident_row(incident_keV, energy_eV)
    nodes = _photon_nodes(energy_eV, cutoff_eV, row, nominal, top)
    scale = _MB_CM2 * atomic_number * atomic_number
    soft, hard_rate, hard_first = 0.0, 0.0, 0.0
    chi_left = _scaled_sdcs_at(nodes[0] / energy_eV, row, fraction, nominal, top, scaled_sdcs)
    for i in range(nodes.size - 1):
        left, right = nodes[i], nodes[i + 1]
        chi_right = _scaled_sdcs_at(right / energy_eV, row, fraction, nominal, top, scaled_sdcs)
        if left == 0.0:
            soft += scale * 0.5 * (chi_left + chi_right) * right
        else:
            rate, first = _linear_cell(left, right, chi_left, chi_right)
            if right <= cutoff_eV:
                soft += scale * first
            else:
                hard_rate += scale * rate
                hard_first += scale * first
        chi_left = chi_right
    return soft, hard_rate, hard_first


@njit(cache=True)
def sample_hard_photon_energy_scalar(
    incident_keV, nominal, top, scaled_sdcs, atomic_number, energy_eV, cutoff_eV, uniform
):
    """Invert the same hard zeroth-moment CDF used by the host sampler.

    Validation: bremslib-radiative-partition
    """
    _, hard_rate, _ = radiative_moments_scalar(
        incident_keV, nominal, top, scaled_sdcs, atomic_number, energy_eV, cutoff_eV
    )
    if hard_rate <= 0.0:
        raise ValueError("partition has no hard photons")
    row, fraction = _incident_row(incident_keV, energy_eV)
    nodes = _photon_nodes(energy_eV, cutoff_eV, row, nominal, top)
    scale = _MB_CM2 * atomic_number * atomic_number
    target = uniform * hard_rate
    cumulative = 0.0
    chi_left = _scaled_sdcs_at(nodes[0] / energy_eV, row, fraction, nominal, top, scaled_sdcs)
    for i in range(nodes.size - 1):
        left, right = nodes[i], nodes[i + 1]
        chi_right = _scaled_sdcs_at(right / energy_eV, row, fraction, nominal, top, scaled_sdcs)
        if right > cutoff_eV:
            mass, _ = _linear_cell(left, right, chi_left, chi_right)
            cell_mass = scale * mass
            if cumulative + cell_mass >= target or i == nodes.size - 2:
                local = target - cumulative
                low, high = left, right
                for _ in range(52):
                    mid = 0.5 * (low + high)
                    chi_mid = chi_left + (chi_right - chi_left) * (mid - left) / (right - left)
                    partial, _ = _linear_cell(left, mid, chi_left, chi_mid)
                    if scale * partial < local:
                        low = mid
                    else:
                        high = mid
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
):
    """Return (soft eV/Angstrom, hard 1/Angstrom) and fill element rates."""
    soft, hard = 0.0, 0.0
    for i in range(n_elements[layer]):
        nT = n_incident[layer, i]
        s, rate, _ = radiative_moments_scalar(
            incident_keV[layer, i, :nT],
            nominal[layer, i],
            top[layer, i, :nT],
            scaled_sdcs[layer, i, :nT],
            atomic_number[layer, i],
            energy_eV,
            cutoff_eV,
        )
        number_per_ang = density_cm3[layer, i] * 1e-8
        soft += number_per_ang * s
        element_rates[i] = number_per_ang * rate
        hard += element_rates[i]
    return soft, hard
