"""Device helpers of the opt-in coupled radiative mode on CUDA.

Scalar twins of the Numba kernels in :mod:`._jit_radiative`, called from the
exact CUDA transport kernel in :mod:`._jit_kernel`. Each follows its host twin
in the same operation order, so the port is checkable against a per-electron
CPU run from identical draws.

Device functions return one value and cannot allocate, so the host's
``(row, fraction)`` pair is split into :func:`_rad_row` and
:func:`_rad_fraction`, the host's sorted photon-node array is walked by
:func:`_rad_next_node` instead, and the host's ``(soft, hard rate, hard
first)`` triple is selected by a ``which`` argument.

Tables are the :func:`.hard_radiative.pack_radiative_layer_tables` arrays
flattened C-order. For element slot ``s = layer * max_el + i``: incident
energies and tips start at ``s * max_t``, nominal reduced energies at
``s * n_red``, and the scaled SDCS row ``r`` at ``(s * max_t + r) * n_red``.

Like its siblings, this module imports ``cupy`` at module scope, so it must
stay out of the package ``__init__``.

Validation: bremslib-radiative-partition, dielectric-suppression
"""

import cupy as xp
import numpy as np

from .._cupy_jit import jit
from ._jit_device import F64_HALF, F64_INF, F64_ONE, F64_ZERO, I32_ONE, I32_TWO, I32_ZERO
from ._jit_radiative import _MB_CM2

F64_MB_CM2 = np.float64(_MB_CM2)
F64_KEV_PER_EV = np.float64(1e-3)
F64_CM_PER_ANG = np.float64(1e-8)
I32_BISECTION_STEPS = np.int32(52)
# ``which`` selectors of :func:`_rad_moment`.
RAD_SOFT = np.int32(0)
RAD_HARD_RATE = np.int32(1)


@jit.rawkernel(device=True)
def _rad_row(incident, base, n_t, energy_eV):
    """Lower incident row of ``energy_eV``; host ``_incident_row``'s first value."""
    log_energy = xp.log(energy_eV * F64_KEV_PER_EV)
    if log_energy <= xp.log(incident[base]):
        return I32_ZERO
    if log_energy >= xp.log(incident[base + n_t - I32_ONE]):
        return n_t - I32_TWO
    lower_idx = I32_ZERO
    upper_idx = n_t - I32_ONE
    while upper_idx - lower_idx > I32_ONE:
        mid_idx = (lower_idx + upper_idx) // I32_TWO
        if xp.log(incident[base + mid_idx]) <= log_energy:
            lower_idx = mid_idx
        else:
            upper_idx = mid_idx
    return lower_idx


@jit.rawkernel(device=True)
def _rad_fraction(incident, base, n_t, energy_eV, row):
    """Log-energy fraction above ``row``; host ``_incident_row``'s second value."""
    log_energy = xp.log(energy_eV * F64_KEV_PER_EV)
    if log_energy <= xp.log(incident[base]):
        return F64_ZERO
    if log_energy >= xp.log(incident[base + n_t - I32_ONE]):
        return F64_ONE
    lower = xp.log(incident[base + row])
    return (log_energy - lower) / (xp.log(incident[base + row + I32_ONE]) - lower)


@jit.rawkernel(device=True)
def _rad_row_chi(reduced, nominal, nom_base, n_red, top_r, chi, chi_row):
    """One incident row's linear scaled SDCS; one side of ``_scaled_sdcs_at``."""
    if reduced >= top_r:
        return chi[chi_row + n_red - I32_ONE]
    lower = I32_ZERO
    upper = n_red - I32_ONE
    while upper - lower > I32_ONE:
        mid_idx = (lower + upper) // I32_TWO
        if reduced >= nominal[nom_base + mid_idx]:
            lower = mid_idx
        else:
            upper = mid_idx
    x0 = nominal[nom_base + lower]
    x1 = nominal[nom_base + lower + I32_ONE]
    if lower == n_red - I32_TWO:
        x1 = top_r
    weight = (reduced - x0) / (x1 - x0)
    return chi[chi_row + lower] + weight * (chi[chi_row + lower + I32_ONE] - chi[chi_row + lower])


@jit.rawkernel(device=True)
def _rad_chi(reduced, row, fraction, nominal, nom_base, n_red, top, t_base, chi, chi_base):
    """Energy-interpolated scaled SDCS; host ``_scaled_sdcs_at``."""
    v0 = _rad_row_chi(
        reduced, nominal, nom_base, n_red, top[t_base + row], chi, chi_base + row * n_red
    )
    v1 = _rad_row_chi(
        reduced,
        nominal,
        nom_base,
        n_red,
        top[t_base + row + I32_ONE],
        chi,
        chi_base + (row + I32_ONE) * n_red,
    )
    return v0 + fraction * (v1 - v0)


@jit.rawkernel(device=True)
def _rad_next_node(x, nominal, nom_base, n_red, top_a, top_b, cut_reduced):
    """Smallest reduced node above ``x`` in host ``_photon_nodes``' union, else inf."""
    lower_idx = I32_ZERO
    upper_idx = n_red - I32_ONE
    while lower_idx < upper_idx:
        mid_idx = (lower_idx + upper_idx) // I32_TWO
        if nominal[nom_base + mid_idx] <= x:
            lower_idx = mid_idx + I32_ONE
        else:
            upper_idx = mid_idx
    best = F64_INF
    if lower_idx < n_red - I32_ONE:
        best = nominal[nom_base + lower_idx]
    if top_a > x and top_a < best:
        best = top_a
    if top_b > x and top_b < best:
        best = top_b
    if cut_reduced > x and cut_reduced < best:
        best = cut_reduced
    if F64_ONE > x and F64_ONE < best:
        best = F64_ONE
    return best


@jit.rawkernel(device=True)
def _rad_photon(reduced, energy_eV, cutoff_eV, cut_reduced):
    """Photon energy of a reduced node, with the cutoff node kept exact."""
    if reduced == cut_reduced:
        return cutoff_eV
    return reduced * energy_eV


@jit.rawkernel(device=True)
def _rad_cell_rate(left, right, chi_left, chi_right):
    """Zeroth moment of one linear chi cell; host ``_linear_cell``'s first value."""
    if right <= left:
        return F64_ZERO
    slope = (chi_right - chi_left) / (right - left)
    intercept = chi_left - slope * left
    return intercept * xp.log(right / left) + slope * (right - left)


# The BremsLib chi(k) is linear in each cell, but chi(k)*F(k) is not.
# Integrate analytically instead of rescaling the widely spaced tabulated nodes.
# The Taylor branch reproduces the host/Numba evaluation near k/(gamma*Ep)=0.
F64_ELECTRON_REST_EV = np.float64(510998.95)
I32_TAYLOR_TERMS = np.int32(6)


@jit.rawkernel(device=True)
def _rad_dielectric_cell(which, left, right, chi_left, chi_right, scale_eV):
    """Suppressed cell first (soft) or zeroth (hard) moment, before Z^2 mb scaling.

    Scalar CUDA equivalent of ``dielectric_cell_moments``. With ``scale_eV=0``
    the caller uses the original unsuppressed device primitives instead.
    """
    if right <= left:
        return F64_ZERO
    slope = (chi_right - chi_left) / (right - left)
    intercept = chi_left - slope * left
    u = left / scale_eV
    v = right / scale_eV
    if v < np.float64(0.1):
        result = F64_ZERO
        sign = F64_ONE
        n = I32_ZERO
        while n < I32_TAYLOR_TERMS:
            power = I32_TWO * n + I32_TWO
            if which == RAD_SOFT:
                result += (
                    sign
                    * scale_eV
                    * (
                        intercept
                        * (v ** (power + I32_ONE) - u ** (power + I32_ONE))
                        / np.float64(power + I32_ONE)
                        + slope
                        * scale_eV
                        * (v ** (power + I32_TWO) - u ** (power + I32_TWO))
                        / np.float64(power + I32_TWO)
                    )
                )
            else:
                result += sign * (
                    intercept * (v**power - u**power) / np.float64(power)
                    + slope
                    * scale_eV
                    * (v ** (power + I32_ONE) - u ** (power + I32_ONE))
                    / np.float64(power + I32_ONE)
                )
            sign = -sign
            n += I32_ONE
        return result
    delta_c = F64_HALF * (xp.log1p(v * v) - xp.log1p(u * u))
    delta_a = (v - u) - (xp.arctan(v) - xp.arctan(u))
    if which == RAD_HARD_RATE:
        return intercept * delta_c + slope * scale_eV * delta_a
    delta_b = F64_HALF * (v * v - u * u) - delta_c
    return scale_eV * (intercept * delta_a + slope * scale_eV * delta_b)


@jit.rawkernel(device=True)
def _rad_moment(
    which,
    incident,
    nominal,
    top,
    chi,
    slot,
    max_t,
    n_red,
    n_t,
    z,
    energy_eV,
    cutoff_eV,
    plasma_energy_eV,
):
    """Soft first moment or hard rate per atom; host ``radiative_moments_scalar``.

    Validation: bremslib-radiative-partition
    """
    incident_offset = slot * max_t
    nominal_offset = slot * n_red
    sdcs_offset = slot * max_t * n_red
    row = _rad_row(incident, incident_offset, n_t, energy_eV)
    fraction = _rad_fraction(incident, incident_offset, n_t, energy_eV, row)
    top_a = top[incident_offset + row]
    top_b = top[incident_offset + row + I32_ONE]
    reduced_cutoff = cutoff_eV / energy_eV
    scale = F64_MB_CM2 * z * z
    suppression_scale = (F64_ONE + energy_eV / F64_ELECTRON_REST_EV) * plasma_energy_eV
    soft = F64_ZERO
    hard_rate = F64_ZERO
    # Each selector retains the original cell arithmetic and summation order,
    # but visits only its side of the cutoff. In particular, soft stopping
    # never evaluates the logarithmic hard-rate integrals.
    if which == RAD_SOFT:
        left_reduced_energy = _rad_next_node(
            -F64_ONE, nominal, nominal_offset, n_red, top_a, top_b, reduced_cutoff
        )
    else:
        left_reduced_energy = reduced_cutoff
    left = _rad_photon(left_reduced_energy, energy_eV, cutoff_eV, reduced_cutoff)
    chi_left = _rad_chi(
        left / energy_eV,
        row,
        fraction,
        nominal,
        nominal_offset,
        n_red,
        top,
        incident_offset,
        chi,
        sdcs_offset,
    )
    right_reduced_energy = _rad_next_node(
        left_reduced_energy, nominal, nominal_offset, n_red, top_a, top_b, reduced_cutoff
    )
    while right_reduced_energy < F64_INF:
        right = _rad_photon(right_reduced_energy, energy_eV, cutoff_eV, reduced_cutoff)
        if which == RAD_SOFT and right > cutoff_eV:
            break
        chi_right = _rad_chi(
            right / energy_eV,
            row,
            fraction,
            nominal,
            nominal_offset,
            n_red,
            top,
            incident_offset,
            chi,
            sdcs_offset,
        )
        if suppression_scale == F64_ZERO:
            # The historical CUDA operations stay bit-for-bit unchanged.
            if left == F64_ZERO:
                soft += scale * F64_HALF * (chi_left + chi_right) * right
            elif right <= cutoff_eV:
                if right > left:
                    soft += scale * (F64_HALF * (chi_left + chi_right) * (right - left))
            else:
                hard_rate += scale * _rad_cell_rate(left, right, chi_left, chi_right)
        elif right <= cutoff_eV:
            soft += scale * _rad_dielectric_cell(
                RAD_SOFT, left, right, chi_left, chi_right, suppression_scale
            )
        else:
            hard_rate += scale * _rad_dielectric_cell(
                RAD_HARD_RATE, left, right, chi_left, chi_right, suppression_scale
            )
        left = right
        chi_left = chi_right
        left_reduced_energy = right_reduced_energy
        right_reduced_energy = _rad_next_node(
            left_reduced_energy, nominal, nominal_offset, n_red, top_a, top_b, reduced_cutoff
        )
    if which == RAD_SOFT:
        return soft
    return hard_rate


@jit.rawkernel(device=True)
def _rad_layer_moment(
    which,
    layer,
    nel,
    nt,
    z_arr,
    ncm3,
    incident,
    nominal,
    top,
    chi,
    max_el,
    max_t,
    n_red,
    energy_eV,
    cutoff_eV,
    plasma_by_layer,
):
    """Layer soft loss (eV/Angstrom) or hard rate (1/Angstrom); host layer moments."""
    total = F64_ZERO
    element_index = I32_ZERO
    while element_index < nel[layer]:
        slot = layer * max_el + element_index
        per_atom = _rad_moment(
            which,
            incident,
            nominal,
            top,
            chi,
            slot,
            max_t,
            n_red,
            nt[slot],
            np.float64(z_arr[slot]),
            energy_eV,
            cutoff_eV,
            plasma_by_layer[layer],
        )
        total += ncm3[slot] * F64_CM_PER_ANG * per_atom
        element_index += I32_ONE
    return total


@jit.rawkernel(device=True)
def _rad_sample_photon_eV(
    incident,
    nominal,
    top,
    chi,
    slot,
    max_t,
    n_red,
    n_t,
    z,
    energy_eV,
    cutoff_eV,
    uniform,
    plasma_energy_eV,
):
    """Invert the hard zeroth-moment CDF; host ``sample_hard_photon_energy_scalar``.

    Validation: bremslib-radiative-partition
    """
    hard_rate = _rad_moment(
        RAD_HARD_RATE,
        incident,
        nominal,
        top,
        chi,
        slot,
        max_t,
        n_red,
        n_t,
        z,
        energy_eV,
        cutoff_eV,
        plasma_energy_eV,
    )
    if hard_rate <= F64_ZERO:
        return energy_eV
    incident_offset = slot * max_t
    nominal_offset = slot * n_red
    sdcs_offset = slot * max_t * n_red
    row = _rad_row(incident, incident_offset, n_t, energy_eV)
    fraction = _rad_fraction(incident, incident_offset, n_t, energy_eV, row)
    top_a = top[incident_offset + row]
    top_b = top[incident_offset + row + I32_ONE]
    reduced_cutoff = cutoff_eV / energy_eV
    scale = F64_MB_CM2 * z * z
    suppression_scale = (F64_ONE + energy_eV / F64_ELECTRON_REST_EV) * plasma_energy_eV
    target = uniform * hard_rate
    cumulative = F64_ZERO
    left_reduced_energy = _rad_next_node(
        -F64_ONE, nominal, nominal_offset, n_red, top_a, top_b, reduced_cutoff
    )
    left = _rad_photon(left_reduced_energy, energy_eV, cutoff_eV, reduced_cutoff)
    chi_left = _rad_chi(
        left / energy_eV,
        row,
        fraction,
        nominal,
        nominal_offset,
        n_red,
        top,
        incident_offset,
        chi,
        sdcs_offset,
    )
    right_reduced_energy = _rad_next_node(
        left_reduced_energy, nominal, nominal_offset, n_red, top_a, top_b, reduced_cutoff
    )
    while right_reduced_energy < F64_INF:
        right = _rad_photon(right_reduced_energy, energy_eV, cutoff_eV, reduced_cutoff)
        chi_right = _rad_chi(
            right / energy_eV,
            row,
            fraction,
            nominal,
            nominal_offset,
            n_red,
            top,
            incident_offset,
            chi,
            sdcs_offset,
        )
        next_reduced_energy = _rad_next_node(
            right_reduced_energy, nominal, nominal_offset, n_red, top_a, top_b, reduced_cutoff
        )
        if right > cutoff_eV:
            if suppression_scale == F64_ZERO:
                cell_mass = scale * _rad_cell_rate(left, right, chi_left, chi_right)
            else:
                cell_mass = scale * _rad_dielectric_cell(
                    RAD_HARD_RATE, left, right, chi_left, chi_right, suppression_scale
                )
            if cumulative + cell_mass >= target or next_reduced_energy == F64_INF:
                local = target - cumulative
                low = left
                high = right
                step = I32_ZERO
                while step < I32_BISECTION_STEPS:
                    midpoint_eV = F64_HALF * (low + high)
                    chi_mid = chi_left + (chi_right - chi_left) * (midpoint_eV - left) / (
                        right - left
                    )
                    if suppression_scale == F64_ZERO:
                        partial = _rad_cell_rate(left, midpoint_eV, chi_left, chi_mid)
                    else:
                        partial = _rad_dielectric_cell(
                            RAD_HARD_RATE,
                            left,
                            midpoint_eV,
                            chi_left,
                            chi_mid,
                            suppression_scale,
                        )
                    if scale * partial < local:
                        low = midpoint_eV
                    else:
                        high = midpoint_eV
                    step += I32_ONE
                return F64_HALF * (low + high)
            cumulative += cell_mass
        left = right
        chi_left = chi_right
        right_reduced_energy = next_reduced_energy
    return energy_eV
