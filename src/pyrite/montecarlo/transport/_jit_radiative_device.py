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

Validation: bremslib-radiative-partition
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
    lo = I32_ZERO
    hi = n_t - I32_ONE
    while hi - lo > I32_ONE:
        mid = (lo + hi) // I32_TWO
        if xp.log(incident[base + mid]) <= log_energy:
            lo = mid
        else:
            hi = mid
    return lo


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
    while lower + I32_ONE < n_red - I32_ONE and reduced >= nominal[nom_base + lower + I32_ONE]:
        lower += I32_ONE
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
    best = F64_INF
    k = I32_ZERO
    while k < n_red - I32_ONE:
        v = nominal[nom_base + k]
        if v > x and v < best:
            best = v
        k += I32_ONE
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


@jit.rawkernel(device=True)
def _rad_moment(
    which, incident, nominal, top, chi, slot, max_t, n_red, n_t, z, energy_eV, cutoff_eV
):
    """Soft first moment or hard rate per atom; host ``radiative_moments_scalar``.

    Validation: bremslib-radiative-partition
    """
    t_base = slot * max_t
    nom_base = slot * n_red
    chi_base = slot * max_t * n_red
    row = _rad_row(incident, t_base, n_t, energy_eV)
    fraction = _rad_fraction(incident, t_base, n_t, energy_eV, row)
    top_a = top[t_base + row]
    top_b = top[t_base + row + I32_ONE]
    cut_reduced = cutoff_eV / energy_eV
    scale = F64_MB_CM2 * z * z
    soft = F64_ZERO
    hard_rate = F64_ZERO
    red_left = _rad_next_node(-F64_ONE, nominal, nom_base, n_red, top_a, top_b, cut_reduced)
    left = _rad_photon(red_left, energy_eV, cutoff_eV, cut_reduced)
    chi_left = _rad_chi(
        left / energy_eV, row, fraction, nominal, nom_base, n_red, top, t_base, chi, chi_base
    )
    red_right = _rad_next_node(red_left, nominal, nom_base, n_red, top_a, top_b, cut_reduced)
    while red_right < F64_INF:
        right = _rad_photon(red_right, energy_eV, cutoff_eV, cut_reduced)
        chi_right = _rad_chi(
            right / energy_eV, row, fraction, nominal, nom_base, n_red, top, t_base, chi, chi_base
        )
        if left == F64_ZERO:
            soft += scale * F64_HALF * (chi_left + chi_right) * right
        elif right <= cutoff_eV:
            if right > left:
                soft += scale * (F64_HALF * (chi_left + chi_right) * (right - left))
        else:
            hard_rate += scale * _rad_cell_rate(left, right, chi_left, chi_right)
        left = right
        chi_left = chi_right
        red_left = red_right
        red_right = _rad_next_node(red_left, nominal, nom_base, n_red, top_a, top_b, cut_reduced)
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
):
    """Layer soft loss (eV/Angstrom) or hard rate (1/Angstrom); host layer moments."""
    total = F64_ZERO
    i = I32_ZERO
    while i < nel[layer]:
        slot = layer * max_el + i
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
        )
        total += ncm3[slot] * F64_CM_PER_ANG * per_atom
        i += I32_ONE
    return total


@jit.rawkernel(device=True)
def _rad_sample_photon_eV(
    incident, nominal, top, chi, slot, max_t, n_red, n_t, z, energy_eV, cutoff_eV, uniform
):
    """Invert the hard zeroth-moment CDF; host ``sample_hard_photon_energy_scalar``.

    Validation: bremslib-radiative-partition
    """
    hard_rate = _rad_moment(
        RAD_HARD_RATE, incident, nominal, top, chi, slot, max_t, n_red, n_t, z, energy_eV, cutoff_eV
    )
    if hard_rate <= F64_ZERO:
        return energy_eV
    t_base = slot * max_t
    nom_base = slot * n_red
    chi_base = slot * max_t * n_red
    row = _rad_row(incident, t_base, n_t, energy_eV)
    fraction = _rad_fraction(incident, t_base, n_t, energy_eV, row)
    top_a = top[t_base + row]
    top_b = top[t_base + row + I32_ONE]
    cut_reduced = cutoff_eV / energy_eV
    scale = F64_MB_CM2 * z * z
    target = uniform * hard_rate
    cumulative = F64_ZERO
    red_left = _rad_next_node(-F64_ONE, nominal, nom_base, n_red, top_a, top_b, cut_reduced)
    left = _rad_photon(red_left, energy_eV, cutoff_eV, cut_reduced)
    chi_left = _rad_chi(
        left / energy_eV, row, fraction, nominal, nom_base, n_red, top, t_base, chi, chi_base
    )
    red_right = _rad_next_node(red_left, nominal, nom_base, n_red, top_a, top_b, cut_reduced)
    while red_right < F64_INF:
        right = _rad_photon(red_right, energy_eV, cutoff_eV, cut_reduced)
        chi_right = _rad_chi(
            right / energy_eV, row, fraction, nominal, nom_base, n_red, top, t_base, chi, chi_base
        )
        red_next = _rad_next_node(red_right, nominal, nom_base, n_red, top_a, top_b, cut_reduced)
        if right > cutoff_eV:
            cell_mass = scale * _rad_cell_rate(left, right, chi_left, chi_right)
            if cumulative + cell_mass >= target or red_next == F64_INF:
                local = target - cumulative
                low = left
                high = right
                step = I32_ZERO
                while step < I32_BISECTION_STEPS:
                    mid = F64_HALF * (low + high)
                    chi_mid = chi_left + (chi_right - chi_left) * (mid - left) / (right - left)
                    if scale * _rad_cell_rate(left, mid, chi_left, chi_mid) < local:
                        low = mid
                    else:
                        high = mid
                    step += I32_ONE
                return F64_HALF * (low + high)
            cumulative += cell_mass
        left = right
        chi_left = chi_right
        red_right = red_next
    return energy_eV
