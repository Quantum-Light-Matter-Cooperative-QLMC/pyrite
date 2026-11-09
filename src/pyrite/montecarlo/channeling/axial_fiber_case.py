"""Fibre-source inputs of an admitted axial case (#994).

Builds the one-cell silicon fibre channels (independent-string continuum TDS
plus electronic kicks), the tilted entrance plane wave and the ownership
boundary exactly as the #992/#993 research driver
``checks/axial_fiber_source.py::build`` does for the numerics stored with an
admitted case, so a production run evaluates the calibrated model and nothing
else. Only ``tds="strings"`` exists here; the research lattice-Kraus variant
is not an admitted model.

Validation: cr-axial-persisted-channel
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...quantum._axial_envelope import AdmittedAxialCase
from .axial_fiber_unraveling import FiberChannels

_GAP_WINDOW_EV = 2.0  # provisional cut-search half-width of the research driver
_SEARCH_STEP_ANG = 200.0


@dataclass(frozen=True)
class AxialFiberInputs:
    """Everything ``fiber_on_axis_source`` needs for one admitted case.

    ``entry_orders`` is the tilted plane wave on the retained order window,
    ``carrier_inv_ang`` its folded transverse carrier, ``boundary_eV`` the
    ``boundary="max"`` final-state cap (potential maximum) and ``depths_ang``
    the uniform depth grid ending at the slab thickness.
    """

    channels: FiberChannels
    entry_orders: np.ndarray
    hbar_v_eV_ang: float
    carrier_inv_ang: tuple[float, float]
    boundary_eV: float
    target_eV: float
    depths_ang: np.ndarray
    k_inv_ang: float
    search_step_ang: float = _SEARCH_STEP_ANG


def gap_cut(energies: np.ndarray, target_eV: float, window_eV: float) -> float:
    """Midpoint of the widest level-free interval meeting ``target ± window`` [eV].

    The provisional numerical cut rule the calibrations were measured with.

    Validation: cr-axial-persisted-channel
    """
    levels = np.sort(np.asarray(energies, dtype=float))
    lo, hi = target_eV - window_eV, target_eV + window_eV
    below, above = levels[:-1], levels[1:]
    meets = (above > lo) & (below < hi)
    if not meets.any():
        raise ValueError("no spectral gap meets the matching window")
    widths = np.where(meets, above - below, -np.inf)
    i = int(np.argmax(widths))
    return float(0.5 * (below[i] + above[i]))


def _kick_columns(dissipator, harmonic_max: int, grid_points: int):
    h = harmonic_max
    index = np.arange(-h, h + 1) % grid_points
    columns, nodes = [], []
    for q, c in zip(dissipator.electronic_q_inv_ang, dissipator.electronic_weights, strict=True):
        harmonics = np.fft.fft2(np.sqrt(dissipator.smoothed_density(float(q))))
        harmonics /= harmonics.size
        columns.append(np.sqrt(c) * harmonics[np.ix_(index, index)])
        nodes.append(float(q))
    if not columns:
        columns, nodes = [np.zeros((2 * h + 1, 2 * h + 1))], [0.0]
    return np.array(columns), np.array(nodes)


def axial_fiber_inputs(case: AdmittedAxialCase) -> AxialFiberInputs:
    """Build the calibrated fibre-source inputs of ``case``.

    Validation: cr-axial-persisted-channel
    """
    from ...materials.channeling.axial import axial_profile, axial_states
    from ...materials.crystal import HBARC_EV_ANG, M_E_EV
    from .axial_dissipator import axial_dissipator
    from .axial_tds_continuum import TdsString
    from .bloch import longitudinal_kinematics
    from .operators import _potential

    if not isinstance(case, AdmittedAxialCase):
        raise TypeError("axial_fiber_inputs requires an admitted case")
    n = case.numerics
    profile = axial_profile(case.crystal, case.axis_uvw, temperature_K=case.temperature_K)
    u_min, u_max = (float(x) for x in profile.extrema())
    target = u_max + n.cut_multiple * (u_max - u_min)
    gamma, beta = longitudinal_kinematics(case.kinetic_energy_eV)
    k_inv_ang = gamma * beta * M_E_EV / HBARC_EV_ANG
    mechanisms = ("tds", "electronic") if n.electronic else ("tds",)
    dissipator = axial_dissipator(
        profile,
        case.kinetic_energy_eV,
        (1, 1),
        (n.grid_points, n.grid_points),
        mechanisms=mechanisms,
    )
    hbar_v = dissipator.hbar_v_eV_ang
    g = np.asarray(profile.g_inv_ang, dtype=float)
    m, box = n.max_order, n.outgoing_order_max
    span = box + m
    window = (m, m)
    strings = tuple(
        TdsString(
            tuple(src.position_ang),
            src.displacement_ang2,
            src.linear_density,
            (lambda element: lambda q: -_potential(element, q) / hbar_v)(src.element),
        )
        for src in dissipator.tds_sources
    )
    kicks, nodes = _kick_columns(dissipator, n.harmonic_max, n.grid_points)
    orders = axial_states(profile, gamma, (0.0, 0.0), window).orders

    def spectrum(kappa):
        states = axial_states(profile, gamma, kappa, window)
        return states.energies_eV, states.coefficients

    channels = FiberChannels(
        order_max=m,
        reciprocal_inv_ang=(float(g[0]), float(g[1])),
        spectrum=spectrum,
        cut=lambda energies: gap_cut(energies, target, _GAP_WINDOW_EV),
        lattice_columns=np.zeros((1, 2 * span + 1, 2 * span + 1)),
        lattice_loss=np.zeros((len(orders),) * 2),
        kick_columns=kicks,
        kick_q_inv_ang=nodes,
        outgoing_order_max=box,
        tds_strings=strings,
    )
    carrier, entry = _tilted_entry(case, k_inv_ang, g, orders)
    steps = int(round(case.thickness_ang / n.step_ang))
    return AxialFiberInputs(
        channels=channels,
        entry_orders=entry,
        hbar_v_eV_ang=hbar_v,
        carrier_inv_ang=carrier,
        boundary_eV=u_max,
        target_eV=target,
        depths_ang=np.linspace(0.0, case.thickness_ang, steps + 1),
        k_inv_ang=k_inv_ang,
    )


def _tilted_entry(case: AdmittedAxialCase, k_inv_ang: float, g: np.ndarray, orders: np.ndarray):
    """Folded carrier and entrance order of the plane wave tilted by ``case.tilt_deg``."""
    theta = np.deg2rad(case.tilt_deg)
    if theta == 0.0:
        return (0.0, 0.0), np.all(orders == 0, axis=1).astype(complex)
    phi = np.deg2rad(case.tilt_azimuth_deg)
    k_perp = k_inv_ang * np.sin(theta) * np.array([np.cos(phi), np.sin(phi)])
    order = np.floor(k_perp / g + 0.5).astype(int)
    carrier = k_perp - order * g
    entry = np.all(orders == order, axis=1).astype(complex)
    if entry.sum() != 1:
        raise ValueError("tilted entrance order lies outside the order window")
    return (float(carrier[0]), float(carrier[1])), entry
