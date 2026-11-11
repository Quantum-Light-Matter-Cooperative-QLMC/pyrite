r"""Carrier-fibred absorbing unraveling of the planar problem (#1056).

Research CPU reference. The planar Lindblad model of :mod:`.operators`
(independent Einstein TDS per atom and local-density electronic kicks, both
with ``q_y`` integrated) is the axial fibre model of
:mod:`.axial_fiber_unraveling` on a rectangular cell ``(d, L_y)`` whose
``y`` direction is a continuum:

* every crystallographic plane site ``s`` (position ``x_s`` in the period,
  volume density ``n_s = 1 / V_cell`` per basis atom) is a row of independent
  strings at spacing ``L_y`` with linear density ``lambda_s = n_s d L_y``,
  so the string channel's ``lambda_s / A = n_s`` is independent of ``L_y``;
* the potential, the electron density and therefore ``H(kappa)`` and the
  kick harmonics have only ``n_y = 0`` orders;
* ``g_y = 2 pi / L_y`` exceeds twice every representable transfer
  (``q_max`` of the strings and kicks) plus any accumulated ``kappa_y``, so
  no output ever folds in ``y`` and ``n_y != 0`` harmonics of a jump vanish
  identically (the transfer cut ``|t| <= q_max``).

Then the carrier's ``y`` component is the accumulated out-of-plane momentum,
a label that never interferes (fibres are block diagonal), and
``H(kappa) = H_x(kappa_x)`` drops the constant ``hbar^2 kappa_y^2 / 2 gamma m``
(a fibre-global phase). Window and outgoing box are ``(M, 0)`` and
``(B, 0)``. Loss: ``Gamma_ab = n_s e^{i(g_b-g_a)x_s} D(|g_a-g_b|) / (2 pi)^2``
per site, the full-transfer ``C(x, x)`` Toeplitz of
:func:`.operators.tds_kernel` for ``S = a^2 I``; kicks reproduce
:func:`.operators.electronic_kernel`'s ``C_el(x, x)``.

Assumptions as :mod:`.operators` (first Born, Einstein isotropic Gaussian
displacements, scalar local-density electronic response) plus a fixed
energy (no stopping) and the absorbing cut ``E_perp < barrier + margin``
of :class:`.density.DensityMatrixSolver`. Isotropic sites only; directional
valence is refused.

:func:`planar_on_axis_source` is the on-axis two-time photon source of this
model (the companion estimator of ``cr-axial-fiber-regression``).

Validation: cr-planar-fiber-unraveling, cr-planar-fiber-regression
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...materials.channeling.potential import PlanarProfile
from ...materials.crystal import CRYSTALS, HBARC_EV_ANG, M_E_EV, reciprocal_g_vector
from .axial_fiber_regression import FiberOnAxisSource, fiber_on_axis_source
from .axial_fiber_unraveling import FiberChannels
from .axial_tds_continuum import TdsString
from .bloch import _hamiltonian, longitudinal_kinematics
from .operators import (
    _Q_MAX_INV_ANG,
    _electronic_nodes,
    _potential,
    _profile_oscillators,
    _transverse_covariances,
    electronic_channels,
)

#: ``g_y`` [1/Å] of the continuum direction: far above ``2 q_max`` (1e4 Å⁻¹)
#: and any accumulated ``kappa_y``, so nothing folds in ``y``.
Y_CONTINUUM_INV_ANG = 1.0e9


@dataclass(frozen=True)
class PlaneSite:
    """One basis atom's plane: element, position ``x_s`` [Å] in ``[0, d)``,
    volume density [Å⁻³] and in-plane mean-square displacement per axis [Å²]."""

    element: str
    position_ang: float
    density_inv_ang3: float
    displacement_ang2: float


@dataclass(frozen=True)
class PlanarFiberInputs:
    """Channels, entrance amplitude and kinematics of one planar fibre model.

    ``orders`` are the window orders ``-M..M``; ``entry_orders`` the
    entrance plane wave on them at ``carrier_inv_ang``; ``k_inv_ang`` the
    longitudinal wavenumber ``p / hbar``; ``barrier_eV`` and ``cutoff_eV``
    the planar barrier and the absorbing cut; ``kinetic_energy_eV`` the
    fixed electron kinetic energy.
    """

    channels: FiberChannels
    orders: np.ndarray
    entry_orders: np.ndarray
    carrier_inv_ang: tuple[float, float]
    hbar_v_eV_ang: float
    k_inv_ang: float
    gamma: float
    barrier_eV: float
    cutoff_eV: float
    sites: tuple[PlaneSite, ...]
    kinetic_energy_eV: float


def plane_sites(profile: PlanarProfile) -> tuple[PlaneSite, ...]:
    """Plane sites of the catalog basis in the convention of ``_site_densities``.

    ``n_h = sum_s n_s e^{-i h g x_s}`` reproduces
    :func:`.operators._site_densities` with ``x_s = (hkl . r_s) 2 pi / g_coprime``.

    Validation: cr-planar-fiber-unraveling
    """
    if profile.crystal not in CRYSTALS:
        raise ValueError(f"profile crystal {profile.crystal!r} is not a catalog crystal")
    info = CRYSTALS[profile.crystal]
    _, g_coprime = reciprocal_g_vector(profile.hkl, info["lattice"])
    covariances = _transverse_covariances(profile)
    d = profile.period_ang
    step = profile.g_inv_ang / g_coprime
    if not np.isclose(step, round(step), rtol=0, atol=1e-9):
        raise ValueError("profile period is not an integer fraction of the coprime spacing")
    merged: dict[tuple[str, float], float] = {}
    for element, position in info["basis"]:
        sxx, sxy, syy = covariances[element]
        if sxy != 0.0 or not np.isclose(sxx, syy, rtol=1e-12):
            raise ValueError("planar fibre strings need isotropic transverse displacements")
        # x_s / d = (hkl . r_s) g / g_coprime (integer, as _site_densities); snap the fraction so coincident
        # planes (and x_s ~ d) merge exactly.
        fraction = float(np.dot(profile.hkl, position)) * round(step)
        x = (round(fraction % 1.0, 12) % 1.0) * d
        merged[(element, x)] = merged.get((element, x), 0.0) + 1.0 / float(info["V_cell"])
    return tuple(
        PlaneSite(element, x, density, covariances[element][0])
        for (element, x), density in sorted(merged.items(), key=lambda item: item[0][1])
    )


def _strings(profile: PlanarProfile, sites, hbar_v: float) -> tuple[TdsString, ...]:
    """One string row per plane site: ``lambda_s = n_s d L_y`` so ``lambda_s / A = n_s``.

    Validation: cr-planar-fiber-unraveling
    """
    area_per_d = 2.0 * np.pi / Y_CONTINUUM_INV_ANG  # L_y
    return tuple(
        TdsString(
            (site.position_ang, 0.0),
            site.displacement_ang2,
            site.density_inv_ang3 * profile.period_ang * area_per_d,
            (lambda element: lambda q: -_potential(element, q) / hbar_v)(site.element),
            _Q_MAX_INV_ANG,
        )
        for site in sites
    )


def root_density_harmonics(profile: PlanarProfile, q_inv_ang: float, half: int) -> np.ndarray:
    """Harmonics ``-half..half`` of ``sqrt(rho_q(x))``, ``rho_q`` smoothed at ``1/q``.

    Same smoothing as :func:`.operators._root_density`, evaluated at ``q``.

    Validation: cr-planar-fiber-unraveling
    """
    g = profile.g_inv_ang
    rho = profile.electron_density
    n_x = max(1024, 8 * half)
    x = np.arange(n_x) * profile.period_ang / n_x
    harmonics = np.arange(rho.size) * g
    smooth = np.exp(-0.5 * (harmonics / q_inv_ang) ** 2) * rho
    density = smooth[0].real + 2.0 * (np.exp(1j * np.outer(x, harmonics[1:])) @ smooth[1:]).real
    spectrum = np.fft.fft(np.sqrt(np.maximum(density, 0.0))) / n_x
    return spectrum[np.arange(-half, half + 1) % n_x]


def _kicks(profile: PlanarProfile, energy_eV: float, half: int):
    """Kick columns ``sqrt(c_i)`` times ``sqrt(rho_q_i)`` harmonics at the electronic nodes.

    Validation: cr-planar-fiber-unraveling
    """
    channels = electronic_channels(_profile_oscillators(profile, None), energy_eV)
    q, c = _electronic_nodes(channels)
    live = c > 0.0
    q, c = q[live], c[live]
    columns = np.array(
        [
            np.sqrt(ci) * root_density_harmonics(profile, float(qi), half)
            for qi, ci in zip(q, c, strict=True)
        ]
    )
    return columns[:, :, None], q


def planar_fiber_inputs(
    profile: PlanarProfile,
    kinetic_energy_eV: float,
    *,
    order_max: int,
    outgoing_order_max: int | None = None,
    margin_eV: float = 300.0,
    mechanisms: tuple[str, ...] = ("tds", "electronic"),
    kick_half: int | None = None,
    incidence_rad: float = 0.0,
) -> PlanarFiberInputs:
    """Build the planar fibre channels of ``profile`` at a fixed energy.

    ``order_max`` = M sets the window ``-M..M``, ``outgoing_order_max`` = B
    (default M) the outgoing box. The retained cut is ``barrier + margin_eV``,
    as :class:`.density.DensityMatrixSolver`. ``kick_half`` (default
    ``B + M``) bounds the kick harmonics; it must cover the electron-density
    harmonics that matter. ``incidence_rad`` tilts the entrance plane wave
    (paraxial ``k_x = k psi``), folded into the zone.

    Validation: cr-planar-fiber-unraveling
    """
    if order_max < 1:
        raise ValueError("order_max must be positive")
    box = order_max if outgoing_order_max is None else int(outgoing_order_max)
    if box < order_max:
        raise ValueError("outgoing_order_max must be >= order_max")
    unknown = set(mechanisms) - {"tds", "electronic"}
    if unknown:
        raise ValueError(f"unknown mechanisms {sorted(unknown)}")
    gamma, beta = longitudinal_kinematics(kinetic_energy_eV)
    hbar_v = HBARC_EV_ANG * beta
    k_inv_ang = gamma * beta * M_E_EV / HBARC_EV_ANG
    g = profile.g_inv_ang
    orders = np.arange(-order_max, order_max + 1)
    _, barrier = profile.extrema()
    cutoff = float(barrier) + float(margin_eV)
    sites = plane_sites(profile)
    strings = _strings(profile, sites, hbar_v) if "tds" in mechanisms else ()
    half = box + order_max if kick_half is None else int(kick_half)
    if "electronic" in mechanisms:
        kicks, nodes = _kicks(profile, kinetic_energy_eV, half)
    else:
        kicks, nodes = np.zeros((1, 1, 1)), np.zeros(1)

    def spectrum(kappa):
        energies, vectors = np.linalg.eigh(_hamiltonian(profile, gamma, float(kappa[0]), orders))
        return energies, vectors

    span = box + order_max
    channels = FiberChannels(
        order_max=(order_max, 0),
        reciprocal_inv_ang=(g, Y_CONTINUUM_INV_ANG),
        spectrum=spectrum,
        cut=lambda energies: cutoff,
        lattice_columns=np.zeros((1, 2 * span + 1, 1)),
        lattice_loss=np.zeros((orders.size, orders.size)),
        kick_columns=kicks,
        kick_q_inv_ang=nodes,
        outgoing_order_max=(box, 0),
        tds_strings=strings,
    )
    k_x = k_inv_ang * float(incidence_rad)
    order = int(np.floor(k_x / g + 0.5))
    if abs(order) > order_max:
        raise ValueError("entrance order lies outside the order window")
    entry = (orders == order).astype(complex)
    return PlanarFiberInputs(
        channels=channels,
        orders=orders,
        entry_orders=entry,
        carrier_inv_ang=(k_x - order * g, 0.0),
        hbar_v_eV_ang=hbar_v,
        k_inv_ang=k_inv_ang,
        gamma=gamma,
        barrier_eV=float(barrier),
        cutoff_eV=cutoff,
        sites=sites,
        kinetic_energy_eV=float(kinetic_energy_eV),
    )


def planar_on_axis_source(
    inputs: PlanarFiberInputs,
    *,
    depths_ang,
    photon_energy_eV,
    incident_ids,
    seed: int = 0,
    absorption_inv_ang=None,
    exit_path_ang=None,
    search_step_ang: float = 100.0,
    max_working_bytes: int = 256 * 1024**2,
) -> FiberOnAxisSource:
    r"""On-axis two-time photon source of the planar fibre model.

    :func:`.axial_fiber_regression.fiber_on_axis_source` on the channels of
    ``inputs``, with lowering currents ending on bound states
    (``boundary_eV`` = the planar barrier): bound -> bound lines plus
    capture from retained above-barrier states, the partition of
    :func:`.emission.channeling_spectrum`. No secular approximation, line
    shape or capture clustering is assumed: line positions, widths, overlaps,
    cross terms between transitions and capture all follow from
    $G(z_i,z_j)=\mathrm{Tr}[J_x\Lambda_{z_i-z_j}(\rho(z_j)J_x^\dagger)]$.

    Planar currents: the window orders are ``(n, 0)`` and the carrier's
    ``y`` component is the accumulated out-of-plane momentum, so
    $J_y=\hbar\kappa_y/(\gamma mc)\,\mathbb 1$ in every fibre and its
    positive-frequency part vanishes; only $J_x=\hbar(\kappa_x+ng)/(\gamma mc)$
    radiates, with $\langle f|J_x|i\rangle=i(E_f-E_i)x_{fi}/\hbar c$
    (``cr-dipole-matrix-element``). The fibre-global
    $\hbar^2\kappa_y^2/2\gamma m$ dropped from $H(\kappa)$ cancels in every
    record (companion and primary share the fibre). Observation is along the
    beam axis; the electron's own out-of-plane angle does not shift the
    Doppler factor (as :func:`.emission.channeling_spectrum` at
    ``theta = 0`` with no divergence).

    Units: eV, Å; photons/(eV sr incident electron).

    Validation: cr-planar-fiber-regression
    """
    return fiber_on_axis_source(
        inputs.channels,
        inputs.entry_orders,
        kinetic_energy_eV=inputs.kinetic_energy_eV,
        hbar_v_eV_ang=inputs.hbar_v_eV_ang,
        depths_ang=depths_ang,
        photon_energy_eV=photon_energy_eV,
        incident_ids=incident_ids,
        seed=seed,
        carrier_inv_ang=inputs.carrier_inv_ang,
        boundary_eV=inputs.barrier_eV,
        absorption_inv_ang=absorption_inv_ang,
        exit_path_ang=exit_path_ang,
        search_step_ang=search_step_ang,
        max_working_bytes=max_working_bytes,
    )
