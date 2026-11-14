"""Zone-row PXR/DCR expected-value source of planar channeling cases (#1166).

The tracked transverse states of a planar zone modulate PXR of every
reflection off the plane's own reciprocal row. Near a Bragg pole the
finite-slab photon mode also diffracts forward channeling radiation into the
detector: diffracted channeling radiation (DCR), up to ~20x PXR at the
(-1 -1 3) mirror of ``channeling_demo`` (#1148). The source
:func:`~pyrite.montecarlo.channeling.zone_pxr.binned_reflection_spectrum`
owns these reflections on zone rows: native line scoring drops them there
(:func:`.planar_zone_lines.planar_zone_line_parts`), as it drops the own row
(#1055), and keeps them on the rows outside the zone.

A reflection is source-owned towards ``n_hat`` when it radiates
(``v.g > 0`` with a physical exact-recoil root) in two-beam Laue geometry
well outside total external reflection: ``n.s`` and ``(k + g).s / |k + g|``
at the root both exceed ``3 sqrt(|chi_0|)`` (three critical angles; ``s`` the
inward face normal), where the slowly varying two-beam equations hold. Any
other reflection keeps native scoring on every row.

Validation: cr-planar-zone-pxr-modulation
"""

from __future__ import annotations

import numpy as np

_ROOT_RANGE_EV = (50.0, 966203.0)


def source_reflections(case, n_hat, beam=None):
    """``[(hkl, g_lab, E0_eV, chi_g, chi_0, u_g), ...]`` the zone source owns towards ``n_hat``.

    ``chi_0`` is returned in the ``exp(-i omega t)`` convention
    (``Im chi_0 > 0`` absorbs) that :mod:`..channeling.zone_pxr` uses;
    ``chi_g`` and ``u_g = U_g / m c^2`` are native line scoring's couplings
    (:func:`~pyrite.materials.crystal.emission_coupling_tables`, anomalous terms
    on), so the PXR--CBS phase matches native.

    ``beam`` overrides the case's mean beam direction (divergence nodes).

    Validation: cr-planar-zone-pxr-modulation, cr-planar-zone-row-ownership
    """
    from ...materials.crystal import (
        CRYSTALS,
        HBARC_EV_ANG,
        chi_0,
        emission_coupling_tables,
        reciprocal_g_vector,
    )
    from ...materials.recoil import recoil_resonance_eV
    from ..channeling.axial_case import _rotation
    from .planar_zone_lines import is_own_row

    record = case["channeling"]
    beam = _beam(case) if beam is None else np.asarray(beam, dtype=float)
    n = np.asarray(n_hat, dtype=float)
    n = n / np.linalg.norm(n)
    s_hat = np.array([0.0, 0.0, 1.0])
    if n @ s_hat <= 0.0:
        return []
    rotation = _rotation(case)
    lattice = CRYSTALS[case["crystal"]]["lattice"]
    out = []
    for hkl in (tuple(int(v) for v in h) for h in case["hkl_list"]):
        if is_own_row(hkl, record["plane_hkl"]):
            continue
        g0, _ = reciprocal_g_vector(hkl, lattice)
        g = rotation @ np.asarray(g0, dtype=float)
        if beam @ g <= 0.0:
            continue
        E0 = float(recoil_resonance_eV(float(case["E0_keV"]) * 1e3, beam, g, n))
        if not (np.isfinite(E0) and _ROOT_RANGE_EV[0] <= E0 <= _ROOT_RANGE_EV[1]):
            continue
        Q = E0 / HBARC_EV_ANG * n + g
        c0 = complex(chi_0(case["crystal"], E0))
        floor = 3.0 * np.sqrt(abs(c0))
        if n @ s_hat <= floor or Q @ s_hat <= floor * np.linalg.norm(Q):
            continue
        tables = emission_coupling_tables(
            case["crystal"], [hkl], np.array([E0]), case.get("B_ang2", 0.0), use_henke=True
        )
        cr, ci, ur, ui = (float(np.asarray(t).ravel()[0]) for t in tables)
        out.append((hkl, g, E0, complex(cr, ci), complex(c0.real, abs(c0.imag)), complex(ur, ui)))
    return out


def _beam(case):
    from ..geometry import tilted_geometry

    beam, _ = tilted_geometry(
        0.0, np.deg2rad(case.get("tilt_deg", 0.0)), np.deg2rad(case.get("tilt_azim_deg", 0.0))
    )
    beam = np.asarray(beam, dtype=float)
    return beam / np.linalg.norm(beam)


def entry_weights(case, solver, incidence_rad, divergence_rad):
    """``(n_cells, n_orders)`` plane-wave diagonal of the zone's entry density.

    The same entry :func:`~pyrite.montecarlo.channeling.emission.channeling_spectrum`
    starts from: the Gaussian-divergence grid of
    :meth:`~pyrite.montecarlo.channeling.density.DensityMatrixSolver.entry_lattice`,
    or the configured source's averaged plane density.

    Validation: cr-planar-zone-pxr-modulation
    """
    if case.get("channeling_source") is not None:
        from ..channeling.planar_injection import source_spec
        from ..channeling.states import planar_entry_density

        spec = source_spec(case["channeling_source"])
        plane, _ = planar_entry_density(
            solver, spec.build(), incidence_rad=incidence_rad, impact="averaged"
        )
        return np.einsum("rnn->rn", np.asarray(plane)).real
    lattice, weights = solver.entry_lattice(incidence_rad, divergence_rad)
    residue = lattice % solver.n_cells
    order = (lattice - residue) // solver.n_cells
    inband = np.abs(order) <= solver.max_order
    out = np.zeros((solver.n_cells, np.asarray(solver.orders).size))
    np.add.at(out, (residue[inband], order[inband] + solver.max_order), weights[inband])
    return out


def zone_states(solver, path_populations, path_ang, entrance=None):
    """:class:`~pyrite.montecarlo.channeling.zone_pxr.ZoneStates` of a solved planar zone.

    ``path_populations`` is ``int dz <i|rho|i>`` (angstrom per electron);
    the source uses its path mean over ``path_ang``. The potential matrix is
    the solver Hamiltonian ``V diag(E) V^dagger`` minus its kinetic diagonal.
    ``entrance`` is :func:`entry_weights`.

    Validation: cr-planar-zone-pxr-modulation
    """
    from ...materials.crystal import HBARC_EV_ANG, M_E_EV
    from ..channeling.zone_pxr import ZoneStates

    basis = solver.basis
    energies = np.asarray(basis.energies, dtype=float)
    vectors = np.asarray(basis.vectors, dtype=complex)
    orders = np.asarray(solver.orders)
    kappa = np.asarray(solver.kappa, dtype=float)
    period = float(solver.profile.period_ang)
    gamma = float(basis.gamma)
    C = HBARC_EV_ANG**2 / (2.0 * gamma * M_E_EV)
    G = 2.0 * np.pi / period
    potentials = np.stack(
        [
            (vectors[r] * energies[r]) @ vectors[r].conj().T
            - np.diag(C * (kappa[r] + orders * G) ** 2)
            for r in range(kappa.size)
        ]
    )
    return ZoneStates(
        potentials=potentials,
        energies=energies,
        vectors=vectors,
        populations=np.clip(np.asarray(path_populations, dtype=float), 0.0, None) / path_ang,
        orders=orders,
        kappa=kappa,
        period_ang=period,
        gamma=gamma,
        entrance=entrance,
    )


def case_zone_pxr_spectrum(
    case,
    solver,
    path_populations,
    path_ang,
    E_grid,
    n_hat,
    *,
    in_plane_rms_rad=0.0,
    incidence_rad=0.0,
    divergence_rad=0.0,
) -> np.ndarray:
    """Zone-row PXR+DCR of ``case`` towards ``n_hat`` [photons/eV/sr/incident electron].

    Averages a circular detector cone of half angle ``dtheta_obs_rad`` and
    the in-plane beam divergence (rotation about the plane normal, which
    leaves the zone states unchanged) by Gauss quadrature, like the CR
    source. ``chi_g`` and ``chi_0`` (absorption, refraction) are taken at
    each reflection's root by :func:`source_reflections`. Ownership is decided
    once, at ``n_hat`` and the mean beam, exactly as native line scoring drops
    it (:func:`.planar_zone_lines.planar_zone_line_parts`): a node scores only
    reflections owned there, and an owned reflection without a Laue root at
    some node is skipped at that node. ``incidence_rad`` and
    ``divergence_rad`` give the entry (:func:`entry_weights`).

    Validation: cr-planar-zone-pxr-modulation
    """
    from ..channeling.emission import _direction_nodes
    from ..channeling.handoff import case_plane_incidence
    from ..channeling.zone_pxr import SlabGeometry, binned_reflection_spectrum

    grid = np.asarray(E_grid, dtype=float)
    out = np.zeros(grid.size)
    if path_populations is None:  # a source without a population history
        return out
    entrance = entry_weights(case, solver, incidence_rad, divergence_rad)
    states = zone_states(solver, path_populations, path_ang, entrance)
    if not np.any(states.populations > 0.0):
        return out
    x_hat = np.asarray(case_plane_incidence(case).plane_normal, dtype=float)
    x_hat = x_hat / np.linalg.norm(x_hat)
    n0 = np.asarray(n_hat, dtype=float)
    n0 = n0 / np.linalg.norm(n0)
    quadrature = case.get("line_quadrature", "node")
    radius, chi, weights = _direction_nodes(float(case.get("dtheta_obs_rad", 0.0)), 6)
    t1 = np.cross(n0, x_hat if abs(n0 @ x_hat) < 0.9 else np.array([0.0, 1.0, 0.0]))
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n0, t1)
    if in_plane_rms_rad > 0.0:
        nodes, psi_weight = np.polynomial.hermite_e.hermegauss(5)
        psi, psi_weight = in_plane_rms_rad * nodes, psi_weight / psi_weight.sum()
    else:
        psi, psi_weight = np.zeros(1), np.ones(1)
    beam0 = _beam(case)
    owned = {row[0] for row in source_reflections(case, n0)}
    for r, c, wa in zip(radius, chi, weights, strict=True):
        n = np.cos(r) * n0 + np.sin(r) * (np.cos(c) * t1 + np.sin(c) * t2)
        for angle, wp in zip(psi, psi_weight, strict=True):
            beam = _rotate(beam0, x_hat, angle)
            geometry = SlabGeometry(beam, x_hat, np.array([0.0, 0.0, 1.0]), float(path_ang))
            for hkl, g, E0, cg, c0, ug in source_reflections(case, n, beam):
                if hkl not in owned or E0 < grid[0] - 1500.0 or E0 > grid[-1] + 1500.0:
                    continue
                out += (wa * wp) * binned_reflection_spectrum(
                    states, geometry, g, n, E0, grid, cg, c0, u_g=ug, quadrature=quadrature
                )
    return out


def _rotate(v, axis, angle):
    """Rodrigues rotation of ``v`` about unit ``axis``."""
    return (
        v * np.cos(angle)
        + np.cross(axis, v) * np.sin(angle)
        + axis * (axis @ v) * (1.0 - np.cos(angle))
    )
