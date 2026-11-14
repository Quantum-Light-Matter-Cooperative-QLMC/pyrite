"""Diffracted bremsstrahlung (DBR) at Bragg-angle detectors (#1166).

Validation: diffracted-bremsstrahlung
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.linalg import expm

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum import diffracted_brem as db

S_HAT = np.array([0.0, 0.0, 1.0])
N_HAT = np.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
G = np.array([-2.4, 0.0, 0.35])  # n.g < 0; Q = k n + g points near +z at E_B
E_B = -HBARC_EV_ANG * (G @ G) / (2.0 * N_HAT @ G)


def _polarizations(E):
    k = E / HBARC_EV_ANG
    Q = k * N_HAT + G
    Qh = Q / np.linalg.norm(Q)
    es = np.cross(N_HAT, G)
    es /= np.linalg.norm(es)
    for e in (es, np.cross(es, N_HAT)):
        yield k, Q, np.linalg.norm(e - Qh * (Qh @ e))


def test_geometry_is_laue_near_the_beam():
    Q = E_B / HBARC_EV_ANG * N_HAT + G
    assert abs(np.linalg.norm(Q) - E_B / HBARC_EV_ANG) < 1e-9 * np.linalg.norm(Q)
    assert Q[2] > 0.5 * np.linalg.norm(Q) and N_HAT[2] > 0.5


def test_diffracted_intensity_solves_the_two_beam_mode():
    chi_g, chi_0 = 3e-5, complex(-2e-5, 4e-7)
    E = E_B + np.array([-3.0, 0.0, 0.7, 4.0])
    u = np.array([0.0, 3000.0, 20000.0])
    got = db.laue_diffracted_intensity(E, N_HAT, G, S_HAT, chi_g, chi_0, u)
    ns = N_HAT @ S_HAT
    want = np.zeros_like(got)
    for j, e in enumerate(E):
        for k, Q, C in _polarizations(e):
            Qs = Q @ S_HAT
            q = -(G @ G + 2 * k * (N_HAT @ G)) / (2 * Qs)
            a = k**2 * chi_g * C / (2 * k * ns)
            b = k**2 * chi_g * C / (2 * Qs)
            N = np.array([[k * chi_0 / (2 * ns), a], [b, q + k**2 * chi_0 / (2 * Qs)]])
            for i, depth in enumerate(u):
                D = (expm(1j * N * depth) @ np.array([1.0, 0.0]))[1]
                want[i, j] += 0.5 * abs(D) ** 2
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-30)


@pytest.mark.parametrize("depth", [2000.0, 20000.0])
def test_kinematic_band_is_linear_in_the_remaining_depth(depth):
    """``int dE |D|^2 = |b|^2 2 pi u hbar c Q_s / |n.g|`` (kinematic Laue reflectivity)."""
    chi_g = 1e-8
    E = E_B + np.arange(-600.0, 600.0, 0.002)
    band = np.trapezoid(db.laue_diffracted_intensity(E, N_HAT, G, S_HAT, chi_g, 0.0, [depth]), E)
    want = 0.0
    for k, Q, C in _polarizations(E_B):
        Qs = Q @ S_HAT
        b2 = k**4 * chi_g**2 * C**2 / (4 * Qs**2)
        want += 0.5 * b2 * 2 * np.pi * depth * HBARC_EV_ANG * Qs / abs(N_HAT @ G)
    assert band[0] == pytest.approx(want, rel=2e-3)


def test_line_bands_are_conserved_by_bin_mean_and_sampled_by_nodes():
    edges = np.linspace(0.0, 20000.0, 9)
    weight = np.linspace(1.0, 2.0, 8)
    E, S = db.reflection_line(weight, edges, [E_B], N_HAT, G, S_HAT, E_B, 3e-5, 0.0)
    grid = np.arange(E[0] - 10.0, E[-1] + 10.0, 2.5)  # coarse bins over the whole line
    band = np.trapezoid(S, E)
    binned = db.deposit(E, S, grid, "bin-mean")
    assert binned.sum() * 2.5 == pytest.approx(band, rel=1e-9)
    nodes = db.deposit(E, S, grid, "node")
    np.testing.assert_allclose(nodes, np.interp(grid, E, S, left=0.0, right=0.0), rtol=0, atol=0)


@pytest.fixture(scope="module")
def plain():
    from pyrite.montecarlo import runner
    from tests.montecarlo.channeling.test_planar_zone_lines import _case

    case = _case(channeling=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        transport = runner._transport_case(case, transport_core="per-electron")
    return case, transport


def _mirror(case, hkl):
    from pyrite.materials.crystal import CRYSTALS, _orientation_R, reciprocal_g_vector
    from pyrite.montecarlo.runner.planar_zone_pxr import _beam

    lattice = CRYSTALS[case["crystal"]]["lattice"]
    R = _orientation_R(
        lattice,
        case.get("beam_uvw"),
        case.get("azimuth_rad", 0.0),
        case.get("recip_miscut_rad"),
        surface_hkl=case.get("surface_hkl"),
    )
    R = np.eye(3) if R is None else R
    g = R @ np.asarray(reciprocal_g_vector(hkl, lattice)[0], dtype=float)
    beam = _beam(case)
    return g, beam - 2 * (beam @ g) * g / (g @ g)


@pytest.mark.parametrize("hkl", [(1, 1, -3), (-1, 3, 1)])
def test_kinematic_dbr_matches_forward_brem_times_integrated_reflectivity(plain, hkl):
    """Independent estimate: native brem toward ``Q_hat`` x kinematic ``R_E`` x mean depth 1/2.

    Kinematic and non-absorbing in the scorer (the estimate has neither
    Pendelloesung nor two-beam absorption, but keeps its escape factor),
    at mirror rays with ``Q_hat`` within the brem cone.
    """
    from pyrite.materials.crystal import chi_g as chi_g_fn
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.spectrum.brem import mc_brem_spectrum

    case, transport = plain
    segs = transport["segs"]
    grid = np.asarray(transport["E_grid"], dtype=float)
    g, n = _mirror(case, hkl)
    E = -HBARC_EV_ANG * (g @ g) / (2 * n @ g)
    Qh = E / HBARC_EV_ANG * n + g
    Qh /= np.linalg.norm(Qh)
    common = dict(
        model=case["bremsstrahlung_model"],
        tables=runner._case_bremslib_tables(case),
        Ne=int(transport["Ne_lines"]),
        E_cut_keV=case.get("E_cut_brem_keV", 1.0),
    )
    forward = mc_brem_spectrum(
        segs,
        np.array([E, E + 1.0]),
        composition=case["composition"],
        n_hat=Qh,
        electron_limit=common["Ne"],
        E_cut_keV=common["E_cut_keV"],
        cross_section_model=common["model"],
        bremslib_tables=common["tables"],
    )[0]
    chi = abs(chi_g_fn("silicon", hkl, E, case["B_ang2"]))
    theta = np.arcsin(HBARC_EV_ANG * np.linalg.norm(g) / (2 * E))
    k = E / HBARC_EV_ANG
    pol = 0.5 * (1 + np.cos(2 * theta) ** 2)
    R_theta = np.pi * k * chi**2 * case["thickness_ang"] / (2 * Qh[2] * np.sin(2 * theta)) * pol
    estimate = forward * E * R_theta / np.tan(theta) / 2
    tiny = 1e-3  # kinematic: scale chi_g down, the band back up by 1/tiny^2
    dbr = db.diffracted_brem_spectrum(
        segs, grid, [(g, chi * tiny, 0j)], n, case["composition"], case["thickness_ang"],
        quadrature="bin-mean", **common,
    )  # fmt: skip
    band = dbr.sum() * (grid[1] - grid[0]) / tiny**2
    assert band == pytest.approx(estimate, rel=0.03)


def test_spectrum_adds_dbr_to_the_line_spectrum_with_channeling_off(plain, monkeypatch):
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import diffracted_brem

    case, transport = plain
    _, n = _mirror(case, (1, 1, -3))
    tp = {**transport, "n_hat": n}
    with_dbr = np.asarray(runner._spectrum_case(case, tp)["spec"])
    dbr = diffracted_brem.case_diffracted_brem(
        case, transport["segs"], transport["E_grid"], n, transport["Ne_brem"]
    )
    assert dbr.max() > 0.0
    monkeypatch.setattr(diffracted_brem, "case_diffracted_brem", lambda *a: np.zeros(dbr.size))
    lines = np.asarray(runner._spectrum_case(case, tp)["spec"])
    np.testing.assert_allclose(with_dbr - lines, dbr, rtol=1e-9, atol=1e-12 * dbr.max())


def test_out_of_claim_cases_add_nothing(plain):
    from pyrite.montecarlo.runner.diffracted_brem import case_diffracted_brem

    case, transport = plain
    _, n = _mirror(case, (1, 1, -3))
    grid = np.asarray(transport["E_grid"], dtype=float)
    for override in ({"mosaic_fwhm_rad": 1e-3}, {"hkl_list": []}, {"abs_layers": [(0, 1, [])]}):
        mapping = {**case.to_dict(), **override}
        dbr = case_diffracted_brem(mapping, transport["segs"], grid, n, transport["Ne_brem"])
        assert not np.any(dbr)
    backward = -n
    assert not np.any(case_diffracted_brem(case, transport["segs"], grid, backward, None))


def test_thick_absorbing_grazing_laue_stays_finite():
    """Strong asymmetric absorption over a millimetre: the eigenvalue branch must not overflow.

    The principal root gave NaN here (fresh-context verifier, #1166).
    """
    n = np.array([np.sqrt(1 - 0.05**2), 0.0, 0.05])  # grazing exit, above 3 sqrt(|chi_0|)
    E = E_B + np.linspace(-5.0, 5.0, 11)
    got = db.laue_diffracted_intensity(E, n, G, S_HAT, 3e-6, complex(-2e-5, 2e-5), [1e5, 1e6, 1e7])
    assert np.all(np.isfinite(got)) and np.all(got >= 0.0)
