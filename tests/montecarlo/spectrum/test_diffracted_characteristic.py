"""Kossel lines: two-beam Laue diffraction of characteristic radiation (#1176).

Validation: diffracted-characteristic-radiation
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.linalg import expm

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum import diffracted_brem as db
from pyrite.montecarlo.spectrum import diffracted_characteristic as dc

S_HAT = np.array([0.0, 0.0, 1.0])
N_HAT = np.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
G = np.array([-2.4, 0.0, 0.35])  # asymmetric Laue: Q_hat.s != n.s
E_B = -HBARC_EV_ANG * (G @ G) / (2.0 * N_HAT @ G)


def _modes(E, chi_g, chi_0):
    """Reference ``P(u) = expm(i N u) (1, 0)`` per polarization, from the matrix ``N``."""
    k = E / HBARC_EV_ANG
    Q = k * N_HAT + G
    Qh = Q / np.linalg.norm(Q)
    ns, Qs = N_HAT @ S_HAT, Q @ S_HAT
    es = np.cross(N_HAT, G)
    es /= np.linalg.norm(es)
    q = -(G @ G + 2 * k * (N_HAT @ G)) / (2 * Qs)
    for e in (es, np.cross(es, N_HAT)):
        C = np.linalg.norm(e - Qh * (Qh @ e))
        a = k**2 * chi_g * C / (2 * k * ns)
        b = k**2 * chi_g * C / (2 * Qs)
        yield np.array([[k * chi_0 / (2 * ns), a], [b, q + k**2 * chi_0 / (2 * Qs)]])


def test_transmitted_intensity_solves_the_two_beam_mode():
    chi_g, chi_0 = 3e-5, complex(-2e-5, 4e-7)
    E = E_B + np.array([-3.0, 0.0, 0.7, 4.0])
    u = np.array([0.0, 3000.0, 20000.0])
    T2, D2 = db.laue_mode_intensities(E, N_HAT, G, S_HAT, chi_g, chi_0, u)
    want_T, want_D = np.zeros_like(T2), np.zeros_like(D2)
    for j, e in enumerate(E):
        for N in _modes(e, chi_g, chi_0):
            for i, depth in enumerate(u):
                P = expm(1j * N * depth) @ np.array([1.0, 0.0])
                want_T[i, j] += 0.5 * abs(P[0]) ** 2
                want_D[i, j] += 0.5 * abs(P[1]) ** 2
    np.testing.assert_allclose(T2, want_T, rtol=1e-8)
    np.testing.assert_allclose(D2, want_D, rtol=1e-8, atol=1e-30)


def test_standing_wave_cross_term_solves_the_mode_with_a_complex_coupling():
    """``Re(phi X)`` equals ``(1/2) sum_s 2 C_s Re(conj(T) D e^{i psi})`` for complex ``chi_g``."""
    chi, psi, chi_0 = 3e-5 * np.exp(0.7j), 1.1, complex(-2e-5, 4e-7)
    E = E_B + np.array([-3.0, 0.0, 0.7, 4.0])
    u = np.array([0.0, 3000.0, 20000.0])
    _, _, X = db.laue_mode_intensities(E, N_HAT, G, S_HAT, abs(chi), chi_0, u, cross=True)
    got = np.real(np.exp(1j * (np.angle(chi) + psi)) * X)
    want = np.zeros_like(got)
    for j, e in enumerate(E):
        k = e / HBARC_EV_ANG
        Q = k * N_HAT + G
        Qh = Q / np.linalg.norm(Q)
        ns, Qs = N_HAT @ S_HAT, Q @ S_HAT
        q = -(G @ G + 2 * k * (N_HAT @ G)) / (2 * Qs)
        es = np.cross(N_HAT, G)
        es /= np.linalg.norm(es)
        for pol in (es, np.cross(es, N_HAT)):
            C = np.linalg.norm(pol - Qh * (Qh @ pol))
            a = k**2 * np.conj(chi) * C / (2 * k * ns)
            b = k**2 * chi * C / (2 * Qs)
            N = np.array([[k * chi_0 / (2 * ns), a], [b, q + k**2 * chi_0 / (2 * Qs)]])
            for i, depth in enumerate(u):
                T, D = expm(1j * N * depth) @ np.array([1.0, 0.0])
                want[i, j] += C * np.real(np.conj(T) * D * np.exp(1j * psi))
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-14)


def test_kinematic_cross_term_band_is_pi_times_the_coupling():
    """Wide line, no absorption: ``int dE Im X = pi sum_s C_s b_s hbar c Q_s/|n.g|``.

    From ``D = b (e^{iqu} - 1)/q``: ``int dq sin(qu)/q = pi``, while
    ``(cos qu - 1)/q`` is odd, so ``Re X`` integrates to a small window
    residual. The band of ``Re(phi X) = Re(phi) Re X - Im(phi) Im X`` is
    therefore ``-pi Im(phi)`` times the coupling, and a centrosymmetric (real)
    emitter phase leaves only a dispersive profile.
    """
    chi_g, depth = 1e-8, 200000.0
    E = E_B + np.arange(-30.0, 30.0, 0.002)  # sin(qu)/q truncation ~ 1/(q u) < 1e-3
    _, _, X = db.laue_mode_intensities(E, N_HAT, G, S_HAT, chi_g, 0j, [depth], cross=True)
    k = E_B / HBARC_EV_ANG
    Q = k * N_HAT + G
    Qh = Q / np.linalg.norm(Q)
    es = np.cross(N_HAT, G)
    es /= np.linalg.norm(es)
    coupling = 0.0
    for pol in (es, np.cross(es, N_HAT)):
        C = np.linalg.norm(pol - Qh * (Qh @ pol))
        coupling += C * k**2 * chi_g * C / (2 * Q @ S_HAT)
    want = np.pi * coupling * HBARC_EV_ANG * (Q @ S_HAT) / abs(N_HAT @ G)
    assert np.trapezoid(X[0].imag, E) == pytest.approx(want, rel=3e-3)
    assert abs(np.trapezoid(X[0].real, E)) < 0.02 * want  # O(dE / E_B) window residual


def test_emitter_phases_follow_the_sublattice():
    ge = dc.emitter_phases("ge", (-4, 0, 0), -1.0 + 0.0j, B_ang2=0.0)
    assert ge == {"Ge": pytest.approx(-1.0 + 0.0j)}
    gaas = dc.emitter_phases("gaas", (2, 0, 0), 1.0 + 0.0j, B_ang2=0.5)
    assert gaas["Ga"] == pytest.approx(-gaas["As"]) and abs(gaas["Ga"].imag) < 1e-12
    assert 0.0 < gaas["Ga"].real < 1.0  # Debye--Waller coherent fraction
    assert dc.emitter_phases("ge", (-4, 0, 0), 0j) == {}


def test_no_coupling_is_plain_attenuation_and_no_kossel_term():
    """``chi_g -> 0``: ``|T|^2 = exp(-k Im chi_0 u / n_s)``, ``D = 0``, Kossel term 0."""
    chi_0 = complex(-2e-5, 4e-7)
    E = E_B + np.linspace(-20.0, 20.0, 9)
    u = np.array([0.0, 5000.0, 40000.0])
    T2, D2 = db.laue_mode_intensities(E, N_HAT, G, S_HAT, 0.0, chi_0, u)
    k = E / HBARC_EV_ANG
    np.testing.assert_allclose(T2, np.exp(-k * chi_0.imag * u[:, None] / N_HAT[2]), rtol=1e-12)
    assert not np.any(D2)
    edges = np.linspace(0.0, 40000.0, 9)
    term = dc.kossel_term(np.ones(8), edges, N_HAT, G, S_HAT, E, 0.0, chi_0)
    np.testing.assert_allclose(term, 0.0, atol=1e-13)


def test_flux_is_conserved_without_absorption():
    """``n_s |T|^2 + (Q_s / k) |D|^2 = n_s`` for real ``chi_0``, at strong coupling."""
    E = E_B + np.linspace(-6.0, 6.0, 25)
    u = np.array([1000.0, 20000.0, 200000.0])
    T2, D2 = db.laue_mode_intensities(E, N_HAT, G, S_HAT, 3e-5, complex(-2e-5, 0.0), u)
    k = E / HBARC_EV_ANG
    gamma_h = ((k[:, None] * N_HAT + G) @ S_HAT) / k
    assert D2.max() > 0.1
    np.testing.assert_allclose(N_HAT[2] * T2 + gamma_h * D2, N_HAT[2], rtol=1e-12)


def test_kossel_band_matches_the_kinematic_laue_estimate():
    """Line band over ``T`` and ``D`` vs ``Y L(E_B) (1 - gamma_h/gamma_n) R_E`` (textbook).

    Kinematic, non-absorbing, line much wider than the Bragg width, uniform
    emission depth (mean remaining depth ``L/2``). ``R_theta = pi k |chi_g|^2
    C^2 t / (2 gamma_h sin 2 theta)``, ``R_E = E R_theta / tan theta``.
    """
    thickness, chi_g, fwhm, Y = 20000.0, 1e-8, 400.0, 1.0
    edges = np.linspace(0.0, thickness, 17)
    weight = np.full(16, Y / 16)
    E = E_B + np.arange(-1500.0, 1500.0, 0.025)
    S = dc._lorentzian(E, E_B, fwhm) * dc.kossel_term(weight, edges, N_HAT, G, S_HAT, E, chi_g, 0j)
    band = np.trapezoid(S, E)
    k = E_B / HBARC_EV_ANG
    Qh = (k * N_HAT + G) / k
    theta = np.arcsin(np.linalg.norm(G) / (2 * k))
    pol = 0.5 * (1 + np.cos(2 * theta) ** 2)
    R_theta = np.pi * k * chi_g**2 * (thickness / 2) / (2 * Qh[2] * np.sin(2 * theta)) * pol
    R_E = E_B * R_theta / np.tan(theta)
    estimate = Y * dc._lorentzian(E_B, E_B, fwhm) * (1 - Qh[2] / N_HAT[2]) * R_E
    assert abs(1 - Qh[2] / N_HAT[2]) > 0.2  # asymmetric: nonzero contrast
    assert band == pytest.approx(estimate, rel=1e-2)


def test_symmetric_laue_without_absorption_redistributes_no_flux():
    n = np.array([np.sin(0.3), 0.0, np.cos(0.3)])
    g = -2.0 * np.array([np.sin(0.3), 0.0, 0.0]) * 1.5  # g along -x: Q_hat mirrors n about z
    E_b = -HBARC_EV_ANG * (g @ g) / (2.0 * n @ g)
    edges = np.linspace(0.0, 20000.0, 9)
    E = E_b + np.linspace(-5.0, 5.0, 41)
    term = dc.kossel_term(np.ones(8), edges, n, g, S_HAT, E, 3e-5, complex(-2e-5, 0.0))
    np.testing.assert_allclose(term, 0.0, atol=1e-12)


# ---- end to end: germanium K-alpha on a Kossel cone -------------------------


@pytest.fixture(scope="module")
def germanium():
    import pyrite as pr
    from pyrite.api import build_case
    from pyrite.config import FixedSampling
    from pyrite.detectors import EnergyBins, PointDetector
    from pyrite.montecarlo import runner

    scene = pr.Scene(
        pr.Beam(energy_keV=100.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab("ge", thickness_ang=20000.0, footprint=pr.geometry.Footprint(5.0, 5.0)),
        PointDetector(),
        physics=pr.Physics(radiative_model="uncoupled", inelastic_model="continuous"),
    )
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=20, brem_histories=1),
        energy_bins=EnergyBins(cxr=np.linspace(9.0e3, 11.5e3, 501)),
    )
    case = build_case(scene, numerics)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        transport = runner._transport_case(case, transport_core="per-electron")
    return case, transport


def _kossel_direction(case, hkl, E_line):
    """Detector direction on the Kossel cone of ``hkl`` at ``E_line``, Laue on both legs."""
    from pyrite.materials.crystal import CRYSTALS, _orientation_R, reciprocal_g_vector

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
    k = E_line / HBARC_EV_ANG
    gh = g / np.linalg.norm(g)
    t = S_HAT - gh[2] * gh
    t /= np.linalg.norm(t)
    sin_b = np.linalg.norm(g) / (2 * k)
    n = -sin_b * gh + np.sqrt(1 - sin_b**2) * t
    return g, n


def _ka1():
    from pyrite.montecarlo.spectrum.characteristic import load_characteristic_cross_sections

    table = load_characteristic_cross_sections("Ge")
    i = list(table.line_labels).index("Ka1")
    return float(table.line_energy_eV[i]), float(table.cxr_fwhm_eV[i])


def test_line_depth_yields_match_native_characteristic(germanium):
    """Depth yields times native escape reproduce the native line band (single slab)."""
    from pyrite.materials.attenuation import _mu_total_inv_ang
    from pyrite.montecarlo.spectrum.characteristic import (
        _energy_bin_edges_and_widths,
        _lorentzian_bin_weights,
        mc_characteristic_spectrum,
    )

    case, transport = germanium
    segs, grid = transport["segs"], np.linspace(9.0e3, 11.5e3, 501)
    _, n = _kossel_direction(case, (-4, 0, 0), _ka1()[0])
    common = dict(Ne=transport["Ne_brem"], E_cut_keV=case.get("E_cut_brem_keV", 1.0))
    edges, lines = dc.line_depth_yields(segs, case["composition"], case["thickness_ang"], **common)
    u = case["thickness_ang"] - 0.5 * (edges[1:] + edges[:-1])
    bin_edges, widths = _energy_bin_edges_and_widths(grid)
    want = np.zeros(grid.size)
    for energy, fwhm, weight, _element in lines:
        mu = float(np.asarray(_mu_total_inv_ang(case["composition"], np.array([energy])))[0])
        want += weight @ np.exp(-mu * u / n[2]) * _lorentzian_bin_weights(bin_edges, energy, fwhm)
    native = mc_characteristic_spectrum(
        segs,
        grid,
        composition=case["composition"],
        n_hat=n,
        electron_limit=common["Ne"],
        E_cut_keV=common["E_cut_keV"],
    )
    assert native.max() > 0.0
    np.testing.assert_allclose(want / widths, native, rtol=2e-3, atol=1e-6 * native.max())


def test_ge_ka1_kossel_line_on_its_cone(germanium):
    """On the asymmetric (-4 0 0) Kossel cone the K-alpha1 line gains; off it, nothing.

    The Ge standing-wave phase ``phi = -exp(-W)`` (real) makes the profile
    dispersive: depleted below the line, enhanced above.
    """
    from pyrite.montecarlo.runner.diffracted_brem import case_diffracted_characteristic

    case, transport = germanium
    grid = np.linspace(9.0e3, 11.5e3, 501)
    E_line, _ = _ka1()
    g, n = _kossel_direction(case, (-4, 0, 0), E_line)
    Qh = (E_line / HBARC_EV_ANG * n + g) / (E_line / HBARC_EV_ANG)
    assert n[2] - Qh[2] > 0.2  # gamma_h < gamma_n: net gain
    kossel = case_diffracted_characteristic(case, transport["segs"], grid, n, transport["Ne_brem"])
    peak = grid[np.argmax(kossel)]
    assert abs(peak - E_line) < 10.0 and kossel.max() > 0.0
    assert np.all(np.isfinite(kossel))
    # The standing-wave term is active (Ge on its own sites) and dispersive.
    reflections = db.bragg_reflections(case, n, grid, detail=True)
    phases = dict(
        (tuple(hkl), dc.emitter_phases(case["crystal"], hkl, chi, case.get("B_ang2", 0.0)))
        for *_, (hkl, chi) in reflections
    )
    assert phases[(-4, 0, 0)]["Ge"].real == pytest.approx(-0.927, abs=0.01)
    intensity_only = dc.diffracted_characteristic_spectrum(
        transport["segs"], grid, [r[:3] for r in reflections], n, case["composition"],
        case["thickness_ang"], Ne=transport["Ne_brem"], E_cut_keV=case.get("E_cut_brem_keV", 1.0),
    )  # fmt: skip
    standing = kossel - intensity_only
    below, above = standing[grid < E_line], standing[grid > E_line]
    assert below[-3:].min() < 0.0 < above[:3].max()
    zero = [(g, 0.0, complex(-2e-5, 4e-7))]
    assert not np.any(
        dc.diffracted_characteristic_spectrum(
            transport["segs"],
            grid,
            zero,
            n,
            case["composition"],
            case["thickness_ang"],
            Ne=transport["Ne_brem"],
            E_cut_keV=case.get("E_cut_brem_keV", 1.0),
        )  # fmt: skip
        > 1e-12 * kossel.max()
    )


def test_spectrum_adds_kossel_term_to_characteristic(germanium, monkeypatch):
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import diffracted_brem

    case, transport = germanium
    _, n = _kossel_direction(case, (-4, 0, 0), _ka1()[0])
    tp = {**transport, "n_hat": n}
    with_kossel = np.asarray(runner._spectrum_case(case, tp)["spec_characteristic"])
    grid = np.asarray(transport["E_grid"], dtype=float)
    kossel = diffracted_brem.case_diffracted_characteristic(
        case, transport["segs"], grid, n, transport["Ne_brem"]
    )
    assert np.abs(kossel).max() > 0.0
    monkeypatch.setattr(
        diffracted_brem, "case_diffracted_characteristic", lambda *a: np.zeros(kossel.size)
    )
    native = np.asarray(runner._spectrum_case(case, tp)["spec_characteristic"])
    np.testing.assert_allclose(
        with_kossel - native, kossel, rtol=1e-9, atol=1e-12 * np.abs(kossel).max()
    )
    for override in ({"mosaic_fwhm_rad": 1e-3}, {"hkl_list": []}):
        mapping = {**case.to_dict(), **override}
        out = diffracted_brem.case_diffracted_characteristic(
            mapping, transport["segs"], grid, n, transport["Ne_brem"]
        )
        assert not np.any(out)
