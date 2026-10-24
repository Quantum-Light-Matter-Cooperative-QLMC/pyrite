"""User-definable channeling entry states (cr-entry-state-projection, #508)."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import eval_hermite, j1

from pyrite.materials.channeling import planar_profile
from pyrite.montecarlo.channeling import DensityMatrixSolver, MCWFSolver, tds_kernel
from pyrite.montecarlo.channeling import states as st

ENERGY_EV = 17e6
GRID = {"n_cells": 4, "max_order": 8}


@pytest.fixture(scope="module")
def solver():
    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    return DensityMatrixSolver(profile, ENERGY_EV, tds_kernel(profile, ENERGY_EV, **GRID))


def _norm(state):
    nodes, weights = zip(*(st._uniform_nodes(e, r) for e, r in state._hints()), strict=True)
    if len(nodes) == 1:
        return float(np.dot(weights[0] / (2 * np.pi), np.abs(state.amplitude(*nodes)) ** 2))
    # Keep every quadrature node and weight, but bound the 2D amplitude scratch.
    # Airy's fine phase resolution otherwise materializes a multi-GiB tensor.
    marginal = np.empty(nodes[1].size)
    for start in range(0, nodes[1].size, 64):
        block = slice(start, start + 64)
        density = np.abs(state.amplitude(nodes[0], nodes[1][block])) ** 2
        marginal[block] = np.dot(weights[0] / (2 * np.pi), density)
    return float(np.dot(weights[1] / (2 * np.pi), marginal))


PRESETS = {
    "gaussian-1d": lambda: st.gaussian(waist_ang=2.0, dimension=1, center_ang=0.3),
    "gaussian-2d": lambda: st.gaussian(waist_ang=2.0),
    "hermite-gauss": lambda: st.hermite_gauss(2, 1, waist_ang=1.5),
    "laguerre-gauss": lambda: st.laguerre_gauss(-2, 1, waist_ang=1.5, center_ang=(0.2, -0.1)),
    "donut": lambda: st.donut(1, waist_ang=1.0),
    "bessel-gauss": lambda: st.bessel_gauss(1, k_radial_inv_ang=3.0, waist_ang=3.0),
    "airy-1d": lambda: st.airy(scale_ang=0.5, truncation=0.1),
    "airy-2d": lambda: st.airy(scale_ang=0.5, truncation=0.2, dimension=2),
    "aperture": lambda: st.aperture_probe(
        200e3, semiangle_rad=20e-3, aberrations={"C10": -20.0, "C30": 1e4}, ell=1
    ),
    "annular": lambda: st.aperture_probe(200e3, semiangle_rad=20e-3, inner_semiangle_rad=10e-3),
    "superposition": lambda: st.superpose(
        [(1.0, st.laguerre_gauss(1, waist_ang=1.0)), (1.0j, st.laguerre_gauss(-1, waist_ang=1.0))]
    ),
    "sampled": lambda: st.from_array(
        np.exp(-(np.linspace(-6, 6, 241) ** 2) + 2j * np.linspace(-6, 6, 241)), spacing_ang=0.05
    ),
}


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets_are_normalized_in_momentum_space(name):
    assert _norm(PRESETS[name]()) == pytest.approx(1.0, abs=2e-3)


def test_analytic_amplitudes_match_the_numerical_transform():
    x = np.linspace(-12.0, 12.0, 961)
    dx = x[1] - x[0]
    k = np.linspace(-4.0, 4.0, 9)
    # 1D Airy: analytic samples vs analytic Fourier transform.
    airy = st.airy(scale_ang=0.5, truncation=0.1)
    samples = st.airy_wavefunction(np.linspace(-40, 12, 5201), scale_ang=0.5, truncation=0.1)
    sampled = st.from_array(samples, spacing_ang=0.01, origin_ang=-40.0)
    np.testing.assert_allclose(sampled.amplitude(k), airy.amplitude(k), atol=2e-4)
    # 2D Laguerre-Gauss and Bessel-Gauss: closed-form real space vs closed-form momentum.
    for state in (
        st.laguerre_gauss(2, 1, waist_ang=1.5, center_ang=(0.4, -0.2)),
        st.bessel_gauss(1, k_radial_inv_ang=2.0, waist_ang=3.0),
    ):
        grid = st.from_array(state.wavefunction(x, x), spacing_ang=dx, origin_ang=x[0])
        np.testing.assert_allclose(grid.amplitude(k, k), state.amplitude(k, k), atol=2e-4)


def test_hermite_gauss_inversion_reproduces_the_mode():
    w = 1.5
    state = st.hermite_gauss(2, waist_ang=w)
    x = np.linspace(-4.0, 4.0, 41)
    ratio = state.wavefunction(x) / (eval_hermite(2, np.sqrt(2) * x / w) * np.exp(-((x / w) ** 2)))
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-6)
    wide = np.linspace(-12.0, 12.0, 2401)
    assert np.trapezoid(np.abs(state.wavefunction(wide)) ** 2, wide) == pytest.approx(1.0, rel=1e-6)


def test_aperture_probe_is_the_airy_disk():
    probe = st.aperture_probe(200e3, semiangle_rad=20e-3)
    k0 = probe.k_outer
    r = np.array([0.05, 0.5, 1.0, 1.5])
    psi = probe.wavefunction(r, np.zeros(1))[:, 0]
    centre = np.sqrt(4 * np.pi / k0**2) * k0**2 / (4 * np.pi)
    np.testing.assert_allclose(psi, centre * 2 * j1(k0 * r) / (k0 * r), rtol=2e-3, atol=2e-4)


def test_aberration_phase_follows_the_polar_convention():
    alpha, phi, p = np.array([0.01, 0.02]), np.array([0.3, 1.1]), 250.0
    chi = st.aberration_phase(alpha, phi, p, {"C10": -50.0, "C12": 20.0, "phi12": 0.4, "C30": 1e5})
    expected = p * (
        -50.0 * alpha**2 / 2 + 20.0 * alpha**2 * np.cos(2 * (phi - 0.4)) / 2 + 1e5 * alpha**4 / 4
    )
    np.testing.assert_allclose(chi, expected, rtol=1e-12)
    with pytest.raises(ValueError, match="unknown aberration"):
        st.aberration_phase(alpha, phi, p, {"defocus": 1.0})


def test_wide_gaussian_is_the_plane_wave_entry(solver):
    incidence = 0.3e-3
    state = st.gaussian(waist_ang=1e4, dimension=1)
    for impact in ("averaged", "fixed"):
        plane, outside = st.planar_entry_density(
            solver, state, incidence_rad=incidence, impact=impact
        )
        rho, free, out = solver.entry_from_plane(plane, outside)
        rho0, free0, out0 = solver.entry_state(incidence)
        np.testing.assert_allclose(rho, rho0, atol=1e-6)
        np.testing.assert_allclose(free, free0, atol=1e-6)
        assert out == pytest.approx(out0, abs=1e-6)


def test_averaged_gaussian_matches_the_divergence_entry(solver):
    divergence = 1e-3
    p = st.momentum_inv_ang(ENERGY_EV)
    state = st.gaussian(waist_ang=1.0 / (p * divergence), dimension=1)
    plane, outside = st.planar_entry_density(solver, state)
    rho, free, out = solver.entry_from_plane(plane, outside)
    rho0, free0, out0 = solver.entry_state(0.0, divergence)
    np.testing.assert_allclose(rho, rho0, atol=3e-5)
    np.testing.assert_allclose(free, free0, atol=3e-5)
    assert out == pytest.approx(out0, abs=3e-5)
    assert np.einsum("rii->", rho).real + free.sum() + out == pytest.approx(1.0, abs=1e-10)


def test_projection_impact_modes_and_translation(solver):
    period = solver.profile.period_ang
    probe = st.gaussian(waist_ang=0.4, dimension=1)
    shifted = st.translate(probe, [0.37 * period])
    whole = st.translate(probe, [period])

    def plane(state, impact):
        return st.planar_entry_density(solver, state, impact=impact)[0]

    np.testing.assert_allclose(plane(shifted, "averaged"), plane(probe, "averaged"), atol=1e-12)
    np.testing.assert_allclose(plane(whole, "fixed"), plane(probe, "fixed"), atol=1e-12)
    assert np.max(np.abs(plane(shifted, "fixed") - plane(probe, "fixed"))) > 1e-3
    averaged = plane(probe, "averaged")
    np.testing.assert_allclose(averaged, np.einsum("rii->ri", averaged)[:, :, None] * np.eye(17))
    # A positioned probe changes the channeled population; an unpositioned one cannot.
    on, off = (
        solver.bound_population(solver.entry_from_plane(plane(s, "fixed"))[0])
        for s in (probe, st.translate(probe, [0.5 * period]))
    )
    assert abs(on - off) > 0.01


def test_vortex_helicity_is_invisible_to_planar_channeling(solver):
    for impact in ("averaged", "fixed"):
        plus, minus = (
            st.planar_entry_density(solver, st.laguerre_gauss(ell, waist_ang=0.6), impact=impact)[0]
            for ell in (1, -1)
        )
        np.testing.assert_allclose(plus, minus, atol=1e-10)


def test_mixture_is_the_weighted_sum(solver):
    a, b = st.gaussian(waist_ang=0.5, dimension=1), st.airy(scale_ang=0.3, truncation=0.2)
    mixed = st.planar_entry_density(solver, st.mixture([(3.0, a), (1.0, b)]), impact="fixed")[0]
    parts = [st.planar_entry_density(solver, s, impact="fixed")[0] for s in (a, b)]
    np.testing.assert_allclose(mixed, 0.75 * parts[0] + 0.25 * parts[1], atol=1e-12)


def test_mcwf_entry_draws_follow_the_projected_density(solver):
    state = st.gaussian(waist_ang=0.4, dimension=1, center_ang=0.25)
    plane, outside = st.planar_entry_density(solver, state, impact="fixed")
    rho, _, _ = solver.entry_from_plane(plane, outside)
    result = MCWFSolver(solver).run(
        4000, 10.0, 10.0, seed=7, entry_state=state, entry_impact="fixed"
    )
    expected = solver.bound_population(rho)
    assert result.bound_population[0] == pytest.approx(
        expected, abs=4 * result.bound_population_sem[0] + 1e-3
    )


def test_from_abtem_reads_waves_and_ensembles():
    frames = np.stack([np.ones((8, 8)), np.exp(1j * np.arange(64.0)).reshape(8, 8)])
    waves = SimpleNamespace(array=frames, sampling=(0.1, 0.2), energy=200e3)
    state = st.from_abtem(waves)
    assert isinstance(state, st.MixedState) and len(state.components) == 2
    single = st.from_abtem(SimpleNamespace(array=frames[0], sampling=(0.1, 0.2), energy=200e3))
    assert single.kinetic_energy_eV == 200e3
    np.testing.assert_allclose(single.axes[1][1], 0.2)
    assert _norm(single) == pytest.approx(1.0, abs=5e-3)


def test_entry_state_refuses_an_extra_divergence(solver):
    with pytest.raises(ValueError, match="own momentum spread"):
        MCWFSolver(solver).run(
            2, 10.0, 10.0, seed=1, divergence_rad=1e-3, entry_state=st.gaussian(waist_ang=1.0)
        )


def test_aperture_probe_matches_abtem_conventions():
    abtem = pytest.importorskip("abtem")
    abtem.config.set({"device": "cpu"})
    aberrations = {"C10": -30.0, "C30": 2e4, "C12": 15.0, "phi12": 0.5}
    waves = (
        abtem.Probe(
            semiangle_cutoff=20, energy=200e3, extent=20, gpts=256, soft=False,
            aberrations=aberrations,
        )
        .build((10.0, 10.0))
        .compute()
    )  # fmt: skip
    reference = np.abs(np.asarray(waves.array)) ** 2
    x = np.arange(256) * waves.sampling[0]

    def mismatch(chi):
        probe = st.aperture_probe(200e3, semiangle_rad=20e-3, aberrations=chi, center_ang=10.0)
        intensity = np.abs(probe.wavefunction(x, x)) ** 2
        return np.max(np.abs(intensity / intensity.sum() - reference / reference.sum())) / (
            reference.max() / reference.sum()
        )

    assert mismatch(aberrations) < 0.03
    assert mismatch({**aberrations, "C10": 30.0}) > 0.06  # the defocus sign is resolved
    imported = st.from_abtem(waves)
    assert imported.kinetic_energy_eV == 200e3


def test_channeling_spectrum_accepts_an_entry_state(solver):
    from pyrite.montecarlo.channeling import channeling_spectrum

    energies = np.linspace(5e3, 60e3, 40)
    kwargs = {"incidence_rad": 0.2e-3, "dz_ang": 250.0, "sample_ang": 1000.0}
    reference = channeling_spectrum(solver, 2000.0, energies, 0.0, **kwargs)
    shaped = channeling_spectrum(
        solver, 2000.0, energies, 0.0, entry_state=st.gaussian(waist_ang=1e4, dimension=1),
        **kwargs,
    )  # fmt: skip
    np.testing.assert_allclose(shaped.spectrum, reference.spectrum, rtol=1e-5, atol=1e-12)
    focused = channeling_spectrum(
        solver, 2000.0, energies, 0.0, entry_state=st.gaussian(waist_ang=0.3, dimension=1),
        entry_impact="fixed", **kwargs,
    )  # fmt: skip
    assert not np.allclose(focused.spectrum, reference.spectrum)


@pytest.mark.parametrize("waist", [3e4, 5e4, 1e5, 1e6])
@pytest.mark.parametrize("bins", [0.0, 0.49, 0.51, -2.7])
def test_narrow_momentum_beams_keep_their_weight(solver, waist, bins):
    # Real 10-100 um beams: the whole state sits inside one Bloch bin.
    p = st.momentum_inv_ang(ENERGY_EV)
    incidence = bins * solver.profile.g_inv_ang / solver.n_cells / p
    plane, outside = st.planar_entry_density(
        solver, st.gaussian(waist_ang=waist, dimension=1), incidence_rad=incidence
    )
    rho, free, out = solver.entry_from_plane(plane, outside)
    rho0, free0, out0 = solver.entry_state(incidence)
    np.testing.assert_allclose(rho, rho0, atol=1e-6)
    np.testing.assert_allclose(free, free0, atol=1e-6)
    assert out == pytest.approx(out0, abs=1e-6)


@pytest.mark.parametrize("separation", [100.0, 200.0])
def test_separated_superposition_resolves_its_fringes(solver, separation):
    left = st.gaussian(waist_ang=2.0, dimension=1)
    pair = st.superpose([(1.0, left), (1.0, st.translate(left, [separation]))])
    assert pair.norm == pytest.approx(np.sqrt(2.0), rel=1e-6)
    plane, outside = st.planar_entry_density(solver, pair, impact="fixed")
    assert np.einsum("rii->", plane).real + outside == pytest.approx(1.0, abs=1e-12)
    assert outside < 1e-3


def test_sampled_states_are_band_limited():
    state = st.from_array(np.exp(-(np.linspace(-3, 3, 61) ** 2)), spacing_ang=0.1)
    nyquist = np.pi / 0.1
    assert np.all(state.amplitude(np.array([1.01, 1.5, 3.0]) * nyquist) == 0.0)
    assert abs(state.amplitude(np.array([0.0]))[0]) > 0.0


@pytest.mark.parametrize("defocus", [-1e7, -2e4])
def test_unresolvable_quadratures_raise(defocus):
    # -1e7 exceeds the per-axis limit; -2e4 fits per axis but not as a 2D tensor.
    probe = st.aperture_probe(200e3, semiangle_rad=20e-3, aberrations={"C10": defocus})
    with pytest.raises(ValueError, match="momentum quadrature needs"):
        probe.wavefunction(np.zeros(1), np.zeros(1))
