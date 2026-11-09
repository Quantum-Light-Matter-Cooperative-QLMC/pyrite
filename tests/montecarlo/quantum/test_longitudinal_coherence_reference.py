"""Regression anchors for the #629 longitudinal-coherence and recoil reference."""

import mpmath as mp
import numpy as np
import pytest

from pyrite.materials.crystal import M_E_EV
from pyrite.materials.recoil import recoil_resonance_eV
from pyrite.quantum import _longitudinal as lc


def _random_pure_state(rng, size):
    c = rng.normal(size=size) + 1j * rng.normal(size=size)
    c /= np.linalg.norm(c)
    return c, np.outer(c, c.conj())


def _slab_amplitudes(orders, spacing, photons, kinetic, g, direction, thickness):
    """Physical per-order amplitudes: exact-recoil slab formation factor."""
    rows = []
    for order in orders:
        mismatch = lc.slab_mismatch_inv_ang(photons, kinetic + order * spacing, g, direction)
        rows.append(lc.slab_formation_amplitude(mismatch, thickness))
    return np.array(rows)


# -- longitudinal-energy-diagonal-emission ---------------------------------


def test_single_electron_photon_counts_ignore_longitudinal_coherence():
    rng = np.random.default_rng(629)
    orders = np.arange(-4, 5)
    spacing, subdivisions = 1.55, 5
    photons = 511.0 + np.arange(60) * spacing / subdivisions
    g = np.array([0.0, 0.0, 0.9])
    direction = np.array([np.sin(2.0), 0.0, np.cos(2.0)])
    amplitudes = _slab_amplitudes(orders, spacing, photons, 30e3, g, direction, 400.0)
    c, pure = _random_pure_state(rng, orders.size)
    dephased = np.diag(np.diag(pure))
    phased = c * np.exp(1j * rng.uniform(0, 2 * np.pi, orders.size))
    rotated = np.outer(phased, phased.conj())  # same populations, new phases
    counts = [
        np.diag(lc.photon_density_matrix(state, amplitudes, subdivisions)).real
        for state in (pure, dephased, rotated)
    ]
    expected = (np.abs(c)[:, None] ** 2 * np.abs(amplitudes) ** 2).sum(axis=0)
    scale = expected.max()
    for spectrum in counts:
        np.testing.assert_allclose(spectrum, expected, rtol=0, atol=1e-13 * scale)


def test_photon_coherence_appears_only_at_comb_multiples():
    rng = np.random.default_rng(1)
    orders, photons, subdivisions = 6, 30, 4
    amplitudes = rng.normal(size=(orders, photons)) + 1j * rng.normal(size=(orders, photons))
    c, pure = _random_pure_state(rng, orders)
    photon = lc.photon_density_matrix(pure, amplitudes, subdivisions)
    dephased = lc.photon_density_matrix(np.diag(np.diag(pure)), amplitudes, subdivisions)
    a, b = np.indices(photon.shape)
    off_comb = (a - b) % subdivisions != 0
    assert np.max(np.abs(photon[off_comb])) == 0.0
    assert np.max(np.abs(dephased[a != b])) == 0.0
    # rho_ph(a, a - l*M) = sum_n c_n conj(c_{n-l}) A_n(a) conj(A_{n-l}(a - l*M)).
    ell, row = 2, 20
    expected = sum(
        c[n]
        * c[n - ell].conj()
        * amplitudes[n, row]
        * amplitudes[n - ell, row - ell * subdivisions].conj()
        for n in range(ell, orders)
    )
    np.testing.assert_allclose(photon[row, row - ell * subdivisions], expected, rtol=1e-12)
    np.testing.assert_allclose(photon, photon.conj().T, atol=1e-12)


@pytest.mark.parametrize(
    ("kinetic", "g", "theta"),
    [(30e3, [0.0, 0.0, 0.9], 2.0), (300e3, [0.4, 0.1, 1.6], 1.2), (5e6, [0.2, 0.0, 1.87], 0.05)],
)
def test_slab_mismatch_vanishes_at_vacuum_recoil_root(kinetic, g, theta):
    direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
    root = float(recoil_resonance_eV(kinetic, [0.0, 0.0, 1.0], g, direction))
    k_scale = np.sqrt(kinetic * (kinetic + 2 * M_E_EV)) / 1973.269804
    at_root = lc.slab_mismatch_inv_ang(root, kinetic, np.array(g), direction)
    assert abs(at_root) <= 1e-15 * k_scale * 64
    above = lc.slab_mismatch_inv_ang(root * 1.001, kinetic, np.array(g), direction)
    assert above > 0  # positive slope: same sign as line-recoil-formation


def test_slab_formation_amplitude_limits():
    thickness = 250.0
    np.testing.assert_allclose(lc.slab_formation_amplitude(0.0, thickness), thickness)
    delta = np.array([1e-3, 0.02, 0.7])
    direct = (np.exp(1j * delta * thickness) - 1) / (1j * delta)
    np.testing.assert_allclose(lc.slab_formation_amplitude(delta, thickness), direct, rtol=1e-12)


# -- longitudinal-quantum-bunching ----------------------------------------


def test_talbot_length_matches_exact_dispersion_curvature():
    kinetic, step = 200e3, 200.0  # coarse step: central difference above roundoff
    k = lc.comb_wavenumbers_inv_ang(kinetic, step, np.arange(-1, 2))
    curvature = k[0] + k[2] - 2 * k[1]
    np.testing.assert_allclose(
        lc.talbot_length_ang(kinetic, step), 4 * np.pi / abs(curvature), rtol=1e-5
    )
    spacing = 1.55
    # 200 keV, 1.55 eV: ~0.48 m, so mm-scale PINEM bunching needs |g| >> 1.
    assert 4.7e9 < lc.talbot_length_ang(kinetic, spacing) < 4.8e9


@pytest.mark.parametrize("harmonic", [1, 3, 8])
@pytest.mark.parametrize("z_over_talbot", [1e-3, 1e-2, 5e-2])
def test_pinem_bunching_matches_neumann_graf_closed_form(harmonic, z_over_talbot):
    kinetic, spacing, coupling = 200e3, 1.55, 3.0
    orders = np.arange(-40, 41)
    c = lc.pinem_amplitudes(coupling, orders)
    assert 1 - np.sum(np.abs(c) ** 2) < 1e-14
    k = lc.comb_wavenumbers_inv_ang(kinetic, spacing, orders)
    z = z_over_talbot * lc.talbot_length_ang(kinetic, spacing)
    exact = abs(lc.bunching_coefficient(np.outer(c, c.conj()), k, z, harmonic))
    closed = lc.pinem_bunching_closed_form(coupling, harmonic, z_over_talbot)
    # Residual: float64 cancellation in k_{n+l} - k_n (cubic dispersion <= 6e-9 here).
    assert abs(exact - closed) < 2e-6


def test_quantum_bunching_reduces_to_klystron_limit():
    harmonic, parameter = 5, 1.0
    classical = lc.klystron_bunching(harmonic, parameter)
    errors = []
    for z_over_talbot in (1e-2, 1e-3, 1e-4):
        coupling = parameter / (8 * np.pi * z_over_talbot)
        quantum = lc.pinem_bunching_closed_form(coupling, harmonic, z_over_talbot)
        errors.append(abs(quantum - classical))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 2e-6
    assert errors[1] / errors[2] == pytest.approx(100, rel=0.05)  # O((z/z_T)^2)


def test_quantum_bunching_vanishes_beyond_coherent_comb_width():
    coupling, harmonic = 3.0, 30  # comb half width ~2|g| = 6 quanta
    z_grid = np.linspace(0, 0.5 / harmonic, 2001)
    quantum = max(lc.pinem_bunching_closed_form(coupling, harmonic, z) for z in z_grid)
    classical = max(lc.klystron_bunching(harmonic, x) for x in np.linspace(0, 3, 3001))
    assert quantum < 1e-5
    assert classical > 0.15


def test_pinem_attosecond_focus_matches_priebe_2017():
    # Priebe et al., Nat. Photon. 11, 793 (2017): 120 keV, 800 nm, g = 3.95,
    # shortest attosecond pulses expected after 2.75 mm. Fixes both the factor 2
    # in J_n(2|g|) and the z_T convention (either off by 2 moves it to ~1.4/5.5 mm).
    kinetic, spacing, coupling = 120e3, 1.55, 3.95
    orders = np.arange(-40, 41)
    c = lc.pinem_amplitudes(coupling, orders)
    k = lc.comb_wavenumbers_inv_ang(kinetic, spacing, orders)
    phase = k - k[40] - orders * (k[41] - k[39]) / 2  # co-moving frame
    tau = np.exp(-1j * np.outer(orders, np.linspace(-np.pi, np.pi, 2001)))
    z_mm = np.arange(2.0, 3.6, 0.05)
    peak = [np.max(np.abs((c * np.exp(1j * phase * z * 1e7)) @ tau) ** 2) for z in z_mm]
    assert 2.65 < z_mm[int(np.argmax(peak))] < 2.85


def test_gaussian_wavepacket_pair_factor_and_dephased_mixture():
    orders = np.arange(-400, 401)
    sigma = 20.0  # coherent energy rms in comb quanta
    c = np.exp(-(orders**2) / (4 * sigma**2))
    c /= np.linalg.norm(c)
    pure = np.outer(c, c)
    ones = np.ones(orders.size)
    for harmonic in (10, 40, 80):
        factor = abs(lc.pair_amplitude(pure, ones, harmonic))
        np.testing.assert_allclose(factor, np.exp(-(harmonic**2) / (8 * sigma**2)), rtol=1e-12)
        assert lc.pair_amplitude(np.diag(np.diag(pure)), ones, harmonic) == 0


def test_pair_amplitude_is_bounded_by_self_term():
    rng = np.random.default_rng(7)
    for _ in range(20):
        weights = rng.dirichlet(np.ones(3))
        states = [_random_pure_state(rng, 12)[1] for _ in range(3)]
        rho = sum(w * s for w, s in zip(weights, states, strict=True))
        amplitudes = rng.normal(size=12) + 1j * rng.normal(size=12)
        harmonic = int(rng.integers(1, 6))
        pair = abs(lc.pair_amplitude(rho, amplitudes, harmonic)) ** 2
        self_term = np.sum(np.diag(rho).real * np.abs(amplitudes) ** 2)
        assert pair <= self_term * (1 + 1e-12)


def test_pair_bound_from_energy_distribution_holds_and_is_attained():
    rng = np.random.default_rng(11)
    for _ in range(20):
        weights = rng.dirichlet(np.ones(4))
        rho = sum(w * _random_pure_state(rng, 16)[1] for w in weights)
        amplitudes = rng.normal(size=16) + 1j * rng.normal(size=16)
        harmonic = int(rng.integers(1, 8))
        bound = lc.pair_amplitude_bound(np.diag(rho).real, harmonic) * np.abs(amplitudes).max()
        assert abs(lc.pair_amplitude(rho, amplitudes, harmonic)) <= bound * (1 + 1e-12)
    populations = rng.dirichlet(np.ones(16))
    root = np.sqrt(populations)
    attained = lc.pair_amplitude(np.outer(root, root), np.ones(16), 3)
    np.testing.assert_allclose(attained, lc.pair_amplitude_bound(populations, 3), rtol=1e-14)


# -- cr-recoil-line-energy -------------------------------------------------

_CR_CASES = [
    (5e6, -5.0, -12.0, 0.0, 0.0, 0.0),
    (5e6, -5.0, -12.0, 3.0, 0.0, 0.0),
    (20e6, -3.0, -9.0, 2.0, 2e-2, 0.0),
    (20e6, -3.0, -9.0, 2.0, 2e-2, np.pi / 2),
    (54e6, -4.0, -15.0, 5.0, 5e-3, 0.3),
]


def _mass_shell_root(kinetic, eps_i, eps_f, t_f, theta, phi):
    with mp.workdps(50):
        kinetic, eps_i, eps_f, t_f, theta, phi = map(
            mp.mpf, (kinetic, eps_i, eps_f, t_f, theta, phi)
        )
        mass = mp.mpf(M_E_EV)
        total = kinetic + mass
        pz = mp.sqrt(total**2 - mass**2 - 2 * total * eps_i)

        def residual(x):
            final = total - x
            level = eps_f - t_f * mp.log(final / total)
            pz_f = pz - x * mp.cos(theta)
            py_f = -x * mp.sin(theta) * mp.sin(phi)
            return pz_f**2 + py_f**2 - (final**2 - mass**2 - 2 * final * level)

        return float(mp.findroot(residual, (eps_i - eps_f) / (1 - pz / total * mp.cos(theta))))


def _doppler(kinetic, eps_i, eps_f, theta):
    total = kinetic + M_E_EV
    beta = np.sqrt(kinetic * (kinetic + 2 * M_E_EV)) / total
    return (eps_i - eps_f) / (1 - beta * np.cos(theta)), total, beta


@pytest.mark.parametrize("case", _CR_CASES)
def test_cr_recoil_root_solves_the_mass_shell(case):
    root = float(lc.channeling_recoil_line_energy_eV(*case))
    np.testing.assert_allclose(root, _mass_shell_root(*case), rtol=1e-14)


def test_cr_forward_fixed_levels_is_baier_katkov():
    kinetic, eps_i, eps_f = 5e6, -5.0, -12.0
    root = float(lc.channeling_recoil_line_energy_eV(kinetic, eps_i, eps_f, 0.0, 0.0, 0.0))
    doppler, total, _ = _doppler(kinetic, eps_i, eps_f, 0.0)
    # Agreement to O(eps/E) ~ 2e-6, the paraxial accuracy.
    np.testing.assert_allclose(root, doppler / (1 + doppler / total), rtol=5e-6)
    assert root / doppler - 1 == pytest.approx(-doppler / total, rel=0.01)


@pytest.mark.parametrize("case", _CR_CASES)
def test_cr_first_order_shift(case):
    kinetic, eps_i, eps_f, t_f, theta, phi = case
    doppler, total, beta = _doppler(kinetic, eps_i, eps_f, theta)
    a = (np.sin(theta) * np.cos(phi)) ** 2
    first = -(doppler / total) * (1 + t_f / (eps_i - eps_f) - a / (2 * (1 - beta * np.cos(theta))))
    shift = float(lc.channeling_recoil_line_energy_eV(*case)) / doppler - 1
    assert shift == pytest.approx(first, rel=1e-2)


def test_cr_recoil_is_anisotropic_in_photon_azimuth():
    base = (20e6, -3.0, -9.0, 0.0, 2e-2)
    normal = float(lc.channeling_recoil_line_energy_eV(*base, 0.0))
    in_plane = float(lc.channeling_recoil_line_energy_eV(*base, np.pi / 2))
    doppler, total, beta = _doppler(20e6, -3.0, -9.0, 2e-2)
    expected = (doppler / total) * np.sin(2e-2) ** 2 / (2 * (1 - beta * np.cos(2e-2)))
    assert normal > in_plane
    assert normal / in_plane - 1 == pytest.approx(expected, rel=1e-2)


def test_cr_forbidden_branch_is_nan():
    # Raising (eps_f > eps_i) has no positive photon root.
    assert np.isnan(lc.channeling_recoil_line_energy_eV(5e6, -12.0, -5.0, 0.0, 0.0, 0.0))
