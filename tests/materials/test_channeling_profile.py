"""Static planar channeling profile: thermal data, potential, states, kinematics."""

import numpy as np
import pytest

from pyrite.materials.channeling import (
    ChannelingDataError,
    PlanarProfile,
    debye_b_factor,
    debye_integral,
    doppler_denominator,
    entry_populations,
    fit_debye_temperature,
    line_energy_eV,
    load_thermal_sites,
    longitudinal_kinematics,
    planar_profile,
    thermal_b_factor,
    transverse_states,
)
from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, U_g, reciprocal_g_vector
from pyrite.materials.emission import photon_energies

SI_A = CRYSTALS["silicon"]["lattice"]["a"]


@pytest.fixture(scope="module")
def si110():
    return planar_profile("silicon", (1, 1, 0), 293.15)


# ---- thermal displacement (cr-debye-b-of-t) ---------------------------------


def test_debye_integral_limits():
    assert debye_integral(1e-6) == pytest.approx(1.0, rel=1e-6)
    # Large-y limit pi^2/(6y); Peng et al. (1996) quote 1.6454/y beyond y = 10.
    assert debye_integral(100.0) == pytest.approx(np.pi**2 / 600.0, rel=1e-9)
    assert debye_integral(10.0) == pytest.approx(0.16444, abs=2e-5)


def test_debye_b_factor_classical_and_zero_point_limits():
    a, theta = 28.0855, 520.0
    high = debye_b_factor(np.array([5000.0, 10000.0]), a, theta)
    assert high[1] / high[0] == pytest.approx(2.0, rel=2e-3)  # B proportional to T
    zero_point = debye_b_factor(1e-3, a, theta)
    assert zero_point == pytest.approx(
        debye_b_factor(1e-2, a, theta), rel=1e-6
    )  # T-independent zero-point motion
    assert zero_point > 0.0


def test_fit_debye_temperature_round_trips():
    theta = fit_debye_temperature(0.4634, 280.0, 28.0855)
    assert debye_b_factor(280.0, 28.0855, theta) == pytest.approx(0.4634, rel=1e-10)
    # Peng et al. (1996) report Theta_D -> ~500 K for Si above 300 K.
    assert 480.0 < theta < 560.0


@pytest.mark.parametrize("key", [("silicon", "Si"), ("diamond", "C")])
def test_thermal_rows_reproduced_and_extrapolation_continuous(key):
    site = load_thermal_sites()[key]
    for t, b in zip(site.temperatures_K, site.B_ang2, strict=True):
        assert site.b_factor(t) == pytest.approx(b, rel=1e-12)
    lo, hi = site.temperatures_K[0], site.temperatures_K[-1]
    assert site.b_factor(lo - 1e-6) == pytest.approx(site.B_ang2[0], rel=1e-6)
    assert site.b_factor(hi + 1e-6) == pytest.approx(site.B_ang2[-1], rel=1e-6)
    assert site.b_factor(80.0) < site.B_ang2[0] < site.B_ang2[-1] < site.b_factor(400.0)


def test_ineligible_crystal_is_refused():
    with pytest.raises(ChannelingDataError, match="eligible crystals"):
        thermal_b_factor("hopg", "C", 293.15)
    with pytest.raises(ChannelingDataError):
        planar_profile("hopg", (0, 0, 1))


# ---- planar potential (cr-planar-continuum-potential, cr-mott-bethe-factor) --


@pytest.mark.parametrize(
    ("hkl", "period"),
    [((1, 1, 0), SI_A / (2 * np.sqrt(2))), ((1, 0, 0), SI_A / 4), ((1, 1, 1), SI_A / np.sqrt(3))],
)
def test_extinctions_set_true_planar_period(hkl, period):
    profile = planar_profile("silicon", hkl, 293.15)
    assert profile.period_ang == pytest.approx(period, rel=1e-12)
    assert planar_profile("silicon", tuple(2 * h for h in hkl)).period_ang == pytest.approx(
        period, rel=1e-12
    )


def test_si110_wells_sit_on_planes_with_literature_depth(si110):
    x = np.linspace(0.0, si110.period_ang, 513)
    u = si110.potential(x)
    assert np.argmin(u) in (0, x.size - 1)  # atomic plane at x = 0 for (220)
    lo, hi = si110.extrema()
    # Si(110) electron well depth ~22 eV at room temperature (Andersen et al.
    # 1983; Azadegan 2013 Fig. 1).
    assert 19.0 < hi - lo < 24.0
    assert np.allclose(si110.potential_eV.imag, 0.0, atol=1e-12)  # centrosymmetric
    assert si110.potential_eV[0] == 0.0


def test_planar_coefficient_is_negative_cbs_coupling():
    # Diamond is not edge-prone, so U_g uses the bare f0 there; for Si the CBS
    # coupling adds anomalous f' and the two intentionally differ.
    profile = planar_profile("diamond", (1, 1, 0), 293.15)
    b = profile.b_factors_ang2["C"]
    expected = -U_g("diamond", (2, 2, 0), 1.0e4, B_ang2=b, use_henke=False)
    assert profile.potential_eV[1] == pytest.approx(expected, rel=1e-12)


def test_density_means_and_heating(si110):
    volume = CRYSTALS["silicon"]["V_cell"]
    assert si110.nuclear_density[0].real == pytest.approx(8 / volume)
    assert si110.electron_density[0].real == pytest.approx(8 * 14 / volume)
    cold = planar_profile("silicon", (1, 1, 0), 100.0)
    hot = planar_profile("silicon", (1, 1, 0), 400.0)
    depth = [p.extrema()[1] - p.extrema()[0] for p in (cold, si110, hot)]
    assert depth[0] > depth[1] > depth[2]  # thermal smearing shallows the well


# ---- transverse states (cr-transverse-eigenproblem, cr-dipole-matrix-element) -


def test_free_limit_has_no_bound_states():
    free = PlanarProfile(
        crystal="free",
        hkl=(1, 0, 0),
        period_ang=2.0,
        potential_eV=np.zeros(4, dtype=complex),
        nuclear_density=np.zeros(4, dtype=complex),
        electron_density=np.zeros(4, dtype=complex),
        temperature_K=0.0,
        b_factors_ang2={},
    )
    states = transverse_states(free, gamma=10.0, k_inv_ang=0.3, n_plane_waves=8)
    g = free.g_inv_ang
    expected = np.sort(3.80998 / 10.0 * (0.3 + np.arange(-8, 9) * g) ** 2)
    np.testing.assert_allclose(states.energies_eV, expected, rtol=1e-4)
    assert states.n_bound == 0


def test_harmonic_limit_gives_equal_spacing():
    v0, period, gamma = 1000.0, 2.0, 100.0
    cosine = PlanarProfile(
        crystal="cosine",
        hkl=(1, 0, 0),
        period_ang=period,
        potential_eV=np.array([0.0, -v0 / 2], dtype=complex),
        nuclear_density=np.zeros(2, dtype=complex),
        electron_density=np.zeros(2, dtype=complex),
        temperature_K=0.0,
        b_factors_ang2={},
    )
    states = transverse_states(cosine, gamma, n_plane_waves=48)
    a = HBARC_EV_ANG**2 / (2 * 510998.95) / gamma
    hbar_omega = np.sqrt(2.0 * a * v0 * cosine.g_inv_ang**2)
    spacing = np.diff(states.energies_eV[:4])
    # Mathieu expansion: E_n = -V0 + hw (n + 1/2) - (a g^2 / 16)(2n^2 + 2n + 1),
    # so successive spacings fall by a g^2 / 4.
    quartic = a * cosine.g_inv_ang**2 / 4.0
    expected = hbar_omega - quartic * np.arange(1, 4)
    np.testing.assert_allclose(spacing, expected, rtol=2e-4)
    assert states.energies_eV[0] == pytest.approx(-v0 + hbar_omega / 2 - quartic / 4, rel=1e-5)


def test_parity_alternates_and_selection_rule_emerges(si110):
    gamma, _ = longitudinal_kinematics(40e6)
    states = transverse_states(si110, gamma)
    assert [states.parity(n) for n in range(6)] == [1, -1, 1, -1, 1, -1]
    assert abs(states.dipole_ang(0, 1)) > 0.05
    assert abs(states.dipole_ang(0, 2)) < 1e-12
    assert abs(states.dipole_ang(1, 3)) < 1e-12


def test_dipole_matches_position_operator_for_deep_states(si110):
    gamma, _ = longitudinal_kinematics(40e6)
    states = transverse_states(si110, gamma)
    x = np.linspace(-si110.period_ang / 2, si110.period_ang / 2, 4001)
    psi0, psi1 = states.wavefunction(x, 0), states.wavefunction(x, 1)
    direct = np.trapezoid(np.conj(psi0) * x * psi1, x)
    assert abs(states.dipole_ang(0, 1)) == pytest.approx(abs(direct), rel=2e-3)


def test_bound_state_count_scales_as_sqrt_gamma(si110):
    counts = [
        transverse_states(si110, longitudinal_kinematics(t)[0], n_plane_waves=48).n_bound
        for t in (40e6, 160e6, 640e6)
    ]
    assert counts[1] == pytest.approx(2 * counts[0], abs=1)
    assert counts[2] == pytest.approx(2 * counts[1], abs=1)


def test_si110_40mev_matches_azadegan(si110):
    # Azadegan, Comput. Phys. Commun. 184, 1064 (2013), Sec. 4: six bound
    # states and E0(1->0) = 2 gamma^2 (e1 - e0) = 76.2071 keV at k = 1.3089/A.
    gamma, _ = longitudinal_kinematics(40e6)
    states = transverse_states(si110, gamma, k_inv_ang=1.3089)
    assert states.n_bound == 6
    e10 = 2 * gamma**2 * (states.energies_eV[1] - states.energies_eV[0])
    assert e10 == pytest.approx(76207.1, rel=0.02)


# ---- entry populations (cr-initial-occupation) -------------------------------


def test_entry_populations_unitary_and_parity_selective(si110):
    angles = np.array([0.0, 1e-4, 3e-4, -6e-4, 1.5e-3])
    populations = entry_populations(si110, 40e6, angles)
    np.testing.assert_allclose(populations.sum(axis=-1), 1.0, atol=1e-12)
    at_zero = populations[0, :6]
    assert np.all(at_zero[1::2] < 1e-20)  # only even states at normal incidence
    assert at_zero[0] == pytest.approx(0.24, abs=0.03)  # Azadegan 2013 Fig. 3
    np.testing.assert_allclose(
        populations[2], entry_populations(si110, 40e6, -3e-4), rtol=1e-9, atol=1e-14
    )


def test_divergence_average_is_normalized_and_mixes_parity(si110):
    sharp = entry_populations(si110, 40e6, 0.0)
    smeared = entry_populations(si110, 40e6, 0.0, divergence_rad=2e-4)
    assert smeared.sum() == pytest.approx(1.0, abs=1e-12)
    assert sharp[1] < 1e-20 < 1e-3 < smeared[1]


# ---- kinematics (cr-doppler-line-energy) -------------------------------------


def test_forward_line_energy_is_doppler_upshift():
    gamma, beta = longitudinal_kinematics(40e6)
    assert line_energy_eV(1.0, 40e6) == pytest.approx(gamma**2 * (1 + beta), rel=1e-12)
    naive = 1.0 - beta * np.cos(0.3)
    assert doppler_denominator(40e6, 0.3) == pytest.approx(naive, rel=1e-12)


@pytest.mark.parametrize(("energy", "theta_deg"), [(5e6, 3.0), (5e6, 20.0), (30e3, 120.0)])
def test_doppler_denominator_matches_pxr_resonance(energy, theta_deg):
    # With zero tilt and the catalog (440) surface reflection, photon_energies
    # returns hbar c beta |g| / (1 - beta cos theta): same denominator.
    _, magnitude = reciprocal_g_vector((4, 4, 0), CRYSTALS["silicon"]["lattice"])
    photon = photon_energies("silicon", energy, theta_obs_deg=theta_deg, hkl=(4, 4, 0))[0]
    _, beta = longitudinal_kinematics(energy)
    pxr_denominator = HBARC_EV_ANG * beta * magnitude / photon
    assert doppler_denominator(energy, np.deg2rad(theta_deg)) == pytest.approx(
        pxr_denominator, rel=1e-12
    )
