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
from pyrite.materials.crystal import (
    CRYSTALS,
    E2_EV_ANG,
    HBARC_EV_ANG,
    M_E_EV,
    U_g,
    reciprocal_g_vector,
)
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


def test_diamond_b_factor_matches_xray_measurements():
    # Takama et al., Acta Cryst. A 46, 514 (1990), abstract: B = 0.142(9) A^2
    # (Pendelloesung); Bindzus et al. (2014) U_iso = 0.001808(10) A^2, i.e.
    # B = 0.1427 A^2. The catalog B_ang2 = 0.21 used by PXR/CBS is not.
    b = thermal_b_factor("diamond", "C", 293.15)
    assert abs(b - 0.142) < 0.009
    assert b == pytest.approx(8 * np.pi**2 * 0.001808, rel=0.02)


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


def test_si111_coefficient_against_cbed_structure_factors():
    # Zuo, Microsc. Res. Tech. 46, 220 (1999), Table 1 "Mean" column: Si at room
    # temperature, 196.7 kV, U(111) = 0.4736(7)e-1 and U(222) = 0.921(50)e-3 A^-2.
    # U_g = 2 m gamma e V_g / h^2 (Zuo & Spence, Ultramicroscopy 35, 185 (1991),
    # Eq. 8), so V_g = U_g (2 pi hbar c)^2 / (2 gamma m c^2).
    gamma = 1.0 + 196.7e3 / M_E_EV
    volts_per_inv_ang2 = (2 * np.pi * HBARC_EV_ANG) ** 2 / (2 * gamma * M_E_EV)
    v111 = 0.4736e-1 * volts_per_inv_ang2  # 5.144 V
    v222 = 0.921e-3 * volts_per_inv_ang2  # 0.100 V
    profile = planar_profile("silicon", (1, 1, 1), 293.15)
    # Independent atoms overestimate the bonding-sensitive (111) by ~6%; the
    # X-ray Pendellosung value in Zuo's Table 2 agrees with the CBED value.
    assert 1.0 < abs(profile.potential_eV[1]) / v111 < 1.08
    # (222) is forbidden for spherical harmonic atoms; the measured 0.1 V is
    # bonding charge the independent-atom potential omits (2% of V_111).
    assert abs(profile.potential_eV[2]) < 1e-12
    assert v222 < 0.03 * v111


def _diamond_structure_modulus(hkl):
    basis = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    basis += [(x + 0.25, y + 0.25, z + 0.25) for x, y, z in basis]
    return abs(sum(np.exp(2j * np.pi * np.dot(hkl, r)) for r in basis))


@pytest.mark.parametrize(
    ("plane", "harmonic", "hkl", "f_measured", "ratio"),
    [
        ((1, 1, 0), 1, (2, 2, 0), 15.404, (0.99, 1.01)),
        ((1, 1, 0), 2, (4, 4, 0), 9.659, (0.99, 1.01)),
        ((1, 1, 0), 3, (6, 6, 0), 5.913, (0.99, 1.01)),
        ((1, 1, 1), 1, (1, 1, 1), 18.574, (1.05, 1.15)),
    ],
)
def test_diamond_coefficients_against_xray_structure_factors(
    plane, harmonic, hkl, f_measured, ratio
):
    # Bindzus et al., Acta Cryst. A 70, 39 (2014), supporting information
    # Sec. 1.4 (HC/Rietveld extracted, cubic cell, thermally smeared with
    # U_iso = 0.001808 A^2, Table 2): F(220) 15.404, F(440) [as 404] 9.659,
    # F(660) [as 606] 5.913, F(111) 18.574. Mott--Bethe on the measured
    # charge: |U_g| = 4 pi e^2 (Z |S| T - F) / (V g^2), T = exp(-B s^2).
    a = 3.566606
    b = 8 * np.pi**2 * 0.001808
    g = 2 * np.pi * np.sqrt(sum(np.square(hkl))) / a
    s = g / (4 * np.pi)
    debye_waller = np.exp(-b * s**2)
    measured = (
        4
        * np.pi
        * E2_EV_ANG
        / (a**3 * g**2)
        * (6 * _diamond_structure_modulus(hkl) * debye_waller - f_measured)
    )
    profile = planar_profile("diamond", plane, 293.15)
    # The (110) family is independent-atom to within 0.5%; bonding charge
    # lowers the measured (111) potential by ~10%.
    assert ratio[0] < abs(profile.potential_eV[harmonic]) / measured < ratio[1]


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


@pytest.mark.parametrize(
    ("plane", "energy_MeV", "transition", "measured_keV", "rel"),
    [
        ((1, 1, 0), 80.23, (1, 0), 313.4, 0.005),
        ((1, 1, 0), 80.23, (2, 1), 213.9, 0.005),
        ((1, 1, 0), 80.23, (3, 2), 164.6, 0.005),
        ((1, 1, 0), 80.23, (4, 3), 130.1, 0.01),
        ((1, 1, 0), 53.2, (1, 0), 160.0, 0.02),
        ((1, 1, 0), 53.2, (2, 1), 103.6, 0.02),
        ((1, 1, 0), 53.2, (3, 2), 78.0, 0.02),
        ((1, 0, 0), 53.2, (1, 0), 119.8, 0.02),
        ((1, 0, 0), 80.23, (1, 0), 239.0, 0.02),
        ((1, 0, 0), 110.2, (1, 0), 405.0, 0.02),
    ],
)
def test_diamond_lines_match_gouanere(plane, energy_MeV, transition, measured_keV, rel):
    # Gouanere et al., Phys. Rev. B 38, 4352 (1988), Table II column 10:
    # forward line energies (E^if)_0 at Psi = 0, corrected for multiple
    # scattering; room temperature, 22 um (110) and 15 um (100) crystals.
    profile = planar_profile("diamond", plane, 293.15)
    gamma, _ = longitudinal_kinematics(energy_MeV * 1e6)
    states = transverse_states(profile, gamma)
    upper, lower = transition
    line = line_energy_eV(states.energies_eV[upper] - states.energies_eV[lower], energy_MeV * 1e6)
    assert line / 1e3 == pytest.approx(measured_keV, rel=rel)


def test_diamond110_9mev_line_within_bloch_band():
    # Genz et al., Phys. Rev. B 53, 8922 (1996), Table VII: (110) 1->0 at
    # 9.0 MeV is 7.80-7.93 keV (+-0.08) over four crystals, detector at 0 deg.
    # State 1 sits near the barrier, so the line spans a Bloch band.
    profile = planar_profile("diamond", (1, 1, 0), 293.15)
    gamma, _ = longitudinal_kinematics(9.0e6)
    lines = []
    for k in np.linspace(0.0, profile.g_inv_ang / 2, 9):
        energies = transverse_states(profile, gamma, k_inv_ang=k).energies_eV
        lines.append(line_energy_eV(energies[1] - energies[0], 9.0e6) / 1e3)
    assert min(lines) < 7.80 and 7.93 < max(lines)
    assert np.mean(lines) == pytest.approx(7.865, rel=0.02)


@pytest.mark.parametrize("hkl", [(1, 1, 0), (1, 1, 1), (1, 0, 0)])
def test_si_planar_lines_below_watson_detector_cutoff(hkl):
    # Watson & Koehler, Phys. Rev. A 24, 861 (1981) and Phys. Rev. B 25, 3079
    # (1982), Sec. IV B: no planar channeling radiation above the ~1 keV Si(Li)
    # cutoff for any Si plane up to 3.0 MeV, detector 2.66 deg off the beam.
    # Every strong (|x_fi| > 0.1 A) bound-bound line must therefore lie below it.
    profile = planar_profile("silicon", hkl, 293.15)
    gamma, _ = longitudinal_kinematics(3.0e6)
    theta = np.deg2rad(2.66)
    highest = 0.0
    for k in np.linspace(0.0, profile.g_inv_ang / 2, 5):
        states = transverse_states(profile, gamma, k_inv_ang=k)
        energies = states.energies_eV
        for i in range(1, states.n_bound):
            for f in range(i):
                if abs(states.dipole_ang(f, i)) > 0.1:
                    line = line_energy_eV(energies[i] - energies[f], 3.0e6, theta)
                    highest = max(highest, line)
    assert highest < 1.05e3


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


def test_planar_wells_sit_on_atomic_planes():
    """``x`` is measured from the cell origin: Si(111) planes at 0 and 3d/4 (narrow gap d/4)."""
    from scipy.signal import argrelmin

    profile = planar_profile("silicon", (1, 1, 1))
    d = profile.period_ang
    x = np.arange(4096) * d / 4096
    u = profile.potential(x)
    wells = x[argrelmin(u, mode="wrap")[0]]
    # Each well is pulled ~0.02 angstrom toward its near neighbour, d/4 away.
    for plane in (0.0, 0.75 * d):
        offset = (wells - plane + 0.5 * d) % d - 0.5 * d
        assert np.min(np.abs(offset)) < 0.03
