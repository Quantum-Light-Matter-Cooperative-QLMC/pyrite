"""Static planar channeling profile: thermal data and continuum potential."""

import numpy as np
import pytest

from pyrite.materials.channeling import (
    ChannelingDataError,
    debye_b_factor,
    debye_integral,
    fit_debye_temperature,
    load_thermal_sites,
    planar_profile,
    thermal_b_factor,
)
from pyrite.materials.crystal import (
    CRYSTALS,
    E2_EV_ANG,
    HBARC_EV_ANG,
    M_E_EV,
    U_g,
)

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


# ---- uniaxial layered rows (cr-uniaxial-thermal-projection) -----------------


def test_mos2_basal_profile_has_sandwich_wells():
    profile = planar_profile("mos2", (0, 0, 1), 293.15)
    c = CRYSTALS["mos2"]["lattice"]["c"]
    # 2H stacking extincts odd 00l: true period is c/2, one S-Mo-S sandwich.
    assert profile.period_ang == pytest.approx(c / 2)
    assert profile.thermal_projected
    assert profile.b_factors_ang2["Mo"] == pytest.approx(8 * np.pi**2 * 0.0077)
    x = np.linspace(0.0, profile.period_ang, 4096, endpoint=False)
    u = profile.potential(x)
    minima = x[(u < np.roll(u, 1)) & (u < np.roll(u, -1))]
    # Wells on the S (z = 0.121, 0.379) and Mo (z = 0.25) planes.
    np.testing.assert_allclose(minima, np.array([0.121, 0.25, 0.379]) * c, atol=0.01)
    # The global maximum (barrier) is in the van der Waals gap at the cell edge.
    assert x[np.argmax(u)] < 0.05 * c or x[np.argmax(u)] > 0.45 * c


def test_mos2_prismatic_profile_uses_basal_displacement():
    profile = planar_profile("mos2", (1, 0, 0), 293.15)
    a = CRYSTALS["mos2"]["lattice"]["a"]
    assert profile.period_ang == pytest.approx(a * np.sqrt(3) / 2)
    assert profile.b_factors_ang2 == pytest.approx(
        {"Mo": 8 * np.pi**2 * 0.0048, "S": 8 * np.pi**2 * 0.0038}
    )


def test_mos2_prismatic_extinctions_and_cutoff_convergence():
    a = CRYSTALS["mos2"]["lattice"]["a"]
    # (110) planes of the hexagonal cell repeat every a/2.
    assert planar_profile("mos2", (1, 1, 0), 295.0).period_ang == pytest.approx(a / 2)
    for hkl in ((0, 0, 1), (1, 0, 0), (1, 0, 1)):
        full = planar_profile("mos2", hkl, 295.0)
        cut = planar_profile("mos2", hkl, 295.0, s_max_inv_ang=4.0)
        assert np.subtract(*full.extrema()) == pytest.approx(np.subtract(*cut.extrema()), abs=0.01)


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


@pytest.mark.parametrize("crystal", ["diamond", "silicon"])
def test_planar_coefficient_is_negative_cbs_coupling(crystal):
    # U_g is the static Z - f0 potential for every element (#513), so edge-prone
    # Si matches the planar profile as diamond does, at any photon energy.
    profile = planar_profile(crystal, (1, 1, 0), 293.15)
    b = next(iter(profile.b_factors_ang2.values()))
    for energy in (1.0e3, 1.839e3, 1.0e4):
        expected = -U_g(crystal, (2, 2, 0), energy, B_ang2=b, use_henke=False)
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
    iam = planar_profile("silicon", (1, 1, 1), 293.15, bonding_charge=False)
    # Independent atoms overestimate the bonding-sensitive (111) by ~6%, and
    # (222) is forbidden for spherical harmonic atoms.
    assert 1.04 < abs(iam.potential_eV[1]) / v111 < 1.08
    assert abs(iam.potential_eV[2]) < 1e-12
    # The X-ray (Cummings--Hart, Alkire) bonding charge reproduces this
    # independent CBED measurement: (111) to 0.1%, (222) within Zuo's 5.4%.
    # Validation: bonding-charge-structure-factors
    profile = planar_profile("silicon", (1, 1, 1), 293.15)
    assert abs(profile.potential_eV[1]) / v111 == pytest.approx(1.0, abs=1e-3)
    assert abs(profile.potential_eV[2]) == pytest.approx(v222, rel=0.050 / 0.921)


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
        ((1, 1, 1), 1, (1, 1, 1), 18.574, (0.98, 1.03)),
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
    # lowers the measured (111) potential by ~10%, which the Takama et al.
    # correction (bonding-charge-structure-factors) reproduces from an
    # independent data set to ~1.5%.
    assert ratio[0] < abs(profile.potential_eV[harmonic]) / measured < ratio[1]
    iam = planar_profile("diamond", plane, 293.15, bonding_charge=False)
    if hkl == (1, 1, 1):
        assert 1.05 < abs(iam.potential_eV[harmonic]) / measured < 1.15


def test_density_means_and_heating(si110):
    volume = CRYSTALS["silicon"]["V_cell"]
    assert si110.nuclear_density[0].real == pytest.approx(8 / volume)
    assert si110.electron_density[0].real == pytest.approx(8 * 14 / volume)
    cold = planar_profile("silicon", (1, 1, 0), 100.0)
    hot = planar_profile("silicon", (1, 1, 0), 400.0)
    depth = [p.extrema()[1] - p.extrema()[0] for p in (cold, si110, hot)]
    assert depth[0] > depth[1] > depth[2]  # thermal smearing shallows the well


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
