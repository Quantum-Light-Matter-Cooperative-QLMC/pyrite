import numpy as np

from pyrite.montecarlo import transport as t


def _synthetic_layer(two_elements=False):
    Z = np.array([6.0, 14.0] if two_elements else [6.0], dtype=np.float64)
    J = np.array([0.081, 0.173] if two_elements else [0.081], dtype=np.float64)
    k = 0.731 + 0.0688 * np.log10(Z)
    density_ang3 = np.array([0.10, 0.05] if two_elements else [0.10], dtype=np.float64)
    n_cm3 = density_ang3 * 1e24
    coeff = (density_ang3 / 0.602214076) * Z

    sr_rate_numer = 5.21e-21 * Z * Z * 4.0 * np.pi * n_cm3
    z17 = Z**1.7
    mott_numer = 3.0e-18 * z17 * n_cm3
    mott_denom1 = 0.005 * z17
    mott_denom2 = 0.0007 * Z * Z
    sr_joy_numer = 3.4e-3 * Z**0.67

    return (
        [J],
        [Z],
        [k],
        [coeff],
        [sr_rate_numer],
        [mott_numer],
        [mott_denom1],
        [mott_denom2],
        [sr_joy_numer],
    )


def _interp_1d(table, lut, E):
    x = np.clip((E - lut.E_min_keV) * lut.inv_dE_keV, 0.0, lut.n_energy - 1.0)
    i = np.minimum(np.floor(x).astype(np.int64), lut.n_energy - 2)
    f = x - i
    return table[i] + f * (table[i + 1] - table[i])


def _interp_2d(table, row, lut, E):
    x = np.clip((E - lut.E_min_keV) * lut.inv_dE_keV, 0.0, lut.n_energy - 1.0)
    i = np.minimum(np.floor(x).astype(np.int64), lut.n_energy - 2)
    f = x - i
    return table[row, i] + f * (table[row, i + 1] - table[row, i])


def test_sr_transport_lut_matches_direct_scalar_physics():
    tables = _synthetic_layer()
    L_Js, L_Zs, L_ks, L_coeffs, L_sr, L_mn, L_d1, L_d2, L_sj = tables
    lut = t.build_transport_energy_lut(
        5.0,
        30.0,
        0,
        L_Js,
        L_Zs,
        L_ks,
        L_coeffs,
        L_sr,
        L_mn,
        L_d1,
        L_d2,
        L_sj,
        [[None]],
        config=t.TransportLUTConfig(step_keV=0.025, min_points=2),
    )

    rng = np.random.default_rng(1234)
    energies = rng.uniform(5.0, 30.0, 2000)
    rate_exact = np.array(
        [t._scatter_rates_sr_scalar(E, L_sr[0][0], L_sj[0][0]) for E in energies]
    )
    dEds_exact = np.array(
        [t._dEds_compound_scalar(L_Js[0], L_ks[0], L_coeffs[0], E) for E in energies]
    )
    inv_beta_exact = 1.0 / np.array([t.beta_from_keV_scalar(E) for E in energies])

    np.testing.assert_allclose(_interp_2d(lut.total_rate, 0, lut, energies), rate_exact, rtol=1e-5)
    np.testing.assert_allclose(_interp_2d(lut.dEds, 0, lut, energies), dEds_exact, rtol=1e-5)
    np.testing.assert_allclose(_interp_1d(lut.inv_beta, lut, energies), inv_beta_exact, rtol=1e-5)


def test_transport_lut_cdf_is_normalized_and_monotone():
    tables = _synthetic_layer(two_elements=True)
    L_Js, L_Zs, L_ks, L_coeffs, L_sr, L_mn, L_d1, L_d2, L_sj = tables
    lut = t.build_transport_energy_lut(
        5.0,
        60.0,
        0,
        L_Js,
        L_Zs,
        L_ks,
        L_coeffs,
        L_sr,
        L_mn,
        L_d1,
        L_d2,
        L_sj,
        [[None, None]],
        config=t.TransportLUTConfig(step_keV=0.05, min_points=2),
    )

    live = lut.cdf[0, :2]
    assert np.all(np.diff(live, axis=0) >= 0.0)
    np.testing.assert_array_equal(live[-1], np.ones(lut.n_energy))


def test_lut_index_clamps_both_endpoints():
    i, f = t._lut_index_frac_scalar(1.0, 5.0, 2.0, 101)
    assert i == 0
    assert f == 0.0

    i, f = t._lut_index_frac_scalar(1000.0, 5.0, 2.0, 101)
    assert i == 99
    assert f == 1.0
