"""Quantity wrappers preserve established deterministic numeric contracts (#816)."""

import numpy as np
import pytest

from pyrite.materials import atomic as a
from pyrite.materials import crystal as c
from pyrite.materials import quantities as q
from pyrite.units import Q_, ureg


def test_constants_use_canonical_values_and_gaussian_energy_length():
    constants = q.crystal_constants()
    for name, unit in (
        ("HC_EV_ANG", "eV*angstrom"),
        ("HBARC_EV_ANG", "eV*angstrom"),
        ("M_E_EV", "eV"),
        ("R_E_ANG", "angstrom"),
        ("E2_EV_ANG", "eV*angstrom"),
    ):
        assert constants[name].to(unit).magnitude == pytest.approx(getattr(c, name))
    constants["M_E_EV"].ito("keV")
    assert q.crystal_constants()["M_E_EV"].units == ureg.eV
    assert constants["E2_EV_ANG"].dimensionality != (ureg.coulomb**2).dimensionality


def test_atomic_wrappers_keep_complex_values_shapes_and_nan_policy():
    g = Q_(np.array([[0.0], [2.0]]), "1/angstrom")
    e = Q_(np.array([[0.0, 500.0, 1000.0]]), "eV")
    f0 = q.cromer_mann_f0_q("Si", g)
    np.testing.assert_allclose(f0.magnitude, a.cromer_mann_f0("Si", g.magnitude))
    fp, fpp = q.henke_dispersion_q("Si", e)
    np.testing.assert_array_equal(
        np.isnan(fp.magnitude), np.isnan(a.henke_dispersion("Si", e.magnitude)[0])
    )
    assert fp.shape == fpp.shape == (1, 3)
    f = q.atomic_form_factor_q("Si", g, e)
    assert f.shape == (2, 3)
    assert f.magnitude.dtype.kind == "c"
    np.testing.assert_allclose(
        f.magnitude[:, 1:], a.atomic_form_factor("Si", g.magnitude, e.magnitude)[:, 1:]
    )
    assert np.isnan(f.magnitude[:, 0]).all()
    f.magnitude[0, 1] = 0
    assert q.atomic_form_factor_q("Si", g, e).magnitude[0, 1] != 0
    with pytest.raises(ValueError):
        q.henke_dispersion_q("Si", e, on_out_of_range="raise")
    with pytest.raises(TypeError):
        q.cromer_mann_f0_q("Si", 2.0)
    with pytest.raises(ValueError):
        q.atomic_form_factor_q("Si", Q_(2, "eV"), e)


def test_crystal_geometry_and_couplings_preserve_units():
    lattice = {"system": "cubic", "a": Q_(0.5431, "nm")}
    vec, norm = q.reciprocal_g_vector_q((1, 1, 1), lattice)
    original, magnitude = c.reciprocal_g_vector((1, 1, 1), {"system": "cubic", "a": 5.431})
    np.testing.assert_allclose(vec.magnitude, original)
    assert norm.to("1/nm").magnitude == pytest.approx(magnitude * 10)
    b = Q_(0.0, "angstrom**2")
    energy = Q_(1.0, "keV")
    sf, gm = q.structure_factor_q("silicon", (1, 1, 1), energy, b_factor=b)
    a0, g0 = c.structure_factor("silicon", (1, 1, 1), 1000.0, 0.0)
    np.testing.assert_allclose(sf.magnitude, a0)
    assert gm.magnitude == pytest.approx(g0)
    assert q.chi_g_q("silicon", (1, 1, 1), energy, b_factor=b).magnitude == pytest.approx(
        c.chi_g("silicon", (1, 1, 1), 1000.0, 0.0)
    )
    assert q.U_g_q("silicon", (1, 1, 1), energy, b_factor=b).to("keV").magnitude == pytest.approx(
        c.U_g("silicon", (1, 1, 1), 1000.0, 0.0) / 1000
    )
    with pytest.raises(ValueError):
        q.reciprocal_g_vector_q((1, 1, 1), {"system": "cubic", "a": 1 * ureg.keV})


def test_absorption_and_reflections_keep_domain_and_indices():
    energy = Q_(np.array([0.0, 1000.0]), "eV")
    length = q.absorption_length_q("Si", energy, Q_(0.05, "1/angstrom**3"))
    assert np.isnan(length.magnitude[0])
    np.testing.assert_allclose(length.magnitude[1], c.absorption_length_ang("Si", 1000.0, 0.05))
    selected = q.dominant_reflections_q(
        "silicon",
        n_families=1,
        reference_energy=Q_(1000, "eV"),
        b_factor=Q_(0, "angstrom**2"),
        g_max=Q_(3, "1/angstrom"),
    )
    assert selected == c.dominant_reflections("silicon", 1, 1000.0, 0.0, g_max_invang=3.0)


def test_inputs_do_not_convert_bare_numbers_or_dimensionless_angles():
    with pytest.raises(TypeError):
        q.chi_g_q("silicon", (1, 1, 1), 1000, b_factor=Q_(0, "angstrom**2"))
    with pytest.raises(ValueError):
        q.photon_energies_q(
            "silicon",
            Q_(60, "keV"),
            tilt=Q_(0, ""),
            tilt_azimuth=Q_(0, "degree"),
            observation_polar=Q_(90, "degree"),
            observation_azimuth=Q_(0, "degree"),
        )
