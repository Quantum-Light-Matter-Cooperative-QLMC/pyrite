"""EEDL bremsstrahlung parsing, interpolation, and backend selection."""

from __future__ import annotations

import numpy as np
import pytest

from pyrite._backend import REAL, xp
from pyrite.montecarlo.spectrum import brem
from tests.helpers import scaled_rtol, to_host


def _zero_mu(_composition, energy):
    """Transparent stand-in for ``_mu_total_inv_ang``.

    The real coefficient returns on the device of its input, so the stub has to
    as well: a NumPy-returning lambda blows up under ``PYRITE_TEST_BACKEND=cuda``
    the moment it is handed a CuPy energy array.
    """
    return xp.zeros_like(xp.asarray(energy, dtype=REAL))


def _single_carbon_segment(T_keV: float, length_ang: float = 100.0):
    return {
        "r_mid": np.array([[0.0, 0.0, 0.5 * length_ang]]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "L_ang": np.array([length_ang]),
        "E_keV": np.array([T_keV]),
        "elec_id": np.array([0]),
        "Ne": 1,
        "thickness_ang": length_ang,
    }


def test_packaged_carbon_total_cross_section_and_photon_tables():
    table = brem.load_bremsstrahlung_cross_sections("C")

    assert table.atomic_number == 6
    sigma_30kev = np.interp(
        30_000.0,
        table.incident_energy_eV,
        table.total_cross_section_cm2,
    )
    assert sigma_30kev == pytest.approx(42.85162635741151e-24, rel=1.0e-13)
    assert len(table.photon_energy_eV_by_incident) == 9
    for energy, density in zip(
        table.photon_energy_eV_by_incident,
        table.photon_probability_density_per_eV_by_incident,
        strict=True,
    ):
        assert np.trapezoid(density, energy) == pytest.approx(1.0, abs=6.0e-7)
        assert not energy.flags.writeable
        assert not density.flags.writeable


def test_eedl_differential_cross_section_integrates_to_total_at_native_panel():
    table = brem.load_bremsstrahlung_cross_sections("C")
    panel = 4
    incident_eV = table.distribution_incident_energy_eV[panel]
    photon_eV = table.photon_energy_eV_by_incident[panel]

    differential = to_host(
        brem._eedl_brem_dsigma_dk("C", np.array([incident_eV / 1.0e3]), photon_eV)
    )[0]
    expected_total = np.interp(
        incident_eV,
        table.incident_energy_eV,
        table.total_cross_section_cm2,
    )

    assert np.trapezoid(differential, photon_eV) == pytest.approx(expected_total, rel=6.0e-7)


def test_eedl_interpolated_distribution_is_normalized_after_physical_cutoff():
    table = brem.load_bremsstrahlung_cross_sections("C")
    incident_eV = 30_000.0
    photon_eV = np.unique(
        np.concatenate(
            (
                np.geomspace(0.1, incident_eV, 50_000),
                [incident_eV],
            )
        )
    )

    differential = to_host(
        brem._eedl_brem_dsigma_dk("C", np.array([incident_eV / 1.0e3]), photon_eV)
    )[0]
    expected_total = np.interp(
        incident_eV,
        table.incident_energy_eV,
        table.total_cross_section_cm2,
    )

    assert np.trapezoid(differential, photon_eV) == pytest.approx(expected_total, rel=2.0e-4)
    assert differential[photon_eV > incident_eV].size == 0


def test_mc_brem_defaults_to_eedl_and_retains_isotropic_angular_model(monkeypatch):
    table = brem.load_bremsstrahlung_cross_sections("C")
    panel = 4
    incident_eV = table.distribution_incident_energy_eV[panel]
    photon_eV = table.photon_energy_eV_by_incident[panel]
    density_ang3 = 0.1
    length_ang = 100.0
    monkeypatch.setattr(brem, "_mu_total_inv_ang", _zero_mu)

    spectrum = brem.mc_brem_spectrum(
        _single_carbon_segment(incident_eV / 1.0e3, length_ang),
        photon_eV,
        composition=[("C", density_ang3)],
    )
    sigma = np.interp(
        incident_eV,
        table.incident_energy_eV,
        table.total_cross_section_cm2,
    )
    expected = (
        density_ang3
        * 1.0e24
        * length_ang
        * 1.0e-8
        * sigma
        * table.photon_probability_density_per_eV_by_incident[panel]
        / (4.0 * np.pi)
    )

    # Each bin is an unreduced product chain evaluated at the backend's REAL, so
    # a few ulps is the whole budget; fp64 keeps the original bound.
    np.testing.assert_allclose(
        spectrum, expected, rtol=scaled_rtol(2.0e-12, eps_multiple=8.0), atol=0.0
    )


def test_bethe_heitler_remains_an_explicit_backend():
    energy = np.array([100.0, 700.0, 2000.0])
    T_keV = np.array([10.0, 30.0])

    selected = brem._bremsstrahlung_dsigma_dk(
        "C", T_keV, energy, cross_section_model="bethe-heitler"
    )
    historical = brem._brem_dsigma_dk(6, T_keV, energy)

    np.testing.assert_array_equal(to_host(selected), to_host(historical))


def test_missing_eedl_element_warns_and_falls_back_to_bethe_heitler(monkeypatch):
    def unavailable(_element, *, data_dir=None):
        del data_dir
        raise brem.EEDLBremsstrahlungDataUnavailable("test tape has no carbon")

    monkeypatch.setattr(brem, "load_bremsstrahlung_cross_sections", unavailable)
    T_keV = np.array([30.0])
    energy = np.array([700.0, 2000.0])

    with pytest.warns(RuntimeWarning, match="falling back to Bethe-Heitler"):
        selected = brem._bremsstrahlung_dsigma_dk("C", T_keV, energy)

    np.testing.assert_array_equal(
        to_host(selected), to_host(brem._brem_dsigma_dk(6, T_keV, energy))
    )


def test_incident_energies_outside_eedl_range_warn_and_fall_back():
    T_keV = np.array([0.005])
    energy = np.array([1.0, 2.0])

    with pytest.warns(RuntimeWarning, match="outside the EEDL range"):
        selected = brem._bremsstrahlung_dsigma_dk("C", T_keV, energy)

    np.testing.assert_array_equal(
        to_host(selected), to_host(brem._brem_dsigma_dk(6, T_keV, energy))
    )


def test_unknown_bremsstrahlung_backend_is_rejected():
    with pytest.raises(ValueError, match="cross_section_model"):
        brem.mc_brem_spectrum(
            _single_carbon_segment(30.0),
            np.array([700.0]),
            composition=[("C", 0.1)],
            cross_section_model="invented",
        )


def test_mc_brem_stages_eedl_grid_once_across_reduction_chunks(monkeypatch):
    segments = {
        "r_mid": np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 20.0], [0.0, 0.0, 30.0]]),
        "v_hat": np.tile([0.0, 0.0, 1.0], (3, 1)),
        "L_ang": np.full(3, 10.0),
        "E_keV": np.array([20.0, 25.0, 30.0]),
        "elec_id": np.arange(3),
        "Ne": 3,
        "thickness_ang": 40.0,
    }
    calls = 0
    original = brem._prepare_eedl_grid

    def counted(table, photon_energy_eV):
        nonlocal calls
        calls += 1
        return original(table, photon_energy_eV)

    monkeypatch.setattr(brem, "_prepare_eedl_grid", counted)
    brem.mc_brem_spectrum(
        segments,
        np.array([100.0, 500.0, 1000.0]),
        composition=[("C", 0.1)],
        chunk=1,
    )

    assert calls == 1
