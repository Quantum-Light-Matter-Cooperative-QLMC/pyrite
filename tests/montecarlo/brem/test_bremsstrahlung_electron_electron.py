"""Electron-electron bremsstrahlung ratio eta from Seltzer and Berger (1985) Table 4.

Validation: bremslib-electron-electron
"""

import tomllib

import numpy as np
import pytest

from pyrite._bremslib_electron import (
    ELECTRON_ELECTRON_DATA,
    electron_electron_eta,
    electron_electron_factor,
)


@pytest.fixture(scope="module")
def raw():
    with ELECTRON_ELECTRON_DATA.open("rb") as handle:
        return tomllib.load(handle)


def test_tabulated_nodes_are_reproduced(raw):
    t1 = np.asarray(raw["incident_energy_MeV"])
    x = np.asarray(raw["reduced_energy"])
    for z in raw["atomic_number"]:
        eta = np.asarray(raw[f"Z{z}"]["eta"])
        np.testing.assert_allclose(electron_electron_eta(z, t1, x), eta, atol=1e-12)


def test_transcription_has_the_source_layout_and_errata(raw):
    assert raw["atomic_number"] == [6, 13, 29, 47, 74, 92]
    assert len(raw["incident_energy_MeV"]) == 31
    for z in raw["atomic_number"]:
        eta = np.asarray(raw[f"Z{z}"]["eta"])
        assert eta.shape == (31, 14)
        np.testing.assert_array_equal(eta[:, -1], 0.0)  # k_max < T for an electron target
        assert np.all(np.diff(raw["incident_energy_MeV"]) > 0)
    t1 = raw["incident_energy_MeV"]
    silver = np.asarray(raw["Z47"]["eta"])
    # Misprints read from their neighbours (see the data file header).
    assert silver[t1.index(5.0), 1] == 1.012
    assert silver[t1.index(60.0), 7] == 1.012


def test_small_reduced_energy_shape_joins_both_tabulated_nodes():
    t1 = np.array([0.1, 1.0, 30.0, 100.0])
    x = np.array([0.0, 1e-6, 0.004, 0.02, 0.06, 0.1 - 1e-12, 0.1])
    eta = electron_electron_eta(14, t1, x)
    np.testing.assert_allclose(eta[:, 1], eta[:, 0], rtol=1e-3)  # soft-photon limit
    np.testing.assert_allclose(eta[:, -2], eta[:, -1], rtol=1e-9)  # continuous at 0.1
    # Above 0.5 MeV eta falls monotonically from its soft-photon value.
    assert np.all(np.diff(eta[1:], axis=1) <= 0.0)


def test_factor_limits_and_z_interpolation():
    t1 = np.array([0.05, 1.0, 30.0, 100.0])
    x = np.linspace(0.0, 1.0, 41)
    for z in (1, 6, 14, 29, 74, 92, 100):
        factor = electron_electron_factor(z, t1, x)
        assert np.all(factor >= 1.0)
        np.testing.assert_array_equal(factor[:, -1], 1.0)
    between = electron_electron_eta(14, t1, x)
    low, high = electron_electron_eta(13, t1, x), electron_electron_eta(29, t1, x)
    assert np.all(between >= np.minimum(low, high) - 1e-12)
    assert np.all(between <= np.maximum(low, high) + 1e-12)
    # Held outside the tabulated Z range.
    np.testing.assert_array_equal(electron_electron_eta(1, t1, x), electron_electron_eta(6, t1, x))


def test_weighted_mean_reproduces_the_tabulated_etabar(raw):
    """Limiting case: the integrated radiative loss is (1 + etabar/Z) times nuclear."""
    from pyrite.validation.brem_sources import (
        SeltzerBergerUnavailableError,
        load_seltzer_berger,
    )

    try:
        sb = load_seltzer_berger()
    except SeltzerBergerUnavailableError as exc:
        pytest.skip(str(exc))
    t1 = np.asarray(raw["incident_energy_MeV"])
    x = np.unique(np.r_[np.geomspace(1e-7, 0.1, 400), np.linspace(0.1, 1.0, 901), 0.0])
    for z in raw["atomic_number"]:
        etabar = np.asarray(raw[f"Z{z}"]["etabar"])
        for row in np.flatnonzero((t1 >= 0.01) & (t1 <= 100.0)):
            eta = electron_electron_eta(z, t1[row : row + 1], x)[0]
            # Seltzer-Berger's digital table is the electron-atom total.
            chi_n = np.interp(x, sb.kappa, sb.chi(z, t1[row])) / (1.0 + eta / z)
            mean = np.trapezoid(chi_n * eta, x) / np.trapezoid(chi_n, x)
            assert abs(mean - etabar[row]) <= 0.005, (z, t1[row], mean, etabar[row])
