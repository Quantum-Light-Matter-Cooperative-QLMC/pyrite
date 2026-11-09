"""Out-of-band coherent energy envelope of optical TR/CTR (#615).

Validation: optical-ctr-population
"""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.materials.crystal import ALPHA_FS, HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.spectrum.optical_transition import C_ANG_PER_FS, OpticalBunchSource
from pyrite.montecarlo.spectrum.optical_transition_validity import (
    out_of_band_coherent_energy_eV,
    pec_crossing_energy_per_eV,
    radiated_energy_fraction,
)

_PC1 = 1.602176634e-7  # one electron [pC]


def _crossings(directions, interface, kinetic_keV=30.0, electron_id=None):
    directions = np.asarray(directions, dtype=float)
    k = len(directions)
    return SimpleNamespace(
        kinetic_energy_keV=np.full(k, kinetic_keV),
        direction=directions,
        interface=np.asarray(interface),
        z_interfaces_ang=np.array([0.0, 1e4, 2e4]),
        electron_id=np.arange(k) if electron_id is None else np.asarray(electron_id),
    )


def _beta(kinetic_keV):
    return np.sqrt(1.0 - 1.0 / (1.0 + kinetic_keV * 1e3 / M_E_EV) ** 2)


def test_normal_incidence_envelope_is_the_ginzburg_frank_hemisphere_energy():
    beta = _beta(30.0)
    mu, w = np.polynomial.legendre.leggauss(400)
    mu, w = 0.5 * (mu + 1.0), 0.5 * w
    # Independent closed form: alpha beta^2 sin^2 / (pi^2 (1 - beta^2 cos^2)^2) per eV per sr.
    closed = (
        2.0
        * np.pi
        * ALPHA_FS
        * beta**2
        / np.pi**2
        * np.sum(w * (1 - mu**2) / (1 - beta**2 * mu**2) ** 2)
    )
    entry = pec_crossing_energy_per_eV(_crossings([[0, 0, 1.0]], [0]), 64, 64)
    backscatter_exit = pec_crossing_energy_per_eV(_crossings([[0, 0, -1.0]], [0]), 64, 64)
    np.testing.assert_allclose(entry, closed, rtol=1e-10)
    np.testing.assert_allclose(backscatter_exit, closed, rtol=1e-10)


def test_inner_face_takes_the_larger_hemisphere_and_last_face_the_forward_one():
    d = [[np.sin(0.5), 0.0, np.cos(0.5)]]
    first, inner, last = (pec_crossing_energy_per_eV(_crossings(d, [i])) for i in (0, 1, 2))
    np.testing.assert_allclose(inner, np.maximum(first, last))
    assert np.all(first > 0) and np.all(last > 0)


def test_gaussian_tails_match_numerical_integration():
    sigma_fs, charge = 200.0, 1.0
    source = OpticalBunchSource.gaussian(charge, "gauss", rms_t_fs=sigma_fs)
    crossings = _crossings(
        [[0, 0, 1.0], [0, 0, 1.0], [0, 0, -1.0]], [0, 0, 0], electron_id=[0, 1, 1]
    )
    band = (0.01, 0.02)
    got = out_of_band_coherent_energy_eV(crossings, band, source, 2)
    per = pec_crossing_energy_per_eV(crossings)
    envelope = 0.5 * (1 * per[0] + 2 * (per[1] + per[2]))  # n_h sum_c U_c, mean over two histories
    sigma = sigma_fs * C_ANG_PER_FS
    e = np.linspace(0.0, 0.2, 400001)
    f2 = np.exp(-((e * sigma / HBARC_EV_ANG) ** 2))
    outside = (e < band[0]) | (e > band[1])
    tails = np.trapezoid(np.where(outside, f2, 0.0), e)
    expected = (source.physical_electrons - 1.0) * envelope * tails
    assert got == pytest.approx(expected, rel=1e-4)
    # 1 pC at 200 fs above the turnover: about 2e-4 of 30 keV (validator estimate).
    fraction = (
        out_of_band_coherent_energy_eV(_crossings([[0, 0, 1.0]], [0]), (1.4, 3.3), source, 1) / 30e3
    )
    assert 1e-4 < fraction < 3e-4


def test_out_of_band_envelope_refusals_and_trivial_cases():
    crossings = _crossings([[0, 0, 1.0]], [0])
    assert (
        out_of_band_coherent_energy_eV(
            crossings, (1.0, 2.0), OpticalBunchSource.sampled(_PC1, "one"), 1
        )
        == 0.0
    )
    with pytest.raises(ValueError, match="source law 'sampled'"):
        out_of_band_coherent_energy_eV(
            crossings, (1.0, 2.0), OpticalBunchSource.sampled(1.0, "x"), 1
        )
    tilted = OpticalBunchSource(
        "gaussian",
        1.0,
        "tilted",
        covariance_ang2=((4.0, 1.0, 0.0), (1.0, 4.0, 0.0), (0.0, 0.0, 1.0)),
    )
    with pytest.raises(ValueError, match="correlated"):
        out_of_band_coherent_energy_eV(crossings, (1.0, 2.0), tilted, 1)


def test_radiated_energy_fraction_adds_and_refuses_out_of_band_energy():
    energy = np.array([1.0, 2.0])

    def score(e, d):
        shape = np.broadcast_shapes(e.shape, d.shape[:-1])
        return np.full(shape, 1e-6), np.full(shape, 1e-6), np.zeros(shape)

    base = radiated_energy_fraction(score, energy, 1e3)
    assert radiated_energy_fraction(score, energy, 1e3, out_of_band_eV=0.1) == pytest.approx(
        base + 1e-4
    )
    with pytest.raises(ValueError, match="kinetic energy"):
        radiated_energy_fraction(score, energy, 1e3, out_of_band_eV=2.0)
    with pytest.raises(ValueError, match="nonnegative"):
        radiated_energy_fraction(score, energy, 1e3, out_of_band_eV=-1.0)


def test_envelope_refuses_cherenkov_media_and_unresolved_gamma():
    source = OpticalBunchSource.gaussian(1.0, "gauss", rms_t_fs=200.0)
    crossings = _crossings([[0, 0, 1.0]], [0])
    beta2 = _beta(30.0) ** 2
    ok = out_of_band_coherent_energy_eV(crossings, (1.0, 2.0), source, 1, [np.array([-30 + 2j])])
    assert ok > 0
    with pytest.raises(ValueError, match="Cherenkov"):
        out_of_band_coherent_energy_eV(
            crossings, (1.0, 2.0), source, 1, [np.array([1.01 / beta2 + 0j])]
        )
    with pytest.raises(ValueError, match="gamma"):
        pec_crossing_energy_per_eV(_crossings([[0, 0, 1.0]], [0], kinetic_keV=3000.0))
