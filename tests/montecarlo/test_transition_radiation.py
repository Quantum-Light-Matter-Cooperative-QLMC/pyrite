"""Exact planar-stack transition radiation and its omission bound (tr-detected-omission-bound)."""

import numpy as np
import pytest

from pyrite.materials.attenuation import plasma_energy_eV
from pyrite.materials.crystal import ALPHA_FS, chi_0
from pyrite.montecarlo.channeling.transition_radiation import transition_radiation
from pyrite.montecarlo.spectrum.transition_radiation import (
    beta_from_kinetic_eV,
    ginzburg_frank_interface,
    medium_dielectric,
    omission_tolerance,
    planar_transition_radiation,
    transition_radiation_omission,
)


def _direction(theta, phi=0.0):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


@pytest.mark.parametrize("eps", [0.6 + 0.05j, 1 - 1e-3 + 1e-5j, -3 + 2j, 2.0 + 0.1j])
@pytest.mark.parametrize("beta", [0.3, 0.9, 0.999])
@pytest.mark.parametrize("sign", [+1, -1])
def test_single_interface_matches_closed_ginzburg_frank(eps, beta, sign):
    # Exit (+) is forward TR; entry (-) is backward TR, the closed form at -beta.
    # eps = 2 at beta = 0.9 is the Cherenkov regime of the medium.
    theta = np.array([0.2, 0.8, 1.3])
    exact = planar_transition_radiation(
        100.0,
        np.stack([_direction(t) for t in theta]),
        np.array([0.0, 0.0, sign * beta]),
        [],
        eps,
    )
    closed = ginzburg_frank_interface(eps, sign * beta, theta, 100.0)
    np.testing.assert_allclose(exact, closed, rtol=1e-9)


@pytest.mark.parametrize("beta,psi", [(0.3, 0.5), (0.9, 0.6), (0.99, 1.0)])
@pytest.mark.parametrize("sign", [+1, -1])
def test_conductor_at_oblique_incidence_is_charge_plus_image(beta, psi, sign):
    # |eps| -> inf: sudden stop of the charge and of its image (-e, mirrored v).
    velocity = beta * np.array([np.sin(psi), 0.0, sign * np.cos(psi)])
    image = velocity * np.array([1.0, 1.0, -1.0])
    for theta, phi in ((0.3, 0.0), (0.9, 2.0), (1.2, 3.0)):
        n = _direction(theta, phi)
        amp = np.cross(n, velocity) / (1 - n @ velocity) - np.cross(n, image) / (1 - n @ image)
        expected = ALPHA_FS / (4 * np.pi**2 * 50.0) * amp @ amp
        exact = planar_transition_radiation(50.0, n, velocity, [], -1e9 + 1e9j)
        assert exact == pytest.approx(expected, rel=3e-4)


def test_contrast_free_and_vanishing_films_radiate_nothing():
    n, v = _direction(0.05), np.array([0.0, 0.0, 0.995])
    eps = 1 - (31.0 / 300.0) ** 2
    single = ginzburg_frank_interface(eps, 0.995, 0.05, 300.0)
    assert planar_transition_radiation(300.0, n, v, [(1.0, 1e3)]) == 0.0
    assert planar_transition_radiation(300.0, n, v, [(eps, 1e-6)]) < 1e-9 * single
    assert planar_transition_radiation(300.0, n, v, [], 1.0) == 0.0


def test_film_interference_limits():
    n, v = _direction(0.05), np.array([0.0, 0.0, 0.995])
    eps = 1 - (31.0 / 300.0) ** 2 + 0j
    single = ginzburg_frank_interface(eps, 0.995, 0.05, 300.0)
    # Transparent film: two interfaces average to twice one and peak at four.
    films = [
        planar_transition_radiation(300.0, n, v, [(eps, d)]) for d in np.linspace(1e3, 3e5, 2000)
    ]
    assert np.mean(films) / single == pytest.approx(2.0, rel=5e-3)
    assert np.max(films) / single == pytest.approx(4.0, rel=1e-3)
    # Opaque film: only the exit interface survives.
    lossy = eps + 0.05j
    thick = planar_transition_radiation(300.0, n, v, [(lossy, 1e6)])
    assert thick == pytest.approx(ginzburg_frank_interface(lossy, 0.995, 0.05, 300.0), rel=1e-10)


def test_relativistic_xray_limit_is_the_channeling_small_angle_form():
    # O(theta^2, gamma^-2, (omega_p/omega)^2) corrections are ~1e-4 here.
    energy = 100e6
    beta = beta_from_kinetic_eV(energy)
    plasma = plasma_energy_eV("silicon")
    photon = 3000.0
    theta = np.array([0.2, 1.0, 3.0]) / (1 + energy / 510998.95)
    exact = planar_transition_radiation(
        photon,
        np.stack([_direction(t) for t in theta]),
        np.array([0, 0, beta]),
        [],
        1 - (plasma / photon) ** 2,
    )
    np.testing.assert_allclose(
        exact, transition_radiation("silicon", energy, photon, theta), rtol=1e-3
    )


@pytest.mark.parametrize("energy", [20e6, 100e6])
def test_interface_energy_scale_from_the_exact_solution(energy):
    # tr-interface-energy-scale: W = alpha gamma hbar omega_p / 3 (1 + O(1/gamma)),
    # integrated over the whole forward hemisphere and down to << hbar omega_p.
    plasma = 31.0
    gamma = 1 + energy / 510998.95
    beta = beta_from_kinetic_eV(energy)
    photon = np.geomspace(1e-4 * gamma * plasma, 1e3 * gamma * plasma, 400)
    theta = np.geomspace(1e-3 / gamma, 0.9999 * np.pi / 2, 600)
    n = np.stack([_direction(t) for t in theta])
    yield_ = planar_transition_radiation(
        photon[:, None],
        n[None, :, :],
        np.array([0, 0, beta]),
        [],
        (1 - (plasma / photon) ** 2 + 0j)[:, None],
    )
    per_eV = np.trapezoid(2 * np.pi * np.sin(theta) * yield_, theta, axis=1)
    ratio = np.trapezoid(per_eV * photon, photon) / (ALPHA_FS * gamma * plasma / 3)
    # Deficit ~1.8/gamma: 0.955 at gamma = 40, 0.991 at gamma = 197 (adaptive quadrature).
    assert 1 - 2.5 / gamma < ratio < 1.0


def test_medium_dielectric_is_conjugate_chi0():
    energy = np.array([100.0, 1000.0, 8000.0])
    np.testing.assert_allclose(
        medium_dielectric("silicon", energy), 1 + np.conj(chi_0("silicon", energy)), rtol=1e-12
    )
    assert np.all(medium_dielectric("silicon", energy).imag > 0)


def test_input_validation():
    with pytest.raises(ValueError, match="n_z > 0"):
        planar_transition_radiation(100.0, [0, 0, -1.0], [0, 0, 0.5], [], 0.9)
    with pytest.raises(ValueError, match="v_z"):
        planar_transition_radiation(100.0, [0, 0, 1.0], [0.5, 0, 0.0], [], 0.9)
    with pytest.raises(ValueError, match="positive"):
        omission_tolerance(0.0)


def test_omission_bound_is_the_worst_pointwise_ratio():
    energy = np.array([50.0, 100.0, 1000.0])
    source = np.array([1e-8, 2e-8, 1e-8])
    tr = np.array([2e-9, 1e-10, 1e-14])
    assert omission_tolerance(0.05) == pytest.approx(0.05 / 3)
    full = transition_radiation_omission(energy, source, tr, target_rse=0.05)
    assert full.bias_bound == pytest.approx(0.2) and not full.omit
    kev = transition_radiation_omission(energy, source, tr, target_rse=0.05, band_eV=(500, 2e3))
    assert kev.bias_bound == pytest.approx(1e-6) and kev.omit
    with pytest.raises(ValueError, match="band"):
        transition_radiation_omission(energy, source, tr, target_rse=0.05, band_eV=(5e3, 6e3))


def test_coherent_multilayer_exceeds_summed_interface_estimates():
    # A resonant 10-foil Si radiator (100 MeV, 1.5 keV) peaks far above 4N
    # times one interface's peak: summed single-interface estimates are not a
    # bound for a coherent stack.
    energy, photon = 100e6, 1500.0
    gamma = 1 + energy / 510998.95
    beta = beta_from_kinetic_eV(energy)
    eps = complex(medium_dielectric("silicon", photon))
    theta = np.linspace(0.05, 4.0, 80) / gamma
    n = np.stack([_direction(t) for t in theta])
    single = np.max(ginzburg_frank_interface(eps, beta, theta, photon))
    foils = 10
    stack = [(eps, 15848.9), (1.0, 199526.2)] * (foils - 1) + [(eps, 15848.9)]
    peak = np.max(planar_transition_radiation(photon, n, np.array([0, 0, beta]), stack))
    assert peak > 3 * (4 * foils) * single


def test_missing_source_with_tr_is_an_unbounded_bias():
    energy = np.array([50.0, 100.0, 1000.0])
    tr = np.array([1e-12, 1e-12, 0.0])
    result = transition_radiation_omission(energy, np.array([0.0, 1e-8, 0.0]), tr, target_rse=0.05)
    assert result.bias_bound == np.inf and not result.omit
    assert result.ratio[2] == 0.0
