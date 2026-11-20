"""Standalone photon scattering laws and recoil accounting (#341)."""

import numpy as np
import pytest

from pyrite.materials.crystal import HC_EV_ANG, M_E_EV
from pyrite.materials.photon_differential import (
    coherent_form_factor,
    incoherent_scattering_function,
)
from pyrite.montecarlo.eadl_relaxation import load_eadl_relaxation
from pyrite.montecarlo.transport import photon_scattering as scattering
from pyrite.montecarlo.transport.photon_scattering import sample_compton, sample_rayleigh


@pytest.mark.parametrize("sampler", [sample_compton, sample_rayleigh])
def test_invalid_incident_ray(sampler):
    with pytest.raises(ValueError, match="unit"):
        sampler("C", 1e5, [0, 0, 2], [])


def test_rayleigh_preserves_energy():
    rng = np.random.default_rng(34110)
    event = sample_rayleigh("Si", 1e5, [0, 0, 1], iter(rng.random, None))
    assert event.photon_energy_eV == 1e5
    assert event.electron_energy_eV == 0
    assert event.electron_direction is None
    assert np.linalg.norm(event.photon_direction) == pytest.approx(1)


def test_compton_conserves_energy():
    rng = np.random.default_rng(34111)
    event = sample_compton("Si", 1e5, [0, 0, 1], iter(rng.random, None))
    binding = {
        s.shell_designator: s.binding_energy_eV for s in load_eadl_relaxation("Si").subshells
    }
    assert event.binding_energy_eV == binding[event.subshell_designator]
    total = event.photon_energy_eV + event.electron_energy_eV + event.binding_energy_eV
    assert total == pytest.approx(1e5, rel=1e-15)
    assert np.linalg.norm(event.photon_direction) == pytest.approx(1)
    assert np.linalg.norm(event.electron_direction) == pytest.approx(1)


def _assert_uniform_cdf(values):
    ordered = np.sort(values)
    n = len(ordered)
    error = max(np.max(np.arange(1, n + 1) / n - ordered), np.max(ordered - np.arange(n) / n))
    # DKW bound: false rejection probability <= 2 exp(-2*2.2²).
    assert error < 2.2 / np.sqrt(n)


def _cdf(grid, density):
    integral = np.r_[0, np.cumsum(np.diff(grid) * (density[:-1] + density[1:]) / 2)]
    return integral / integral[-1]


def _assert_uniform_azimuth(events, ray):
    # Independent fixed transverse axis for the oblique ray used below.
    transverse = np.array([2, -1, 0]) / np.sqrt(5)
    other = np.cross(ray, transverse)
    directions = np.array([event.photon_direction for event in events])
    angle = np.arctan2(directions @ transverse, directions @ other)
    _assert_uniform_cdf((angle % (2 * np.pi)) / (2 * np.pi))


@pytest.mark.parametrize("element,energy", [("C", 1e3), ("Si", 511e3), ("Pb", 1e8)])
def test_rayleigh_distribution_against_independent_quadrature(element, energy):
    rng = np.random.default_rng(34112)
    ray = np.array([1, 2, -2]) / 3
    events = [sample_rayleigh(element, energy, ray, iter(rng.random, None)) for _ in range(2000)]
    limit = energy / HC_EV_ANG
    x = np.r_[0, np.geomspace(limit * 1e-9, limit, 30_000)]
    mu = 1 - 2 * (x / limit) ** 2
    density = x * (1 + mu**2) * coherent_form_factor(element, x) ** 2
    sampled_x = limit * np.sqrt([e.one_minus_cosine / 2 for e in events])
    _assert_uniform_cdf(np.interp(sampled_x, x, _cdf(x, density)))
    _assert_uniform_azimuth(events, ray)
    np.testing.assert_allclose(
        np.linalg.norm([e.photon_direction for e in events], axis=1), 1, rtol=0, atol=2e-14
    )


@pytest.mark.parametrize("element,energy", [("C", 1e3), ("Si", 511e3), ("Pb", 1e8)])
def test_compton_angles_energy_closure_and_recoil_direction(element, energy):
    rng = np.random.default_rng(34113)
    ray = np.array([1, 2, -2]) / 3
    events = [sample_compton(element, energy, ray, iter(rng.random, None)) for _ in range(2000)]
    delta = np.r_[0, np.geomspace(1e-10, 2, 30_000)]
    epsilon = 1 / (1 + energy / M_E_EV * delta)
    s = incoherent_scattering_function(element, energy / HC_EV_ANG * np.sqrt(delta / 2))
    density = epsilon**2 * (epsilon + 1 / epsilon - delta * (2 - delta)) * s
    _assert_uniform_cdf(
        np.interp([e.one_minus_cosine for e in events], delta, _cdf(delta, density))
    )
    _assert_uniform_azimuth(events, ray)
    photon_energy = np.array([e.photon_energy_eV for e in events])
    kinetic = np.array([e.electron_energy_eV for e in events])
    binding = np.array([e.binding_energy_eV for e in events])
    assert np.all(kinetic >= 0) and np.all(binding > 0)
    np.testing.assert_allclose(photon_energy + kinetic + binding, energy, rtol=3e-16)
    # Bound electron: the atom takes the momentum imbalance; the electron
    # leaves along the momentum transfer q = E n - E' n' (PENELOPE-2024 Eq. 2.72).
    transfer = energy * ray - photon_energy[:, None] * np.array(
        [e.photon_direction for e in events]
    )
    transfer /= np.linalg.norm(transfer, axis=1)[:, None]
    np.testing.assert_allclose(
        np.array([e.electron_direction for e in events]), transfer, rtol=0, atol=1e-9
    )


def test_constant_form_factor_is_thomson(monkeypatch):
    monkeypatch.setattr(
        scattering,
        "coherent_form_factor_table",
        lambda el: (np.array([0, 1e9]), np.array([1.0, 1.0])),
    )
    rng = np.random.default_rng(34114)
    mu = np.array(
        [
            1 - sample_rayleigh("C", 1e8, [0, 0, 1], iter(rng.random, None)).one_minus_cosine
            for _ in range(2000)
        ]
    )
    _assert_uniform_cdf(0.5 + 3 * mu / 8 + mu**3 / 8)


def test_constant_scattering_function_is_free_klein_nishina(monkeypatch):
    monkeypatch.setattr(
        scattering,
        "incoherent_scattering_function_table",
        lambda el: (np.array([0, 1e9]), np.array([1.0, 1.0])),
    )
    rng = np.random.default_rng(34115)
    energy = 2e6
    delta = np.linspace(0, 2, 30_000)
    epsilon = 1 / (1 + energy / M_E_EV * delta)
    density = epsilon**2 * (epsilon + 1 / epsilon - delta * (2 - delta))
    sampled = [
        sample_compton("C", energy, [0, 0, 1], iter(rng.random, None)).one_minus_cosine
        for _ in range(2000)
    ]
    _assert_uniform_cdf(np.interp(sampled, delta, _cdf(delta, density)))


@pytest.mark.parametrize(
    "sampler,energy",
    [
        (sample_rayleigh, 1.0),
        (sample_rayleigh, 1e11),
        (sample_compton, 1e3),
        (sample_compton, 1e11),
    ],
)
def test_epdl_domain_endpoints(sampler, energy):
    rng = np.random.default_rng(34116)
    for _ in range(10):
        event = sampler("Pb", energy, [1, 0, 0], iter(rng.random, None))
        assert 0 <= event.one_minus_cosine <= 2
        assert np.isfinite(event.photon_direction).all()
        total = event.photon_energy_eV + event.electron_energy_eV + event.binding_energy_eV
        assert total == pytest.approx(energy)


@pytest.mark.parametrize("energy", [1.0, 20.0])
def test_compton_without_impulse_support_has_no_final_state(energy):
    # Pb's outermost EADL binding is 7.4 eV: nothing can be ionized at 1 eV,
    # and at 20 eV even backscatter needs |p_z| >~ 40 a.u. (n_i underflows).
    with pytest.raises(ValueError, match="no Compton subshell"):
        sample_compton("Pb", energy, [1, 0, 0], iter(np.random.default_rng(34117).random, None))


@pytest.mark.parametrize("sampler", [sample_rayleigh, sample_compton])
@pytest.mark.parametrize("draws", [[], [1], [np.nan], [-0.1]])
def test_invalid_or_exhausted_uniforms(sampler, draws):
    with pytest.raises(ValueError, match="uniform"):
        sampler("C", 1e5, [0, 0, 1], draws)


@pytest.mark.parametrize("sampler", [sample_rayleigh, sample_compton])
@pytest.mark.parametrize("energy", [0, np.nan, np.inf, 2e11])
def test_invalid_energy(sampler, energy):
    with pytest.raises(ValueError, match="energy"):
        sampler("C", energy, [0, 0, 1], [])


def test_endpoint_draws_and_exact_consumption(monkeypatch):
    monkeypatch.setattr(
        scattering,
        "coherent_form_factor_table",
        lambda el: (np.array([0, 1e9]), np.array([1.0, 1.0])),
    )
    tape = iter([0, 0, 0, 0.5, 0.123])
    event = sample_rayleigh("C", 1e5, [0, 0, 1], tape)
    assert event.one_minus_cosine == 0 and event.draws_used == 4
    assert next(tape) == 0.123
    np.testing.assert_array_equal(event.photon_direction, [0, 0, 1])
    back = sample_rayleigh("C", 1e5, [0, 0, 1], [0, np.nextafter(1.0, 0), 0, 0])
    assert back.one_minus_cosine == pytest.approx(2)


def test_zero_angle_compton_has_no_bound_final_state(monkeypatch):
    # theta = 0 transfers no momentum, so no bound electron can be ejected.
    monkeypatch.setattr(
        scattering,
        "incoherent_scattering_function_table",
        lambda el: (np.array([0, 1e9]), np.array([1.0, 1.0])),
    )
    # Each accepted zero angle is rejected again, until the explicit tape ends.
    with pytest.raises(ValueError, match="uniform sequence exhausted"):
        sample_compton("C", 1e5, [0, 1, 0], [0.0] * 8)


@pytest.mark.parametrize(
    "sampler,accessor",
    [
        (sample_rayleigh, "coherent_form_factor_table"),
        (sample_compton, "incoherent_scattering_function_table"),
    ],
)
def test_zero_support_and_no_forced_acceptance(monkeypatch, sampler, accessor):
    monkeypatch.setattr(
        scattering, accessor, lambda el: (np.array([0.0, 1e9]), np.array([0.0, 0.0]))
    )
    with pytest.raises(ValueError):
        sampler("C", 1e5, [0, 0, 1], [])
    monkeypatch.setattr(
        scattering, accessor, lambda el: (np.array([0.0, 1e9]), np.array([1.0, 1.0]))
    )
    monkeypatch.setattr(scattering, "MAX_REJECTIONS", 0)
    with pytest.raises(RuntimeError, match="budget"):
        sampler("C", 1e5, [0, 0, 1], [])
