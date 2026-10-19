"""Two-beam Maxwell field limits; no transport or external table dependency."""

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.dtr import dtr_crossing_amplitudes, pxr_straight_amplitude
from pyrite.montecarlo.spectrum.dtr_two_beam import two_beam_straight_field
from pyrite.montecarlo.spectrum.transition_radiation import planar_transition_radiation


def _config(thickness=300.0, chi=0.0):
    return dict(
        photon_energy_eV=np.linspace(800.0, 1200.0, 9),
        direction=np.array([np.sin(0.2), 0.0, np.cos(0.2)]),
        layers=[(0.995 + 0.0001j, thickness)],
        substrate_eps=1.0,
        velocity=np.array([0.0, 0.0, -0.99]),
        time_ang=0.0,
        position_ang=np.array([0.0, 0.0, thickness]),
        crystal_layer=0,
        reciprocal_vector_ang=np.array([0.0, 0.0, -1.0]),
        chi_transfer=chi,
        chi_reverse=chi,
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_zero_harmonics_recover_homogeneous_tr(sign):
    config = _config()
    config["velocity"] = np.array([0.1, 0.05, sign * 0.98])
    config["position_ang"] = np.zeros(3)
    field = two_beam_straight_field(**config)
    _, expected = planar_transition_radiation(
        config["photon_energy_eV"],
        config["direction"],
        config["velocity"],
        config["layers"],
        return_amplitude=True,
    )
    np.testing.assert_allclose(field.transition, expected, rtol=1e-10, atol=1e-13)
    np.testing.assert_array_equal(field.line, 0.0)


def test_weak_harmonics_recover_born_pxr_plus_dtr():
    chi = 1e-7
    config = _config(chi=chi)
    field = two_beam_straight_field(**config)
    born = dict(config)
    born.pop("chi_reverse")
    own = pxr_straight_amplitude(**born)
    dtr = dict(born)
    dtr.update(
        velocity=np.tile(config["velocity"], (2, 1)),
        time_ang=np.array([300.0 / 0.99, 0.0]),
        position_ang=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 300.0]]),
        interface=np.array([0, 1]),
    )
    expected = own + dtr_crossing_amplitudes(**dtr).sum(-2)
    np.testing.assert_allclose(field.line, expected, rtol=1e-5, atol=1e-14)


def test_passive_opaque_slab_stays_finite():
    config = _config(thickness=1e8, chi=1e-4)
    with np.errstate(over="raise", invalid="raise"):
        field = two_beam_straight_field(**config)
    assert np.all(np.isfinite(field.total))


def test_surface_termination_rotates_the_diffracted_row():
    config = _config(chi=1e-4)
    field = two_beam_straight_field(**config)
    shifted = two_beam_straight_field(**config, lattice_origin_ang=[0.0, 0.0, 0.3])
    floor = 1e-12 * np.max(np.abs(field.total))
    np.testing.assert_allclose(shifted.line, field.line * np.exp(-0.3j), rtol=1e-9, atol=floor)
    np.testing.assert_allclose(shifted.transition, field.transition, rtol=1e-9, atol=floor)


def test_active_permittivity_refuses():
    config = _config(chi=0.001j)
    with pytest.raises(ValueError, match="passive"):
        two_beam_straight_field(**config)


@pytest.mark.parametrize("medium", ["layer", "substrate"])
@pytest.mark.parametrize("eps", [np.nan, np.inf, 1.0 + np.nan * 1j])
def test_nonfinite_permittivity_refuses_before_solving(medium, eps):
    config = _config(chi=1e-4)
    if medium == "layer":
        config["layers"] = [(eps, 300.0)]
    else:
        config["substrate_eps"] = eps
    with pytest.raises(ValueError, match="permittivit.*finite"):
        two_beam_straight_field(**config)


@pytest.mark.parametrize("eps", [1.0, 0.995, -0.5, 0.995 + 0.0001j])
@pytest.mark.parametrize("g_parallel", [0.0, 2.0])
def test_uncoupled_propagating_and_evanescent_harmonics(eps, g_parallel):
    config = _config()
    config["layers"] = [(eps, 300.0)]
    config["reciprocal_vector_ang"] = np.array([g_parallel, 0.0, -1.0])
    config["position_ang"] = np.zeros(3)
    field = two_beam_straight_field(**config)
    _, expected = planar_transition_radiation(
        config["photon_energy_eV"],
        config["direction"],
        config["velocity"],
        config["layers"],
        return_amplitude=True,
    )
    np.testing.assert_allclose(field.transition, expected, rtol=1e-10, atol=1e-13)
    np.testing.assert_array_equal(field.line, 0.0)


def test_vanishing_coupled_film_in_vacuum():
    config = _config(chi=1e-3)
    config["layers"] = [(1.0, 1e-8)]
    config["position_ang"] = np.zeros(3)
    field = two_beam_straight_field(**config)
    assert np.linalg.norm(field.total) < 1e-12


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("caps", [False, True])
@pytest.mark.parametrize("g", [(0.015, -0.01, 0.04), (0.015, -0.01, -1.0)])
def test_one_way_coupling_matches_born_field(sign, caps, g):
    """With no reverse harmonic, repeated scattering cannot return to the row.

    The triangular Maxwell system must reproduce the first-Born vector at
    finite transfer strength, including caps, both polarizations and phase.
    This is an internal consistency check, not an independent physics anchor.
    """
    config = _config(thickness=80.0, chi=1e-4 + 2e-5j)
    config.update(
        velocity=np.array([0.25, 0.1, sign * 0.8]),
        direction=np.array([0.6, 0.2, np.sqrt(0.6)]),
        reciprocal_vector_ang=np.asarray(g),
        position_ang=np.array([7.0, -4.0, 11.0]),
        time_ang=19.0,
        chi_reverse=0.0,
        lattice_origin_ang=np.array([2.0, 1.0, -3.0]),
    )
    if caps:
        config["layers"] = [(0.95 + 0.001j, 17.0), *config["layers"], (0.99 + 0.0005j, 19.0)]
        config["crystal_layer"] = 1
    field = two_beam_straight_field(**config)
    born = {key: value for key, value in config.items() if key != "chi_reverse"}
    own = pxr_straight_amplitude(**born)
    z = np.r_[0.0, np.cumsum([d for _, d in config["layers"]])]
    dt = (z - config["position_ang"][2]) / config["velocity"][2]
    born.update(
        velocity=np.tile(config["velocity"], (len(z), 1)),
        time_ang=config["time_ang"] + dt,
        position_ang=config["position_ang"] + dt[:, None] * config["velocity"],
        interface=np.arange(len(z)),
    )
    expected = own + dtr_crossing_amplitudes(**born).sum(-2)
    floor = 1e-10 * np.max(np.abs(expected))
    np.testing.assert_allclose(field.line, expected, rtol=1e-9, atol=floor)
    homogeneous = two_beam_straight_field(**(config | {"chi_transfer": 0.0}))
    np.testing.assert_allclose(field.transition, homogeneous.transition, rtol=1e-10, atol=floor)


def test_two_beam_translation_and_straight_point_reference():
    config = _config(chi=1e-4)
    config["reciprocal_vector_ang"] = np.array([0.015, -0.01, 0.04])
    field = two_beam_straight_field(**config)
    dt = 37.0
    reparametrized = two_beam_straight_field(
        **(
            config
            | {"time_ang": dt, "position_ang": config["position_ang"] + dt * config["velocity"]}
        )
    )
    np.testing.assert_allclose(reparametrized.transition, field.transition, rtol=1e-11, atol=1e-15)
    np.testing.assert_allclose(reparametrized.line, field.line, rtol=1e-11, atol=1e-15)
    delta = np.array([7.0, -4.0, 0.0])
    translated = two_beam_straight_field(
        **(config | {"time_ang": dt, "position_ang": config["position_ang"] + delta})
    )
    phase = np.exp(
        1j * config["photon_energy_eV"] * (dt - config["direction"] @ delta) / HBARC_EV_ANG
    )
    np.testing.assert_allclose(translated.transition, field.transition * phase[:, None], rtol=1e-10)
    phase *= np.exp(-1j * config["reciprocal_vector_ang"] @ delta)
    np.testing.assert_allclose(translated.line, field.line * phase[:, None], rtol=1e-10)
    np.testing.assert_allclose(field.total @ config["direction"], 0.0, atol=1e-16)


def test_reverse_harmonic_adds_repeated_scattering_orders():
    """A return trip changes TR at second order and the line at third order."""
    config = _config(thickness=3000.0, chi=2e-5)
    bare = two_beam_straight_field(**(config | {"chi_transfer": 0.0, "chi_reverse": 0.0}))
    corrections = []
    for factor in (1, 2):
        strength = factor * config["chi_transfer"]
        coupled = two_beam_straight_field(
            **(config | {"chi_transfer": strength, "chi_reverse": strength})
        )
        one_way = two_beam_straight_field(
            **(config | {"chi_transfer": strength, "chi_reverse": 0.0})
        )
        corrections.append(
            [
                np.linalg.norm(coupled.transition - bare.transition),
                np.linalg.norm(coupled.line - one_way.line),
            ]
        )
    assert np.all(np.asarray(corrections) > 0)
    # The omitted orders are respectively fourth and fifth, so the relative
    # error in these leading-order ratios is quadratic in the coupling.
    np.testing.assert_allclose(np.asarray(corrections)[1] / corrections[0], [4, 8], rtol=0.01)


def test_spectral_angular_broadcast_matches_individual_solves():
    config = _config()
    config["photon_energy_eV"] = np.array([900.0, 1000.0, 1100.0])
    angle = np.array([[0.1], [0.2]])
    config["direction"] = np.stack([np.sin(angle), np.zeros_like(angle), np.cos(angle)], axis=-1)
    config["layers"] = [(np.array([0.995, 0.996, 0.997]) + 0.0001j, 300.0)]
    config["chi_transfer"] = np.array([1e-5, 2e-5, 3e-5])
    config["chi_reverse"] = np.array([[1e-5], [2e-5]])
    batch = two_beam_straight_field(**config)
    assert batch.total.shape == (2, 3, 3)
    for i, j in np.ndindex(2, 3):
        scalar = two_beam_straight_field(
            **(
                config
                | dict(
                    photon_energy_eV=config["photon_energy_eV"][j],
                    direction=config["direction"][i, 0],
                    layers=[(config["layers"][0][0][j], 300.0)],
                    chi_transfer=config["chi_transfer"][j],
                    chi_reverse=config["chi_reverse"][i, 0],
                )
            )
        )
        np.testing.assert_allclose(
            batch.transition[i, j], scalar.transition, rtol=1e-12, atol=1e-16
        )
        np.testing.assert_allclose(batch.line[i, j], scalar.line, rtol=1e-12, atol=1e-16)
