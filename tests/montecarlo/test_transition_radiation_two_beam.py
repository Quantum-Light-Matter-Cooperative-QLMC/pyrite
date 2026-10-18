"""Two-beam Maxwell field limits; no transport or external table dependency."""

import numpy as np
import pytest

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
