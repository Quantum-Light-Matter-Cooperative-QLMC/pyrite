"""Transported-current Maxwell limits, phase and lattice-force source.

Validation: dtr-transported-current, line-tr-phase-reconciliation
"""

import numpy as np
import pytest

from pyrite.materials.crystal import ALPHA_FS, HBARC_EV_ANG
from pyrite.montecarlo.spectrum.dtr_history import (
    TransportedCurrent,
    _cbs_dipole,
    two_beam_current_fields,
)
from pyrite.montecarlo.spectrum.dtr_two_beam import two_beam_straight_field


def _passage(velocity, thickness=80.0, pieces=1):
    velocity = np.asarray(velocity)
    duration = thickness / abs(velocity[2])
    start = np.array([3.0, -2.0, 0 if velocity[2] > 0 else thickness])
    time = np.arange(pieces) * duration / pieces
    end = start + velocity * duration
    return TransportedCurrent(
        position=start + time[:, None] * velocity,
        velocity=np.tile(velocity, (pieces, 1)),
        time=time,
        duration=np.full(pieces, duration / pieces),
        history=np.zeros(pieces, int),
        layer=np.zeros(pieces, int),
        charge=np.ones(pieces),
        boundary_position=np.array([start, end]),
        boundary_velocity=np.tile(velocity, (2, 1)),
        boundary_time=np.array([0, duration]),
        boundary_history=np.zeros(2, int),
        boundary_side=np.array([0, 1] if velocity[2] > 0 else [1, 0]),
        boundary_charge=np.ones(2),
        histories=1,
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_current_solve_recovers_straight_two_beam(sign):
    current = _passage([0.2, 0.1, sign * 0.8], pieces=7)
    energy = np.array([800.0, 1000.0, 1200.0])
    n = np.array([0.3, 0.2, np.sqrt(0.87)])
    layers = [(0.995 + 0.0001j, 80.0)]
    params = dict(
        crystal_layer=0,
        reciprocal_vector_ang=[0.015, -0.01, -1.0],
        chi_transfer=1e-4 + 2e-5j,
        chi_reverse=1e-4 + 2e-5j,
        lattice_origin_ang=[1, 2, 3],
    )
    coupled, bare = two_beam_current_fields(energy, n, layers, current, **params)
    expected = two_beam_straight_field(
        energy, n, layers, 1.0, current.velocity[0], 0.0, current.position[0], **params
    )
    np.testing.assert_allclose(coupled[:, 0], expected.total, rtol=1e-9, atol=1e-12)
    uncoupled = two_beam_straight_field(
        energy,
        n,
        layers,
        1.0,
        current.velocity[0],
        0.0,
        current.position[0],
        **(params | dict(chi_transfer=0.0, chi_reverse=0.0)),
    )
    np.testing.assert_allclose(bare[:, 0], uncoupled.total, rtol=1e-9, atol=1e-12)


def test_cbs_vacuum_resonance_has_feranchuk_amplitude_and_absolute_phase():
    v = np.array([0.1, 0.0, 0.8])
    n = np.array([0.3, 0.2, np.sqrt(0.87)])
    g_ang = np.array([0.01, -0.005, 0.04])
    g = g_ang * HBARC_EV_ANG
    omega = float(v @ g / (1 - v @ n))
    potential = 2e-5
    current = _passage(v, pieces=5)
    coupled, bare = two_beam_current_fields(
        [omega],
        n,
        [(1.0, 80.0)],
        current,
        crystal_layer=0,
        reciprocal_vector_ang=g_ang,
        chi_transfer=0.0,
        chi_reverse=0.0,
        potential_transfer_over_m=potential,
    )
    k = omega * n
    vdg, gamma = v @ g, 1 / np.sqrt(1 - v @ v)
    # Feranchuk Eq. 14 (relativistic form), independently written as a vector.
    amplitude = -potential / (gamma * vdg) * (g - vdg * v + v * (k @ g - (k @ v) * vdg) / vdg)
    amplitude -= n * (amplitude @ n)
    phase = np.exp(
        1j
        * (omega * n[2] * 80 / HBARC_EV_ANG - (omega * n + g) @ current.position[0] / HBARC_EV_ANG)
    )
    expected = (
        2
        * np.pi
        * np.sqrt(ALPHA_FS)
        / n[2]
        * amplitude
        * (current.duration.sum() / HBARC_EV_ANG)
        * phase
    )
    np.testing.assert_allclose((coupled - bare)[0, 0], expected, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(
        _cbs_dipole(v, g, potential) @ v,
        1j * (potential / (gamma * vdg) * (g - v * vdg)) @ v / vdg,
    )


def test_pxr_and_cbs_have_the_same_feranchuk_to_maxwell_phase():
    v = np.array([0.1, 0.0, 0.8])
    n = np.array([0.3, 0.2, np.sqrt(0.87)])
    g_ang = np.array([0.01, -0.005, 0.04])
    g = g_ang * HBARC_EV_ANG
    omega = float(v @ g / (1 - v @ n))
    current = _passage(v, pieces=5)
    chi, potential = 1e-7, 2e-7
    coupled, bare = two_beam_current_fields(
        [omega],
        n,
        [(1.0, 80.0)],
        current,
        crystal_layer=0,
        reciprocal_vector_ang=g_ang,
        chi_transfer=chi,
        chi_reverse=chi,
        potential_transfer_over_m=potential,
    )
    k = omega * n
    vdg, gamma = v @ g, 1 / np.sqrt(1 - v @ v)
    pxr = chi / (g @ g + 2 * k @ g) * ((v @ (k + g)) * g - omega**2 * v)
    cbs = -potential / (gamma * vdg) * (g - vdg * v + v * (k @ g - (k @ v) * vdg) / vdg)
    amplitude = pxr + cbs
    amplitude -= n * (amplitude @ n)
    phase = np.exp(1j * (omega * n[2] * 80 - (omega * n + g) @ current.position[0]) / HBARC_EV_ANG)
    expected = (
        2
        * np.pi
        * np.sqrt(ALPHA_FS)
        / n[2]
        * amplitude
        * current.duration.sum()
        / HBARC_EV_ANG
        * phase
    )
    # Finite chi includes repeated scattering, absent from the linear formula.
    np.testing.assert_allclose((coupled - bare)[0, 0], expected, rtol=1e-5, atol=1e-13)


def test_scattered_history_with_horizontal_flight_is_substep_invariant():
    from dataclasses import replace

    velocity = np.array([[0.1, 0.0, 0.8], [0.8, 0.1, 0.0], [-0.1, 0.1, -0.8]])
    duration = np.array([50.0, 30.0, 50.0])
    time = np.r_[0.0, np.cumsum(duration)[:-1]]
    displacement = velocity * duration[:, None]
    position = np.vstack([np.zeros(3), np.cumsum(displacement, axis=0)[:-1]])
    current = replace(
        _passage([0.0, 0.0, 0.8]),
        position=position,
        velocity=velocity,
        time=time,
        duration=duration,
        history=np.zeros(3, int),
        layer=np.zeros(3, int),
        charge=np.ones(3),
        boundary_position=np.array([position[0], displacement.sum(axis=0)]),
        boundary_velocity=velocity[[0, -1]],
        boundary_time=np.array([0.0, duration.sum()]),
        boundary_side=np.array([0, 0]),
    )
    pieces = 7
    offset = np.arange(pieces)[None, :] * duration[:, None] / pieces
    refined = replace(
        current,
        position=(position[:, None] + offset[..., None] * velocity[:, None]).reshape(-1, 3),
        velocity=np.repeat(velocity, pieces, axis=0),
        time=(time[:, None] + offset).ravel(),
        duration=np.repeat(duration / pieces, pieces),
        history=np.zeros(3 * pieces, int),
        layer=np.zeros(3 * pieces, int),
        charge=np.ones(3 * pieces),
    )
    params = dict(
        crystal_layer=0,
        reciprocal_vector_ang=[0.01, -0.005, 0.04],
        chi_transfer=1e-4,
        chi_reverse=1e-4,
        potential_transfer_over_m=1e-5,
        potential_reverse_over_m=1e-5,
    )
    energy, n, layers = [500.0, 1000.0], [0.3, 0.2, np.sqrt(0.87)], [(0.995 + 0.0001j, 80.0)]
    for coarse, fine in zip(
        two_beam_current_fields(energy, n, layers, current, **params),
        two_beam_current_fields(energy, n, layers, refined, **params),
        strict=True,
    ):
        np.testing.assert_allclose(coarse, fine, rtol=1e-10, atol=1e-12)
