"""Planar two-time source vs independent planar dipoles and the dense regression oracle.

Oracles: ``bloch.transverse_states(...).dipole_ang`` (``cr-dipole-matrix-element``,
independent of the fibre's velocity currents) and
``quantum_regression.regression_correlation`` (dense GKSL quantum regression).
"""

from functools import cache

import numpy as np
import pytest

from pyrite.materials.channeling.potential import planar_profile
from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.channeling.axial_fiber_regression import (
    fiber_lowering_currents,
    signed_on_axis_spectrum,
)
from pyrite.montecarlo.channeling.bloch import transverse_states
from pyrite.montecarlo.channeling.planar_fiber import planar_fiber_inputs, planar_on_axis_source
from pyrite.montecarlo.channeling.quantum_regression import regression_correlation

ENERGY = 17.0e6
DEPTHS = np.linspace(0.0, 2400.0, 25)
PHOTONS = np.linspace(4000.0, 24000.0, 200)


@cache
def _profile():
    return planar_profile("silicon", (1, 1, 0), 293.15)


def _momenta(inputs, kappa):
    orders = np.stack([inputs.orders, np.zeros_like(inputs.orders)], -1)
    g = np.array([_profile().g_inv_ang, 0.0])
    return np.asarray(kappa, dtype=float) + orders * g


@pytest.mark.parametrize("kappa", [(0.0, 0.0), (0.37, 0.0), (0.21, 4.0e-3)])
def test_planar_currents_are_the_in_plane_dipoles(kappa):
    inputs = planar_fiber_inputs(_profile(), ENERGY, order_max=8, mechanisms=())
    energies, vectors = inputs.channels.spectrum(kappa)
    currents = fiber_lowering_currents(vectors, energies, _momenta(inputs, kappa), inputs.gamma)
    # The out-of-plane current is kappa_y times the identity: no positive-frequency
    # part beyond the eigenvectors' rounding.
    assert np.max(np.abs(currents[1])) <= 1e-12 * np.max(np.abs(currents[0]))
    states = transverse_states(_profile(), inputs.gamma, kappa[0], n_plane_waves=8)
    np.testing.assert_allclose(states.energies_eV, energies, rtol=0, atol=1e-9)
    checked = 0
    for final in range(5):
        for initial in range(final + 1, 9):
            delta = energies[initial] - energies[final]
            # |<f|J_x|i>| = |E_i - E_f| |x_fi| / hbar c (gauge-free magnitudes).
            expected = delta * abs(states.dipole_ang(final, initial)) / HBARC_EV_ANG
            assert abs(currents[0, final, initial]) == pytest.approx(expected, rel=1e-9, abs=1e-15)
            assert currents[0, initial, final] == 0.0
            checked += 1
    assert checked == 30


def test_without_scattering_the_planar_source_is_the_exact_regression():
    inputs = planar_fiber_inputs(
        _profile(), ENERGY, order_max=6, mechanisms=(), incidence_rad=2.0e-5
    )
    source = planar_on_axis_source(
        inputs, depths_ang=DEPTHS, photon_energy_eV=PHOTONS, incident_ids=range(3)
    )
    energies, vectors = inputs.channels.spectrum(inputs.carrier_inv_ang)
    keep = energies < inputs.cutoff_eV
    vectors, energies = vectors[:, keep], energies[keep]
    u = vectors.conj().T @ inputs.entry_orders
    currents = fiber_lowering_currents(
        vectors,
        energies,
        _momenta(inputs, inputs.carrier_inv_ang),
        inputs.gamma,
        boundary_eV=inputs.barrier_eV,
    )
    oracle = regression_correlation(
        np.diag(energies),
        (),
        np.outer(u, u.conj()),
        currents,
        DEPTHS,
        hbar_v_eV_ang=inputs.hbar_v_eV_ang,
    )
    scale = np.max(np.abs(oracle.values))
    assert scale > 0
    np.testing.assert_allclose(source.correlation.values, oracle.values, rtol=0, atol=1e-9 * scale)
    # Every primary is the same deterministic history: no spread.
    assert np.max(source.sem) <= 1e-12 * np.max(source.spectrum)
    expected = signed_on_axis_spectrum(oracle.values, DEPTHS, PHOTONS, ENERGY)
    np.testing.assert_allclose(source.spectrum, expected, rtol=1e-9, atol=1e-12 * expected.max())
    # The bound-final partition keeps only lines ending below the barrier.
    assert np.any(energies >= inputs.barrier_eV)
    assert np.all(currents[0][energies >= inputs.barrier_eV] == 0.0)
