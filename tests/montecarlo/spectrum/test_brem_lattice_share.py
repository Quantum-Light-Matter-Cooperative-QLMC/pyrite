"""Lattice-coherent share of the BremsLib continuum (#1250).

Validation: brem-lattice-coherent-fraction
"""

import numpy as np
import pytest

from pyrite._bremslib_table import solid_angle_integral
from pyrite.montecarlo.spectrum.brem_lattice_share import (
    LatticeShareKernel,
    kernel_x,
    lattice_share_density,
    lattice_share_table,
    reflection_shell_weights,
    shell_kernel_x,
)

X_EDGES = np.linspace(0.0, 1.05, 106)
N_SI = 0.049942993373787194


@pytest.fixture(scope="module")
def kernel():
    """A 2x2x2-node kernel around Si (111)/(220) at 20-45 keV, 30-60 degrees."""
    g = np.array([1.5, 3.4])
    T = np.array([20.0, 45.0])
    th = np.array([30.0, 60.0])
    mass = np.array(
        [
            [
                [shell_kernel_x(gi, Ti, np.deg2rad(t), X_EDGES, n_mu=200, n_phi=12) for t in th]
                for Ti in T
            ]
            for gi in g
        ]
    )
    return LatticeShareKernel(g, T, th, X_EDGES, mass, 200, 12)


def _si_low_order():
    hkls = [(1, 1, 1), (-1, -1, -1), (2, 2, 0), (-2, -2, 0), (3, 1, 1)]
    return reflection_shell_weights("silicon", hkls, 0.46, [("Si", N_SI)])


def test_resonance_variable_is_g_dot_v_for_the_classical_root():
    # Classical root: hbar omega = hbar c g beta mu / (1 - beta cos theta).
    T, th, g, mu = 30.0, np.deg2rad(45.0), 2.0, 0.37
    from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV

    gamma = 1 + T * 1e3 / M_E_EV
    beta = np.sqrt(1 - 1 / gamma**2)
    E = HBARC_EV_ANG * g * beta * mu / (1 - beta * np.cos(th))
    assert kernel_x(E, g, T, th) == pytest.approx(mu, rel=1e-12)


def test_shell_kernel_lives_on_the_physical_x_range():
    mass = shell_kernel_x(2.2, 30.0, np.deg2rad(45.0), X_EDGES, n_mu=200, n_phi=12)
    assert mass.sum() > 0
    assert np.all(mass >= 0)
    # x = mu <= 1 classically; recoil only lowers the photon energy.
    assert mass[X_EDGES[1:] > 1.0 + 1e-9].sum() == 0.0


def test_large_debye_waller_and_empty_sets_remove_nothing(kernel):
    g, w = reflection_shell_weights("silicon", [(1, 1, 1)], 1e4, [("Si", N_SI)])
    assert w["Si"].max() < 1e-30  # B -> infinity: e^{-2W} -> 0
    S = lattice_share_density(kernel, g, w["Si"], 30.0, np.linspace(500.0, 9000.0, 30))
    assert np.abs(S).max() < 1e-30
    empty = lattice_share_density(kernel, np.array([]), np.array([]), 30.0, [1000.0])
    assert not empty.any()  # G -> 0: no reflections, no share


def test_share_is_additive_over_reflections(kernel):
    g, w = _si_low_order()
    k = np.linspace(300.0, 20000.0, 40)
    whole = lattice_share_density(kernel, g, w["Si"], 30.0, k)
    parts = sum(
        lattice_share_density(kernel, g[i : i + 1], w["Si"][i : i + 1], 30.0, k)
        for i in range(g.size)
    )
    np.testing.assert_allclose(whole, parts, rtol=1e-12, atol=0)


def test_kernel_nodes_reproduce_the_direct_shell(kernel):
    # At a node (g, T, theta), the interpolated density is the direct kernel's.
    g0, T0, th0 = 1.5, 20.0, 30.0
    mass = shell_kernel_x(g0, T0, np.deg2rad(th0), X_EDGES, n_mu=200, n_phi=12)
    from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV

    beta = np.sqrt(1 - 1 / (1 + T0 * 1e3 / M_E_EV) ** 2)
    scale = HBARC_EV_ANG * g0 * beta / (1 - beta * np.cos(np.deg2rad(th0)))
    centres = 0.5 * (X_EDGES[:-1] + X_EDGES[1:])
    k = centres[10:90] * scale
    S = lattice_share_density(kernel, np.array([g0]), np.array([1.0]), T0, k)[0]
    expected = mass[10:90] / np.diff(X_EDGES)[10:90] / scale
    np.testing.assert_allclose(S, expected, rtol=1e-10)


def test_out_of_range_energy_is_left_alone(kernel):
    g, w = _si_low_order()
    assert not lattice_share_density(kernel, g, w["Si"], 5.0, [1000.0]).any()
    assert not lattice_share_density(kernel, g, w["Si"], 100.0, [1000.0]).any()


def test_table_loses_exactly_k_S_over_n_Z2_and_stays_self_consistent(kernel):
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

    table = load_bremsstrahlung_tables(["Si"])["Si"]
    g, w = _si_low_order()
    corrected, info = lattice_share_table(table, kernel, "Si", N_SI, g, w["Si"])
    assert info.clamped_nodes == 0
    assert 0.0 < info.removed_fraction < 0.2
    # Keys name the BremsLib parent (the coupled partition check compares them).
    assert (corrected.key, corrected.digest) == (table.key, table.digest)
    T = np.asarray(table.incident_energy_keV)
    i = int(np.argmin(np.abs(T - 30.0)))
    kap = np.array(table.nominal_reduced_energy, dtype=float)
    kap[-1] = float(table.top_reduced_energy[i])
    k = kap * T[i] * 1e3
    S = lattice_share_density(kernel, g, w["Si"], T[i], k)
    j = int(np.argmin(np.abs(np.rad2deg(table.theta_rad) - 45.0)))
    assert np.rad2deg(table.theta_rad[j]) == pytest.approx(45.0, abs=1e-9)
    # Kernel theta nodes are 30 and 60 degrees; the table interpolates between them.
    delta = 1e11 * k * 0.5 * (S[0] + S[1]) / (N_SI * 14.0**2)
    removed = table.scaled_ddcs_mb_sr[i, :, j] - corrected.scaled_ddcs_mb_sr[i, :, j]
    np.testing.assert_allclose(removed, delta, rtol=1e-5, atol=1e-12 * delta.max())
    # SDCS remains the linear-in-theta solid-angle integral of the DDCS.
    np.testing.assert_allclose(
        solid_angle_integral(corrected.theta_rad, corrected.scaled_ddcs_mb_sr)[i],
        corrected.scaled_sdcs_mb[i],
        rtol=1e-5,
    )
    # Energies outside the kernel keep the parent table bit for bit.
    outside = (T < 20.0) | (T > 45.0)
    np.testing.assert_array_equal(
        corrected.scaled_ddcs_mb_sr[outside], table.scaled_ddcs_mb_sr[outside]
    )
