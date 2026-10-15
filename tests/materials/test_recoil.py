"""Exact vacuum recoil compared with the unsquared electron mass shell."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.materials.recoil import (
    recoil_mismatch_inv_ang,
    recoil_mismatch_slope,
    recoil_resonance_eV,
)


def _mass_shell_root(kinetic, g, direction, *, photon=None):
    """65-digit bisection of source Eq (6), without squaring or a root formula."""
    with localcontext() as ctx:
        ctx.prec = 65

        def dec(value):
            return Decimal(str(value))

        mass, hc, kinetic = dec(M_E_EV), dec(HBARC_EV_ANG), dec(kinetic)
        total = kinetic + mass
        momentum = (kinetic * (kinetic + 2 * mass)).sqrt()
        transfer = [hc * dec(value) for value in g]
        normal = [dec(value) for value in direction]

        def residual(photon):
            final = [-transfer[i] - photon * normal[i] for i in range(3)]
            final[2] += momentum
            return (mass * mass + sum(value * value for value in final)).sqrt() + photon - total

        if photon is not None:
            return float(residual(dec(photon)) / hc)
        if residual(Decimal(0)) >= 0:
            return np.nan
        lo, hi = Decimal(0), kinetic
        for _ in range(190):
            midpoint = (lo + hi) / 2
            if residual(midpoint) > 0:
                hi = midpoint
            else:
                lo = midpoint
        return float((lo + hi) / 2)


@pytest.mark.parametrize("kinetic", [30e3, 1e6, 5e6, 50e6, 1e9])
@pytest.mark.parametrize("direction", [[0.0, 0.0, 1.0], [0.6, 0.0, 0.8], [0.0, 0.0, -1.0]])
def test_resonance_matches_unsquared_mass_shell(kinetic, direction):
    g = [0.4, 0.2, 1.8]
    expected = _mass_shell_root(kinetic, g, direction)
    actual = recoil_resonance_eV(kinetic, [0, 0, 1], g, direction)
    # Float64 arithmetic versus 65-digit unsquared conservation; tolerance
    # includes rounded repository constants and geometric normalization.
    np.testing.assert_allclose(actual, expected, rtol=3e-13)
    assert 0 < actual <= kinetic


def test_hopg_forward_root_no_longer_exceeds_the_beam_energy():
    from pyrite.materials.crystal import CRYSTALS, reciprocal_g_vector

    g, _ = reciprocal_g_vector((0, 0, 4), CRYSTALS["hopg"]["lattice"])
    kinetic = 50e6
    beta = np.sqrt(kinetic * (kinetic + 2 * M_E_EV)) / (kinetic + M_E_EV)
    classical = HBARC_EV_ANG * beta * g[2] / (1 - beta)
    assert classical > kinetic
    actual = recoil_resonance_eV(kinetic, [0, 0, 1], g, [0, 0, 1])
    assert 0 < actual < kinetic
    assert actual == pytest.approx(_mass_shell_root(kinetic, g, [0, 0, 1]), rel=3e-13)


@pytest.mark.parametrize("g", [[0, 0, 0], [0, 0, -2], [10, 0, 0.001]])
def test_forbidden_branches_return_nan(g):
    assert np.isnan(recoil_resonance_eV(1e6, [0, 0, 1], g, [0, 0, 1]))


def test_all_kinetic_energy_endpoint_and_broadcasting():
    kinetic = np.array([1e5, 1e6])
    momentum = np.sqrt(kinetic * (kinetic + 2 * M_E_EV))
    g = np.zeros((2, 3))
    g[:, 2] = (momentum - kinetic) / HBARC_EV_ANG
    actual = recoil_resonance_eV(kinetic, [0, 0, 1], g, [0, 0, 1])
    np.testing.assert_allclose(actual, kinetic, rtol=3e-15)
    assert actual.shape == (2,)


def test_forward_denominator_survives_beta_rounding_to_unity():
    # At this energy beta itself rounds to one in float64. The stable mass
    # contribution to 1-beta must still leave a finite physical root.
    root = recoil_resonance_eV(1e15, [0, 0, 1], [0, 0, 2e-8], [0, 0, 1])
    expected = _mass_shell_root(1e15, [0, 0, 2e-8], [0, 0, 1])
    assert 0 < root < 1e15
    assert root == pytest.approx(expected, rel=3e-13)


def test_classical_limit_requires_small_lattice_recoil_too():
    kinetic = 1e6
    tiny = np.array([0.4, 0.2, 1.8]) * 1e-8
    beta = np.sqrt(kinetic * (kinetic + 2 * M_E_EV)) / (kinetic + M_E_EV)
    classical = HBARC_EV_ANG * beta * tiny[2] / (1 + beta)
    root = recoil_resonance_eV(kinetic, [0, 0, 1], tiny, [0, 0, -1])
    assert root == pytest.approx(classical, rel=1e-9)
    # A tiny classical photon/beam-energy ratio alone is insufficient: a
    # large transverse lattice transfer can forbid the line altogether.
    assert np.isnan(recoil_resonance_eV(kinetic, [0, 0, 1], [10, 0, 0.001], [0, 0, -1]))


def test_exact_mismatch_zero_and_derivative_jacobian():
    kinetic, g, direction = 50e6, [0.4, 0.2, 1.8], [0.6, 0, 0.8]
    root = recoil_resonance_eV(kinetic, [0, 0, 1], g, direction)
    assert abs(recoil_mismatch_inv_ang(root, kinetic, [0, 0, 1], g, direction)) < 1e-12
    step = 1.0
    numeric = (
        recoil_mismatch_inv_ang(root + step, kinetic, [0, 0, 1], g, direction)
        - recoil_mismatch_inv_ang(root - step, kinetic, [0, 0, 1], g, direction)
    ) / (2 * step)
    slope = recoil_mismatch_slope(root, kinetic, [0, 0, 1], g, direction)
    assert slope > 0
    assert slope == pytest.approx(numeric, rel=1e-9)


def test_rotations_and_paired_leading_axes():
    kinetic = np.array([30e3, 1e6, 50e6])[:, None]
    normal = np.array([[0, 0, 1], [0, 0, -1]])
    g = np.array([0.4, 0.2, 1.8])
    expected = np.array(
        [[_mass_shell_root(energy, g, n) for n in normal] for energy in kinetic[:, 0]]
    )
    rotation = np.array([[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]])
    direction = rotation @ [0, 0, 1]
    rotated_normal, rotated_g = normal @ rotation.T, rotation @ g
    actual = recoil_resonance_eV(kinetic, direction, rotated_g, rotated_normal)
    assert actual.shape == (3, 2)
    np.testing.assert_allclose(actual, expected, rtol=3e-13)
    mismatch = recoil_mismatch_inv_ang(actual, kinetic, direction, rotated_g, rotated_normal)
    np.testing.assert_allclose(mismatch, 0, atol=1e-12)
    assert recoil_mismatch_slope(actual, kinetic, direction, rotated_g, rotated_normal).shape == (
        3,
        2,
    )


@pytest.mark.parametrize("photon", [0, 5e5, 1e6])
def test_off_resonance_mismatch_matches_unsquared_mass_shell(photon):
    g, direction = [0.4, 0.2, 1.8], [0.6, 0, 0.8]
    expected = _mass_shell_root(1e6, g, direction, photon=photon)
    actual = recoil_mismatch_inv_ang(photon, 1e6, [0, 0, 1], g, direction)
    assert actual == pytest.approx(expected, rel=3e-13, abs=1e-12)


def test_forward_jacobian_stays_positive_after_beta_rounding():
    args = (1e15, [0, 0, 1], [0, 0, 2e-8], [0, 0, 1])
    root = recoil_resonance_eV(*args)
    slope = recoil_mismatch_slope(root, *args)
    assert np.isfinite(slope) and slope > 0


def test_constant_momentum_flight_factor_is_split_invariant():
    args = (1e6, [0, 0, 1], [0.4, 0.2, 1.8], [0.6, 0, 0.8])
    root = recoil_resonance_eV(*args)
    mismatch = recoil_mismatch_inv_ang(root + 25.0, *args)
    duration = 150.0  # c*dt in angstroms, not spatial flight length

    def field(length, midpoint):
        return length * np.sinc(mismatch * length / (2 * np.pi)) * np.exp(1j * mismatch * midpoint)

    whole = field(duration, duration / 2)
    pieces = sum(field(duration / 7, (i + 0.5) * duration / 7) for i in range(7))
    assert pieces == pytest.approx(whole, rel=2e-14)


@pytest.mark.parametrize("kinetic", [0, -1, np.nan, np.inf])
def test_invalid_kinetic_energy_is_rejected(kinetic):
    with pytest.raises(ValueError, match="kinetic"):
        recoil_resonance_eV(kinetic, [0, 0, 1], [0, 0, 2], [0, 0, 1])


def test_nonunit_direction_and_unphysical_photon_are_rejected():
    with pytest.raises(ValueError, match="unit"):
        recoil_resonance_eV(1e6, [0, 0, 2], [0, 0, 2], [0, 0, 1])
    with pytest.raises(ValueError, match="photon"):
        recoil_mismatch_inv_ang(2e6, 1e6, [0, 0, 1], [0, 0, 2], [0, 0, 1])
