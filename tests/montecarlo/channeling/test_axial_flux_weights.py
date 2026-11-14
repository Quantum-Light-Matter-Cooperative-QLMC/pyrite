"""Axial zone-row local-emission flux weights (#1148).

Validation: cr-zone-incoherent-flux-weight
"""

import numpy as np
import pytest

from pyrite.materials.channeling.axial import axial_profile
from pyrite.montecarlo.channeling.axial_flux_weights import (
    _string_densities,
    axial_flux_kernels,
)
from pyrite.montecarlo.channeling.flux_weights import BREM_KEY

ORDER = (3, 3)


@pytest.fixture(scope="module")
def profile():
    return axial_profile("silicon", (0, 0, 1), temperature_K=300.0)


@pytest.fixture(scope="module")
def kernels(profile):
    return axial_flux_kernels(profile, 3.5e6, ORDER)


def test_string_densities_sum_to_the_profile_nuclear_density(profile):
    m_max, n_max = profile.max_order
    m, n = np.meshgrid(np.arange(-m_max, m_max + 1), np.arange(-n_max, n_max + 1), indexing="ij")
    gx, gy = profile.g_inv_ang
    vectors = np.stack([m * gx, n * gy], axis=-1)
    total = sum(_string_densities(profile, vectors).values())
    live = profile.nuclear_density != 0.0
    np.testing.assert_allclose(total[live], profile.nuclear_density[live], rtol=1e-9, atol=0)


def test_kernels_are_normalized_real_and_carry_brem_and_shells(kernels):
    assert kernels.keys[0] == BREM_KEY and "14:K" in kernels.keys
    centre = (2 * ORDER[0], 2 * ORDER[1])
    np.testing.assert_allclose(kernels.harmonics[:, centre[0], centre[1]], 1.0)
    # Real kernel: K_{-Q} = conj(K_Q).
    np.testing.assert_allclose(
        kernels.harmonics[:, ::-1, ::-1], kernels.harmonics.conj(), atol=1e-12
    )


def test_a_single_bloch_order_is_uniform_and_amorphous(kernels):
    size = (2 * ORDER[0] + 1) * (2 * ORDER[1] + 1)
    for index in (0, size // 2, size - 1):
        state = np.zeros(size, dtype=complex)
        state[index] = 0.7j
        np.testing.assert_allclose(kernels.factors(state), 1.0, atol=1e-12)
    np.testing.assert_allclose(kernels.factors(np.zeros(size)), 1.0)


def test_a_state_on_the_strings_enhances_nuclear_local_emission(profile, kernels):
    """A Gaussian on a string beats amorphous; one between strings falls below."""
    mx, my = ORDER
    m, n = np.meshgrid(np.arange(-mx, mx + 1), np.arange(-my, my + 1), indexing="ij")
    gx, gy = profile.g_inv_ang
    width = 0.25 * min(profile.period_ang)

    def gaussian(x0, y0):
        q2 = (m * gx) ** 2 + (n * gy) ** 2
        return (np.exp(-0.25 * width**2 * q2) * np.exp(-1j * (m * gx * x0 + n * gy * y0))).ravel()

    string = profile.strings[0].position_ang
    on = kernels.factors(gaussian(*string))
    between = kernels.factors(gaussian(string[0] + 0.25 * profile.period_ang[0], string[1]))
    brem, k_shell = kernels.keys.index(BREM_KEY), kernels.keys.index("14:K")
    assert on[brem] > 1.2 and on[k_shell] > 1.05
    assert on[brem] > on[k_shell]  # K-shell ionization is less local than the nucleus
    assert between[brem] < on[brem]


@pytest.mark.parametrize(("axis", "hkl"), [(0, (1, 1, 0)), (1, (1, -1, 0))])
def test_a_state_uniform_along_one_cell_axis_gives_the_planar_weight(profile, kernels, axis, hkl):
    """Planar-symmetric limit (#1175): orders on one cell axis reproduce the planar F.

    The Si <001> cell is the 1.92 A square of the {110} planes, so a density
    uniform along ``y`` (``x``) is a (110) ((1-10)) planar density; its axial
    ``F_c`` must equal :func:`.flux_weights.flux_factors` of the same 1D state.
    """
    from pyrite.materials.channeling.potential import planar_profile
    from pyrite.montecarlo.channeling.flux_weights import flux_factors, flux_kernels

    mx, my = ORDER
    plane = planar_profile("silicon", hkl, 300.0)
    assert plane.g_inv_ang == pytest.approx(profile.g_inv_ang[axis], rel=1e-12)
    planar = flux_kernels(plane, 3.5e6, 2 * ORDER[axis])
    assert planar.keys == kernels.keys
    rng = np.random.default_rng(1175)
    line = rng.normal(size=2 * ORDER[axis] + 1) + 1j * rng.normal(size=2 * ORDER[axis] + 1)
    orders = np.zeros((2 * mx + 1, 2 * my + 1), dtype=complex)
    if axis == 0:
        orders[:, my] = line
    else:
        orders[mx, :] = line
    x = np.arange(256) * (2.0 * np.pi / plane.g_inv_ang) / 256
    m = np.arange(-ORDER[axis], ORDER[axis] + 1)
    psi = np.exp(1j * plane.g_inv_ang * np.multiply.outer(x, m)) @ line
    expected = flux_factors(planar, x, (np.abs(psi) ** 2)[None])[0]
    np.testing.assert_allclose(kernels.factors(orders.ravel()), expected, rtol=1e-12)
    assert np.ptp(expected) > 0.1  # a non-trivial planar density
