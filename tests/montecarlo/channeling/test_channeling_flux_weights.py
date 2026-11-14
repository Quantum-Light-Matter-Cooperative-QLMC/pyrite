"""Channeling flux weights of local incoherent emission on zone rows (#1148)."""

import numpy as np
import pytest

from pyrite.materials.channeling import planar_profile
from pyrite.montecarlo import mc_brem_spectrum, mc_characteristic_spectrum
from pyrite.montecarlo.channeling.flux_weights import (
    BREM_KEY,
    _element_densities,
    _screened_transform,
    flux_factors,
    flux_kernels,
    row_flux_weights,
    shell_key,
    slater_form_factor,
)
from pyrite.montecarlo.transport.shell_gos import shell_gos_moments
from pyrite.montecarlo.transport.shell_rates import catalog_shell_oscillators

E_EV = 17e6
H = 12


@pytest.fixture(scope="module")
def profile():
    return planar_profile("silicon", (1, -1, 0), 293.15)


@pytest.fixture(scope="module")
def kernels(profile):
    return flux_kernels(profile, E_EV, 2 * H)


def _grid(profile, n=8 * H):
    return np.arange(n) * profile.period_ang / n


def test_kernels_are_normalized_nonnegative_and_cover_brem_and_si_shells(profile, kernels):
    assert kernels.keys == (BREM_KEY, "14:L2", "14:L3", "14:L1", "14:K")
    np.testing.assert_allclose(kernels.harmonics[:, 0], 1.0, rtol=0, atol=1e-14)
    values = kernels.values(_grid(profile, 512))
    assert values.min() > -1e-6  # localization densities
    np.testing.assert_allclose(values.mean(axis=1), 1.0, rtol=1e-12)


def test_uniform_density_has_unit_factor_for_every_kernel(profile, kernels):
    x = _grid(profile)
    density = np.full((3, x.size), 0.7)
    np.testing.assert_allclose(flux_factors(kernels, x, density), 1.0, rtol=1e-12)


def test_density_on_the_planes_raises_local_emission_and_k_beats_l(profile, kernels):
    x = _grid(profile)
    centred = np.minimum(x, profile.period_ang - x)  # distance to the plane at x = 0
    density = np.exp(-0.5 * (centred / 0.2) ** 2)[None, :]
    factor = dict(zip(kernels.keys, flux_factors(kernels, x, density)[0], strict=True))
    # K-shell close collisions sit on the nucleus; L shells are delocalized.
    assert factor[BREM_KEY] > factor["14:K"] > factor["14:L1"] > factor["14:L3"] > 1.0
    between = np.exp(-0.5 * ((x - 0.5 * profile.period_ang) / 0.2) ** 2)[None, :]
    assert np.all(flux_factors(kernels, x, between)[0] < 1.0)


def test_zero_retained_population_gets_unit_factor(profile, kernels):
    x = _grid(profile)
    np.testing.assert_array_equal(flux_factors(kernels, x, np.zeros((1, x.size))), 1.0)


def test_element_site_densities_reproduce_the_profile_nuclear_density(profile):
    densities = _element_densities(profile, 6)
    total = sum(densities.values())
    count = min(total.size, profile.nuclear_density.size)
    np.testing.assert_allclose(
        total[:count], profile.nuclear_density[:count], rtol=1e-12, atol=1e-15
    )


def test_slater_form_factor_closed_forms():
    g = np.array([0.0, 0.3, 1.0, 4.0, 20.0])
    zeta = np.sqrt(2.0 * 13.6 / 27.211386245988) / 0.529177210903
    hydrogenic = (1.0 + (g / (2.0 * zeta)) ** 2) ** -2
    np.testing.assert_allclose(slater_form_factor(g, 13.6, 1), hydrogenic, rtol=1e-12)
    # Small-G limit: 1 - G^2 <r^2>/6 with <r^2> = (2n+2)(2n+1)/(4 zeta^2).
    for n in (1, 2, 3):
        small = 1e-3
        expected = 1.0 - small**2 * (2 * n + 2) * (2 * n + 1) / (4 * zeta**2) / 6.0
        np.testing.assert_allclose(slater_form_factor(small, 13.6, n), expected, rtol=1e-9)


def test_shell_kernel_mean_is_the_full_gos_strength():
    """The localized q_perp integral plus the transverse term is the shell's sigma_0."""
    from dataclasses import replace

    from pyrite.montecarlo.channeling.operators import (
        _electronic_transform,
        electronic_channels,
    )

    osc = catalog_shell_oscillators("silicon")
    channels = electronic_channels(osc, E_EV)
    moments = shell_gos_moments(osc, E_EV)
    for index, shell in enumerate(osc.oscillators):
        if shell.label != "K":
            continue
        one = replace(
            channels,
            **{
                name: getattr(channels, name)[index : index + 1]
                for name in (
                    "distant_ang2",
                    "distant_qmin_inv_ang",
                    "distant_qmax_inv_ang",
                    "close_ang2_eV",
                    "close_lower_eV",
                    "close_upper_eV",
                    "close_prime_eV",
                )
            },
        )
        localized = float(_electronic_transform(one, np.array([0.0]))[0])
        expected = (
            (moments.distant_longitudinal[index, 0] + moments.close[index, 0])
            * 1e16
            / osc.electrons_per_formula
        )
        assert localized == pytest.approx(expected, rel=2e-3)


def test_screened_brem_transfer_is_unit_at_zero_and_decreasing():
    g = np.array([0.0, 3.3, 10.0, 50.0])
    for electron_field in (False, True):
        value = _screened_transform("Si", g, electron_field)
        assert value[0] == pytest.approx(1.0, abs=1e-14)
        assert np.all(np.diff(value) < 0.0)


def test_row_weights_are_exact_depth_averages_of_piecewise_linear_factors():
    z = np.array([0.0, 100.0, 300.0])
    factors = np.array([[1.0, 2.0], [3.0, 2.0], [1.0, 2.0]])
    lengths = np.array([0.0, 50.0, 100.0, 200.0, 300.0, 600.0])
    weights = row_flux_weights(z, factors, lengths)
    # F0 = 1 + z/50 on [0, 100], then 3 - (z - 100)/100.
    integral = {50.0: 50 + 25, 100.0: 200.0, 200.0: 200 + 250, 300.0: 200 + 400}
    for row, length in enumerate(lengths):
        if length == 0.0:
            expected = 1.0
        elif length <= 300.0:
            expected = integral[length] / length
        else:
            expected = (integral[300.0] + (length - 300.0) * 1.0) / length
        assert weights[row, 0] == pytest.approx(expected, rel=1e-14)
    np.testing.assert_allclose(weights[:, 1], 2.0, rtol=1e-14)


def _segments(weights=None, keys=None):
    n = 4
    segs = {
        "r_mid": np.column_stack([np.zeros(n), np.zeros(n), np.linspace(1e3, 4e3, n)]),
        "v_hat": np.tile([0.0, 0.0, 1.0], (n, 1)),
        "L_ang": np.full(n, 1e3),
        "E_keV": np.full(n, 5000.0),
        "t_ang": np.zeros(n),
        "t0_ang": np.zeros(n),
        "elec_id": np.arange(n),
        "layer": np.zeros(n, dtype=int),
        "Ne": n,
        "thickness_ang": 5e3,
    }
    if weights is not None:
        segs["channeling_flux"] = np.asarray(weights, dtype=float)
        segs["channeling_flux_keys"] = keys
    return segs


def test_brem_scorer_scales_each_row_by_its_brem_weight():
    grid = np.linspace(2000.0, 20000.0, 7)
    kw = dict(element="Si", n_atoms_per_ang3=0.05, n_hat=np.array([0.0, 0.0, 1.0]))
    plain = mc_brem_spectrum(_segments(), grid, **kw)
    keys = (BREM_KEY, "14:K")
    weighted = mc_brem_spectrum(_segments(np.tile([2.0, 5.0], (4, 1)), keys), grid, **kw)
    np.testing.assert_allclose(weighted, 2.0 * plain, rtol=1e-12)
    half = np.array([[2.0, 1.0], [2.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    mixed = mc_brem_spectrum(_segments(half, keys), grid, **kw)
    assert np.all(mixed > plain) and np.all(mixed < weighted)


def test_characteristic_scorer_weights_each_ionized_shell():
    grid = np.arange(1600.0, 1900.0, 2.0)  # Si K alpha/beta
    kw = dict(composition=[("Si", 0.05)], n_hat=np.array([0.0, 0.0, 1.0]))
    plain = mc_characteristic_spectrum(_segments(), grid, **kw)
    assert plain.max() > 0.0
    shells = ("14:K", "14:L1", "14:L2", "14:L3")
    doubled = mc_characteristic_spectrum(
        _segments(np.tile([2.0, 2.0, 2.0, 2.0, 9.0], (4, 1)), (*shells, BREM_KEY)), grid, **kw
    )
    np.testing.assert_allclose(doubled, 2.0 * plain, rtol=1e-12)
    # K alone: the K lines double, only the L-line Lorentzian tails stay put.
    k_only = mc_characteristic_spectrum(_segments(np.full((4, 1), 2.0), ("14:K",)), grid, **kw)
    np.testing.assert_allclose(k_only, 2.0 * plain, rtol=1e-4)
    assert not np.array_equal(k_only, 2.0 * plain)
    # A key absent from the rows leaves its shell unweighted.
    other = mc_characteristic_spectrum(_segments(np.tile([3.0], (4, 1)), ("13:K",)), grid, **kw)
    np.testing.assert_array_equal(other, plain)


def test_shell_key_spelling():
    assert shell_key(14, "K") == "14:K"
