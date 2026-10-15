"""Lindblad scattering kernels for planar channeling (cr-lindblad-jump-operators)."""

from dataclasses import replace

import numpy as np
import pytest
import xraydb
from scipy import integrate

from pyrite.materials.atomic import Z_TABLE
from pyrite.materials.channeling import longitudinal_kinematics, planar_profile
from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.channeling import (
    decoherence_rate,
    electronic_channels,
    electronic_kernel,
    tds_kernel,
)
from pyrite.montecarlo.channeling.operators import _site_densities
from pyrite.montecarlo.transport.shell_gos import (
    shell_gos_moments,
    windowed_shell_gos_moments,
)
from pyrite.montecarlo.transport.shell_rates import catalog_shell_oscillators

ENERGY_EV = 17e6
BOHR_ANG = 0.529177210544
SMALL = {"n_cells": 4, "max_order": 8}


@pytest.fixture(scope="module")
def si110():
    return planar_profile("silicon", (1, 1, 0), 293.15)


@pytest.fixture(scope="module")
def tds(si110):
    return tds_kernel(si110, ENERGY_EV, **SMALL)


@pytest.fixture(scope="module")
def electronic(si110):
    return electronic_kernel(si110, ENERGY_EV, **SMALL)


def _electron_factor(element, s):
    """Relativistic Mott--Bethe ``f^e(s)`` [angstrom] with ``f0`` normalized to ``Z`` at ``s = 0``.

    Independent of the kernel code: International Tables Vol. C §4.3,
    ``f^e = gamma (Z - f_x(s)) / (8 pi^2 a0 s^2)``, with ``f_x`` beyond the
    Waasmaier--Kirfel range ``s > 6`` decaying as ``s^-4``.
    """
    gamma, _ = longitudinal_kinematics(ENERGY_EV)
    z = Z_TABLE[element]
    s = np.maximum(np.asarray(s, dtype=float), 0.02 / (4 * np.pi))
    fx = xraydb.f0(element, np.minimum(s, 6.0).ravel()).reshape(s.shape)
    fx = fx * z / xraydb.f0(element, [0.0])[0]
    fx = np.where(s > 6.0, fx * (6.0 / s) ** 4, fx)
    return gamma * (z - fx) / (8 * np.pi**2 * BOHR_ANG * s**2)


def _wavelength_ang():
    pc = np.sqrt(ENERGY_EV * (ENERGY_EV + 2 * M_E_EV))
    return 2 * np.pi * HBARC_EV_ANG / pc


# ---- anchors: absorptive potential ------------------------------------------


def test_mean_tds_rate_is_pennycook_jesson_cross_section(si110, tds):
    # Pennycook & Jesson (1991); Mendis, Acta Cryst. A80, 178 (2024), Eqs. 8-9:
    # dsigma/dOmega = |f^e|^2 (1 - exp(-2 B s^2)), 1/lambda = N sigma.
    # Small-angle-exact solid angle: dOmega = 8 pi lambda^2 s ds.
    b = si110.b_factors_ang2["Si"]
    lam = _wavelength_ang()

    def integrand(log_s):
        s = np.exp(log_s)
        return s * s * _electron_factor("Si", s) ** 2 * -np.expm1(-2 * b * s * s)

    sigma = 8 * np.pi * lam**2 * integrate.quad(integrand, np.log(1e-5), np.log(1e3), limit=400)[0]
    density = len(CRYSTALS["silicon"]["basis"]) / CRYSTALS["silicon"]["V_cell"]
    assert tds.diagonal_coefficients[0].real == pytest.approx(density * sigma, rel=2e-3)


@pytest.mark.parametrize("harmonic", [1, 2, 3])
def test_tds_absorptive_harmonics_are_hall_hirsch(si110, tds, harmonic):
    # Hall & Hirsch (1965); Bird & King (1990): the absorptive Fourier
    # coefficient is n_G 4 lambda^2 int d^2s f^e(|s + s_G/2|) f^e(|s - s_G/2|)
    # [exp(-B s_G^2) - exp(-2 B s^2 - B s_G^2 / 2)], with s = q / 4 pi.
    b = si110.b_factors_ang2["Si"]
    s_g = harmonic * si110.g_inv_ang / (4 * np.pi)
    lam = _wavelength_ang()

    def radial(log_s):
        s = np.exp(log_s)

        def angular(phi):
            cross = s * s_g * np.cos(phi)
            base = s * s + 0.25 * s_g * s_g
            ket = _electron_factor("Si", np.sqrt(base + cross))
            bra = _electron_factor("Si", np.sqrt(np.maximum(base - cross, 0.0)))
            return ket * bra

        ring = integrate.quad(angular, 0.0, np.pi, limit=100, epsrel=1e-6)[0] * 2
        bracket = np.exp(-b * s_g * s_g) - np.exp(-2 * b * s * s - 0.5 * b * s_g * s_g)
        return s * s * ring * bracket

    breaks = np.log([0.5 * s_g, s_g, 2 * s_g])
    edges = np.concatenate([[np.log(1e-4)], breaks, [np.log(1e3)]])
    value = sum(
        integrate.quad(radial, lo, hi, limit=100, epsrel=1e-6)[0]
        for lo, hi in zip(edges, edges[1:], strict=False)
    )
    density = _site_densities(si110, harmonic)["Si"][2 * harmonic]
    expected = density * 4 * lam**2 * value
    assert tds.diagonal_coefficients[harmonic] == pytest.approx(expected, rel=2e-3, abs=1e-12)


def test_absorptive_potential_scales_as_inverse_velocity(si110):
    slow = tds_kernel(si110, 2e6, n_cells=2, max_order=2)
    fast = tds_kernel(si110, 50e6, n_cells=2, max_order=2)
    x = np.linspace(0.0, si110.period_ang, 7)
    _, beta_slow = longitudinal_kinematics(2e6)
    _, beta_fast = longitudinal_kinematics(50e6)
    ratio = slow.absorptive_potential_eV(x) / fast.absorptive_potential_eV(x)
    np.testing.assert_allclose(ratio, beta_fast / beta_slow, rtol=1e-12)


def test_tds_absorption_peaks_on_planes(si110, tds):
    x = np.linspace(0.0, si110.period_ang, 64, endpoint=False)
    w = tds.absorptive_potential_eV(x)
    assert np.all(w > 0.0)
    assert np.argmax(w) == 0  # wells, and atoms, sit at x = 0
    assert w[0] / w[32] > 100.0


# ---- kernel structure --------------------------------------------------------


def test_site_densities_reproduce_profile_nuclear_density(si110):
    sites = _site_densities(si110, 6)
    smeared = sum(
        density[6:]
        * np.exp(-si110.b_factors_ang2[el] * (np.arange(7) * si110.g_inv_ang / (4 * np.pi)) ** 2)
        for el, density in sites.items()
    )
    np.testing.assert_allclose(smeared, si110.nuclear_density[:7], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("name", ["tds", "electronic"])
def test_kernels_are_hermitian_and_positive_semidefinite(request, name):
    kernel = request.getfixturevalue(name).momentum_kernel
    np.testing.assert_allclose(kernel, np.conj(np.swapaxes(kernel, 1, 2)), atol=0.0)
    eigen = np.array([np.linalg.eigvalsh(block) for block in kernel])
    assert eigen.min() >= -1e-12 * eigen.max()


@pytest.mark.parametrize("name", ["tds", "electronic"])
def test_cell_kernel_is_position_space_form(request, name, si110):
    kernel = request.getfixturevalue(name)
    cells = kernel.cell_kernel(24)
    x = np.arange(24) * si110.period_ang / 24
    np.testing.assert_allclose(np.diagonal(cells[0]).real, kernel.kept_diagonal(x), rtol=1e-10)
    # C(x', x) = C(x, x')*: K[-j] = K[j]^dagger.
    mirrored = np.conj(np.swapaxes(cells[(-np.arange(kernel.n_cells)) % kernel.n_cells], 1, 2))
    np.testing.assert_allclose(cells, mirrored, atol=1e-12 * np.abs(cells).max())
    with pytest.raises(ValueError, match="n_grid"):
        kernel.cell_kernel(2 * kernel.max_order + 1)


def test_kept_band_converges_to_full_diagonal(si110, tds):
    x = np.linspace(0.0, si110.period_ang, 128, endpoint=False)
    coarse = tds.kept_diagonal(x).mean() / tds.diagonal(x).mean()
    wide = tds_kernel(si110, ENERGY_EV, n_cells=4, max_order=24)
    fine = wide.kept_diagonal(x).mean() / wide.diagonal(x).mean()
    assert coarse < fine <= 1.0
    assert fine > 0.99


def test_kernels_add(tds, electronic, si110):
    total = tds + electronic
    x = np.linspace(0.0, si110.period_ang, 5)
    np.testing.assert_allclose(total.diagonal(x), tds.diagonal(x) + electronic.diagonal(x))
    assert total.label == "tds+electronic"
    with pytest.raises(ValueError, match="share"):
        tds + tds_kernel(si110, ENERGY_EV, n_cells=2, max_order=8)


# ---- electronic channels -----------------------------------------------------


def test_electronic_channels_integrate_to_penelope_moments():
    oscillators = catalog_shell_oscillators("silicon")
    channels = electronic_channels(oscillators, ENERGY_EV)
    per_electron = 1e16 / oscillators.electrons_per_formula
    moments = shell_gos_moments(oscillators, ENERGY_EV)
    for k in range(len(oscillators.oscillators)):
        ridge = np.sqrt(
            channels.distant_qmax_inv_ang[k] ** 2 - channels.distant_qmin_inv_ang[k] ** 2
        )
        if ridge == 0.0:
            continue
        one = replace(
            channels,
            distant_ang2=np.where(
                np.arange(channels.distant_ang2.size) == k, channels.distant_ang2, 0.0
            ),
        )
        value = integrate.quad(
            lambda q, one=one: 2 * np.pi * q * one.distant(q),
            0.0,
            ridge,
            points=[channels.distant_qmin_inv_ang[k]],
            limit=200,
        )[0]
        assert value == pytest.approx(moments.distant_longitudinal[k, 0] * per_electron, rel=1e-6)
    # Close collisions mapped onto q_perp: (hbar c q)^2 = 2 mc^2 W (1 - W/E).
    w_top = 0.25 * ENERGY_EV
    q_top = float(channels.close_transfer(w_top))
    breaks = channels.close_transfer(channels.close_lower_eV)
    edges = np.unique(np.concatenate([[breaks.min()], breaks, [q_top]]))
    close = sum(
        integrate.quad(lambda q: 2 * np.pi * q * channels.close(q), lo, hi, limit=200)[0]
        for lo, hi in zip(edges, edges[1:], strict=False)
    )
    window = windowed_shell_gos_moments(oscillators, ENERGY_EV, 0.0, w_top)
    assert close == pytest.approx(window.close[:, 0].sum() * per_electron, rel=1e-6)


def test_uniform_density_gives_pure_momentum_diffusion(si110):
    flat = replace(si110, electron_density=si110.electron_density[:1].copy())
    kernel = electronic_kernel(flat, ENERGY_EV, **SMALL)
    off = kernel.momentum_kernel - np.einsum("rii->ri", kernel.momentum_kernel)[..., None] * np.eye(
        kernel.orders.size
    )
    assert np.abs(off).max() < 1e-12 * np.abs(kernel.momentum_kernel).max()


def test_catalog_oscillators_required(si110):
    diamond = planar_profile("diamond", (1, 1, 0))
    with pytest.raises(ValueError, match="conduction-band"):
        electronic_kernel(diamond, ENERGY_EV, **SMALL)


# ---- decoherence diagnostic --------------------------------------------------


def test_decoherence_rate_limits(si110, tds, electronic):
    rate = decoherence_rate(si110, ENERGY_EV, [0.0, 1e-3, 2e-3, 1e5])
    assert rate.total[0] == pytest.approx(0.0, abs=1e-15)
    # Soft limit: Gamma ~ Delta^2 ln(1/Delta); the Coulomb log comes from the
    # 1/q^4 Rutherford tails (point nucleus, close collisions).
    for part in (rate.tds, rate.electronic):
        assert 3.0 < part[2] / part[1] < 4.0
    # Far limit: every jump decoheres, so Gamma -> mean loss rate.
    assert rate.tds[-1] == pytest.approx(tds.diagonal_coefficients[0].real, rel=1e-3)
    assert rate.electronic[-1] == pytest.approx(electronic.diagonal_coefficients[0].real, rel=1e-4)
    np.testing.assert_allclose(rate.total, rate.tds + rate.electronic)


@pytest.mark.parametrize(
    ("delta", "reference"), [(1.92, 6.40792e-5), (3.0, 6.38278e-5), (5.0, 6.37912e-5)]
)
def test_tds_decoherence_rate_matches_converged_reference(si110, delta, reference):
    # Reference: fresh-context verifier's dense quadrature of the same Hankel
    # integral (docs/validation/channeling/cr-lindblad-jump-operators.md).
    assert decoherence_rate(si110, ENERGY_EV, [delta]).tds[0] == pytest.approx(reference, rel=2e-5)


def test_tds_decoherence_rate_matches_cell_kernel(si110):
    # Position-space route: Gamma(j d) = <C(x, x)> - <Re C(x, x + j d)>, with the
    # full mean loss and the kept-band cell kernel at offset j.
    kernel = tds_kernel(si110, ENERGY_EV, n_cells=8, max_order=24)
    cells = kernel.cell_kernel(80)
    for j in (1, 2):
        expected = kernel.diagonal_coefficients[0].real - np.diagonal(cells[j]).real.mean()
        got = decoherence_rate(si110, ENERGY_EV, [j * si110.period_ang]).tds[0]
        assert got == pytest.approx(expected, rel=1e-3)
