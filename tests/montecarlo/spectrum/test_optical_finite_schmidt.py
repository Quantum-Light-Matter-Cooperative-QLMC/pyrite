"""Published finite-disc TR references: Schmidt et al., arXiv:1803.00608, Sect. 2.3 (#616).

Reference: S. Schmidt, M. Dohlus, B. Schmidt et al. (2018), "Longitudinal bunch
diagnostics using coherent transition radiation spectroscopy", DESY report,
Eqs. (12)-(15), (18) and Figs. 6-7. Eq. (14) is the far-field Ginzburg-Frank
yield of a circular PEC disc of radius ``a`` times ``(1 - T)^2``; Eq. (15) the
paraxial (Fresnel) near-field Hankel integral at distance ``R``; both carry the
absolute prefactor ``e^2 k^4 / (4 pi^3 eps0 c beta^4 gamma^2)`` of d2U/(domega
dOmega). Fig. 7 values were read from the published plot (peak positions only).

The solver is the vector PEC surface-current source observed with the exact
retarded Green function (no paraxial expansion).

Validation: optical-finite-boundary-field
"""

import numpy as np
import pytest
from scipy.special import j0, j1, k0, k1, roots_legendre

from pyrite.materials.crystal import HC_EV_ANG
from pyrite.montecarlo.spectrum.optical_finite_screen import (
    _E_CHARGE_C,
    _HBAR_EV_S,
    _Z0_OHM,
    PECCircularScreen,
    circular_pec_transition_field,
    ginzburg_frank_photons_per_eV_sr,
    observe_screen_green,
)

_EPS0 = 8.8541878128e-12
_C = 299792458.0
_GAMMA = 100.0
_BETA = float(np.sqrt(1.0 - 1.0 / _GAMMA**2))
_LAMBDA_M = 0.3e-3
_K = 2.0 * np.pi / _LAMBDA_M
_E_EV = HC_EV_ANG * 1e-10 / _LAMBDA_M
_THETA = np.linspace(0.002, 0.09, 45)  # gamma theta = 0.2 ... 9


def _t_correction(theta, a, k=_K, beta=_BETA, gamma=_GAMMA):
    """Schmidt Eq. (12)."""
    s, x = np.sin(theta), k * a / (beta * gamma)
    return x * j0(k * a * s) * k1(x) + k * a / (beta**2 * gamma**2 * s) * j1(k * a * s) * k0(x)


def _hankel_photons(theta, a, distance=None, n=6000):
    """Schmidt Eq. (13) (far field) or Eq. (15) (``distance`` given) [ph/eV/sr]."""
    t, w = roots_legendre(n)
    rho, w = 0.5 * a * (t + 1.0), 0.5 * a * w
    phase = 1.0 if distance is None else np.exp(1j * _K * rho[None] ** 2 / (2 * distance[:, None]))
    integrand = j1(_K * rho[None] * np.sin(theta)[:, None]) * k1(_K * rho / (_BETA * _GAMMA))
    integral = np.sum(w * integrand * phase * rho, axis=1)
    prefactor = _E_CHARGE_C**2 * _K**4 / (4 * np.pi**3 * _EPS0 * _C * _BETA**4 * _GAMMA**2)
    return prefactor * np.abs(integral) ** 2 / (_HBAR_EV_S * _E_EV * _E_CHARGE_C)


def _solver_photons(a, distance, dx, *, transverse_only=False):
    """Vector PEC source, exact Green field on the x axis at ``distance`` [ph/eV/sr].

    ``transverse_only`` drops the longitudinal ``E_z`` (absent from scalar Eq. 15).
    """
    n = int(2 * a / dx) + 8
    n += n % 2
    screen = PECCircularScreen(outer_radius_m=a, beta_z=-_BETA)
    field = circular_pec_transition_field(_E_EV, screen, nx=n, ny=n, dx_m=dx, dy_m=dx)
    x = distance * np.tan(_THETA)
    points = np.stack([x, 0.0 * x, np.full_like(x, distance)], axis=-1)
    e = observe_screen_green(field, points, max_interactions=1 << 21)
    radius = np.hypot(distance, x)
    components = e[:, :2] if transverse_only else e
    du = radius**2 * np.sum(np.abs(components) ** 2, axis=-1) / (np.pi * _Z0_OHM)
    return du / (_HBAR_EV_S * _E_EV * _E_CHARGE_C), radius


def test_closed_form_correction_is_the_far_field_hankel_integral_and_fig6():
    a = 3e-3
    gf = ginzburg_frank_photons_per_eV_sr(_E_EV, _THETA, _BETA)
    np.testing.assert_allclose(
        gf * (1 - _t_correction(_THETA, a)) ** 2, _hankel_photons(_THETA, a), rtol=1e-9
    )
    # Fig. 6 (small-angle Eq. (18)): a / r_eff = 1 and 0.5, r_eff = gamma lambda.
    theta_s = np.array([0.01, 1.4, 2.4])
    for xi, expected in ((1.0, (0.955, 1.0, 1.0)), (0.5, (0.56, 1.10, 0.94))):
        a = xi * _GAMMA * _LAMBDA_M
        factor = (1 - _t_correction(theta_s / _GAMMA, a)) ** 2
        np.testing.assert_allclose(factor, expected, atol=0.02)


def test_fig7a_small_disc_far_field_absolute_yield():
    a, distance = 3e-3, 4.0  # a << r_eff = 30 mm; D > gamma r_eff = 3 m
    solver, radius = _solver_photons(a, distance, 0.05e-3)
    gf = ginzburg_frank_photons_per_eV_sr(_E_EV, _THETA, _BETA)
    np.testing.assert_allclose(solver, gf * (1 - _t_correction(_THETA, a)) ** 2, rtol=3e-3)
    np.testing.assert_allclose(solver, _hankel_photons(_THETA, a, radius), rtol=3e-3)
    # Published Fig. 7(a): finite-disc lobe at gamma theta ~ 4.6, Ginzburg-Frank at 1.
    assert _GAMMA * _THETA[np.argmax(solver)] == pytest.approx(4.6, abs=0.25)
    assert _GAMMA * _THETA[np.argmax(gf)] == pytest.approx(1.0, abs=0.1)


def test_fig7b_large_disc_near_field_follows_fresnel_integral():
    a, distance = 30e-3, 0.2  # a = r_eff; D << gamma r_eff
    solver, radius = _solver_photons(a, distance, 0.1e-3)
    near = _hankel_photons(_THETA, a, radius)
    # Full vector |E|^2 includes the longitudinal E_z ~ (a / D) E_rho of the near
    # field, absent from scalar Eq. (15); it matters only near the on-axis minimum
    # (+5.7 % at gamma theta = 0.2, falling as theta^-2). The transverse field alone
    # follows Eq. (15) closely.
    useful = _GAMMA * _THETA >= 0.5
    np.testing.assert_allclose(solver[useful], near[useful], rtol=1e-2)
    transverse, _ = _solver_photons(a, distance, 0.1e-3, transverse_only=True)
    inner = _GAMMA * _THETA <= 3.0
    np.testing.assert_allclose(transverse[inner], near[inner], rtol=2e-3)
    far = (
        ginzburg_frank_photons_per_eV_sr(_E_EV, _THETA, _BETA) * (1 - _t_correction(_THETA, a)) ** 2
    )
    assert np.max(np.abs(solver / far - 1)) > 0.5  # near field is not the far-field formula
    # Published Fig. 7(b): near-field peak ~ 3.2 and a secondary bump ~ 6.5; far field ~ 1.
    gt = _GAMMA * _THETA
    assert gt[np.argmax(solver)] == pytest.approx(3.2, abs=0.25)
    assert gt[np.argmax(far)] == pytest.approx(1.0, abs=0.25)
    local = (solver[1:-1] > solver[:-2]) & (solver[1:-1] > solver[2:])
    peaks = gt[1:-1][local]
    assert peaks[0] == pytest.approx(3.2, abs=0.25)
    assert np.any((peaks > 6.0) & (peaks < 7.2))
