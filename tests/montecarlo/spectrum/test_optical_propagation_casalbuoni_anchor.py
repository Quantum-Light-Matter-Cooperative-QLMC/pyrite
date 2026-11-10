"""Published near-field TR through angular-spectrum propagation: TESLA Report 2005-15.

S. Casalbuoni, B. Schmidt and P. Schmüser, "Far-Infrared Transition and
Diffraction Radiation, Part I" (DESY, 2005). A relativistic electron crossing
a metal disc of radius ``a`` drives the radial source field of Eq. (11); the
field at distance ``D`` is the Huygens integral of Eq. (12), whose second-order
(Fresnel) expansion in the source radius is Eq. (25),

    d^2U/(dw dOmega) = e^2 k^4 / (4 pi^3 eps0 c beta^4 gamma^2)
                       |int_0^a J_1(k rho sin th) K_1(k rho/(beta gamma))
                        exp(i k rho^2 / (2R)) rho drho|^2,

with the absolute prefactor from Eqs. (11), (12) and (19). Two published
results are pinned, both with every parameter stated and nothing fitted:

- Fig. 8: the near-field angular distribution for ``lambda`` = 0.3 mm,
  ``gamma`` = 1000, ``a`` = 20 mm, ``D`` = 250 mm (Fresnel number 5.3; the
  far-field curve of the same figure differs from it by 0.96 of full scale).
  The red line is Eq. (25), the red diamonds Eq. (12) with the exact distance.
  The ordinate is in arbitrary units; every curve of the figure peaks at 1.00.
- Fig. 15 (right), dashed: the absolute energy in µJ per GHz of an infinitely
  short 1 nC bunch of radius 150 µm from an ``a`` = 12.5 mm disc inside a
  100 mrad detector 600 mm away.

Here the PEC disc source is propagated with the periodic FFT angular spectrum
(``propagate_screen_field``; ``propagate_vacuum`` is the same operator) and
collected with ``collect_screen_field``. Fig. 8 tests the propagator phase.
Fig. 15 does not: the detector takes nearly all of the beam, so the report's
near- and far-field energies differ by under 1 %; it tests the absolute
collected-energy normalization (Faraday ``H``, Poynting flux, aperture, N^2)
behind the propagator.

Window convergence. A square grid of pitch ``dx`` carries plane waves up to
``k_x = k_y = pi/dx``; with ``s = lambda/(2 dx)`` the corner mode walks off by
``D s / sqrt(1 - 2 s^2)`` along each axis. The periodic images of the disc stay
out of an observation radius ``r_obs`` when ``W >= a + r_obs + D s /
sqrt(1 - 2 s^2)``. With ``dx <= lambda/sqrt(2)`` grazing waves are carried and
no window satisfies this (the wraparound then decays only algebraically); the
pitch must be refined together with the window. The slow test below shows
both the converged sequence and the failure.

Scope. Fig. 8 is paraxial (numerical aperture <= 0.05) at Fresnel number ~5.
The ``1e-3`` complex comparison with the non-periodic Green sum fixes the phase
sign, but a paraxial transfer function deviates by only ``1.6e-3`` there, so
exact-versus-Fresnel transfer is not discriminated by the published curves.
Only the transverse field is compared, as in the report's scalar Eq. (12); the
omitted ``|E_z|^2`` is 0.2 % at ``gamma theta`` = 46. In Fig. 15 the 150 µm
bunch radius changes the energy by under 0.4 %, below the tolerance, so the
transverse bunch field of Eq. (33) is carried along but not tested.

Data: ``tests/data/optical/casalbuoni-2005-fig8-15.json``, curve vertices from
the PDF's vector paths. Not covered: the solid curve of Fig. 15 right (it is
not the dashed curve times the stated Gaussian form factor, by up to 10 %),
optical elements, ``collect_screen_green``, any measurement. The
physical-optics disc source is shared with the report, so that approximation
is not tested. Fresh-context review: ``docs/validation/radiation-physics/
optical-wavefront-propagation.md``.

Validation: optical-wavefront-propagation
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import i1, j1, k1, roots_legendre

from pyrite.montecarlo.spectrum.optical_finite_collection import (
    PlaneAcceptance,
    collect_screen_field,
)
from pyrite.montecarlo.spectrum.optical_finite_screen import (
    PECCircularScreen,
    circular_pec_transition_field,
    observe_screen_green,
    propagate_screen_field,
)
from pyrite.montecarlo.spectrum.optical_wavefront import OpticalWavefront, propagate_vacuum

_DATA = json.loads(
    (Path(__file__).parents[2] / "data" / "optical" / "casalbuoni-2005-fig8-15.json").read_text()
)
# CODATA constants, independent of the implementation's module constants.
_E_C = 1.602176634e-19
_EPS0 = 8.8541878128e-12
_C = 299792458.0
_H_EV_S = 4.135667696e-15
_Z0 = 1.0 / (_EPS0 * _C)
_GAMMA = 1000.0
_BETA = float(np.sqrt(1.0 - _GAMMA**-2))


def _photon_energy_eV(wavelength_m):
    return _H_EV_S * _C / wavelength_m


def _eq25(theta, k, a, distance, *, bunch_radius=None, n=3000):
    """Eq. (25) ``d^2U/(dw dOmega)`` per electron [J s/sr]; ``distance=None`` is Eq. (26).

    ``bunch_radius`` replaces the single-electron ``K_1`` by the uniform-disc
    bunch field of Eq. (33), normalized per electron.
    """
    t, w = roots_legendre(n)
    alpha = k / (_BETA * _GAMMA)
    if bunch_radius is None:
        rho, w = 0.5 * a * (t + 1), 0.5 * a * w
        source = k1(alpha * rho)
    else:
        inner, outer = (
            0.5 * bunch_radius * (t + 1),
            bunch_radius + 0.5 * (a - bunch_radius) * (t + 1),
        )
        rho = np.concatenate([inner, outer])
        w = np.concatenate([0.5 * bunch_radius * w, 0.5 * (a - bunch_radius) * w])
        scale = 2.0 / (alpha * bunch_radius)
        source = scale * np.where(
            rho < bunch_radius,
            k1(alpha * bunch_radius) * i1(alpha * rho),
            i1(alpha * bunch_radius) * k1(alpha * rho),
        )
    theta = np.atleast_1d(theta)[:, None]
    phase = 1.0 if distance is None else np.exp(1j * k * rho**2 * np.cos(theta) / (2 * distance))
    integral = np.sum(w * rho * source * j1(k * rho * np.sin(theta)) * phase, axis=1)
    prefactor = _E_C**2 * k**4 / (4 * np.pi**3 * _EPS0 * _C * _BETA**4 * _GAMMA**2)
    return prefactor * np.abs(integral) ** 2


def _source(wavelength_m, radius_m, window_m, pitch_m, offset_m=(0.0, 0.0)):
    n = int(round(window_m / pitch_m))
    n += n % 2
    screen = PECCircularScreen(
        outer_radius_m=radius_m, beta_z=-_BETA, electron_x_m=offset_m[0], electron_y_m=offset_m[1]
    )
    return circular_pec_transition_field(
        _photon_energy_eV(wavelength_m), screen, nx=n, ny=n, dx_m=pitch_m, dy_m=pitch_m
    )


def _walkoff_window(wavelength_m, pitch_m, radius_m, observed_radius_m, distance_m):
    """Smallest window keeping periodic images out of the observed radius (docstring)."""
    s = wavelength_m / (2 * pitch_m)
    if 2 * s * s >= 1:
        return np.inf
    return radius_m + observed_radius_m + distance_m * s / np.sqrt(1 - 2 * s * s)


# --- Fig. 8: near-field angular distribution (propagator phase) --------------------

_FIG8 = _DATA["fig8"]["parameters"]
_LAMBDA = _FIG8["wavelength_mm"] * 1e-3
_K = 2 * np.pi / _LAMBDA
_A = _FIG8["disc_radius_mm"] * 1e-3
_D = _FIG8["distance_mm"] * 1e-3
_R_OBS = _D * np.tan(52.0 / _GAMMA)


def _fig8_row(window_m, pitch_m):
    """Propagated transverse ``d^2U/(dw dOmega)`` [J s/sr] on the grid row next to the axis."""
    field = _source(_LAMBDA, _A, window_m, pitch_m)
    assert _FIG8["gamma"] == _GAMMA
    electric = propagate_screen_field(field, _D).electric_xy_v_s_m
    n = electric.shape[-1]
    x = (np.arange(n) - (n - 1) / 2) * pitch_m
    row = n // 2
    keep = (x > 0) & (x < _R_OBS)
    rho = np.hypot(x[keep], x[row])
    intensity = np.abs(electric[0, row, keep]) ** 2 + np.abs(electric[1, row, keep]) ** 2
    # Two-sided transform: dU/dw = |E|^2/(pi Z0) per area; dS = R^2 dOmega (report Eq. 20).
    density = intensity * (_D**2 + rho**2) / (np.pi * _Z0)
    return field, x[keep], x[row], np.arctan(rho / _D), density, electric[:, row, keep].T


def test_fft_propagation_reproduces_fig8_near_field_distribution():
    window, pitch = 0.32, 0.3e-3
    assert window >= _walkoff_window(_LAMBDA, pitch, _A, _R_OBS, _D)
    field, x, y, theta, density, electric = _fig8_row(window, pitch)
    scaled = _GAMMA * theta
    lobe = (scaled > 8) & (scaled < 46)  # intensity above ~4 % of the peak
    # Absolute Eq. (25): measured deviation 0.58 % (pitch and window sequence below).
    equation = _eq25(theta, _K, _A, _D)
    np.testing.assert_allclose(density[lobe], equation[lobe], rtol=1e-2)
    # The non-periodic Green sum of the same sampled source has no window at all.
    # Complex comparison: the intensity alone is blind to the sign of the phase.
    some = np.flatnonzero(lobe)[::12]
    points = np.column_stack([x[some], np.full(some.size, y), np.full(some.size, _D)])
    green = observe_screen_green(field, points)[:, :2]
    # Measured 1.9e-4 (source quadrature); a paraxial transfer would give 1.6e-3.
    assert np.max(np.abs(electric[some] - green)) < 1e-3 * np.max(np.abs(green))
    # Published curves (peak-normalized): ordinate digitization <= 0.003 of full
    # scale; the figure itself departs from its Eq. (25) by 0.011 near gamma theta = 42.
    normalized = density / density.max()
    for name in ("second_order_line", "exact_distance_diamonds"):
        curve = np.array(_DATA["fig8"][name])
        curve = curve[(curve[:, 0] > scaled[0]) & (curve[:, 0] < scaled[-1])]
        computed = np.interp(curve[:, 0], scaled, normalized)
        assert np.max(np.abs(computed - curve[:, 1])) < 0.02, name
        assert curve[np.argmax(curve[:, 1]), 0] == pytest.approx(
            scaled[np.argmax(density)], abs=0.7
        )
    # The far field of the same disc (Eq. 26) is a different curve altogether.
    line = np.array(_DATA["fig8"]["second_order_line"])
    far = _eq25(line[:, 0] / _GAMMA, _K, _A, None)
    assert np.max(np.abs(far / far.max() - line[:, 1])) > 0.9
    # The V/m phasor entry point is the same operator.
    phasor = OpticalWavefront(field.energy_eV, pitch, pitch, np.array(field.electric_xy_v_s_m))
    np.testing.assert_allclose(
        propagate_vacuum(phasor, _D).electric_xy_v_m,
        propagate_screen_field(field, _D).electric_xy_v_s_m,
        rtol=1e-12,
        atol=0,
    )


@pytest.mark.slow
def test_fig8_propagation_converges_only_with_the_walk_off_window():
    def lobe(window, pitch):
        """Lobe density on its own grid points and its largest deviation from Eq. (25)."""
        _, _, _, theta, density, _ = _fig8_row(window, pitch)
        keep = (_GAMMA * theta > 8) & (_GAMMA * theta < 46)
        return density[keep], np.max(np.abs(density[keep] / _eq25(theta[keep], _K, _A, _D) - 1))

    base, base_error = lobe(0.32, 0.3e-3)
    # A 1.5x window at fixed pitch (same grid points), and a finer pitch whose
    # walk-off window still fits.
    wide, _ = lobe(0.48, 0.3e-3)
    np.testing.assert_allclose(wide, base, rtol=2e-3)
    assert 0.32 >= _walkoff_window(_LAMBDA, 0.25e-3, _A, _R_OBS, _D)
    _, fine_error = lobe(0.32, 0.25e-3)
    _, coarse_error = lobe(0.32, 0.6e-3)
    # Measured 2.5 % -> 0.58 % -> 0.56 % at pitch 0.6, 0.3, 0.25 mm: the floor is the
    # report's Fresnel Eq. (25) against exact propagation (FFT vs exact Eq. (12): 0.17 %).
    assert fine_error < 8e-3 and base_error < 8e-3
    assert 1.5e-2 < coarse_error < 4e-2
    # Below dx = lambda/sqrt(2) a square grid carries grazing corner waves, whose
    # wraparound no window removes: refining the pitch at a fixed window fails.
    assert np.isinf(_walkoff_window(_LAMBDA, 0.2e-3, _A, _R_OBS, _D))
    assert lobe(0.32, 0.2e-3)[1] > 0.01
    assert lobe(0.32, 0.15e-3)[1] > 0.05
    # Nor is a window inside the walk-off of the base pitch converged.
    assert 0.16 < _walkoff_window(_LAMBDA, 0.3e-3, _A, _R_OBS, _D)
    assert lobe(0.16, 0.3e-3)[1] > 0.015


# --- Fig. 15 right: absolute collected bunch energy at 600 mm ----------------------

_FIG15 = _DATA["fig15_right"]["parameters"]


def _disc_points(radius_m, count):
    """Equal-area (sunflower) points of a uniform disc."""
    i = np.arange(count) + 0.5
    r, angle = radius_m * np.sqrt(i / count), i * np.pi * (3 - np.sqrt(5))
    return np.column_stack([r * np.cos(angle), r * np.sin(angle)])


def _cone_uJ_per_GHz(f_thz, distance, population):
    """Eqs. (25) and (33) integrated over the detector cone, for the whole bunch."""
    theta_det = _FIG15["theta_det_rad"]
    t, w = roots_legendre(400)
    theta, w = 0.5 * theta_det * (t + 1), 0.5 * theta_det * w
    density = _eq25(
        theta,
        2 * np.pi * f_thz * 1e12 / _C,
        _FIG15["disc_radius_mm"] * 1e-3,
        distance,
        bunch_radius=_FIG15["bunch_radius_um"] * 1e-6,
    )
    return population**2 * np.sum(w * 2 * np.pi * np.sin(theta) * density) * 2 * np.pi * 1e9 * 1e6


@pytest.mark.parametrize(("f_thz", "pitch_m"), [(0.6, 1.0e-3), (1.0, 0.6e-3)])
def test_fft_plane_collection_reproduces_fig15_near_field_bunch_energy(f_thz, pitch_m):
    assert _FIG15["gamma"] == _GAMMA
    wavelength = _C / (f_thz * 1e12)
    radius = _FIG15["disc_radius_mm"] * 1e-3
    distance = _FIG15["distance_mm"] * 1e-3
    detector = distance * np.tan(_FIG15["theta_det_rad"])
    window = 0.48
    assert window >= _walkoff_window(wavelength, pitch_m, radius, detector, distance)
    # Infinitely short bunch: the coherent field is the mean single-electron field.
    points = _disc_points(_FIG15["bunch_radius_um"] * 1e-6, 7)
    fields = [_source(wavelength, radius, window, pitch_m, tuple(p)) for p in points]
    mean = replace(
        fields[0], electric_xy_v_s_m=np.mean([f.electric_xy_v_s_m for f in fields], axis=0)
    )
    collected = collect_screen_field(
        mean, plane_z_m=distance, aperture=PlaneAcceptance("circle", radius_m=detector)
    )
    population = _FIG15["charge_nC"] * 1e-9 / _E_C
    bandwidth_eV = _H_EV_S * _FIG15["bandwidth_GHz"] * 1e9
    energy = population**2 * collected.spectral_energy_j_per_eV * bandwidth_eV * 1e6
    curve = np.array(_DATA["fig15_right"]["f_THz_uJ_per_GHz"])
    published = np.interp(np.log(f_thz), np.log(curve[:, 0]), curve[:, 1])
    # Measured: halving the 0.6 THz pitch moves the energy +0.3 % (from -0.04 % to
    # +0.28 % of the equation), a 1.5x window 3e-5; figure vs its equations <= 0.8 %.
    assert energy == pytest.approx(published, rel=1.5e-2)
    assert energy == pytest.approx(_cone_uJ_per_GHz(f_thz, distance, population), rel=1e-2)
    # Why this is a normalization test, not a phase test: far and near energies agree.
    assert _cone_uJ_per_GHz(f_thz, None, population) == pytest.approx(
        _cone_uJ_per_GHz(f_thz, distance, population), rel=1e-2
    )
