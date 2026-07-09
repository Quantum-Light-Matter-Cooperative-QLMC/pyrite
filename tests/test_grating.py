"""Tests for the grazing-incidence grating forward model (P3 #8). Pure numpy."""

import numpy as np
import pytest

from cxr_mc.crystallography import HC_EV_ANG, optical_constants
from cxr_mc.grating import (
    Grating,
    coating_number_density_per_ang3,
    detector_position_mm,
    disperse_spectrum,
    groove_spacing_angstrom,
    resolving_power,
    wavelength_angstrom,
)


def test_wavelength_and_spacing():
    assert wavelength_angstrom(HC_EV_ANG) == pytest.approx(1.0)  # E = hc -> 1 A
    assert groove_spacing_angstrom(1000.0) == pytest.approx(1.0e4)  # 1000/mm -> 1 um


def test_zero_order_is_specular():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(88.0), order=0)
    beta = g.diffraction_angle_rad(np.array([300.0, 900.0]))
    assert np.allclose(beta, -g.alpha_rad)  # sin beta = -sin alpha


def test_first_order_known_value():
    # alpha = 0, d = 1e4 A (1000/mm), order 1, lambda = 100 A (E = hc/100):
    # sin beta = 1 * 100 / 1e4 = 0.01
    g = Grating(groove_density_per_mm=1000.0, alpha_rad=0.0, order=1)
    E = HC_EV_ANG / 100.0
    assert float(g.diffraction_angle_rad(E)) == pytest.approx(np.arcsin(0.01))


def test_dispersion_monotonic_in_energy():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(85.0), order=1)
    # longer wavelength (lower energy) diffracts to a larger beta
    assert float(g.diffraction_angle_rad(300.0)) > float(g.diffraction_angle_rad(1200.0))


def test_evanescent_orders_are_nan():
    # extreme spacing/order so m*lambda/d > 1 -> no propagating order -> NaN.
    # (Realistic soft X-ray gratings stay well clear of this -- that's why they
    # work; here we just exercise the cutoff branch.) d = 50 A, lambda ~ 124 A.
    g = Grating(groove_density_per_mm=200000.0, alpha_rad=0.0, order=1)
    assert np.isnan(float(g.diffraction_angle_rad(100.0)))


def test_detector_position_sign_and_zero():
    assert detector_position_mm(0.5, 0.5, 100.0) == pytest.approx(0.0)
    assert detector_position_mm(0.6, 0.5, 100.0) > 0.0


def test_disperse_spectrum_conserves_flux_and_localizes_line():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(86.0), order=1)
    E = np.arange(500.0, 1200.0, 1.0)
    spec = np.exp(-0.5 * ((E - 850.0) / 6.0) ** 2)  # a line at 850 eV
    x, inten = disperse_spectrum(E, spec, g, distance_mm=200.0)
    # flux conserved: integral(I dx) ~ integral(spec dE)
    order = np.argsort(x)
    flux_x = np.trapezoid(inten[order], x[order])
    flux_E = np.trapezoid(spec, E)
    assert flux_x == pytest.approx(flux_E, rel=0.02)
    # the line maps to a localized band (intensity-weighted std is small in mm)
    w = inten[order] / inten[order].sum()
    xc = (w * x[order]).sum()
    spread = np.sqrt((w * (x[order] - xc) ** 2).sum())
    assert spread < 0.05 * (x.max() - x.min())


def test_resolving_power_positive_and_rises_with_density():
    coarse = Grating(groove_density_per_mm=600.0, alpha_rad=np.deg2rad(86.0), order=1)
    fine = Grating(groove_density_per_mm=2400.0, alpha_rad=np.deg2rad(86.0), order=1)
    R_coarse = float(resolving_power(900.0, coarse, distance_mm=200.0, pixel_mm=0.015))
    R_fine = float(resolving_power(900.0, fine, distance_mm=200.0, pixel_mm=0.015))
    assert R_coarse > 0 and R_fine > R_coarse


# ---- reflectivity / throughput (docs/grazing-grating.md phased-plan step 2) --


def test_reflectivity_total_external_reflection_below_critical_angle():
    # theta << theta_c = sqrt(2 delta): total external reflection, R -> 1
    # (Au at 1500 eV: theta_c ~ 2.5 deg; use a grazing angle ~25x smaller).
    g = Grating(
        groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(90.0 - 0.001), order=1, coating="Au"
    )
    assert float(g.reflectivity(1500.0)) == pytest.approx(1.0, abs=2e-3)


def test_reflectivity_asymptotic_falloff_above_critical_angle():
    # theta >> theta_c: R -> (theta_c / (2 theta))**4 (Als-Nielsen & McMorrow
    # steep power-law falloff, beta->0 idealization). Au at 1500 eV has
    # beta/delta ~ 0.25 (not negligible), so this idealization only converges
    # to <10% error for grazing angle >~20 deg (8% at 20, 14% at 10, 41% at 5)
    # -- 60 deg here is a deliberate, necessary choice, not arbitrary slack.
    # NOTE this checks the code's OWN algebraic theta>>theta_c asymptote (a
    # Taylor-series identity valid at any theta as long as delta,beta << theta^2)
    # -- it is not a check of true large-angle Fresnel physics, since 60 deg is
    # well outside the theta<<1 rad domain the small-angle reduction itself
    # assumes. Real grating operation stays at few-degree grazing throughout;
    # see docs/validation/grazing-reflectivity.md for the full derivation.
    E = 1500.0
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(60.0), order=1, coating="Au")
    n = coating_number_density_per_ang3("Au")
    delta, _ = optical_constants("Au", E, n)
    theta_c = np.sqrt(2.0 * float(delta))
    theta = g.grazing_angle_rad
    expected = (theta_c / (2.0 * theta)) ** 4
    assert float(g.reflectivity(E)) == pytest.approx(expected, rel=0.1)


def test_reflectivity_bounded_and_decreases_away_from_critical_angle():
    E = 1500.0
    grazing_deg = [3.0, 5.0, 10.0, 20.0, 40.0, 60.0]
    Rs = []
    for gd in grazing_deg:
        g = Grating(
            groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(90.0 - gd), order=1, coating="Au"
        )
        R = float(g.reflectivity(E))
        assert 0.0 <= R <= 1.0
        Rs.append(R)
    assert all(a > b for a, b in zip(Rs, Rs[1:], strict=False))  # monotonic decrease


def test_throughput_wiring_reduces_flux_vs_raw_geometry():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(86.0), order=1, coating="Au")
    E = np.arange(500.0, 1200.0, 1.0)
    spec = np.exp(-0.5 * ((E - 850.0) / 6.0) ** 2)  # a line at 850 eV
    x_geom, inten_geom = disperse_spectrum(E, spec, g, distance_mm=200.0)
    x_thr, inten_thr = disperse_spectrum(E, spec, g, distance_mm=200.0, weight_by_throughput=True)
    assert np.allclose(x_geom, x_thr)  # throughput only rescales intensity, not position
    order = np.argsort(x_geom)
    flux_geom = np.trapezoid(inten_geom[order], x_geom[order])
    flux_thr = np.trapezoid(inten_thr[order], x_thr[order])
    # R <= 1 everywhere -> throughput-weighted flux is strictly less than raw geometric flux
    assert 0.0 < flux_thr < flux_geom
    # spot check against a direct hand computation of the weighted integral
    expected_flux = np.trapezoid(spec * g.throughput(E), E)
    assert flux_thr == pytest.approx(expected_flux, rel=0.02)
