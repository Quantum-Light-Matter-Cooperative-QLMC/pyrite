"""Tests for the grazing-incidence grating forward model (P3 #8). Pure numpy."""

import os

import numpy as np
import pytest

from pyrite.detectors import TabulatedEfficiency
from pyrite.detectors._si_sensor import SI_N_PER_ANG3
from pyrite.detectors.grating import (
    ALEXS_SENSORS,
    FlatFieldMount,
    Grating,
    SimpleCCD,
    bin_to_pixels,
    charge_cloud_sigma_um,
    coating_number_density_per_ang3,
    detected_image,
    detected_image_physical,
    detector_position_mm,
    disperse_spectrum,
    energy_fwhm_eV,
    groove_spacing_angstrom,
    qe_absorption,
    resolving_power,
    wavelength_angstrom,
)
from pyrite.materials import load_material_catalog
from pyrite.materials.crystal import HC_EV_ANG, absorption_length_ang, optical_constants


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


# ---- reflectivity / throughput (docs/research/instrumentation/grazing-grating.md phased-plan step 2) --


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
    # see docs/validation/detectors/grazing-reflectivity.md for the full derivation.
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


# ---- simple CCD pixel binning (docs/research/instrumentation/grazing-grating.md phased-plan step 3) ---


def test_alexs_sensor_formats_match_datasheet_geometry():
    # greateyes ALEX-s 1k256 / 2k512 (docs/research/instrumentation/grazing-grating.md "Hardware targets")
    assert ALEXS_SENSORS["1k256"]["n_pix"] == 1024
    assert ALEXS_SENSORS["1k256"]["pixel_um"] == pytest.approx(26.0)
    assert ALEXS_SENSORS["2k512"]["n_pix"] == 2048
    assert ALEXS_SENSORS["2k512"]["pixel_um"] == pytest.approx(13.5)


def test_simple_ccd_from_alexs_geometry():
    ccd = SimpleCCD.from_alexs("1k256")
    assert ccd.n_pix == 1024
    assert ccd.pixel_mm == pytest.approx(0.026)
    assert ccd.width_mm == pytest.approx(1024 * 0.026)
    edges = ccd.pixel_edges_mm(center_mm=0.0)
    centers = ccd.pixel_centers_mm(center_mm=0.0)
    assert edges.shape == (1025,)
    assert centers.shape == (1024,)
    assert edges[0] == pytest.approx(-ccd.width_mm / 2.0)
    assert edges[-1] == pytest.approx(ccd.width_mm / 2.0)
    assert centers[0] == pytest.approx(edges[0] + ccd.pixel_mm / 2.0)

    with pytest.raises(KeyError):
        SimpleCCD.from_alexs("not-a-real-variant")


def test_bin_to_pixels_conserves_flux_when_fully_contained():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(86.0), order=1)
    E = np.arange(500.0, 1200.0, 1.0)
    spec = np.exp(-0.5 * ((E - 850.0) / 6.0) ** 2)  # a line at 850 eV
    x, inten = disperse_spectrum(E, spec, g, distance_mm=200.0)
    # a CCD much wider than the dispersed band -> nothing falls off the edge
    ccd = SimpleCCD(n_pix=2000, pixel_mm=0.05)
    centers, flux = bin_to_pixels(x, inten, ccd)
    order = np.argsort(x)
    flux_expected = np.trapezoid(inten[order], x[order])
    assert flux.sum() == pytest.approx(flux_expected, rel=0.02)
    assert centers.shape == (2000,)
    assert np.all(np.diff(centers) > 0)  # centers strictly increasing


def test_bin_to_pixels_clips_light_outside_sensor():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(86.0), order=1)
    E = np.arange(500.0, 1200.0, 1.0)
    spec = np.exp(-0.5 * ((E - 850.0) / 6.0) ** 2)
    x, inten = disperse_spectrum(E, spec, g, distance_mm=200.0)
    flux_full = np.trapezoid(inten[np.argsort(x)], x[np.argsort(x)])
    # a CCD much narrower than the dispersed band -> most flux falls off the edge
    ccd = SimpleCCD(n_pix=10, pixel_mm=0.001)
    _, flux = bin_to_pixels(x, inten, ccd)
    assert 0.0 < flux.sum() < flux_full


def test_bin_to_pixels_localizes_narrow_line_to_few_pixels():
    ccd = SimpleCCD.from_alexs("2k512")
    # a narrow synthetic peak, already in position/intensity space
    x = np.linspace(-1.0, 1.0, 2001)
    inten = np.exp(-0.5 * (x / 0.01) ** 2)
    _, flux = bin_to_pixels(x, inten, ccd)
    assert flux.sum() > 0
    # essentially all flux lands within a handful of pixels around the centroid
    frac_in_top10 = np.sort(flux)[-10:].sum() / flux.sum()
    assert frac_in_top10 > 0.99


# ---- combined forward-model entry (docs/research/instrumentation/grazing-grating.md phased-plan step 4) --


def test_detected_image_matches_manual_disperse_then_bin():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(86.0), order=1, coating="Au")
    ccd = SimpleCCD.from_alexs("2k512")
    E = np.arange(500.0, 1200.0, 1.0)
    spec = np.exp(-0.5 * ((E - 850.0) / 6.0) ** 2)  # a line at 850 eV
    centers, counts = detected_image(E, spec, g, ccd, distance_mm=200.0)
    # same contract as bin_to_pixels: fixed pixel grid, one value per pixel
    assert centers.shape == (ccd.n_pix,)
    assert counts.shape == (ccd.n_pix,)
    assert counts.sum() > 0
    # matches manually chaining disperse_spectrum (throughput-weighted, since
    # detected_image defaults weight_by_throughput=True) then bin_to_pixels
    x, inten = disperse_spectrum(E, spec, g, distance_mm=200.0, weight_by_throughput=True)
    centers_manual, counts_manual = bin_to_pixels(x, inten, ccd)
    assert np.allclose(centers, centers_manual)
    assert np.allclose(counts, counts_manual)


def test_detected_image_throughput_weighting_reduces_counts_vs_geometry_only():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(86.0), order=1, coating="Au")
    ccd = SimpleCCD(n_pix=2000, pixel_mm=0.05)  # wide enough to contain the line
    E = np.arange(500.0, 1200.0, 1.0)
    spec = np.exp(-0.5 * ((E - 850.0) / 6.0) ** 2)
    _, counts_default = detected_image(E, spec, g, ccd, distance_mm=200.0)
    _, counts_geom = detected_image(E, spec, g, ccd, distance_mm=200.0, weight_by_throughput=False)
    # default (throughput-weighted) total is strictly less than geometry-only,
    # since R <= 1 everywhere (mirrors test_throughput_wiring_reduces_flux_vs_raw_geometry)
    assert 0.0 < counts_default.sum() < counts_geom.sum()


def test_detected_image_matches_geometry_only_when_throughput_disabled():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(86.0), order=1)
    ccd = SimpleCCD(n_pix=2000, pixel_mm=0.05)
    E = np.arange(500.0, 1200.0, 1.0)
    spec = np.exp(-0.5 * ((E - 850.0) / 6.0) ** 2)
    _, counts = detected_image(E, spec, g, ccd, distance_mm=200.0, weight_by_throughput=False)
    x, inten = disperse_spectrum(E, spec, g, distance_mm=200.0, weight_by_throughput=False)
    _, counts_manual = bin_to_pixels(x, inten, ccd)
    assert np.allclose(counts, counts_manual)


# ---- physical CCD: QE, charge sharing, energy resolution (phased-plan step 5) --


def test_qe_absorption_bounded_and_thickness_limits():
    E = np.array([200.0, 900.0, 4000.0])
    # t/L_abs ~ 1e-5 - 1e-7 here (see absorption lengths of ~0.04-9.7 um over
    # this band) -- deep enough into the optically-thin regime that the
    # leading-order linear approximation holds to << 1e-3 relative.
    t_thin_um = 1.0e-4
    qe_thin = qe_absorption(E, active_um=t_thin_um, peak=1.0)  # t << L_abs everywhere
    qe_thick = qe_absorption(E, active_um=1.0e4, peak=1.0)  # t >> L_abs everywhere
    assert np.all(qe_thin >= 0.0) and np.all(qe_thin <= 1.0)
    assert np.all(qe_thick >= 0.0) and np.all(qe_thick <= 1.0)
    # optically-thin limit: QE -> t/L_abs (linear regime)
    L_ang = absorption_length_ang("Si", E, SI_N_PER_ANG3)
    expected_thin = (t_thin_um * 1.0e4) / L_ang
    assert qe_thin == pytest.approx(expected_thin, rel=1e-3)
    # optically-thick limit: QE -> peak (full capture)
    assert qe_thick == pytest.approx(1.0, abs=1e-6)
    # peak scales the whole curve
    assert qe_absorption(900.0, active_um=1.0e4, peak=0.5) == pytest.approx(0.5, abs=1e-6)


def test_qe_absorption_increases_with_thickness():
    E = 900.0
    thin = float(qe_absorption(E, active_um=1.0, peak=1.0))
    thick = float(qe_absorption(E, active_um=100.0, peak=1.0))
    assert 0.0 < thin < thick <= 1.0


def test_charge_cloud_sigma_soft_photon_is_maximum_blur():
    # L_abs(10 eV, Si) ~ 0.04 um: with a (deliberately oversized, for this
    # limit test) t=1000 um active layer, L_abs/t ~ 4e-5 -- deep enough into
    # the "absorbed right at the back surface" regime that drift ~ t almost
    # exactly, so sigma -> t*sqrt(2 kT/(q v_dep)) (see derivation) to << 1e-3.
    t_um, v_dep, temp_c = 1000.0, 40.0, -60.0
    sigma_soft = float(charge_cloud_sigma_um(10.0, active_um=t_um, v_dep=v_dep, temp_c=temp_c))
    k_b_ev_per_k = 8.617333262e-5
    kT_over_q = k_b_ev_per_k * (temp_c + 273.15)
    expected_max = t_um * np.sqrt(2.0 * kT_over_q / v_dep)
    assert sigma_soft == pytest.approx(expected_max, rel=1e-3)


def test_charge_cloud_sigma_zero_when_absorbed_at_front():
    # L_abs(E) >= t: absorbed right at the front electrodes -> zero drift -> sigma -> 0.
    t_um = 1e-6  # vanishingly thin sensor: L_abs(E) >= t for any real E
    sigma = float(charge_cloud_sigma_um(900.0, active_um=t_um, v_dep=40.0, temp_c=-60.0))
    assert sigma == pytest.approx(0.0, abs=1e-9)


def test_charge_cloud_sigma_positive_and_bounded_by_soft_limit():
    t_um, v_dep, temp_c = 30.0, 40.0, -60.0
    k_b_ev_per_k = 8.617333262e-5
    kT_over_q = k_b_ev_per_k * (temp_c + 273.15)
    max_sigma = t_um * np.sqrt(2.0 * kT_over_q / v_dep)
    for E in [50.0, 300.0, 900.0, 2000.0, 4000.0]:
        sigma = float(charge_cloud_sigma_um(E, active_um=t_um, v_dep=v_dep, temp_c=temp_c))
        assert 0.0 <= sigma <= max_sigma + 1e-9


def test_energy_fwhm_ev_positive_and_grows_with_energy():
    fwhm_low = float(energy_fwhm_eV(200.0))
    fwhm_high = float(energy_fwhm_eV(2000.0))
    assert fwhm_low > 0.0 and fwhm_high > fwhm_low


def test_detected_image_physical_matches_manual_qe_dispersion_bin():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(86.0), order=1, coating="Au")
    ccd = SimpleCCD.from_alexs("2k512")
    E = np.arange(500.0, 1200.0, 1.0)
    spec = np.exp(-0.5 * ((E - 850.0) / 6.0) ** 2)  # a line at 850 eV
    centers, counts = detected_image_physical(E, spec, g, ccd, distance_mm=200.0)
    assert centers.shape == (ccd.n_pix,)
    assert counts.shape == (ccd.n_pix,)
    assert counts.sum() > 0
    # manual QE weighting before the existing (already-tested) dispersion+binning chain
    spec_qe = spec * qe_absorption(E)
    x, inten = disperse_spectrum(E, spec_qe, g, distance_mm=200.0, weight_by_throughput=True)
    centers_manual, counts_manual_unblurred = bin_to_pixels(x, inten, ccd)
    assert np.allclose(centers, centers_manual)
    # the charge-cloud blur conserves total flux (up to edge losses,
    # negligible for a line well inside the sensor). Under these DEFAULT
    # device constants (deep-depletion, cold, high-field -- by design a small
    # charge cloud) the blur sigma is sub-pixel at 850 eV, so the blurred and
    # unblurred images can legitimately coincide; see
    # test_detected_image_physical_blur_redistributes_flux_across_pixels for a
    # case with parameters chosen to make the blur pixel-visible.
    assert counts.sum() == pytest.approx(counts_manual_unblurred.sum(), rel=1e-3)


def test_detected_image_physical_reduces_counts_vs_geometry_only():
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(86.0), order=1, coating="Au")
    ccd = SimpleCCD(n_pix=2000, pixel_mm=0.05)
    E = np.arange(500.0, 1200.0, 1.0)
    spec = np.exp(-0.5 * ((E - 850.0) / 6.0) ** 2)
    _, counts_physical = detected_image_physical(E, spec, g, ccd, distance_mm=200.0)
    _, counts_geom = detected_image(E, spec, g, ccd, distance_mm=200.0, weight_by_throughput=False)
    # QE <= 1 and throughput <= 1 both reduce flux relative to raw geometry
    assert 0.0 < counts_physical.sum() < counts_geom.sum()


def test_charge_cloud_sigma_decreases_with_energy_over_Si_absorption_band():
    # sigma(E) tracks drift = t - min(L_abs(E), t): softer photons absorb
    # closer to the entrance surface (small L_abs) and so drift further, all
    # else equal -- the well-known "back-illuminated CCDs blur soft X-rays
    # more" result. 200 eV -> 4 keV both have L_abs << t=30 um (drift ~ t,
    # near the max-blur plateau); 8 keV has L_abs=68.7 um > t, so drift -> 0.
    t_um, v_dep, temp_c = 30.0, 40.0, -60.0
    sigma_200 = float(charge_cloud_sigma_um(200.0, active_um=t_um, v_dep=v_dep, temp_c=temp_c))
    sigma_4000 = float(charge_cloud_sigma_um(4000.0, active_um=t_um, v_dep=v_dep, temp_c=temp_c))
    sigma_8000 = float(charge_cloud_sigma_um(8000.0, active_um=t_um, v_dep=v_dep, temp_c=temp_c))
    assert sigma_200 > sigma_4000 > sigma_8000 == pytest.approx(0.0, abs=1e-9)


def test_detected_image_physical_blur_redistributes_flux_across_pixels():
    # Deliberately EXAGGERATED (non-physical) v_dep/active_um -- just to make
    # the charge-cloud sigma several pixels wide so the convolution mechanism
    # itself is pixel-visible; the default device constants give a sub-pixel
    # cloud by design (see test_detected_image_physical_matches_manual_qe_dispersion_bin).
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(86.0), order=1, coating="Au")
    ccd = SimpleCCD(n_pix=4000, pixel_mm=0.01)
    E = np.arange(500.0, 1200.0, 1.0)
    spec = np.exp(-0.5 * ((E - 850.0) / 0.5) ** 2)  # a very narrow input line
    kwargs = dict(active_um=100.0, v_dep=0.5, temp_c=20.0)
    centers, counts = detected_image_physical(E, spec, g, ccd, distance_mm=200.0, **kwargs)
    x, inten = disperse_spectrum(
        E, spec * qe_absorption(E, active_um=100.0), g, distance_mm=200.0, weight_by_throughput=True
    )
    _, counts_unblurred = bin_to_pixels(x, inten, ccd)
    assert counts.sum() > 0
    # total flux conserved (this line sits well inside the wide sensor, so no
    # edge losses either from binning or from the blur's zero-padded edges)
    assert counts.sum() == pytest.approx(counts_unblurred.sum(), rel=1e-2)
    # the blur visibly redistributes flux: pixels that were exactly zero
    # before blurring now carry some counts near the peak
    zero_before = counts_unblurred == 0.0
    assert np.any(counts[zero_before] > 0.0)


# ---- flat-field grating: efficiency curve, focal plane, catalog (#1081) -------
# Every grating, mount and curve below is synthetic. Real vendor data stays in
# the user's private catalog and is checked only by the skipped test at the end.
_EFFICIENCY = TabulatedEfficiency((100.0, 200.0, 300.0), (0.10, 0.20, 0.05), name="synthetic")


def test_throughput_unchanged_without_efficiency_curve():
    E = np.array([300.0, 600.0, 900.0])
    g = Grating(groove_density_per_mm=1200.0, alpha_rad=np.deg2rad(88.0), groove_efficiency=0.3)
    assert g.throughput(E) == pytest.approx(g.reflectivity(E) * 0.3)


def test_efficiency_curve_is_applied_once_without_fresnel():
    g = Grating(1200.0, np.deg2rad(88.0), efficiency_curve=_EFFICIENCY)
    E = np.array([100.0, 150.0, 300.0])
    assert g.throughput(E) == pytest.approx([0.10, 0.15, 0.05])
    # Detected flux scales by eta exactly once: no reflectivity factor.
    E_grid = np.linspace(120.0, 280.0, 801)
    spec = np.ones_like(E_grid)
    x, inten = disperse_spectrum(E_grid, spec, g, 500.0, weight_by_throughput=True)
    assert abs(np.trapezoid(inten, x)) == pytest.approx(
        np.trapezoid(_EFFICIENCY(E_grid), E_grid), rel=1e-9
    )


def test_efficiency_curve_is_zero_outside_its_band():
    g = Grating(1200.0, np.deg2rad(88.0), efficiency_curve=_EFFICIENCY)
    assert g.throughput(np.array([50.0, 99.9, 300.1, 1000.0])) == pytest.approx(np.zeros(4))
    assert not _EFFICIENCY.covers(50.0) and _EFFICIENCY.covers(200.0)


def test_efficiency_curve_refuses_a_second_groove_efficiency():
    with pytest.raises(ValueError, match="already includes"):
        Grating(1200.0, 1.5, groove_efficiency=0.5, efficiency_curve=_EFFICIENCY)


def test_flat_field_position_is_the_plane_intersection():
    mount = FlatFieldMount(entrance_arm_mm=300.0, focal_distance_mm=280.0, focal_normal_rad=0.3)
    beta = np.array([0.1, 0.3, 0.5])
    s = mount.position_mm(beta)
    rays = mount.exit_arm_mm(beta)[:, None] * np.column_stack([np.sin(beta), np.cos(beta)])
    normal = np.array([np.sin(0.3), np.cos(0.3)])
    tangent = np.array([np.cos(0.3), -np.sin(0.3)])
    assert rays @ normal == pytest.approx(np.full(3, 280.0))  # on the plane
    assert rays @ tangent == pytest.approx(s)  # offset from the foot
    assert s[1] == pytest.approx(0.0)  # beta = beta_H lands at the foot


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_flat_field_reduces_to_cot_beta_for_a_plane_normal_to_the_grating(sign):
    # tan has period pi, so beta_H -> +90 and -90 deg both give -r_H' cot(beta).
    # Rays reach the plane on the same side as its normal.
    mount = FlatFieldMount(300.0, 280.0, sign * np.deg2rad(89.999999))
    beta = sign * np.deg2rad(np.array([80.0, 84.0]))
    assert mount.position_mm(beta) == pytest.approx(-280.0 / np.tan(beta), rel=1e-5, abs=1e-4)


def test_flat_field_rays_that_miss_the_plane_are_nan():
    mount = FlatFieldMount(300.0, 280.0, 0.2)
    assert np.isnan(mount.position_mm(0.2 + np.pi / 2 + 0.1))
    assert np.isnan(mount.exit_arm_mm(0.2 - np.pi / 2 - 0.1))
    assert np.isfinite(mount.position_mm(0.5))


# Synthetic mount (arbitrary values): 1000/mm, m = +1, alpha = 86 deg, band 6-24 nm. Its tabulated
# betas and focal-plane length are this module's equations rounded to the
# precision a vendor table prints, so the check is a consistency check.
_SYNTH_GRATING = """\
label = "Synthetic flat-field grating"
groove_density_per_mm = 1000
order = 1
order_convention = "{convention}"
coating = "Au"
efficiency_curve = "synthetic-eta"

[mount]
r_mm = 300.0
alpha_deg = 86.0
band_nm = [6.0, 24.0]
r_prime_mm = [280.67, 284.06]
beta_deg = [{b1}, {b2}]
beta_H_deg = {bh}
r_H_prime_mm = 280.0
focal_plane_length_mm = 28.56
"""
_SYNTH_ETA = """\
source = "synthetic absolute efficiency"
digitization = "analytic toy"
confidential = false
quantity = "efficiency"
abscissa = "wavelength"
abscissa_unit = "nm"
x = [4.0, 12.0, 21.0]
y = [0.05, 0.12, 0.08]
"""


def _write_synthetic_grating(root, convention):
    sign = 1.0 if convention == "sum" else -1.0
    (root / "gratings").mkdir(exist_ok=True)
    (root / "curves").mkdir(exist_ok=True)
    (root / "curves" / "synthetic-eta.toml").write_text(_SYNTH_ETA)
    text = _SYNTH_GRATING.format(
        convention=convention, b1=sign * -82.5525, b2=sign * -76.7963, bh=sign * -86.5
    )
    (root / "gratings" / "synthetic-ff.toml").write_text(text)
    load_material_catalog.cache_clear()


@pytest.mark.parametrize("convention", ["sum", "difference"])
def test_from_catalog_builds_a_flat_field_grating(_isolate_user_catalog, convention):
    _write_synthetic_grating(_isolate_user_catalog, convention)
    try:
        g = Grating.from_catalog("synthetic-ff")
    finally:
        load_material_catalog.cache_clear()
    assert g.order == 1 and g.alpha_rad == pytest.approx(np.deg2rad(86.0))
    assert g.mount is not None
    assert g.mount.focal_normal_rad == pytest.approx(np.deg2rad(-86.5))
    assert g.mount.entrance_arm_mm == 300.0
    assert g.efficiency_curve is not None and g.efficiency_curve.name == "synthetic-eta"
    # 12 nm -> its curve value, no reflectivity factor.
    assert g.throughput(HC_EV_ANG / 120.0) == pytest.approx(0.12)
    residuals = g.mount_residuals()
    assert max(abs(e) for e in residuals["beta_error_deg"]) < 0.05
    assert abs(residuals["focal_plane_length_rel_error"]) < 0.01
    assert max(abs(e) for e in residuals["exit_arm_rel_error"]) < 0.01


def test_from_catalog_without_focal_plane_has_no_mount(_isolate_user_catalog):
    _write_synthetic_grating(_isolate_user_catalog, "sum")
    path = _isolate_user_catalog / "gratings" / "synthetic-ff.toml"
    path.write_text(path.read_text().replace("beta_H_deg = -86.5\n", ""))
    load_material_catalog.cache_clear()
    try:
        assert Grating.from_catalog("synthetic-ff").mount is None
    finally:
        load_material_catalog.cache_clear()


@pytest.mark.skipif(
    not (
        os.environ.get("PYRITE_INTERNAL_USER_CATALOG") and os.environ.get("PYRITE_INTERNAL_GRATING")
    ),
    reason="needs the private instrument catalog (PYRITE_INTERNAL_USER_CATALOG, "
    "PYRITE_INTERNAL_GRATING)",
)
def test_internal_grating_mount_matches_its_vendor_table(monkeypatch):
    """Same consistency check against the real grating; numbers never enter the repo."""
    monkeypatch.setenv("PYRITE_USER_CATALOG", os.environ["PYRITE_INTERNAL_USER_CATALOG"])
    load_material_catalog.cache_clear()
    try:
        g = Grating.from_catalog(os.environ["PYRITE_INTERNAL_GRATING"])
    finally:
        load_material_catalog.cache_clear()
    residuals = g.mount_residuals()
    assert max(abs(e) for e in residuals["beta_error_deg"]) < 0.05
    assert abs(residuals["focal_plane_length_rel_error"]) < 0.01
