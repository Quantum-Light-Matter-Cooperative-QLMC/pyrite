"""Slit-image kernel, 2D focal-plane image and ALEX-s spectrometer frame (#1081).

All mounts, curves and spreads are synthetic.
"""

import numpy as np
import pytest

from pyrite.detectors import AlexS, TabulatedEfficiency
from pyrite.detectors.alexs_response import AlexsFormat, alexs_frame
from pyrite.detectors.grating import (
    FlatFieldMount,
    Grating,
    SimpleCCD,
    bin_to_pixels,
    detected_image,
    disperse_spectrum,
)
from pyrite.detectors.grating_image import (
    AlongSlitSpread,
    SlitImage,
    alexs_spectrometer_image,
    detected_image_2d,
    slit_image_pixels,
    slit_image_width_mm,
)

MOUNT = FlatFieldMount(entrance_arm_mm=300.0, focal_distance_mm=280.0, focal_normal_rad=-1.5097)
GRATING = Grating(1000.0, np.deg2rad(86.0), order=1, mount=MOUNT)
FMT = AlexsFormat.named("2k512")
CCD = SimpleCCD(n_pix=FMT.columns, pixel_mm=FMT.pitch_mm)
E = np.linspace(60.0, 250.0, 6001)


def _lines():
    return sum(a * np.exp(-0.5 * ((E - e0) / 0.3) ** 2) for a, e0 in ((1.0, 100.0), (0.5, 150.0)))


def test_slit_image_width_matches_anamorphic_formula():
    w, E0 = 0.02, 150.0
    beta = float(GRATING.diffraction_angle_rad(E0))
    r_prime = float(MOUNT.exit_arm_mm(beta))
    anamorphic = w * np.cos(GRATING.alpha_rad) / np.cos(beta) * r_prime / 300.0
    tilt = np.cos(beta - MOUNT.focal_normal_rad)
    assert slit_image_width_mm(E0, GRATING, w) == pytest.approx(anamorphic / tilt, rel=1e-12)
    # At normal incidence on the plane (beta_H = beta, r' = r_H') it is w' itself.
    normal = Grating(1000.0, np.deg2rad(86.0), mount=FlatFieldMount(300.0, 280.0, beta))
    w_prime = w * np.cos(GRATING.alpha_rad) / np.cos(beta) * 280.0 / 300.0
    assert slit_image_width_mm(E0, normal, w) == pytest.approx(w_prime, rel=1e-12)
    assert slit_image_width_mm(E0, GRATING, 0.0) == 0.0


def test_slit_image_width_explicit_distance_map():
    beta = float(GRATING.diffraction_angle_rad(150.0))
    width = slit_image_width_mm(150.0, GRATING, 0.01, entrance_arm_mm=500.0, distance_mm=300.0)
    expected = 0.01 * np.cos(GRATING.alpha_rad) / np.cos(beta) * 300.0 / 500.0
    assert width == pytest.approx(expected, rel=1e-12)  # beta_ref = beta: normal incidence


def test_zero_width_slit_is_the_point_slit_dispersion():
    spec = _lines()
    x, f = slit_image_pixels(E, spec, GRATING, CCD, SlitImage(0.0))
    x0, f0 = detected_image(E, spec, GRATING, CCD)
    assert x == pytest.approx(x0) and f == pytest.approx(f0, rel=0, abs=0)
    xd, density = disperse_spectrum(E, spec, GRATING, weight_by_throughput=True)
    assert f == pytest.approx(bin_to_pixels(xd, density, CCD)[1])


def test_narrow_slit_converges_to_point_slit():
    spec = _lines()
    centre = float(np.mean(detected_image(E, spec, GRATING, CCD)[0]))
    _, f0 = slit_image_pixels(E, spec, GRATING, CCD, SlitImage(0.0), center_mm=centre)
    _, f1 = slit_image_pixels(E, spec, GRATING, CCD, SlitImage(1.0e-7), center_mm=centre)
    assert np.abs(f1 - f0).sum() < 1e-6 * f0.sum()


@pytest.mark.parametrize("slit", [SlitImage(0.05), SlitImage(0.05, resolving_power=300.0)])
def test_slit_convolution_conserves_flux_and_broadens(slit):
    spec = _lines()
    centre = float(np.mean(detected_image(E, spec, GRATING, CCD)[0]))
    x, f0 = slit_image_pixels(E, spec, GRATING, CCD, SlitImage(0.0), center_mm=centre)
    _, f = slit_image_pixels(E, spec, GRATING, CCD, slit, center_mm=centre)
    assert f.sum() == pytest.approx(f0.sum(), rel=1e-10)

    def rms(profile):
        half = (
            slice(None, profile.size // 2)
            if x[np.argmax(f0)] < np.mean(x)
            else slice(profile.size // 2, None)
        )
        p, xs = profile[half], x[half]
        mean = np.dot(p, xs) / p.sum()
        return np.sqrt(np.dot(p, (xs - mean) ** 2) / p.sum())

    assert rms(f) > rms(f0)


def test_rect_kernel_width_on_a_single_line():
    # A delta-like line through a wide slit becomes a rect of the image width.
    spec = np.exp(-0.5 * ((E - 150.0) / 0.01) ** 2)
    w = 0.5
    x, f = slit_image_pixels(E, spec, GRATING, CCD, SlitImage(w), weight_by_throughput=False)
    expected = float(slit_image_width_mm(150.0, GRATING, w))
    lit = x[f > 0.5 * f.max()]
    assert lit.max() - lit.min() + CCD.pixel_mm == pytest.approx(expected, abs=2 * CCD.pixel_mm)


def test_2d_image_row_sum_is_1d_times_capture():
    spec = _lines()
    slit = SlitImage(0.03)
    _, f1d = slit_image_pixels(E, spec, GRATING, CCD, slit)
    height = 20.0
    spread = AlongSlitSpread(height_mm=(height,))
    columns, rows, image = detected_image_2d(E, spec, GRATING, FMT, slit, along_slit=spread)
    assert image.shape == (515, 2048) and rows.size == 515 and columns.size == 2048
    capture = FMT.height_mm / height
    assert image.sum(axis=0) == pytest.approx(f1d * capture, rel=1e-12, abs=1e-15)
    # Spread wider than the sensor: uniform over rows.
    lit = image[:, np.argmax(image.sum(axis=0))]
    assert lit == pytest.approx(np.full(515, lit.mean()), rel=1e-12)


def test_tabulated_capture_and_small_height_profiles():
    spec = _lines()
    slit = SlitImage(0.0)
    _, f1d = slit_image_pixels(E, spec, GRATING, CCD, slit)
    tabulated = AlongSlitSpread(capture_fraction=(0.25,))
    image = detected_image_2d(E, spec, GRATING, FMT, slit, along_slit=tabulated)[2]
    assert image.sum(axis=0) == pytest.approx(0.25 * f1d, rel=1e-12, abs=1e-15)
    narrow = AlongSlitSpread(height_mm=(1.0,))
    image = detected_image_2d(E, spec, GRATING, FMT, slit, along_slit=narrow)[2]
    assert image.sum(axis=0) == pytest.approx(f1d, rel=1e-12, abs=1e-15)
    rows_lit = np.count_nonzero(image[:, np.argmax(f1d)])
    assert rows_lit == pytest.approx(1.0 / FMT.pitch_mm, abs=2)


def test_energy_dependent_spread_outside_table_is_not_delivered():
    spread = AlongSlitSpread(energy_eV=(80.0, 200.0), height_mm=(10.0, 30.0))
    capture = spread.capture(np.array([70.0, 80.0, 140.0, 200.0]), FMT.height_mm)
    assert capture == pytest.approx(
        [0.0, FMT.height_mm / 10.0, FMT.height_mm / 20.0, FMT.height_mm / 30.0]
    )
    with pytest.raises(ValueError, match="exactly one"):
        AlongSlitSpread(height_mm=(1.0,), capture_fraction=(0.5,))


def test_alexs_spectrometer_frame_is_qe_e_over_w_weighted_image():
    qe = TabulatedEfficiency((50.0, 300.0), (0.5, 0.5), name="synthetic-qe")
    response = AlexS(qe=qe, gain_mode="high-capacity")
    photons = 50.0 * _lines()
    slit = SlitImage(0.02)
    spread = AlongSlitSpread(height_mm=(10.0,))
    frame = alexs_spectrometer_image(
        E, photons, GRATING, response, slit, exposure_s=10.0, along_slit=spread
    )
    image = detected_image_2d(E, 0.5 * photons, GRATING, FMT, slit, along_slit=spread)[2]
    np.testing.assert_allclose(frame.expected_photons, image, rtol=1e-12, atol=0)
    total_e = np.trapezoid(0.5 * photons * GRATING.throughput(E) * E / 3.66, E)
    assert frame.expected_electrons.sum() == pytest.approx(total_e * FMT.height_mm / 10.0, rel=1e-3)
    expected = alexs_frame(frame.expected_electrons, exposure_s=10.0, gain_mode="high-capacity")
    np.testing.assert_allclose(frame.adu, expected, rtol=1e-12, atol=0)
    assert frame.row_summed() == pytest.approx(frame.adu.sum(axis=0))


def test_alexs_spectrometer_noise_is_reproducible_with_seed():
    response = AlexS()
    args = (E, 10.0 * _lines(), GRATING, response, SlitImage(0.02))
    kwargs = dict(exposure_s=1.0, shot_noise=True, read_noise=True)
    a = alexs_spectrometer_image(*args, rng=np.random.default_rng(5), **kwargs).adu
    b = alexs_spectrometer_image(*args, rng=np.random.default_rng(5), **kwargs).adu
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="sensor format"):
        alexs_spectrometer_image(
            *args[:3], AlexS(sensor_format=(1024, 256)), args[4], exposure_s=1.0
        )
