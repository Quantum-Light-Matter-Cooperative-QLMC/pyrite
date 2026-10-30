"""Detector/grating wrappers preserve physical densities and pixel integrals (#816)."""

import numpy as np
import pytest

from pyrite.detectors import quantities as q
from pyrite.detectors import response
from pyrite.detectors.grating import SimpleCCD, detected_image
from pyrite.units import Q_, ureg


def test_convolution_keeps_density_and_outside_integral_units():
    energy = Q_(np.linspace(100, 200, 11), "eV")
    values = Q_(np.ones(11), "1/eV")
    output, (below, above) = q.convolve_detector_q(
        energy, values, Q_(10, "eV"), return_outside=True
    )
    np.testing.assert_allclose(
        output.magnitude, response.convolve_detector(energy.magnitude, values.magnitude, 10)
    )
    assert output.units == ureg.Unit("1/eV")
    assert below.dimensionality == above.dimensionality == ureg.dimensionless.dimensionality
    assert below.magnitude >= 0 and above.magnitude >= 0
    output.magnitude[0] = 10**6
    assert q.convolve_detector_q(energy, values, Q_(10, "eV")).magnitude[0] < 10**6
    with pytest.raises(TypeError):
        q.convolve_detector_q(energy.magnitude, values, Q_(10, "eV"))


def test_grating_geometry_and_pixel_integrals():
    grating = q.grating_q(groove_density=Q_(1200, "1/mm"), incidence_angle=Q_(87, "degree"))
    params = q.grating_parameters(grating)
    assert params["incidence_angle"].to("degree").magnitude == pytest.approx(87)
    assert params["groove_spacing"].to("nm").magnitude == pytest.approx(grating.d_angstrom / 10)
    params["groove_spacing"].ito("nm")
    assert grating.d_angstrom == pytest.approx(1e7 / 1200)
    ccd = q.ccd_q(n_pix=16, pixel_pitch=Q_(0.1, "mm"))
    assert isinstance(ccd, SimpleCCD)
    assert q.ccd_parameters(ccd)["width"].magnitude == pytest.approx(1.6)
    x = Q_(np.linspace(-1.5, 1.5, 101), "mm")
    intensity = Q_(np.ones(101), "1/mm")
    centers, integral = q.bin_to_pixels_q(x, intensity, ccd, center=Q_(0, "mm"))
    assert centers.shape == integral.shape == (16,)
    assert integral.units == ureg.dimensionless
    assert integral.magnitude.sum() == pytest.approx(1.6)
    with pytest.raises(ValueError):
        q.grating_q(groove_density=Q_(1200, "1/mm"), incidence_angle=Q_(87, ""))


def test_dispersion_unit_jacobian_and_refractive_helpers():
    grating = q.grating_q(groove_density=Q_(1200, "1/mm"), incidence_angle=Q_(87, "degree"))
    energy = Q_(np.linspace(150, 300, 100), "eV")
    source = Q_(np.ones(100), "1/eV")
    x, intensity = q.disperse_spectrum_q(energy, source, grating, Q_(200, "mm"))
    assert x.units == ureg.mm
    assert intensity.units == ureg.Unit("1/mm")
    assert x.shape == intensity.shape
    assert q.wavelength_q(Q_(200, "eV")).to("nm").magnitude > 0
    assert q.groove_spacing_q(Q_(1200, "1/mm")).magnitude == pytest.approx(grating.d_angstrom)
    assert q.diffraction_angle_q(grating, energy).units == ureg.radian
    assert q.angular_dispersion_q(grating, energy).units == ureg.radian / ureg.angstrom
    assert q.resolving_power_q(energy, grating, Q_(200, "mm"), Q_(0.1, "mm")).dimensionless
    np.testing.assert_allclose(
        q.grating_reflectivity_q(grating, energy).magnitude,
        grating.reflectivity(energy.magnitude),
        equal_nan=True,
    )


def test_detected_image_q_preserves_integral_units():
    grating = q.grating_q(groove_density=Q_(1200, "1/mm"), incidence_angle=Q_(87, "degree"))
    ccd = SimpleCCD(n_pix=256, pixel_mm=0.02)
    energies = Q_(np.linspace(150, 300, 101), "eV")
    spectrum = Q_(np.ones(101), "1/eV")
    pixels, counts = q.detected_image_q(
        energies, spectrum, grating, ccd, Q_(200, "mm"), weight_by_throughput=False
    )
    x_ref, counts_ref = detected_image(
        energies.magnitude, spectrum.magnitude, grating, ccd, 200, weight_by_throughput=False
    )
    np.testing.assert_allclose(pixels.magnitude, x_ref)
    np.testing.assert_allclose(counts.magnitude, counts_ref)
    assert counts.units == ureg.dimensionless
    with pytest.raises(ValueError):
        q.detected_image_q(
            energies, spectrum, grating, ccd, Q_(200, "mm"), active_thickness=Q_(30, "um")
        )
