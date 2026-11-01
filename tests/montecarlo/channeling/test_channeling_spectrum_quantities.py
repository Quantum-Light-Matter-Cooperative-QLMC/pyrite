"""Channeling-spectrum quantity view keeps measure and integrals (#755)."""

import numpy as np
import pytest

from pyrite.montecarlo.channeling.emission import ChannelingSpectrum
from pyrite.montecarlo.channeling.quantified_emission import QuantifiedChannelingSpectrum
from pyrite.units import ureg


def _spectrum() -> ChannelingSpectrum:
    energy = np.linspace(1_000.0, 3_000.0, 5)
    theta = np.array([0.0, 1e-3])
    density = np.outer([1.0, 0.5], np.exp(-(((energy - 2_000.0) / 400.0) ** 2)))
    stokes = np.zeros(density.shape + (2,))
    return ChannelingSpectrum(
        photon_energy_eV=energy,
        theta_rad=theta,
        phi_rad=np.zeros(2),
        spectrum=density,
        capture_spectrum=0.1 * density,
        stokes=stokes,
        lines=None,
        z_ang=np.array([0.0, 50.0]),
        line_rates=np.array([[1e-3], [2e-3]]),
        bound_population=np.array([1.0, 0.9]),
        line_yield=np.array([0.05]),
    )


def test_view_units_measure_and_metadata() -> None:
    view = _spectrum().quantify()
    assert isinstance(view, QuantifiedChannelingSpectrum)
    assert view.measure == "per_energy_per_steradian"
    assert view.count_basis == "photons_per_incident_electron"
    assert view.sampling == "nodes"
    assert view.photon_energy.units == ureg.eV
    assert view.density.units == 1 / ureg.eV / ureg.steradian
    assert view.theta.units == ureg.radian
    assert view.depth.units == ureg.angstrom
    assert view.line_rates.units == 1 / ureg.angstrom
    assert view.line_yield.dimensionless
    assert view.bound_population.dimensionless
    assert view.stokes.shape == (2, 5, 2)


def test_energy_conversion_scales_density_inversely_and_preserves_integral() -> None:
    spectrum = _spectrum()
    view = spectrum.quantify()
    kev = view.to_energy("keV")
    np.testing.assert_allclose(kev.photon_energy.magnitude, spectrum.photon_energy_eV / 1e3)
    np.testing.assert_allclose(kev.density.magnitude, spectrum.spectrum * 1e3)
    assert kev.density.units == 1 / ureg.keV / ureg.steradian
    np.testing.assert_allclose(kev.capture_density.magnitude, spectrum.capture_spectrum * 1e3)
    integral_ev = np.trapezoid(view.density, view.photon_energy, axis=-1)
    integral_kev = np.trapezoid(kev.density, kev.photon_energy, axis=-1)
    np.testing.assert_allclose(integral_kev.to("1/sr").magnitude, integral_ev.to("1/sr").magnitude)
    with pytest.raises(ValueError, match="energy unit"):
        view.to_energy("mm")


def test_view_is_isolated_from_source_and_callers() -> None:
    spectrum = _spectrum()
    view = spectrum.quantify()
    spectrum.spectrum[...] = 0
    assert np.any(view.density.magnitude != 0)
    readback = view.density
    readback.magnitude[...] = -1
    assert np.all(view.density.magnitude >= 0)
