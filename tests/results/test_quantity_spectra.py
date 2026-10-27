"""#755: coupled energy/density conversion and legacy result invariants."""

from dataclasses import asdict

import numpy as np
import pytest

from pyrite.results import CxrSpectra, QuantifiedSpectrum, Result, SpectralDensity
from pyrite.units import ureg


def _integrated(spectrum: QuantifiedSpectrum) -> float:
    """The unit of this integral is per sr and incident electron."""
    return float(np.trapezoid(spectrum.density.magnitude, spectrum.photon_energy.magnitude))


def test_quantify_and_to_energy_convert_both_axes_on_nonuniform_grid() -> None:
    component = SpectralDensity([200, 400, 1000, 2000], [1.0, 3.0, 2.0, 4.0])
    ev = component.quantify()
    kev = ev.to_energy("keV")
    assert isinstance(kev, QuantifiedSpectrum)
    assert ev.photon_energy.units == ureg.eV
    assert kev.photon_energy.units == ureg.keV
    assert kev.density.units == ureg("1 / keV / steradian").units
    np.testing.assert_allclose(kev.photon_energy.magnitude, [0.2, 0.4, 1.0, 2.0])
    np.testing.assert_allclose(kev.density.magnitude, [1000, 3000, 2000, 4000])
    np.testing.assert_allclose(kev.density.to("1/eV/steradian").magnitude, component.density)
    assert _integrated(kev) == pytest.approx(_integrated(ev))
    assert _integrated(kev.to_energy("MeV")) == pytest.approx(_integrated(ev))
    assert ev.measure == kev.measure == "per_energy_per_steradian"
    assert kev.sampling == "nodes"
    assert kev.count_basis == "photons_per_incident_electron"
    assert kev.normalization == "source"


def test_mutating_returned_pint_quantities_cannot_break_spectrum_invariant() -> None:
    component = SpectralDensity([100.0, 300.0], [2.0, 4.0])
    original = asdict(component)
    quantified = component.quantify()
    converted = quantified.to_energy("keV")
    returned_energy = converted.photon_energy
    returned_density = converted.density
    returned_energy.magnitude[0] = -999
    returned_density.magnitude[0] = -999
    np.testing.assert_array_equal(converted.photon_energy.magnitude, [0.1, 0.3])
    np.testing.assert_array_equal(converted.density.magnitude, [2000, 4000])
    np.testing.assert_array_equal(quantified.photon_energy.magnitude, [100, 300])
    np.testing.assert_array_equal(component.photon_energy_eV, original["photon_energy_eV"])
    np.testing.assert_array_equal(component.density, original["density"])


def test_cxr_and_brem_keep_their_native_grids_and_legacy_flux_is_unchanged() -> None:
    cxr = SpectralDensity([100.0, 300.0], [2.0, 4.0])
    brem = SpectralDensity([100.0, 200.0, 400.0], [1.0, 1.0, 1.0])
    result = Result(
        cxr=CxrSpectra(cxr),
        brem=brem,
        case={"domega_sr": 0.01, "bunch_charge_pc": 2.0, "rep_rate_hz": 1.0e4},
        provenance={},
    )
    before = result.spectral_flux(units="electron")
    c = cxr.quantify().to_energy("keV")
    b = brem.quantify().to_energy("keV")
    assert c.photon_energy.magnitude.shape == (2,)
    assert b.photon_energy.magnitude.shape == (3,)
    assert _integrated(c) == pytest.approx(np.trapezoid(cxr.density, cxr.photon_energy_eV))
    assert _integrated(b) == pytest.approx(np.trapezoid(brem.density, brem.photon_energy_eV))
    assert before.equals(result.spectral_flux(units="electron"))
    assert "photon_energy_keV" in before.columns  # Density remains per eV.


def test_observation_average_and_nan_support_are_preserved() -> None:
    spectrum = SpectralDensity([10.0, 20.0], [np.nan, 2.0], normalization="observation-average")
    view = spectrum.quantify().to_energy("keV")
    assert view.normalization == "observation-average"
    assert np.isnan(view.density.magnitude[0])
    assert view.density.magnitude[1] == pytest.approx(2000.0)
    assert np.isnan(spectrum.density[0])
    assert spectrum.density[1] == 2.0


@pytest.mark.parametrize("unit", ["s", "mm", "dimensionless", "radian", "Hz"])
def test_to_energy_rejects_nonenergy_dimensions(unit: str) -> None:
    spectrum = SpectralDensity([1.0, 2.0], [3.0, 4.0]).quantify()
    with pytest.raises(ValueError, match="energy_unit.*energy"):
        spectrum.to_energy(unit)


@pytest.mark.parametrize("unit", [None, 1, ""])
def test_to_energy_requires_a_unit_string(unit: object) -> None:
    spectrum = SpectralDensity([1.0, 2.0], [3.0, 4.0]).quantify()
    with pytest.raises(TypeError, match="energy_unit"):
        spectrum.to_energy(unit)  # type: ignore[arg-type]
