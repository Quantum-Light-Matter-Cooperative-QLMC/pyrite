"""Machine-readable export units leave table schemas unchanged (#755)."""

import json

import pytest

from pyrite import CxrSpectra, Result, SpectralDensity
from pyrite.results import FLUX_UNITS
from pyrite.units import ureg


def _result():
    return Result(
        cxr=CxrSpectra(SpectralDensity([1, 2], [3, 4]), SpectralDensity([1, 2], [7, 8])),
        brem=SpectralDensity([1, 2, 3], [5, 6, 7]),
        characteristic=SpectralDensity([1, 2], [0.25, 0.5]),
        case={},
        provenance={},
    )


def test_dataframe_units_cover_physical_columns_and_parse() -> None:
    result = _result()
    frame = result.to_dataframe()
    units = result.dataframe_units()
    assert set(units) <= set(frame.columns)
    assert set(frame.columns) - set(units) == {"component", "mode", "normalization"}
    assert ureg.Unit(units["density"]["unit"]) == 1 / ureg.eV / ureg.steradian
    assert units["density"]["basis"] == "photons_per_incident_electron"
    assert json.loads(json.dumps(units)) == units
    units["density"]["unit"] = "m"
    assert result.dataframe_units()["density"]["unit"] == "1 / eV / sr"


@pytest.mark.parametrize("choice", sorted(FLUX_UNITS))
def test_spectral_flux_units_match_columns(choice) -> None:
    result = _result()
    if choice in {"detector", "sr"}:
        result.case["bunch_charge_pc"] = 1.0
        result.case["rep_rate_hz"] = 1000.0
    if choice in {"detector", "nA"}:
        result.case["domega_sr"] = 1e-3
    frame = result.spectral_flux(units=choice)
    units = result.spectral_flux_units(units=choice)
    assert list(units) == list(frame.columns)
    family = units["total"]
    ureg.Unit(family["unit"])
    assert "electron" not in family["unit"]
    assert (family["basis"] == "photons_per_incident_electron") == (choice == "electron")
    assert units["photon_energy_keV"] == {"unit": "keV", "basis": None}


def test_spectral_flux_units_refuse_unknown_choice() -> None:
    with pytest.raises(ValueError, match="units must be one of"):
        _result().spectral_flux_units(units="W")
