"""Machine-readable export units leave table schemas unchanged (#755)."""

import json

from pyrite import CxrSpectra, Result, SpectralDensity
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
