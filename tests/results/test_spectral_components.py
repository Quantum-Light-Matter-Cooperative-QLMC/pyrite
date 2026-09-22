import numpy as np
import pytest

from pyrite._spectral_components import incident_spectrum, line_spectrum, separate_legacy
from pyrite.results.model import Result


def _record():
    return {
        "E_grid": np.array([1.0, 2.0]),
        "spec": np.array([1.0, 2.0]),
        "spec_coherent": np.array([3.0, 4.0]),
        "spec_characteristic": np.array([0.5, 0.25]),
        "brem": np.array([10.0, 20.0]),
    }


def test_line_spectrum_adds_characteristic_on_request():
    record = _record()

    np.testing.assert_array_equal(line_spectrum(record), [1.5, 2.25])
    np.testing.assert_array_equal(line_spectrum(record, characteristic=False), [1.0, 2.0])
    np.testing.assert_array_equal(line_spectrum(record, coherent=True), [3.5, 4.25])
    np.testing.assert_array_equal(incident_spectrum(record), [11.5, 22.25])


def test_line_spectrum_without_characteristic_component_is_the_line():
    record = _record()
    del record["spec_characteristic"]

    np.testing.assert_array_equal(line_spectrum(record), record["spec"])


def test_separate_legacy_subtracts_once_from_nested_store_totals():
    legacy = _record()
    legacy["spec"] = legacy["spec"] + legacy["spec_characteristic"]
    legacy["spec_coherent"] = legacy["spec_coherent"] + legacy["spec_characteristic"]
    store = {"cfg": {30.0: legacy}, "bare": {30.0: {"E_grid": np.array([1.0])}}}

    separate_legacy(store)

    np.testing.assert_array_equal(store["cfg"][30.0]["spec"], [1.0, 2.0])
    np.testing.assert_array_equal(store["cfg"][30.0]["spec_coherent"], [3.0, 4.0])
    np.testing.assert_array_equal(store["cfg"][30.0]["spec_characteristic"], [0.5, 0.25])


def test_result_line_total_combines_components():
    record = _record()
    result = Result(
        energy_eV=record["E_grid"],
        spectrum=record["spec"],
        background_energy_eV=record["E_grid"],
        background=record["brem"],
        case={},
        provenance={},
        characteristic_spectrum=record["spec_characteristic"],
    )

    np.testing.assert_array_equal(result.spectrum, [1.0, 2.0])
    np.testing.assert_array_equal(result.line_total(), [1.5, 2.25])
    np.testing.assert_array_equal(result.line_total(characteristic=False), [1.0, 2.0])
    with pytest.raises(ValueError, match="coherent spectrum is not available"):
        result.line_total(coherent=True)
