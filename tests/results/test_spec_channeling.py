"""``spec_channeling`` as a persisted additive line-grid channel (#507, ADR-0018 P5)."""

import numpy as np
import pytest

from pyrite._spectral_components import cxr_spectrum, incident_spectrum
from pyrite.checkpoints import _checkpoint_store
from pyrite.results.model import CxrSpectra, Result, SpectralDensity
from pyrite.results.selection import merge_dataset, project_dataset
from pyrite.results.spatial import PixelRayMap, SpatialResult, SpectralFactors
from pyrite.results.store import store_result


def _record(*, channeling=True):
    record = {
        "case": {"name": "cfg", "E0_keV": 5000.0},
        "E_grid": np.array([1.0, 2.0]),
        "spec": np.array([1.0, 2.0]),
        "spec_coherent": np.array([3.0, 4.0]),
        "spec_characteristic": np.array([0.5, 0.25]),
        "brem": np.array([10.0, 20.0]),
        "E_grid_brem": np.array([1.0, 2.0]),
        "brem_wide": np.array([10.0, 20.0]),
    }
    if channeling:
        record["spec_channeling"] = np.array([0.125, 4.0])
    return record


def test_line_totals_add_channeling_once_to_either_mode():
    record = _record()

    np.testing.assert_array_equal(cxr_spectrum(record), [1.625, 6.25])
    np.testing.assert_array_equal(cxr_spectrum(record, coherent=True), [3.625, 8.25])
    np.testing.assert_array_equal(cxr_spectrum(record, channeling=False), [1.5, 2.25])
    np.testing.assert_array_equal(
        cxr_spectrum(record, characteristic=False, channeling=False), record["spec"]
    )
    np.testing.assert_array_equal(incident_spectrum(record), [11.625, 26.25])


def test_record_without_channeling_is_unchanged():
    record = _record(channeling=False)

    np.testing.assert_array_equal(cxr_spectrum(record), [1.5, 2.25])


def test_checkpoint_round_trips_channeling_component(tmp_path):
    results = {"cfg": {5000.0: _record()}}

    _checkpoint_store.save("si", tmp_path, results)

    assert _checkpoint_store.component_path("si", "channeling", tmp_path).is_file()
    loaded = _checkpoint_store.load("si", tmp_path)["cfg"][5000.0]
    np.testing.assert_array_equal(loaded["spec_channeling"], [0.125, 4.0])
    np.testing.assert_array_equal(loaded["spec"], [1.0, 2.0])
    np.testing.assert_array_equal(loaded["spec_characteristic"], [0.5, 0.25])
    line = _checkpoint_store._component_store(results, "line")["cfg"][5000.0]
    assert "spec_channeling" not in line


def test_checkpoint_without_channeling_keeps_historical_layout(tmp_path):
    _checkpoint_store.save("si", tmp_path, {"cfg": {5000.0: _record(channeling=False)}})

    files = {path.name for path in (tmp_path / "si").iterdir()}
    assert files == {"line.h5", "brem.h5", "characteristic.h5"}
    loaded = _checkpoint_store.load("si", tmp_path)["cfg"][5000.0]
    assert "spec_channeling" not in loaded


def test_channeling_grid_mismatch_is_refused(tmp_path):
    record = _record()
    del record["spec_characteristic"]
    _checkpoint_store.save("si", tmp_path, {"cfg": {5000.0: record}})
    shifted = {**record, "E_grid": np.array([1.0, 3.0])}
    _checkpoint_store.save("si", tmp_path, {"cfg": {5000.0: shifted}}, components=("line",))

    with pytest.raises(ValueError, match="channeling checkpoint grid"):
        _checkpoint_store.load("si", tmp_path)


def test_store_result_keeps_channeling_and_peaks_on_the_total():
    case = {
        "name": "cfg",
        "E0_keV": 5000.0,
        "domega_sr": 1e-3,
        "theta_obs_rad": 0.0,
        "dtheta_obs_rad": 0.0,
    }
    out = {**_record(), "eta": 0.0}
    results: dict = {}

    store_result(results, case, out)

    stored = results["cfg"][5000.0]
    np.testing.assert_array_equal(stored["spec_channeling"], out["spec_channeling"])
    assert stored["E_pk"] == 2.0


def test_line_dataset_projection_and_merge_carry_channeling():
    local = {"cfg": {5000.0: _record()}}
    incoming = project_dataset({"cfg": {5000.0: _record(channeling=False)}}, "line")

    assert "spec_channeling" in project_dataset(local, "line")["cfg"][5000.0]
    merge_dataset(local, incoming, "line")

    assert "spec_channeling" not in local["cfg"][5000.0]


def test_result_combines_channeling_and_lists_it_as_a_family():
    record = _record()
    result = Result(
        cxr=CxrSpectra(SpectralDensity(record["E_grid"], record["spec"])),
        brem=SpectralDensity(record["E_grid"], record["brem"]),
        characteristic=SpectralDensity(record["E_grid"], record["spec_characteristic"]),
        channeling=SpectralDensity(record["E_grid"], record["spec_channeling"]),
        case={"domega_sr": 1.0},
        provenance={},
    )

    np.testing.assert_array_equal(result.combined_cxr().density, [1.625, 6.25])
    np.testing.assert_array_equal(
        result.combined_cxr(include_channeling=False).density, [1.5, 2.25]
    )
    assert "channeling" in set(result.to_dataframe()["component"].to_list())
    flux = result.spectral_flux(units="electron")
    np.testing.assert_allclose(flux["total"].to_numpy(), [11.625, 26.25])


def test_result_refuses_channeling_on_another_grid():
    record = _record()
    with pytest.raises(ValueError, match="channeling must use the CXR photon-energy grid"):
        Result(
            cxr=CxrSpectra(SpectralDensity(record["E_grid"], record["spec"])),
            brem=SpectralDensity(record["E_grid"], record["brem"]),
            channeling=SpectralDensity(np.array([1.0, 3.0]), record["spec_channeling"]),
            case={},
            provenance={},
        )


def test_spatial_total_adds_characteristic_and_channeling():
    from pyrite.instrument import PixelGrid, PlanarDetector, PlanarPose

    energy = np.array([1.0, 2.0])
    mu = np.zeros((0, 2))

    def factors(values):
        return SpectralFactors(energy, np.array([values]), mu)

    detector = PlanarDetector(
        pose=PlanarPose((0.0, 0.0, 10.0)), pixels=PixelGrid((1, 1), (1.0, 1.0))
    )
    spatial = SpatialResult(
        ray_map=PixelRayMap(np.zeros((1, 1), int), np.ones((1, 1)), np.zeros((1, 1, 0))),
        cxr=factors([1.0, 2.0]),
        brem=factors([10.0, 20.0]),
        detector=detector,
        characteristic=factors([0.5, 0.25]),
        channeling=factors([0.125, 4.0]),
    )

    np.testing.assert_array_equal(spatial._factor("cxr_total").intrinsic_by_tile, [[1.625, 6.25]])
    np.testing.assert_array_equal(spatial._factor("channeling").intrinsic_by_tile, [[0.125, 4.0]])
