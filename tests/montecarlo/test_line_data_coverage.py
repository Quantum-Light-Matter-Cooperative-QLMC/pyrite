"""Missing high-energy optical data must remain visible on clipped axes."""

import json
import warnings

import numpy as np
import pytest

from pyrite.montecarlo import mc_cxr_spectrum
from pyrite.montecarlo.spectrum.cxr import (
    SpectrumRequest,
    _accumulate_batched,
    _finalize_spectrum,
    _prepare_spectrum,
)
from pyrite.montecarlo.spectrum.cxr._coverage import (
    LineOpticalDataWarning,
    collect_line_data_coverage,
)


def _segments(energy_keV):
    energy = np.atleast_1d(energy_keV)
    count = energy.size
    return {
        "r_mid": np.tile([0.0, 0.0, 50.0], (count, 1)),
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 10.0),
        "E_keV": energy,
        "t_ang": np.zeros(count),
        "elec_id": np.arange(count),
        "Ne": count,
        "thickness_ang": 100.0,
    }


def _spectrum(energy=(50_000.0,), **kwargs):
    return mc_cxr_spectrum(
        _segments(energy),
        np.array([1000.0, 1200.0, 1500.0]),
        crystal="hopg",
        hkl_list=[(0, 0, 2), (0, 0, -2)],
        B_ang2=0.8,
        n_hat=np.array([0.0, 0.0, 1.0]),
        **kwargs,
    )


@pytest.mark.parametrize("coherent", [False, True])
@pytest.mark.parametrize("route", ["batched", "per_hkl"])
@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt:RuntimeWarning")
def test_forward_high_energy_miss_survives_a_low_explicit_axis(coherent, route, monkeypatch):
    from pyrite.montecarlo.spectrum.cxr import _spectrum as spectrum_module

    monkeypatch.setattr(spectrum_module, "_needs_per_hkl_route", lambda st: route == "per_hkl")
    # 50 MeV forward HOPG (002) has a vacuum seed of tens of MeV, beyond
    # Chantler. The negative reflection has no positive seed. The deliberately
    # low output grid must not suppress the warning or the denominator.
    with pytest.warns(LineOpticalDataWarning, match="1/2.*not lost photon yield"):
        with collect_line_data_coverage() as records:
            _spectrum(coherent=coherent)
    (record,) = records
    counts = record["counts"]
    assert counts["pairs"] == 2
    assert counts["vacuum_seed_outside_optical_data"] == 1
    assert counts["no_positive_vacuum_seed"] == 1
    assert counts["finite_root"] == 0
    assert record["coherent"] is coherent
    outcomes = (
        "invalid_vacuum_seed",
        "no_positive_vacuum_seed",
        "vacuum_seed_outside_optical_data",
        "root_missing_optical_data",
        "critical_angle_rejected",
        "root_nonconverged",
        "below_line_floor",
        "finite_root",
    )
    assert sum(counts[key] for key in outcomes) == counts["pairs"]
    json.dumps(records, allow_nan=False)


def test_electron_limit_and_mosaic_population_are_explicit():
    with pytest.warns(LineOpticalDataWarning):
        with collect_line_data_coverage() as records:
            _spectrum(
                (50_000.0, 50_000.0),
                electron_limit=1,
                mosaic_fwhm_rad=0.001,
                mosaic_nodes=3,
            )
    # One selected electron, two reflections, nine quadrature orientations.
    assert records[0]["counts"]["pairs"] == 18
    assert records[0]["counts"]["vacuum_seed_outside_optical_data"] == 9


def test_supported_spectrum_is_bitwise_unchanged_and_does_not_warn():
    kwargs = dict(
        segments=_segments(30.0),
        E_grid_eV=np.arange(700.0, 1500.0),
        crystal="hopg",
        hkl_list=[(0, 0, 2)],
        B_ang2=0.8,
        n_hat=np.array([1.0, 0.0, -0.05]) / np.linalg.norm([1.0, 0.0, -0.05]),
    )
    # Existing phases bypass diagnostic collection and supply the numerical
    # baseline; the assertion catches any effect on the actual spectrum.
    st = _prepare_spectrum(SpectrumRequest(**kwargs))
    _accumulate_batched(st)
    expected = _finalize_spectrum(st)
    with warnings.catch_warnings():
        warnings.simplefilter("error", LineOpticalDataWarning)
        with collect_line_data_coverage() as records:
            actual = mc_cxr_spectrum(**kwargs)
    np.testing.assert_array_equal(actual, expected)
    assert records[0]["counts"]["finite_root"] == 1
    assert records[0]["counts"]["vacuum_seed_outside_optical_data"] == 0


def test_failed_nested_attempt_does_not_publish_counts_or_warnings():
    with pytest.warns(LineOpticalDataWarning, match="1/2") as emitted:
        with collect_line_data_coverage() as records:
            with pytest.raises(MemoryError):
                with collect_line_data_coverage():
                    _spectrum()
                    raise MemoryError("simulated later electron-block OOM")
            _spectrum()
    assert len(records) == 1
    assert len(emitted) == 1


def test_low_level_call_warns_without_a_collector():
    with pytest.warns(LineOpticalDataWarning):
        _spectrum()


def test_runner_directions_and_checkpoint_reuse_retain_coverage(monkeypatch, tmp_path):
    from pyrite.checkpoints import _checkpoint_store
    from pyrite.checkpoints.persistence import _cas_payload_from_record
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import directions
    from pyrite.results.store import store_result
    from tests.montecarlo.test_coherent_emission import _runner_case, _runner_tp

    segs = _segments(50_000.0)
    segs.update(n_backscattered=0, n_missed=0, layer=np.zeros(1, dtype=int))
    tp = _runner_tp(segs)
    tp["n_hat"] = np.array([0.0, 0.0, 1.0])
    case = {**_runner_case(), "E0_keV": 50_000.0}
    monkeypatch.setattr(
        runner, "_brem_wide_from_segments", lambda *a, **k: np.zeros_like(tp["E_grid"])
    )
    monkeypatch.setattr(
        runner, "_characteristic_from_segments", lambda *a, **k: np.zeros_like(tp["E_grid"])
    )
    monkeypatch.setattr(directions, "resolve_observation_cxr_grid", lambda *a: (tp["E_grid"], None))
    with pytest.warns(LineOpticalDataWarning):
        output = directions.directional_outputs(
            case, tp, [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], runner._spectrum_case_impl
        )
    forward, backward = output["line_data_coverage"]
    assert forward["counts"]["vacuum_seed_outside_optical_data"] == 1
    assert backward["counts"]["vacuum_seed_outside_optical_data"] == 0
    assert forward["direction"] != backward["direction"]
    # Scalar storage uses exactly the same runner diagnostic records.
    case.update(name="coverage", theta_obs_rad=0.1, dtheta_obs_rad=0.01, domega_sr=0.01)
    with pytest.warns(LineOpticalDataWarning):
        scalar = runner._spectrum_case_impl(case, tp)
    results = {}
    store_result(results, case, scalar)
    _checkpoint_store.save("coverage", tmp_path, results)
    loaded = _checkpoint_store.load("coverage", tmp_path)
    record = loaded["coverage"][50_000.0]
    assert record["line_data_coverage"] == scalar["line_data_coverage"]
    assert _cas_payload_from_record(record)["line_data_coverage"] == scalar["line_data_coverage"]
    json.dumps(record["line_data_coverage"], allow_nan=False)


def test_each_crystalline_layer_contributes_its_own_population():
    from pyrite.montecarlo import runner
    from tests.montecarlo.test_coherent_emission import _runner_case

    segments = _segments([50_000.0, 50_000.0])
    segments.update(
        layer=np.array([0, 1]),
        thickness_ang=200.0,
        r_mid=np.array([[0.0, 0.0, 50.0], [0.0, 0.0, 150.0]]),
        n_layers=2,
    )
    layers = [(0.0, 100.0, [("C", 0.1)]), (100.0, 200.0, [("C", 0.1)])]
    radiator = _runner_case()
    case = {**radiator, "layer_radiators": [radiator, radiator]}
    with pytest.warns(LineOpticalDataWarning, match="2/2"):
        with collect_line_data_coverage() as records:
            runner._cxr_for_segments(
                segments,
                np.array([1000.0, 1200.0]),
                case,
                np.array([0.0, 0.0, 1.0]),
                layers,
                None,
            )
    assert len(records) == 2
    assert all(record["counts"]["pairs"] == 1 for record in records)
    assert all(record["counts"]["vacuum_seed_outside_optical_data"] == 1 for record in records)
