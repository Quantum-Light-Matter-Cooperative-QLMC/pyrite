"""Deterministic channeling source survives replay and checkpoint repair (#521)."""

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite._spectral_components import cxr_spectrum
from pyrite.checkpoints import _checkpoint_store, recompute
from pyrite.detectors import EnergyBins
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner.channeling_emission import rescore_channeling_spectrum
from pyrite.montecarlo.trajectories import read_trajectory_artifact, write_trajectory_artifact
from pyrite.results.store import store_result

# Real channeling transport/table setup and smallest-block HDF5 replay exceed
# the fast suite budget; retain the full end-to-end compatibility check in CI.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def channeling_run():
    scene = pr.Scene(
        pr.Beam(energy_keV=5000.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()),
        pr.geometry.Slab(
            "silicon",
            thickness_ang=5000.0,
            tilt_deg=3.0,
            tilt_azim_deg=45.0,
            footprint=pr.geometry.Footprint(5.0, 5.0),
        ),
        pr.Detector(),
        physics=pr.Physics(
            channeling_model="planar", channeling_plane=(1, -1, 0), radiative_model="uncoupled"
        ),
    )
    numerics = pr.Numerics(
        cxr_trials=4,
        brem_trials=6,
        energy_bins=EnergyBins(
            cxr=np.linspace(1500.0, 3000.0, 31), brem=np.array([5000.0, 10000.0])
        ),
        channeling_basis_size=8,
        channeling_bloch_cells=4,
        channeling_dz_ang=500.0,
    )
    case = api.build_case(scene, numerics).to_dict()
    transport = runner._transport_case(case)
    return case, transport, runner._spectrum_case(case, transport)


def test_case_rescore_matches_live_source_without_transport(channeling_run, monkeypatch):
    case, transport, live = channeling_run
    monkeypatch.setattr(runner, "_transport_case", lambda *_a, **_k: pytest.fail("transported"))
    # Independent of sampled handoff fates; the entrance population has more
    # bremsstrahlung than line primaries, matching the live divergence solve.
    actual = rescore_channeling_spectrum(case, transport["E_grid"])
    np.testing.assert_array_equal(actual, live["spec_channeling"])
    assert np.any(actual > 0)


@pytest.mark.parametrize("block_size", [1, 100000])
def test_full_and_streamed_artifact_replay_score_cr_once(
    channeling_run, tmp_path, monkeypatch, block_size
):
    case, transport, live = channeling_run
    # Capture is separately refused by the runner. The artifact writer accepts
    # an existing transport result; replay must reconstruct CR from its case.
    inputs = {k: v for k, v in transport.items() if k not in {"segs", "channeling_beam_dir"}}
    path = write_trajectory_artifact(
        tmp_path / "channeling.h5", transport["segs"], case=case, spectrum_inputs=inputs
    )
    monkeypatch.setattr(runner, "_transport_case", lambda *_a, **_k: pytest.fail("transported"))
    full = runner.spectrum_from_artifact(read_trajectory_artifact(path))
    streamed = runner.stream_spectrum_from_artifact(path, max_segments=block_size)
    for actual in (full, streamed):
        np.testing.assert_array_equal(actual["spec_channeling"], live["spec_channeling"])
        # CR has no block summation error; native estimators may reassociate.
        for key in ("spec", "spec_characteristic", "brem"):
            np.testing.assert_allclose(actual[key], live[key], rtol=1e-12, atol=0)
        np.testing.assert_allclose(
            cxr_spectrum(actual),
            actual["spec"] + actual["spec_characteristic"] + live["spec_channeling"],
            rtol=1e-14,
        )


def test_reline_rescores_target_grid_and_round_trips_once(channeling_run, tmp_path, monkeypatch):
    case, transport, live = channeling_run
    records = {}
    store_result(records, dict(case), live)
    record = next(iter(next(iter(records.values())).values()))
    # Native components remain separate from CR while the source moves to the
    # finer line grid. Use constants so total accounting is independent.
    monkeypatch.setattr(
        runner,
        "_cxr_pair_for_case",
        lambda _c, grid, **_kw: (np.ones_like(grid), None, np.full_like(grid, 2.0)),
    )
    assert (
        recompute.repair_cxr_spec(
            records, "silicon", from_config=False, line_step_eV=25.0, redo_all=True
        )
        == 1
    )
    target = record["E_grid"]
    reference = runner._spectrum_case(case, {**transport, "E_grid": target})["spec_channeling"]
    np.testing.assert_array_equal(record["spec_channeling"], reference)
    assert target.size > live["E_grid"].size
    np.testing.assert_allclose(cxr_spectrum(record), 3.0 + reference)
    _checkpoint_store.save("silicon", tmp_path, records)
    loaded = _checkpoint_store.load("silicon", tmp_path)
    restored = next(iter(next(iter(loaded.values())).values()))
    np.testing.assert_array_equal(restored["spec_channeling"], reference)
    np.testing.assert_allclose(cxr_spectrum(restored), 3.0 + reference)


def test_brem_repair_keeps_channeling_source_untouched(channeling_run, monkeypatch):
    case, _transport, live = channeling_run
    records = {}
    store_result(records, dict(case), live)
    record = next(iter(next(iter(records.values())).values()))
    original = record["spec_channeling"]
    monkeypatch.setattr(
        "pyrite.montecarlo._brem_for_case", lambda _c, grid: np.full_like(grid, 4.0)
    )
    assert recompute.repair_brem_wide(records, only_nonfinite=False, progress=False) == 1
    assert record["spec_channeling"] is original
    np.testing.assert_array_equal(record["brem"], 4.0)


@pytest.mark.parametrize("broken", [None, np.array([np.nan]), np.array([1.0])])
def test_reline_repairs_stale_channeling_on_an_unchanged_grid(channeling_run, monkeypatch, broken):
    from pyrite.montecarlo.runner import channeling_emission

    case, _transport, live = channeling_run
    records = {}
    store_result(records, dict(case), live)
    record = next(iter(next(iter(records.values())).values()))
    record["spec_channeling"] = broken
    monkeypatch.setattr(
        runner,
        "_cxr_pair_for_case",
        lambda _c, grid, **_kw: (np.ones_like(grid), None, np.zeros_like(grid)),
    )
    monkeypatch.setattr(
        channeling_emission, "rescore_channeling_spectrum", lambda _c, grid: np.full_like(grid, 3.0)
    )
    assert recompute.repair_cxr_spec(records, "silicon", from_config=False) == 1
    np.testing.assert_array_equal(record["spec_channeling"], 3.0)
    assert recompute.repair_cxr_spec(records, "silicon", from_config=False) == 0
