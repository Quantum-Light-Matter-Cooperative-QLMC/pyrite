"""Tests for run.py: checkpoint resume, group-streaming, and brem repair.

Monte Carlo work is replaced with a stub — these tests cover the orchestration
layer (resume/skip, checkpoint writes, on_chunk dispatch), not the physics.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from pyrite.checkpoints import _checkpoint_io, _checkpoint_store
from pyrite.montecarlo import runner
from pyrite.runs.run import (
    _checkpoint_save,
    _load_checkpoint_cached,
    _manifest_save,
    _material_analysis_cache,
    cached_material_analysis,
    cases_from_results,
    checkpoint_manifest,
    checkpoint_path_for,
    load_checkpoint,
    repair_brem_wide,
    run_sweep,
)
from tests.helpers import (
    fake_out,
    fake_segments,
    runner_transport_payload,
    stub_run_cases,
    tracking_run_cases_factory,
)

# ---------------------------------------------------------------------------
# The load_checkpoint cache is module-global (functools.lru_cache) -- clear it
# around every test so a checkpoint loaded in one test doesn't sit cached (by
# path/mtime) and leak into an unrelated test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_checkpoint_cache():
    _load_checkpoint_cached.cache_clear()
    _material_analysis_cache.clear()
    yield
    _load_checkpoint_cached.cache_clear()
    _material_analysis_cache.clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fake_case(
    name: str, E0_keV: float, tilt_deg: float = 0.0, crystal: str = "hopg"
) -> dict[str, Any]:
    """Minimal case dict that satisfies store_result and run_sweep internals."""
    E_grid = (100.0, 200.0, 10.0)
    E_brem = (0.0, 500.0, 50.0)
    return dict(
        name=name,
        crystal=crystal,
        composition=[("C", 0.113)],
        hkl_list=[],
        B_ang2=0.8,
        E0_keV=float(E0_keV),
        thickness_ang=1e4,
        E_grid=E_grid,
        E_grid_line=E_grid,
        E_grid_brem=E_brem,
        theta_obs_rad=np.deg2rad(90.0),
        dtheta_obs_rad=np.deg2rad(2.0),
        tilt_deg=float(tilt_deg),
        tilt_azim_deg=0.0,
        domega_sr=1e-4,
        beam_uvw=(0, 0, 1),
        mosaic_fwhm_rad=None,
        mosaic_mc_fwhm_rad=None,
        mosaic_mc_nodes=1,
        abs_layers=None,
        layer_radiators=None,
        brem_file=None,
        Ne=10,
        Ne_brem=5,
        seed=42,
        spec_chunk=None,
        brem_chunk=None,
    )


def test_run_case_directions_transports_once_and_stacks_direction_outputs(monkeypatch):
    case = _fake_case("directions", 30.0)
    calls = {"transport": 0, "spectrum": 0}

    def fake_transport(payload, **kwargs):
        assert payload is case
        assert kwargs["keep_segments_on_device"] is True
        calls["transport"] += 1
        return {"n_hat": np.array([0.0, 0.0, 1.0]), "segments": object()}

    def fake_spectrum(payload, transport):
        assert payload is case
        calls["spectrum"] += 1
        value = transport["n_hat"][0]
        return {
            "E_grid": np.array([1.0, 2.0]),
            "E_grid_brem": np.array([1.0, 2.0, 3.0]),
            "spec": np.array([value, value + 1.0]),
            "spec_characteristic": np.array([0.25, 0.5]),
            "brem": np.array([value + 2.0, value + 3.0]),
            "brem_wide": np.array([value + 2.0, value + 3.0, value + 4.0]),
        }

    monkeypatch.setattr(runner, "_transport_case", fake_transport)
    monkeypatch.setattr(runner, "_spectrum_case", fake_spectrum)

    output = runner.run_case_directions(case, np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))

    assert calls == {"transport": 1, "spectrum": 2}
    np.testing.assert_array_equal(output["spec_by_direction"], [[1.0, 2.0], [-1.0, 0.0]])
    np.testing.assert_array_equal(
        output["spec_characteristic_by_direction"],
        [[0.25, 0.5], [0.25, 0.5]],
    )
    assert output["brem_wide_by_direction"].shape == (2, 3)


def test_one_direction_runner_matches_scalar_runner_bit_for_bit() -> None:
    case = _fake_case("one-direction", 30.0)
    case["hkl_list"] = [(0, 0, 2)]
    _, direction = runner.tilted_geometry(
        case["theta_obs_rad"],
        np.deg2rad(case["tilt_deg"]),
        np.deg2rad(case["tilt_azim_deg"]),
    )

    scalar = runner.run_case(case)
    directional = runner.run_case_directions(case, direction[None, :])

    np.testing.assert_array_equal(directional["spec_by_direction"][0], scalar["spec"])
    np.testing.assert_array_equal(
        directional["spec_characteristic_by_direction"][0],
        scalar["spec_characteristic"],
    )
    np.testing.assert_array_equal(directional["brem_wide_by_direction"][0], scalar["brem_wide"])


def test_transport_case_forwards_finite_footprint_to_shared_transport(monkeypatch):
    case = _fake_case("finite", 30.0)
    case.update(
        crystal_width_mm=0.1,
        crystal_height_mm=0.2,
    )

    captured = []

    def _transport(*args, **kwargs):
        captured.append((args, kwargs))
        return {"transport": len(captured)}

    monkeypatch.setattr(runner, "simulate_trajectories", _transport)

    tp = runner._transport_case(case)

    assert len(captured) == 1

    args, kwargs = captured[0]

    assert kwargs["crystal_width_mm"] == 0.1
    assert kwargs["crystal_height_mm"] == 0.2

    assert args[1] == max(case["Ne"], case["Ne_brem"])
    assert tp["segs"] == {"transport": 1}
    assert tp["Ne_lines"] == case["Ne"]
    assert tp["Ne_brem"] == case["Ne_brem"]


def test_transport_case_combines_line_and_brem_cutoffs(monkeypatch):
    case = _fake_case("combined", 30.0)
    case.update(
        Ne=3,
        Ne_brem=2,
        E_cut_lines_keV=5.0,
        E_cut_brem_keV=1.0,
    )

    captured = {}

    def _transport(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"segments": True}

    monkeypatch.setattr(runner, "simulate_trajectories", _transport)

    runner._transport_case(case)

    np.testing.assert_array_equal(
        captured["kwargs"]["E_cut_by_electrons"],
        np.array(
            [
                1.0,  # shared line+brem electron: lower cutoff
                1.0,  # shared line+brem electron
                5.0,  # line-only electron
            ]
        ),
    )

    assert captured["args"][1] == 3


def test_transport_case_handles_brem_only_electrons(monkeypatch):
    case = _fake_case("combined", 30.0)
    case.update(
        Ne=2,
        Ne_brem=4,
        E_cut_lines_keV=5.0,
        E_cut_brem_keV=1.0,
    )

    captured = {}

    def _transport(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(runner, "simulate_trajectories", _transport)

    runner._transport_case(case)

    np.testing.assert_array_equal(
        captured["kwargs"]["E_cut_by_electrons"],
        np.array([1.0, 1.0, 1.0, 1.0]),
    )

    assert captured["args"][1] == 4


def test_brem_for_case_forwards_finite_footprint(monkeypatch):
    case = _fake_case("finite", 30.0)
    case.update(crystal_width_mm=0.1, crystal_height_mm=0.2)
    captured = []

    def _transport(*args, **kwargs):
        captured.append(kwargs)
        return {"transport": True}

    monkeypatch.setattr(runner, "simulate_trajectories", _transport)
    monkeypatch.setattr(runner, "_brem_wide_from_segments", lambda *args, **kwargs: np.array([0.0]))

    runner._brem_for_case(case, np.array([100.0]))

    assert len(captured) == 1
    assert captured[0]["crystal_width_mm"] == pytest.approx(0.1)
    assert captured[0]["crystal_height_mm"] == pytest.approx(0.2)


def test_spectrum_case_forwards_groove_to_brem(monkeypatch):
    groove = object()
    segments = fake_segments()
    grid = np.array([1000.0, 2000.0])
    seen = []

    monkeypatch.setattr(
        runner,
        "_lines_for_segments",
        lambda *_args, **_kwargs: np.zeros_like(grid),
    )
    monkeypatch.setattr(
        runner,
        "_characteristic_from_segments",
        lambda *_args, **_kwargs: np.zeros_like(grid),
    )

    def _brem(*_args, **kwargs):
        seen.append(kwargs)
        return np.zeros_like(grid)

    monkeypatch.setattr(runner, "mc_brem_spectrum", _brem)
    case = _fake_case("grooved", 30.0)
    tp = runner_transport_payload(
        segments,
        E_grid=grid,
        E_brem=grid,
        n_hat=np.array([0.0, 0.0, -1.0]),
        groove=groove,
    )

    runner._spectrum_case(case, tp)

    assert len(seen) == 1
    assert seen[0]["groove"] is groove


def test_brem_for_case_forwards_identical_groove_to_brem_rebuild(monkeypatch):
    case = _fake_case("grooved", 30.0, tilt_deg=45.0)
    case.update(tilt_azim_deg=180.0, groove_spacing_ang=20_000.0)
    segments = fake_segments()
    transported = []
    radiated = []

    def _transport(*_args, **kwargs):
        transported.append(kwargs)
        return segments

    def _brem(*_args, **kwargs):
        radiated.append(kwargs)
        return np.zeros(2)

    monkeypatch.setattr(runner, "simulate_trajectories", _transport)
    monkeypatch.setattr(runner, "mc_brem_spectrum", _brem)

    runner._brem_for_case(case, np.array([1000.0, 2000.0]))

    assert len(transported) == len(radiated) == 1
    assert radiated[0]["groove"] is transported[0]["groove"]


def test_brem_wide_from_segments_forwards_groove_to_brem(monkeypatch):
    groove = object()
    segments = fake_segments()
    grid = np.array([1000.0, 2000.0])
    seen = []

    def _brem(*_args, **kwargs):
        seen.append(kwargs)
        return np.zeros_like(grid)

    monkeypatch.setattr(runner, "mc_brem_spectrum", _brem)

    runner._brem_wide_from_segments(
        segments,
        grid,
        _fake_case("grooved", 30.0),
        np.array([0.0, 0.0, -1.0]),
        None,
        groove=groove,
    )

    assert len(seen) == 1
    assert seen[0]["groove"] is groove


# ---------------------------------------------------------------------------
# checkpoint_path_for
# ---------------------------------------------------------------------------


def test_checkpoint_path_for_includes_material_and_dir():
    p = checkpoint_path_for("hopg", checkpoint_dir="ckpts")
    assert Path(p) == Path("ckpts/hopg")


# ---------------------------------------------------------------------------
# _checkpoint_save
# ---------------------------------------------------------------------------


def test_checkpoint_save_roundtrips(tmp_path):
    rec = {"cfg_a": {30.0: {"case": {"crystal": "hopg"}, "spec": np.array([1.0])}}}
    ckpt = tmp_path / "hopg.pkl"
    _checkpoint_save(str(ckpt), rec)
    assert set(_checkpoint_io.load(str(ckpt))) == {"cfg_a"}


def test_checkpoint_save_is_hdf5(tmp_path):
    """Result writes use the versioned HDF5 container despite the path suffix."""
    ckpt = tmp_path / "hopg.pkl"
    _checkpoint_save(str(ckpt), {"cfg_a": {30.0: {"case": {}, "spec": np.ones(1000)}}})
    assert ckpt.read_bytes()[:8] == b"\x89HDF\r\n\x1a\n"


def test_checkpoint_save_is_atomic(tmp_path):
    """A successful save leaves no stray .tmp and replaces the prior file."""
    ckpt = tmp_path / "hopg.pkl"
    _checkpoint_save(str(ckpt), {"old": 1})
    _checkpoint_save(str(ckpt), {"new": 2})
    assert not (tmp_path / "hopg.pkl.tmp").exists()
    assert _checkpoint_io.load(str(ckpt)) == {"new": 2}


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------


def test_load_checkpoint_missing_returns_empty(tmp_path, capsys):
    result = load_checkpoint("hopg", checkpoint_dir=str(tmp_path))
    assert result == {}
    assert "hopg" in capsys.readouterr().out


def test_load_checkpoint_reads_existing_pickle(tmp_path):
    rec = {"cfg_a": {30.0: {"case": {"crystal": "hopg"}, "spec": np.array([1.0])}}}
    pkl = tmp_path / "hopg.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(rec, f)
    loaded = load_checkpoint("hopg", checkpoint_dir=str(tmp_path))
    assert set(loaded) == {"cfg_a"}
    assert 30.0 in loaded["cfg_a"]


def test_load_checkpoint_caches_across_calls(tmp_path, monkeypatch):
    """Two calls for the same (path, mtime) hit disk once -- the second is
    served from the module-level lru_cache."""
    rec = {"cfg_a": {30.0: {"case": {"crystal": "hopg"}, "spec": np.array([1.0])}}}
    pkl = tmp_path / "hopg.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(rec, f)

    orig_load = _checkpoint_io.load
    calls = []

    def _counting_load(path):
        calls.append(path)
        return orig_load(path)

    monkeypatch.setattr(_checkpoint_io, "load", _counting_load)

    first = load_checkpoint("hopg", checkpoint_dir=str(tmp_path))
    second = load_checkpoint("hopg", checkpoint_dir=str(tmp_path))
    assert len(calls) == 1
    assert first is second  # cache hit returns the same object, not a re-load


def test_load_checkpoint_reloads_after_mtime_change(tmp_path, monkeypatch):
    """Touching the checkpoint to a new mtime invalidates the cache entry and
    forces a fresh disk load (e.g. after a re-run scan overwrites the pickle)."""
    rec = {"cfg_a": {30.0: {"case": {"crystal": "hopg"}, "spec": np.array([1.0])}}}
    pkl = tmp_path / "hopg.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(rec, f)

    orig_load = _checkpoint_io.load
    calls = []

    def _counting_load(path):
        calls.append(path)
        return orig_load(path)

    monkeypatch.setattr(_checkpoint_io, "load", _counting_load)

    load_checkpoint("hopg", checkpoint_dir=str(tmp_path))
    load_checkpoint("hopg", checkpoint_dir=str(tmp_path))
    assert len(calls) == 1

    new_mtime = os.path.getmtime(pkl) + 5.0
    os.utime(pkl, (new_mtime, new_mtime))

    load_checkpoint("hopg", checkpoint_dir=str(tmp_path))
    assert len(calls) == 2


def test_load_checkpoint_cache_hit_is_silent(tmp_path, capsys):
    """The 'loaded N records' print lives inside the cached function body, so a
    cache hit prints nothing (only the first, real disk load does)."""
    rec = {"cfg_a": {30.0: {"case": {"crystal": "hopg"}, "spec": np.array([1.0])}}}
    pkl = tmp_path / "hopg.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(rec, f)

    load_checkpoint("hopg", checkpoint_dir=str(tmp_path))
    capsys.readouterr()  # discard the first (real) load's print
    load_checkpoint("hopg", checkpoint_dir=str(tmp_path))
    assert capsys.readouterr().out == ""


def test_cached_material_analysis_survives_process_cache_reset(tmp_path, monkeypatch):
    """Small analysis results persist after module-memory state disappears."""
    checkpoint = tmp_path / "hopg.pkl"
    with open(checkpoint, "wb") as f:
        pickle.dump({"cfg": {30.0: {"value": 7}}}, f)

    first = cached_material_analysis(
        "hopg",
        lambda results: results["cfg"][30.0]["value"],
        ("quality_peak", None, 0.5),
        checkpoint_dir=str(tmp_path),
    )
    assert first == 7

    _material_analysis_cache.clear()  # model app exit + fresh process
    monkeypatch.setattr(
        "pyrite.runs.run.load_checkpoint",
        lambda *args, **kwargs: pytest.fail("persistent cache should avoid checkpoint reload"),
    )

    second = cached_material_analysis(
        "hopg",
        lambda results: results["cfg"][30.0]["value"],
        ("quality_peak", None, 0.5),
        checkpoint_dir=str(tmp_path),
    )
    assert second == 7


def test_cached_material_analysis_invalidates_when_checkpoint_changes(tmp_path):
    checkpoint = tmp_path / "hopg.pkl"
    with open(checkpoint, "wb") as f:
        pickle.dump({"cfg": {30.0: {"value": 7}}}, f)

    analyze_calls = []

    def analyze(results):
        analyze_calls.append(results)
        return results["cfg"][30.0]["value"]

    assert cached_material_analysis("hopg", analyze, ("summary", 1), str(tmp_path)) == 7
    with open(checkpoint, "wb") as f:
        pickle.dump({"cfg": {30.0: {"value": 9}}}, f)
    new_mtime = os.path.getmtime(checkpoint) + 5.0
    os.utime(checkpoint, (new_mtime, new_mtime))

    assert cached_material_analysis("hopg", analyze, ("summary", 1), str(tmp_path)) == 9
    assert len(analyze_calls) == 2


# ---------------------------------------------------------------------------
# checkpoint_manifest / _manifest_save
# ---------------------------------------------------------------------------


def test_checkpoint_manifest_none_when_no_checkpoint(tmp_path):
    assert checkpoint_manifest("hopg", checkpoint_dir=str(tmp_path)) is None


def test_checkpoint_manifest_backfills_when_sidecar_missing(tmp_path):
    """No meta.json alongside an existing .pkl (e.g. a checkpoint written before
    this feature existed) -- checkpoint_manifest loads it once, computes the
    manifest, and writes the sidecar for next time."""
    results = {
        "cfg_a": {
            30.0: {"case": _fake_case("cfg_a", 30.0), "spec": np.array([1.0])},
            45.0: {"case": _fake_case("cfg_a", 45.0), "spec": np.array([1.0])},
        }
    }
    pkl = tmp_path / "hopg.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(results, f)
    meta_path = tmp_path / "hopg.meta.json"
    assert not meta_path.exists()

    manifest = checkpoint_manifest("hopg", checkpoint_dir=str(tmp_path))

    assert manifest["n_records"] == 2
    assert manifest["energies_keV"] == [30.0, 45.0]
    assert manifest["sweep"]["crystal"] == ["hopg"]
    assert meta_path.exists()  # backfilled for next call
    with open(meta_path) as f:
        assert json.load(f) == manifest


def test_checkpoint_manifest_regenerates_when_sidecar_stale(tmp_path):
    """A meta.json older than the .pkl (e.g. left over from before a re-run
    scan overwrote the checkpoint) is treated as stale and rebuilt, not trusted
    as-is."""
    results = {
        "cfg_a": {
            30.0: {"case": _fake_case("cfg_a", 30.0), "spec": np.array([1.0])},
            45.0: {"case": _fake_case("cfg_a", 45.0), "spec": np.array([1.0])},
        }
    }
    pkl = tmp_path / "hopg.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(results, f)
    meta_path = tmp_path / "hopg.meta.json"
    meta_path.write_text(json.dumps({"energies_keV": [999.0], "n_records": 999, "sweep": {}}))
    pkl_mtime = os.path.getmtime(pkl)
    os.utime(meta_path, (pkl_mtime - 10.0, pkl_mtime - 10.0))

    manifest = checkpoint_manifest("hopg", checkpoint_dir=str(tmp_path))

    assert manifest["n_records"] == 2
    assert manifest["energies_keV"] == [30.0, 45.0]


def test_checkpoint_manifest_fresh_sidecar_skips_unpickling(tmp_path, monkeypatch):
    """A meta.json at least as new as the .pkl is read directly -- the whole
    point of the sidecar is enumerating a checkpoint's contents without paying
    for the (140-225 MB) unpickle."""
    pkl = tmp_path / "hopg.pkl"
    pkl.write_bytes(b"not a real pickle")  # would blow up if load() were ever called
    meta_path = tmp_path / "hopg.meta.json"
    meta_path.write_text(json.dumps({"energies_keV": [30.0], "n_records": 1, "sweep": {}}))
    pkl_mtime = os.path.getmtime(pkl)
    os.utime(meta_path, (pkl_mtime + 10.0, pkl_mtime + 10.0))

    def _boom(path):
        raise AssertionError("checkpoint_manifest must not unpickle a fresh sidecar")

    monkeypatch.setattr(_checkpoint_io, "load", _boom)

    manifest = checkpoint_manifest("hopg", checkpoint_dir=str(tmp_path))
    assert manifest == {"energies_keV": [30.0], "n_records": 1, "sweep": {}}


def test_checkpoint_manifest_prefers_component_store_over_stale_legacy_pkl(tmp_path):
    """A leftover legacy ``<material>.pkl`` monolith beside a migrated
    ``<material>/line.h5`` component checkpoint (mid-migration: `save()`
    leaves the old .pkl in place) must not shadow the component data --
    `_manifest_path_for` has to recognize `line.h5`, not just `line.pkl`, as
    "already migrated", or it reads/writes the wrong (legacy) sidecar and
    stale results win."""
    legacy_results = {"cfg_a": {30.0: {"case": _fake_case("cfg_a", 30.0), "spec": np.array([1.0])}}}
    pkl = tmp_path / "hopg.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(legacy_results, f)

    fresh_results = {
        "cfg_a": {
            30.0: {"case": _fake_case("cfg_a", 30.0), "spec": np.array([1.0])},
            45.0: {"case": _fake_case("cfg_a", 45.0), "spec": np.array([1.0])},
        }
    }
    _checkpoint_store.save("hopg", tmp_path, fresh_results)

    manifest = checkpoint_manifest("hopg", checkpoint_dir=str(tmp_path))

    assert manifest["n_records"] == 2
    assert manifest["energies_keV"] == [30.0, 45.0]
    assert (tmp_path / "hopg" / "meta.json").is_file()
    assert not (tmp_path / "hopg.meta.json").exists()


def test_manifest_save_coerces_numpy_scalars_to_json_safe(tmp_path):
    case = _fake_case("cfg_a", 30.0)
    case["thickness_ang"] = np.float64(1e4)  # simulate a numpy scalar in a case field
    results = {"cfg_a": {30.0: {"case": case, "spec": np.array([1.0])}}}
    ckpt = tmp_path / "hopg.pkl"

    manifest = _manifest_save(str(ckpt), results)

    json.dumps(manifest)  # raises TypeError if a numpy scalar leaked through
    assert manifest["sweep"]["thickness_ang"] == [1e4]
    assert (tmp_path / "hopg.meta.json").exists()


def test_run_sweep_writes_manifest_alongside_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    cases = [
        _fake_case("cfg_a", 30.0),
        _fake_case("cfg_a", 45.0),
        _fake_case("cfg_b", 30.0),
    ]
    run_sweep(cases, {}, checkpoint_dir=str(tmp_path), progress=False)

    meta_path = tmp_path / "hopg" / "meta.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        manifest = json.load(f)
    assert manifest["n_records"] == 3
    assert manifest["energies_keV"] == [30.0, 45.0]
    assert manifest["sweep"]["crystal"] == ["hopg"]


def test_shard_manifest_scans_accumulated_results_only_at_consolidation(tmp_path, monkeypatch):
    """Per-config shard publication must not rebuild the manifest from every
    accumulated record; one full scan at final consolidation is sufficient."""
    import pyrite.runs.run as run
    from pyrite.checkpoints import persistence

    monkeypatch.setattr(run, "run_cases", stub_run_cases)
    real_manifest_for = persistence._manifest_for
    scanned_record_counts = []

    def counted_manifest_for(results, dataset_identity=None):
        scanned_record_counts.append(sum(len(by_energy) for by_energy in results.values()))
        return real_manifest_for(results, dataset_identity)

    monkeypatch.setattr(persistence, "_manifest_for", counted_manifest_for)
    cases = [
        _fake_case("cfg_a", 30.0),
        _fake_case("cfg_a", 45.0),
        _fake_case("cfg_b", 30.0),
    ]

    run.run_sweep(cases, {}, checkpoint_dir=str(tmp_path), progress=False)

    assert scanned_record_counts == [3]
    manifest = json.loads((tmp_path / "hopg" / "meta.json").read_text())
    assert manifest["n_records"] == 3
    assert manifest["completed_case_set"]["count"] == 3


def test_run_sweep_reports_checkpoint_timing(tmp_path, monkeypatch):
    def profile_run_cases(
        cases,
        *,
        callback=None,
        on_timing=None,
        **kwargs,
    ):
        for i, case in enumerate(cases):
            out = fake_out(case)

            if on_timing is not None:
                on_timing(
                    {
                        "case_index": i,
                        "case": case,
                        "transport_seconds": 0.01,
                        "spectrum_seconds": 0.02,
                        "driver_wait_seconds": 0.0,
                    }
                )

            if callback is not None:
                callback(i, case, out)

        return [None] * len(cases)

    monkeypatch.setattr("pyrite.runs.run.run_cases", profile_run_cases)
    timings = []

    run_sweep(
        [_fake_case("cfg_a", 30.0)],
        {},
        checkpoint_dir=str(tmp_path),
        progress=False,
        on_timing=timings.append,
    )

    # one per-config shard write + one end-of-sweep consolidation into the monolith
    checkpoint = [timing for timing in timings if "checkpoint_seconds" in timing]
    assert len(checkpoint) == 2
    assert all(timing["checkpoint_seconds"] >= 0 for timing in checkpoint)


def test_run_sweep_persists_dataset_identity_in_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    identity = {
        "schema": "cxr.dataset-identity.v1",
        "material": "hopg",
        "profile": "survey",
        "parameter_sha256": "a" * 64,
        "resolved_parameters": {"settings": {}, "sweep": {}},
    }
    run_sweep(
        [_fake_case("cfg_a", 30.0)],
        {},
        checkpoint_dir=str(tmp_path),
        progress=False,
        dataset_identity=identity,
    )

    manifest = json.loads((tmp_path / "hopg" / "meta.json").read_text())
    assert manifest["schema"] == "cxr.checkpoint-manifest.v2"
    assert manifest["identity_version"] == 1
    assert manifest["dataset_identity"] == {**identity, "identity_version": 1}


def test_run_sweep_exact_metadata_hit_skips_decode_launch_and_writes(tmp_path, monkeypatch):
    cases = [_fake_case("cfg_a", 30.0), _fake_case("cfg_b", 45.0)]
    identity = {
        "schema": "cxr.dataset-identity.v1",
        "material": "hopg",
        "profile": "survey",
        "parameter_sha256": "a" * 64,
        "resolved_parameters": {"settings": {}, "sweep": {}},
    }
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    run_sweep(cases, {}, checkpoint_dir=str(tmp_path), progress=False, dataset_identity=identity)
    paths = [
        tmp_path / "hopg" / name
        for name in ("line.h5", "brem.h5", "characteristic.h5", "meta.json")
    ]
    before = {path: path.stat().st_mtime_ns for path in paths}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("exact metadata hit must not decode, launch, or write")

    monkeypatch.setattr("pyrite.runs.run._checkpoint_load", forbidden)
    monkeypatch.setattr("pyrite.runs.run.run_cases", forbidden)
    monkeypatch.setattr("pyrite.runs.run._checkpoint_components_save", forbidden)
    progress = []
    assert run_sweep(
        cases,
        {},
        checkpoint_dir=str(tmp_path),
        progress=False,
        dataset_identity=identity,
        metadata_only_complete=True,
        on_progress=lambda *values: progress.append(values),
    )
    assert progress == [(0, 2, 2)]
    assert {path: path.stat().st_mtime_ns for path in paths} == before


def test_budget_pause_keeps_shards_until_completion(tmp_path, monkeypatch):
    cases = [_fake_case("cfg_a", 30.0), _fake_case("cfg_b", 45.0)]

    def first_only(cases, callback=None, **_kwargs):
        callback(0, cases[0], fake_out(cases[0]))

    monkeypatch.setattr("pyrite.runs.run.run_cases", first_only)
    assert not run_sweep(cases, {}, checkpoint_dir=str(tmp_path), progress=False)
    assert _checkpoint_store.has_parts("hopg", tmp_path)
    assert not (tmp_path / "hopg" / "line.h5").exists()
    assert sum(len(items) for items in _checkpoint_store.load("hopg", tmp_path).values()) == 1

    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    results = {}
    assert run_sweep(cases, results, checkpoint_dir=str(tmp_path), progress=False)
    assert sum(len(items) for items in results.values()) == 2
    assert (tmp_path / "hopg" / "line.h5").is_file()
    assert not _checkpoint_store.has_parts("hopg", tmp_path)


def test_manifest_refresh_preserves_existing_dataset_identity(tmp_path):
    identity = {"profile": "survey", "parameter_sha256": "a" * 64}
    results = {"cfg": {30.0: {"case": _fake_case("cfg", 30.0)}}}
    checkpoint = tmp_path / "hopg"
    checkpoint.mkdir()

    _manifest_save(str(checkpoint), results, identity)
    refreshed = _manifest_save(str(checkpoint), results)

    assert refreshed["dataset_identity"] == {**identity, "identity_version": 1}


def test_run_sweep_refuses_resume_across_dataset_identities(tmp_path, monkeypatch):
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    case = _fake_case("cfg", 30.0)
    checkpoint = tmp_path / "hopg"
    old = {"profile": "full", "parameter_sha256": "a" * 64}
    new = {"profile": "full", "parameter_sha256": "b" * 64}
    run_sweep(
        [case],
        {},
        checkpoint_path=str(checkpoint),
        progress=False,
        dataset_identity=old,
    )

    with pytest.raises(ValueError, match="dataset identity mismatch"):
        run_sweep(
            [case],
            {},
            checkpoint_path=str(checkpoint),
            progress=False,
            dataset_identity=new,
        )


# ---------------------------------------------------------------------------
# cases_from_results
# ---------------------------------------------------------------------------


def test_cases_from_results_flat_list():
    case_a = _fake_case("cfg_a", 30.0)
    case_b = _fake_case("cfg_b", 45.0)
    results = {
        "cfg_a": {30.0: {"case": case_a}},
        "cfg_b": {45.0: {"case": case_b}},
    }
    cases = cases_from_results(results)
    assert len(cases) == 2
    assert {c["name"] for c in cases} == {"cfg_a", "cfg_b"}


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------


def test_run_sweep_stores_all_cases(tmp_path, monkeypatch):
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    cases = [
        _fake_case("cfg_a", 30.0),
        _fake_case("cfg_a", 45.0),
        _fake_case("cfg_b", 30.0),
    ]
    results = {}
    run_sweep(cases, results, checkpoint_dir=str(tmp_path), progress=False)
    assert "cfg_a" in results and "cfg_b" in results
    assert 30.0 in results["cfg_a"] and 45.0 in results["cfg_a"]
    assert 30.0 in results["cfg_b"]


def test_run_sweep_writes_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    run_sweep([_fake_case("cfg_a", 30.0)], {}, checkpoint_dir=str(tmp_path), progress=False)
    ckpt = tmp_path / "hopg"
    assert (ckpt / "line.h5").exists()
    assert (ckpt / "brem.h5").exists()
    saved = load_checkpoint("hopg", checkpoint_dir=str(tmp_path))
    assert "cfg_a" in saved


def test_run_sweep_consolidates_and_clears_shards(tmp_path, monkeypatch):
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    run_sweep([_fake_case("cfg_a", 30.0)], {}, checkpoint_dir=str(tmp_path), progress=False)
    # crash-safety shards are folded into the monolith and removed on a clean run
    assert not (tmp_path / "hopg" / "parts").exists()


def test_checkpoint_load_recovers_unconsolidated_shards(tmp_path):
    from pyrite.checkpoints import _checkpoint_store
    from pyrite.runs.run import _checkpoint_exists, _checkpoint_load

    # simulate a sweep killed after writing a shard but before consolidation:
    # no line.h5/brem.h5 monolith, only the parts directory
    _checkpoint_store.save_part(
        "hopg",
        tmp_path,
        "cfg_a",
        {30.0: {"case": _fake_case("cfg_a", 30.0), "spec": np.array([1.0])}},
    )
    ckpt = str(tmp_path / "hopg")
    assert not (tmp_path / "hopg" / "line.h5").exists()
    assert _checkpoint_exists(ckpt)
    loaded = _checkpoint_load(ckpt)
    assert 30.0 in loaded["cfg_a"]


def test_checkpoint_shards_win_over_stale_monolith(tmp_path):
    from pyrite.checkpoints import _checkpoint_store
    from pyrite.runs.run import _checkpoint_load

    # a monolith from a prior run holds cfg_a with an old spectrum
    stale = {"cfg_a": {30.0: {"case": _fake_case("cfg_a", 30.0), "spec": np.array([1.0])}}}
    _checkpoint_store.save("hopg", tmp_path, stale)
    # a newer shard supersedes cfg_a@30 and adds cfg_b
    _checkpoint_store.save_part(
        "hopg",
        tmp_path,
        "cfg_a",
        {30.0: {"case": _fake_case("cfg_a", 30.0), "spec": np.array([9.0])}},
    )
    _checkpoint_store.save_part(
        "hopg",
        tmp_path,
        "cfg_b",
        {45.0: {"case": _fake_case("cfg_b", 45.0), "spec": np.array([2.0])}},
    )
    loaded = _checkpoint_load(str(tmp_path / "hopg"))
    assert np.array_equal(loaded["cfg_a"][30.0]["spec"], np.array([9.0]))
    assert 45.0 in loaded["cfg_b"]


def test_run_sweep_splits_line_and_brem_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    run_sweep([_fake_case("cfg_a", 30.0)], {}, checkpoint_dir=str(tmp_path), progress=False)

    line = _checkpoint_io.load(str(tmp_path / "hopg" / "line.h5"))
    brem = _checkpoint_io.load(str(tmp_path / "hopg" / "brem.h5"))
    line_record = line["cfg_a"][30.0]
    brem_record = brem["cfg_a"][30.0]
    assert "spec" in line_record and "brem_wide" not in line_record
    assert "brem_wide" in brem_record and "spec" not in brem_record


def test_checkpoint_stores_characteristic_in_independent_component(tmp_path):
    case = _fake_case("cfg_a", 30.0)
    record = {
        "case": case,
        "E_grid": np.array([100.0, 110.0]),
        "spec": np.array([5.0, 7.0]),
        "spec_coherent": np.array([9.0, 11.0]),
        "spec_characteristic": np.array([1.0, 2.0]),
        "E_grid_brem": np.array([100.0, 110.0]),
        "brem_wide": np.array([0.5, 0.25]),
        "brem": np.array([0.5, 0.25]),
        "scale": 1.0,
    }
    results = {"cfg_a": {30.0: record}}

    _checkpoint_store.save("hopg", tmp_path, results)

    characteristic_path = tmp_path / "hopg" / "characteristic.h5"
    assert characteristic_path.is_file()
    line = _checkpoint_io.load(str(tmp_path / "hopg" / "line.h5"))["cfg_a"][30.0]
    characteristic = _checkpoint_io.load(str(characteristic_path))["cfg_a"][30.0]
    assert "spec_characteristic" not in line
    np.testing.assert_array_equal(line["spec"], [4.0, 5.0])
    np.testing.assert_array_equal(line["spec_coherent"], [8.0, 9.0])
    assert set(characteristic) == {"case", "E_grid", "scale", "spec_characteristic"}

    merged = _checkpoint_store.load("hopg", tmp_path)["cfg_a"][30.0]
    np.testing.assert_array_equal(merged["spec"], record["spec"])
    np.testing.assert_array_equal(merged["spec_coherent"], record["spec_coherent"])
    np.testing.assert_array_equal(merged["spec_characteristic"], record["spec_characteristic"])

    characteristic_path.unlink()
    without_characteristic = _checkpoint_store.load("hopg", tmp_path)["cfg_a"][30.0]
    assert "spec_characteristic" not in without_characteristic
    np.testing.assert_array_equal(without_characteristic["spec"], [4.0, 5.0])


def test_legacy_checkpoint_migrates_to_components_on_save(tmp_path, monkeypatch):
    # The default save creates all three current components, even when an older
    # record has no characteristic field.
    existing = {"cfg_a": {30.0: {"case": _fake_case("cfg_a", 30.0), "spec": np.array([1.0])}}}
    with open(tmp_path / "hopg.pkl", "wb") as f:
        pickle.dump(existing, f)
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)

    run_sweep(
        [_fake_case("cfg_a", 30.0), _fake_case("cfg_b", 30.0)],
        {},
        checkpoint_dir=str(tmp_path),
        resume=True,
        progress=False,
    )

    assert (tmp_path / "hopg" / "line.h5").is_file()
    assert (tmp_path / "hopg" / "characteristic.h5").is_file()
    assert set(load_checkpoint("hopg", checkpoint_dir=str(tmp_path))) == {"cfg_a", "cfg_b"}


def test_legacy_line_component_with_characteristic_is_not_double_added(tmp_path):
    record = {
        "case": _fake_case("cfg_a", 30.0),
        "E_grid": np.array([100.0, 110.0]),
        "spec": np.array([5.0, 7.0]),
        "spec_coherent": np.array([9.0, 11.0]),
        "spec_characteristic": np.array([1.0, 2.0]),
    }
    store = {"cfg_a": {30.0: record}}
    line_path = tmp_path / "hopg" / "line.h5"
    line_path.parent.mkdir()
    _checkpoint_io.dump(store, str(line_path))
    _checkpoint_store.save("hopg", tmp_path, store, components=("characteristic",))

    merged = _checkpoint_store.load("hopg", tmp_path)["cfg_a"][30.0]

    np.testing.assert_array_equal(merged["spec"], record["spec"])
    np.testing.assert_array_equal(merged["spec_coherent"], record["spec_coherent"])
    np.testing.assert_array_equal(merged["spec_characteristic"], record["spec_characteristic"])


def test_partial_component_save_fully_migrates_legacy_checkpoint(tmp_path):
    from pyrite.checkpoints import _checkpoint_store

    record = {
        "case": _fake_case("cfg_a", 30.0),
        "E_grid": np.array([1.0, 2.0]),
        "spec": np.array([3.0, 4.0]),
        "E_grid_brem": np.array([1.0, 2.0]),
        "brem_wide": np.array([5.0, 6.0]),
        "brem": np.array([5.0, 6.0]),
    }
    legacy = {"cfg_a": {30.0: record}}
    with open(tmp_path / "hopg.pkl", "wb") as f:
        pickle.dump(legacy, f)

    record["brem_wide"] = np.array([7.0, 8.0])
    _checkpoint_store.save("hopg", tmp_path, legacy, components=("brem",))

    assert (tmp_path / "hopg" / "line.h5").is_file()
    assert (tmp_path / "hopg" / "brem.h5").is_file()
    loaded = _checkpoint_store.load("hopg", tmp_path)
    assert np.array_equal(loaded["cfg_a"][30.0]["spec"], np.array([3.0, 4.0]))
    assert np.array_equal(loaded["cfg_a"][30.0]["brem_wide"], np.array([7.0, 8.0]))


def test_component_and_shard_pkl_payloads_remain_readable(tmp_path):
    record = {"cfg_a": {30.0: {"case": _fake_case("cfg_a", 30.0), "spec": np.array([1.0])}}}
    _checkpoint_store.save("hopg", tmp_path, record)
    for component in _checkpoint_store.COMPONENTS:
        path = _checkpoint_store.component_path("hopg", component, tmp_path)
        path.replace(path.with_suffix(".pkl"))
    _checkpoint_store.save_part(
        "hopg",
        tmp_path,
        "cfg_b",
        {45.0: {"case": _fake_case("cfg_b", 45.0), "spec": np.array([2.0])}},
    )
    shard = next((tmp_path / "hopg" / "parts").glob("*.h5"))
    shard.replace(shard.with_suffix(".pkl"))

    loaded = _checkpoint_store.load("hopg", tmp_path)
    assert set(loaded) == {"cfg_a", "cfg_b"}


def test_run_sweep_resume_skips_cached_cases(tmp_path, monkeypatch):
    existing = {"cfg_a": {30.0: {"case": _fake_case("cfg_a", 30.0), "spec": np.array([1.0])}}}
    with open(tmp_path / "hopg.pkl", "wb") as f:
        pickle.dump(existing, f)

    ran = []
    monkeypatch.setattr(
        "pyrite.runs.run.run_cases",
        tracking_run_cases_factory(ran),
    )
    cases = [_fake_case("cfg_a", 30.0), _fake_case("cfg_b", 30.0)]
    results = {}
    run_sweep(cases, results, checkpoint_dir=str(tmp_path), resume=True, progress=False)
    assert ran == ["cfg_b"]  # cfg_a was cached
    assert "cfg_a" in results and "cfg_b" in results


# ---------------------------------------------------------------------------
# Cross-profile content-addressable case reuse (--no-cache / --recompute gates)
# ---------------------------------------------------------------------------


def test_run_sweep_reuses_cases_across_stems_by_content_key(tmp_path, monkeypatch):
    """A case computed under one profile's stem is replayed under a DIFFERENT
    stem sharing the same material + content key, instead of recomputed."""
    from pyrite.campaign.profiles import case_content_key

    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    cases = [_fake_case("cfg_a", 30.0), _fake_case("cfg_b", 45.0)]

    # Profile A run: populates the shared per-material CAS.
    run_sweep(
        cases,
        {},
        checkpoint_path=str(tmp_path / "hopg@a-000000000000"),
        content_key_fn=case_content_key,
        progress=False,
    )
    material = cases[0]["crystal"]
    blobs = list((tmp_path / material).glob("*/*.h5"))
    assert len(blobs) == 2  # one blob per case

    # Profile B run: a different stem, same cases -> everything reused, run_cases
    # sees NOTHING to compute.
    ran = []
    monkeypatch.setattr("pyrite.runs.run.run_cases", tracking_run_cases_factory(ran))
    results_b = {}
    run_sweep(
        cases,
        results_b,
        checkpoint_path=str(tmp_path / "hopg@b-111111111111"),
        content_key_fn=case_content_key,
        progress=False,
    )
    assert ran == []  # all reused from the CAS
    assert results_b["cfg_a"][30.0]["spec"] is not None
    assert results_b["cfg_b"][45.0]["spec"] is not None
    # The reused record is rebuilt through store_result with the REQUESTED case.
    assert results_b["cfg_a"][30.0]["case"] is cases[0]


def test_no_cache_neither_reads_nor_writes(tmp_path, monkeypatch):
    from pyrite.campaign.profiles import case_content_key

    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    cases = [_fake_case("cfg_a", 30.0)]
    # Pre-populate the CAS from a normal run.
    run_sweep(
        cases,
        {},
        checkpoint_path=str(tmp_path / "hopg@a-000000000000"),
        content_key_fn=case_content_key,
        progress=False,
    )
    material = cases[0]["crystal"]
    before = {p.name for p in (tmp_path / material).glob("*/*.h5")}
    assert before

    ran = []
    monkeypatch.setattr("pyrite.runs.run.run_cases", tracking_run_cases_factory(ran))
    run_sweep(
        cases,
        {},
        checkpoint_path=str(tmp_path / "hopg@b-111111111111"),
        content_key_fn=case_content_key,
        cache_read=False,
        cache_write=False,
        resume=False,
        progress=False,
    )
    assert ran == ["cfg_a"]  # no read -> recomputed
    after = {p.name for p in (tmp_path / material).glob("*/*.h5")}
    assert after == before  # no write -> CAS untouched
    assert not (tmp_path / "hopg@b-111111111111" / "cases.json").exists()  # ephemeral


def test_recompute_skips_read_but_repopulates(tmp_path, monkeypatch):
    from pyrite.campaign.profiles import case_content_key

    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    cases = [_fake_case("cfg_a", 30.0)]
    run_sweep(
        cases,
        {},
        checkpoint_path=str(tmp_path / "hopg@a-000000000000"),
        content_key_fn=case_content_key,
        progress=False,
    )
    material = cases[0]["crystal"]
    blob = next((tmp_path / material).glob("*/*.h5"))
    blob.unlink()  # remove so we can prove --recompute rewrites it

    ran = []
    monkeypatch.setattr("pyrite.runs.run.run_cases", tracking_run_cases_factory(ran))
    run_sweep(
        cases,
        {},
        checkpoint_path=str(tmp_path / "hopg@b-111111111111"),
        content_key_fn=case_content_key,
        cache_read=False,
        cache_write=True,
        resume=False,
        progress=False,
    )
    assert ran == ["cfg_a"]  # skipped read -> recomputed
    assert list((tmp_path / material).glob("*/*.h5"))  # write -> repopulated
    assert (tmp_path / "hopg@b-111111111111" / "cases.json").exists()  # manifest written


def test_existing_checkpoint_seeds_shared_cache_on_resume(tmp_path, monkeypatch):
    from pyrite.campaign.profiles import case_content_key

    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    cases = [_fake_case("cfg_a", 30.0)]
    stem = tmp_path / "hopg@legacy-000000000000"
    run_sweep(cases, {}, checkpoint_path=str(stem), progress=False)
    assert not list((tmp_path / "hopg").glob("*/*.h5"))

    run_sweep(
        cases,
        {},
        checkpoint_path=str(stem),
        content_key_fn=case_content_key,
        progress=False,
    )
    assert len(list((tmp_path / "hopg").glob("*/*.h5"))) == 1


def test_invalid_cached_payload_falls_back_to_recompute(tmp_path, monkeypatch):
    from pyrite.campaign.profiles import case_content_key
    from pyrite.checkpoints import _checkpoint_io

    cases = [_fake_case("cfg_a", 30.0)]
    key = case_content_key(cases[0])
    blob = tmp_path / "hopg" / key[:2] / f"{key}.pkl"
    blob.parent.mkdir(parents=True)
    _checkpoint_io.dump({"not": "a case payload"}, str(blob))
    ran = []
    monkeypatch.setattr("pyrite.runs.run.run_cases", tracking_run_cases_factory(ran))

    run_sweep(
        cases,
        {},
        checkpoint_path=str(tmp_path / "hopg@new-111111111111"),
        content_key_fn=case_content_key,
        progress=False,
    )
    assert ran == ["cfg_a"]


def test_cas_rejects_non_sha256_content_key(tmp_path):
    from pyrite.checkpoints import _checkpoint_store

    with pytest.raises(ValueError, match="64-character SHA-256"):
        _checkpoint_store.cas_blob_path("hopg", "../escape", tmp_path)


def test_content_key_fn_none_leaves_cas_inert(tmp_path, monkeypatch):
    """Without content_key_fn the CAS path is entirely dormant (byte-identical to
    the pre-feature per-stem-only behavior): no blobs, no cases.json."""
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    cases = [_fake_case("cfg_a", 30.0)]
    run_sweep(cases, {}, checkpoint_path=str(tmp_path / "hopg"), progress=False)
    assert not list((tmp_path / "hopg").glob("*/*.h5"))  # no sharded CAS blobs
    assert not (tmp_path / "hopg" / "cases.json").exists()
    # only the component store + its meta sidecar exist
    assert (tmp_path / "hopg" / "line.h5").exists()


def test_run_sweep_reports_initial_and_per_case_progress(tmp_path, monkeypatch):
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    cases = [
        _fake_case("cfg_a", 30.0),
        _fake_case("cfg_a", 45.0),
        _fake_case("cfg_b", 30.0),
    ]
    progress = []

    run_sweep(
        cases,
        {},
        checkpoint_dir=str(tmp_path),
        progress=False,
        on_progress=lambda completed, total, cached: progress.append((completed, total, cached)),
    )

    assert progress == [(0, 3, 0), (1, 3, 0), (2, 3, 0), (3, 3, 0)]


def test_run_sweep_on_case_fires_with_each_finished_case(tmp_path, monkeypatch):
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    cases = [_fake_case("cfg_a", 30.0), _fake_case("cfg_b", 45.0)]
    seen = []

    run_sweep(
        cases,
        {},
        checkpoint_dir=str(tmp_path),
        progress=False,
        on_case=lambda case: seen.append((case["name"], case["E0_keV"])),
    )

    assert seen == [("cfg_a", 30.0), ("cfg_b", 45.0)]


def test_run_sweep_progress_counts_cached_cases_on_resume(tmp_path, monkeypatch):
    cached_case = _fake_case("cfg_a", 30.0)
    existing = {"cfg_a": {30.0: {"case": cached_case, "spec": np.array([1.0])}}}
    with open(tmp_path / "hopg.pkl", "wb") as f:
        pickle.dump(existing, f)
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    progress = []

    run_sweep(
        [cached_case, _fake_case("cfg_b", 30.0)],
        {},
        checkpoint_dir=str(tmp_path),
        progress=False,
        on_progress=lambda completed, total, cached: progress.append((completed, total, cached)),
    )

    assert progress == [(0, 2, 1), (1, 2, 1)]


def test_run_sweep_on_cost_reports_cached_seed_and_per_case_totals(tmp_path, monkeypatch):
    """item 6: on_cost seeds the cached case's exact cost (identity, not just a
    count) then accumulates each newly completed case's cost -- see
    scan._run_material, which wires sweep.case_cost through this callback into
    the remote progress JSON's done_cost/total_cost fields."""
    cached_case = _fake_case("cfg_a", 30.0)
    existing = {"cfg_a": {30.0: {"case": cached_case, "spec": np.array([1.0])}}}
    with open(tmp_path / "hopg.pkl", "wb") as f:
        pickle.dump(existing, f)
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    cost = []

    run_sweep(
        [cached_case, _fake_case("cfg_b", 30.0)],
        {},
        checkpoint_dir=str(tmp_path),
        progress=False,
        case_cost_fn=lambda case: 1.0 if case["name"] == "cfg_a" else 3.0,
        on_cost=lambda done, total: cost.append((done, total)),
    )

    # total = 1 (cfg_a) + 3 (cfg_b) = 4; cfg_a is cached so the initial call
    # already seeds done_cost=1, not 0.
    assert cost == [(1.0, 4.0), (4.0, 4.0)]


def test_run_sweep_on_cost_never_fires_without_case_cost_fn(tmp_path, monkeypatch):
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    cost = []

    run_sweep(
        [_fake_case("cfg_a", 30.0)],
        {},
        checkpoint_dir=str(tmp_path),
        progress=False,
        on_cost=lambda done, total: cost.append((done, total)),
    )

    assert cost == []


def test_run_sweep_on_chunk_fires_per_group(tmp_path, monkeypatch):
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    # same (crystal, thickness, tilt, omitted footprint) -> one group;
    # different tilt -> another
    c1 = _fake_case("cfg_a", 30.0, tilt_deg=30.0)
    c2 = _fake_case("cfg_b", 30.0, tilt_deg=30.0)
    c3 = _fake_case("cfg_c", 30.0, tilt_deg=0.0)
    chunks = []
    run_sweep(
        [c1, c2, c3], {}, checkpoint_dir=str(tmp_path), on_chunk=chunks.append, progress=False
    )
    assert len(chunks) == 2
    tilt_group = next(ch for ch in chunks if "cfg_a" in ch)
    assert set(tilt_group) == {"cfg_a", "cfg_b"}


def test_run_sweep_separates_finite_footprint_chunk_groups(tmp_path, monkeypatch):
    """Finite footprints need independent streaming groups at one tilt."""
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    narrow = _fake_case("narrow", 30.0, tilt_deg=30.0)
    wide = _fake_case("wide", 30.0, tilt_deg=30.0)
    narrow.update(crystal_width_mm=0.1, crystal_height_mm=0.2)
    wide.update(crystal_width_mm=0.3, crystal_height_mm=0.2)

    chunks = []
    run_sweep(
        [narrow, wide], {}, checkpoint_dir=str(tmp_path), on_chunk=chunks.append, progress=False
    )

    assert {frozenset(chunk) for chunk in chunks} == {frozenset({"narrow"}), frozenset({"wide"})}


def test_run_sweep_resume_replays_cached_chunks(tmp_path, monkeypatch):
    case = _fake_case("cfg_a", 30.0)
    existing = {"cfg_a": {30.0: {"case": case, "spec": np.array([1.0])}}}
    with open(tmp_path / "hopg.pkl", "wb") as f:
        pickle.dump(existing, f)
    monkeypatch.setattr("pyrite.runs.run.run_cases", stub_run_cases)
    chunks = []
    run_sweep(
        [case],
        {},
        checkpoint_dir=str(tmp_path),
        resume=True,
        on_chunk=chunks.append,
        progress=False,
    )
    assert chunks == [["cfg_a"]]  # fully-cached group replayed immediately


def test_run_sweep_budget_stops_early_and_resumes(tmp_path, monkeypatch):
    from pyrite.runs import run as run_mod

    clock = {"t": 0.0}

    def fake_run_cases(
        todo,
        *,
        callback=None,
        should_stop=None,
        keep_results=True,
        **_unused,
    ):
        for i, c in enumerate(todo):
            if should_stop is not None and should_stop():
                break
            clock["t"] += 10.0  # each case takes 10 "seconds"
            callback(i, c, {"out": c["name"]})
        return []

    monkeypatch.setattr(run_mod, "run_cases", fake_run_cases)
    monkeypatch.setattr(
        run_mod,
        "store_result",
        lambda results, case, out: results.setdefault(case["name"], {}).__setitem__(
            case["E0_keV"], {"case": case}
        ),
    )
    cases = [
        {
            "name": f"cfg{i}",
            "E0_keV": 30,
            "crystal": "hopg",
            "thickness_ang": 1e4,
            "tilt_deg": 0.0,
        }
        for i in range(4)
    ]
    ckpt = tmp_path / "hopg.pkl"

    results = {}
    complete = run_mod.run_sweep(
        cases,
        results,
        checkpoint_path=str(ckpt),
        max_seconds=25.0,
        time_fn=lambda: clock["t"],
    )
    assert complete is False
    assert len(results) == 3  # budget passed after case 3 (t=30 > 25)
    # the final save persisted the partial state: a fresh resume skips them
    results2 = {}
    complete2 = run_mod.run_sweep(
        cases,
        results2,
        checkpoint_path=str(ckpt),
        max_seconds=1000.0,
        time_fn=lambda: clock["t"],
    )
    assert complete2 is True
    assert len(results2) == 4


def test_run_sweep_no_budget_returns_complete(tmp_path, monkeypatch):
    from pyrite.runs import run as run_mod

    clock = {"t": 0.0}

    def fake_run_cases(
        todo,
        *,
        callback=None,
        should_stop=None,
        keep_results=True,
        **_unused,
    ):
        results = [None] * len(todo)

        for i, case in enumerate(todo):
            if should_stop is not None and should_stop():
                break

            clock["t"] += 10.0
            out = {"out": case["name"]}

            if callback is not None:
                callback(i, case, out)

            if keep_results:
                results[i] = out

        return results

    monkeypatch.setattr(run_mod, "run_cases", fake_run_cases)

    monkeypatch.setattr(
        run_mod,
        "store_result",
        lambda results, case, out: results.setdefault(case["name"], {}).__setitem__(
            case["E0_keV"],
            {"case": case},
        ),
    )

    cases = [
        {
            "name": f"cfg{i}",
            "E0_keV": 30,
            "crystal": "hopg",
            "thickness_ang": 1e4,
            "tilt_deg": 0.0,
        }
        for i in range(4)
    ]

    results = {}

    complete = run_mod.run_sweep(
        cases,
        results,
        checkpoint_path=str(tmp_path / "hopg.pkl"),
        max_seconds=None,
    )

    assert complete is True
    assert len(results) == 4
    assert clock["t"] == 40.0


# ---------------------------------------------------------------------------
# repair_brem_wide
# ---------------------------------------------------------------------------


def test_repair_brem_wide_skips_already_finite():
    E = np.arange(100.0, 200.0, 10.0)
    Eb = np.arange(0.0, 500.0, 50.0)
    record = dict(
        E_grid=E,
        spec=np.ones_like(E),
        brem=np.ones_like(E) * 0.01,
        E_grid_brem=Eb,
        brem_wide=np.full_like(Eb, 0.001),
        eta=0.05,
        scale=1.0,
        case=_fake_case("cfg_a", 30.0),
    )
    n = repair_brem_wide({"cfg_a": {30.0: record}}, only_nonfinite=True, progress=False)
    assert n == 0


def test_repair_brem_wide_recomputes_all_zero_placeholder(monkeypatch):
    E = np.arange(100.0, 200.0, 10.0)
    Eb = np.arange(0.0, 500.0, 50.0)
    record = dict(
        E_grid=E,
        spec=np.ones_like(E),
        brem=np.zeros_like(E),
        E_grid_brem=Eb,
        brem_wide=np.zeros_like(Eb),
        eta=0.05,
        scale=1.0,
        case=_fake_case("cfg_a", 30.0),
    )
    monkeypatch.setattr(
        "pyrite.montecarlo._brem_for_case",
        lambda c, E_brem: np.full(np.asarray(E_brem, float).shape, 0.002),
    )

    n = repair_brem_wide({"cfg_a": {30.0: record}}, only_nonfinite=True, progress=False)

    assert n == 1
    assert np.allclose(record["brem_wide"], 0.002)
    assert np.allclose(record["brem"], 0.002)


def test_repair_brem_wide_delegates_stacked_case_to_runner(monkeypatch):
    """A stale film-on-substrate record is repaired through the runner's
    ``_brem_for_case`` (the live-sweep path), which is handed the case's full
    ``abs_layers`` stack -- NOT the old hand-rolled single-slab brem that
    dropped ``layers=``. Regression for the multilayer-drift bug (H1)."""
    Eb = np.arange(0.0, 500.0, 50.0)
    case = _fake_case("cfg_a", 30.0)
    # two-layer stack: (name, thickness_ang, composition) -- what the old repair
    # path silently ignored, regenerating brem for the film slab alone.
    case["abs_layers"] = [
        ("film", 1e4, [("Mo", 1.0), ("S", 2.0)]),
        ("substrate", 5e6, [("Al", 2.0), ("O", 3.0)]),
    ]
    record = dict(
        E_grid=np.arange(*case["E_grid"]),
        spec=np.ones(10),
        brem=np.ones(10) * 0.01,
        E_grid_brem=Eb,
        brem_wide=np.full_like(Eb, np.nan),  # stale -> selected for repair
        eta=0.05,
        scale=1.0,
        case=case,
    )
    seen: dict[str, Any] = {}

    def _spy(c, E_brem):
        seen["abs_layers"] = c["abs_layers"]
        return np.full(np.asarray(E_brem, float).shape, 0.002)

    # run.repair_brem_wide does `from .montecarlo import _brem_for_case` at call
    # time, so patch the name on the package it resolves against.
    monkeypatch.setattr("pyrite.montecarlo._brem_for_case", _spy)
    n = repair_brem_wide({"cfg_a": {30.0: record}}, only_nonfinite=True, progress=False)
    assert n == 1
    assert seen["abs_layers"] == case["abs_layers"]  # full stack reached the runner
    assert np.allclose(record["brem_wide"], 0.002)  # repaired in place


def test_repair_brem_wide_frees_gpu_pool_per_record(monkeypatch):
    """rebrem's repair loop must hand the CuPy pool back to the card between
    records, the SAME way the live sweep's _spectrum_case does -- else the
    reserved pool grows and fragments across hundreds of records until VRAM
    fills (Task 8 GPU-memory regression). One guarded free per repaired record."""
    Eb = np.arange(0.0, 500.0, 50.0)

    def _stale(cfg, e0):
        return dict(
            E_grid=np.arange(100.0, 200.0, 10.0),
            spec=np.ones(10),
            brem=np.ones(10) * 0.01,
            E_grid_brem=Eb,
            brem_wide=np.full_like(Eb, np.nan),  # nonfinite -> selected for repair
            eta=0.05,
            scale=1.0,
            case=_fake_case(cfg, e0),
        )

    results = {
        "cfg_a": {30.0: _stale("cfg_a", 30.0), 40.0: _stale("cfg_a", 40.0)},
        "cfg_b": {50.0: _stale("cfg_b", 50.0)},
    }
    # keep the repair off the GPU: dummy brem, no real transport/spectrum.
    monkeypatch.setattr(
        "pyrite.montecarlo._brem_for_case",
        lambda c, E_brem: np.full(np.asarray(E_brem, float).shape, 0.002),
    )
    # pretend a live GPU so the guarded inter-case cadence path executes, and
    # count how often the pool is released instead of touching CuPy.
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    freed = {"n": 0}
    monkeypatch.setattr(runner, "_maybe_free_pool", lambda: freed.__setitem__("n", freed["n"] + 1))

    n = repair_brem_wide(results, only_nonfinite=True, progress=False)
    assert n == 3  # every stale record repaired
    assert freed["n"] == 3  # ...and the pool released once per repaired record


def test_repair_brem_wide_skips_pool_free_off_gpu(monkeypatch):
    """CPU-default path (runner._RESOURCE_POLICY.gpu False) must NOT touch the pool -- the free
    is guarded, so the byte-identical local repair never calls into CuPy."""
    Eb = np.arange(0.0, 500.0, 50.0)
    record = dict(
        E_grid=np.arange(100.0, 200.0, 10.0),
        spec=np.ones(10),
        brem=np.ones(10) * 0.01,
        E_grid_brem=Eb,
        brem_wide=np.full_like(Eb, np.nan),
        eta=0.05,
        scale=1.0,
        case=_fake_case("cfg_a", 30.0),
    )
    monkeypatch.setattr(
        "pyrite.montecarlo._brem_for_case",
        lambda c, E_brem: np.full(np.asarray(E_brem, float).shape, 0.002),
    )
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", False)

    def _boom():
        raise AssertionError("pool free must not run off-GPU")

    monkeypatch.setattr(runner, "_maybe_free_pool", _boom)
    n = repair_brem_wide({"cfg_a": {30.0: record}}, only_nonfinite=True, progress=False)
    assert n == 1


def test_lines_for_case_matches_spectrum_case_single_slab():
    import numpy as np

    from pyrite.montecarlo import _lines_for_case, runner

    case = dict(
        E0_keV=30.0,
        Ne=4,
        Ne_brem=2,
        thickness_ang=1.0e4,
        composition=[("Mo", 1.0), ("S", 2.0)],
        crystal="mos2",
        hkl_list=[(1, 0, 0)],
        B_ang2=0.5,
        theta_obs_rad=np.deg2rad(90.0),
        seed=7,
        E_cut_lines_keV=5.0,
        E_cut_brem_keV=1.0,
    )
    E_grid = np.linspace(1000.0, 30000.0, 16)
    tp = runner._transport_case({**case, "E_grid": E_grid})
    spec_ref = runner._spectrum_case({**case, "E_grid": E_grid}, tp)["spec"]
    spec = _lines_for_case(case, E_grid)
    assert spec.shape == E_grid.shape
    np.testing.assert_allclose(spec, spec_ref, rtol=0, atol=0)


def test_repair_brem_wide_preserves_exact_nonuniform_case_grid(monkeypatch):
    case = _fake_case("cfg_a", 30.0)
    case["E_grid_brem"] = np.array([10.0, 100.0, 1000.0])
    record = dict(
        E_grid=np.arange(*case["E_grid"]),
        spec=np.ones(10),
        brem=np.ones(10) * 0.01,
        E_grid_brem=np.array([10.0, 100.0, 1000.0]),
        brem_wide=np.full(3, np.nan),
        eta=0.05,
        scale=1.0,
        case=case,
    )
    seen: dict[str, np.ndarray] = {}

    def _spy(_case, E_brem):
        seen["E_brem"] = np.asarray(E_brem, float)
        return np.ones(3)

    monkeypatch.setattr("pyrite.montecarlo._brem_for_case", _spy)

    assert repair_brem_wide({"cfg_a": {30.0: record}}, only_nonfinite=True, progress=False) == 1
    np.testing.assert_array_equal(seen["E_brem"], [10.0, 100.0, 1000.0])
    np.testing.assert_array_equal(record["E_grid_brem"], [10.0, 100.0, 1000.0])


def _finite_record(case):
    Eb = np.arange(*case["E_grid_brem"])
    return dict(
        E_grid=np.arange(*case["E_grid"]),
        spec=np.ones(10),
        brem=np.ones(10) * 0.01,
        E_grid_brem=Eb,
        brem_wide=np.full_like(Eb, 0.001),
        eta=0.05,
        scale=1.0,
        case=case,
    )


def test_repair_brem_wide_retunes_finite_record_and_persists_params(monkeypatch):
    """New ne_brem/brem_step_eV select an all-finite record (parameter mismatch),
    rebuild the grid at the sweep-convention stop, and persist both into the
    case so a re-run skips it (resumability)."""
    record = _finite_record(_fake_case("cfg_a", 30.0))  # Ne_brem=5, step 50 eV
    seen: dict[str, Any] = {}

    def _spy(c, E_brem):
        seen["Ne_brem"] = c["Ne_brem"]
        seen["E_brem"] = np.asarray(E_brem, float)
        return np.full(np.asarray(E_brem, float).shape, 0.002)

    monkeypatch.setattr("pyrite.montecarlo._brem_for_case", _spy)
    results = {"cfg_a": {30.0: record}}
    n = repair_brem_wide(results, progress=False, ne_brem=1000, brem_step_eV=25.0)
    assert n == 1
    assert seen["Ne_brem"] == 1000
    # grid: old start (0.0) -> E0*1e3 + step, at the new spacing
    np.testing.assert_allclose(seen["E_brem"], np.arange(0.0, 30_025.0, 25.0))
    case = record["case"]
    assert case["Ne_brem"] == 1000
    assert case["E_grid_brem"] == (0.0, 30_025.0, 25.0)
    np.testing.assert_array_equal(record["E_grid_brem"], seen["E_brem"])
    assert np.allclose(record["brem_wide"], 0.002)
    # brem re-interpolated onto the line grid from the new wide curve
    assert np.allclose(record["brem"], 0.002)
    # re-run at the same target: already-at-target record is skipped
    assert repair_brem_wide(results, progress=False, ne_brem=1000, brem_step_eV=25.0) == 0


def test_repair_brem_wide_persists_profile_and_explicit_bounds(monkeypatch):
    record = _finite_record(_fake_case("cfg_a", 30.0))
    monkeypatch.setattr(
        "pyrite.montecarlo._brem_for_case",
        lambda _case, grid: np.ones(np.asarray(grid).shape),
    )

    n = repair_brem_wide(
        {"cfg_a": {30.0: record}},
        progress=False,
        ne_brem=30,
        brem_start_eV=100.0,
        brem_stop_eV=500.0,
        brem_step_eV=100.0,
        fidelity="survey",
    )

    assert n == 1
    np.testing.assert_array_equal(record["E_grid_brem"], [100.0, 200.0, 300.0, 400.0])
    assert record["case"]["E_grid_brem"] == (100.0, 500.0, 100.0)
    assert record["case"]["Ne_brem"] == 30
    assert record["case"]["brem_profile"] == "survey"


def test_repair_brem_wide_retune_skips_record_already_at_target():
    record = _finite_record(_fake_case("cfg_a", 30.0))
    n = repair_brem_wide({"cfg_a": {30.0: record}}, progress=False, ne_brem=5, brem_step_eV=50.0)
    assert n == 0  # Ne_brem and spacing already match; nothing recomputed


def test_rebrem_checkpoints_enumerates_pkls_and_passes_params(monkeypatch, tmp_path):
    from pyrite.checkpoints.recompute import rebrem_checkpoints

    (tmp_path / "MoS2.pkl").write_bytes(b"")
    (tmp_path / "W_grooved.pkl").write_bytes(b"")
    (tmp_path / "MoS2.slim.pkl").write_bytes(b"")  # transfer copy: excluded
    calls: list[tuple[str, dict[str, Any]]] = []

    def _spy(path, **kw):
        calls.append((path, kw))
        return {}

    monkeypatch.setattr("pyrite.checkpoints.recompute.repair_checkpoint", _spy)
    out = rebrem_checkpoints(checkpoint_dir=str(tmp_path), ne_brem=1000, brem_step_eV=25.0)
    assert sorted(out) == ["MoS2", "W_grooved"]
    assert all(kw["ne_brem"] == 1000 and kw["brem_step_eV"] == 25.0 for _, kw in calls)
    assert all(kw["only_nonfinite"] is True for _, kw in calls)  # resumable default

    calls.clear()
    rebrem_checkpoints(
        materials=["MoS2"], checkpoint_dir=str(tmp_path), ne_brem=1000, redo_all=True
    )
    assert [Path(p).name for p, _ in calls] == ["MoS2"]
    assert calls[0][1]["only_nonfinite"] is False


def test_rebrem_profile_defaults_match_for_explicit_materials_and_all(monkeypatch, tmp_path):
    from pyrite.checkpoints import recompute as rebrem

    for material in ("hopg", "hbn"):
        (tmp_path / f"{material}.pkl").write_bytes(b"")
    calls = []
    monkeypatch.setattr(
        "pyrite.checkpoints.recompute.repair_checkpoint",
        lambda path, **kwargs: calls.append((Path(path).name, kwargs)) or {},
    )

    rebrem.rebrem_checkpoints(materials=["hopg", "hbn"], checkpoint_dir=tmp_path, fidelity="survey")
    explicit = calls.copy()
    calls.clear()
    rebrem.rebrem_checkpoints(checkpoint_dir=tmp_path, fidelity="survey")

    assert {name: kwargs for name, kwargs in calls} == {name: kwargs for name, kwargs in explicit}
    assert {kwargs["ne_brem"] for _, kwargs in calls} == {30}
    assert {kwargs["fidelity"] for _, kwargs in calls} == {"survey"}


def test_rebrem_cli_requires_materials_xor_all(monkeypatch):
    """`pyrite checkpoint recompute brem` refuses no-selection and materials+--all;
    accepts either alone."""
    from pyrite.checkpoints import recompute
    from pyrite.cli.commands.recompute import brem_command

    seen: list[Any] = []
    monkeypatch.setattr(recompute, "rebrem_checkpoints", lambda **kw: seen.append(kw) or {})
    runner = CliRunner()

    neither = runner.invoke(brem_command, ["--ne-brem", "1000"], catch_exceptions=False)
    both = runner.invoke(brem_command, ["MoS2", "--all"], catch_exceptions=False)
    assert neither.exit_code == 2
    assert both.exit_code == 2
    assert "needs material name(s)" in neither.stderr
    assert "--all does not take material names" in both.stderr
    assert seen == []

    material = runner.invoke(brem_command, ["MoS2"], catch_exceptions=False)
    assert material.exit_code == 0
    assert material.stderr == ""
    assert seen[-1]["materials"] == ["MoS2"]

    all_materials = runner.invoke(brem_command, ["-a"], catch_exceptions=False)
    assert all_materials.exit_code == 0
    assert all_materials.stderr == ""
    assert seen[-1]["materials"] is None


def test_reline_cli_requires_materials_xor_all(monkeypatch):
    """`pyrite checkpoint recompute line` preserves brem's exclusive
    material-selection contract."""
    from pyrite.checkpoints import recompute
    from pyrite.cli.commands.recompute import line_command

    seen: list[Any] = []
    monkeypatch.setattr(recompute, "reline_checkpoints", lambda **kw: seen.append(kw) or {})
    runner = CliRunner()

    neither = runner.invoke(line_command, [], catch_exceptions=False)
    both = runner.invoke(line_command, ["MoS2", "--all"], catch_exceptions=False)
    assert neither.exit_code == 2
    assert both.exit_code == 2
    assert "needs material name(s)" in neither.stderr
    assert "--all does not take material names" in both.stderr
    assert seen == []

    material = runner.invoke(line_command, ["MoS2"], catch_exceptions=False)
    assert material.exit_code == 0
    assert material.stderr == ""
    assert seen[-1]["materials"] == ["MoS2"]

    all_materials = runner.invoke(line_command, ["-a"], catch_exceptions=False)
    assert all_materials.exit_code == 0
    assert all_materials.stderr == ""
    assert seen[-1]["materials"] is None


def test_repair_brem_wide_on_progress_reports_skipped_and_done(monkeypatch):
    """on_progress fires with (done, todo_total, skipped): one record already at
    target counts as skipped, the stale one ticks through the loop."""
    at_target = _finite_record(_fake_case("cfg_a", 30.0))  # Ne_brem=5, 50 eV: matches
    stale = _finite_record(_fake_case("cfg_b", 30.0))
    stale["case"]["Ne_brem"] = 7  # mismatch -> selected
    monkeypatch.setattr(
        "pyrite.montecarlo._brem_for_case",
        lambda c, E_brem: np.ones(np.asarray(E_brem, float).shape),
    )
    calls: list[tuple[int, int, int]] = []
    n = repair_brem_wide(
        {"cfg_a": {30.0: at_target}, "cfg_b": {30.0: stale}},
        progress=False,
        ne_brem=5,
        brem_step_eV=50.0,
        on_progress=lambda *a: calls.append(a),
    )
    assert n == 1
    assert calls == [(0, 1, 1), (1, 1, 1)]


def test_repair_brem_wide_max_seconds_stops_early(monkeypatch):
    """An immediate deadline (max_seconds=0.0) breaks before the sole stale
    record: returns 0 and marks status incomplete (mirror of the reline path)."""
    stale = _finite_record(_fake_case("cfg_a", 30.0))
    stale["case"]["Ne_brem"] = 7  # parameter mismatch -> selected for repair
    monkeypatch.setattr(
        "pyrite.montecarlo._brem_for_case",
        lambda c, E_brem: np.ones(np.asarray(E_brem, float).shape),
    )
    n = repair_brem_wide(
        {"cfg_a": {30.0: stale}},
        progress=False,
        ne_brem=5,
        brem_step_eV=50.0,
        max_seconds=0.0,
        status=(s := {}),
    )
    assert n == 0
    assert s["complete"] is False


def test_rebrem_checkpoints_progress_file_writes_dashboard_records(monkeypatch, tmp_path):
    from pyrite.checkpoints.recompute import rebrem_checkpoints

    (tmp_path / "hopg.pkl").write_bytes(b"")
    progress = tmp_path / "hopg.json"
    states: list[dict[str, Any]] = []

    def _spy(path, **kw):
        kw["on_progress"](2, 5, 3)  # mid-run tick
        return {}

    def _snoop_write(path):
        states.append(json.loads(Path(path).read_text()))

    monkeypatch.setattr("pyrite.checkpoints.recompute.repair_checkpoint", _spy)
    rebrem_checkpoints(
        materials=["hopg"], checkpoint_dir=str(tmp_path), ne_brem=1000, progress_file=str(progress)
    )
    final = json.loads(progress.read_text())
    assert final["state"] == "done"
    assert final["material"] == "hopg"
    assert final["total_cases"] == 8  # todo 5 + skipped 3
    assert final["cached_cases"] == 3 and final["completed_new_cases"] == 2


def test_rebrem_checkpoints_progress_file_marks_failed_and_rejects_multi(monkeypatch, tmp_path):
    from pyrite.checkpoints.recompute import rebrem_checkpoints

    (tmp_path / "hopg.pkl").write_bytes(b"")
    (tmp_path / "hbn.pkl").write_bytes(b"")
    progress = tmp_path / "p.json"

    with pytest.raises(SystemExit, match="exactly one material"):
        rebrem_checkpoints(
            materials=["hopg", "hbn"], checkpoint_dir=str(tmp_path), progress_file=str(progress)
        )

    def _boom(path, **kw):
        raise RuntimeError("kaput")

    monkeypatch.setattr("pyrite.checkpoints.recompute.repair_checkpoint", _boom)
    with pytest.raises(RuntimeError):
        rebrem_checkpoints(
            materials=["hopg"], checkpoint_dir=str(tmp_path), progress_file=str(progress)
        )
    assert json.loads(progress.read_text())["state"] == "failed"


# ---------------------------------------------------------------------------
# repair_line_spec / reline_checkpoint
# ---------------------------------------------------------------------------


def _line_record(E0=30.0, ne=200):
    import numpy as np

    E_grid = np.linspace(1000.0, 30000.0, 32)
    E_brem = np.linspace(1000.0, 30500.0, 16)
    brem_wide = np.linspace(1.0, 0.1, 16)
    case = dict(
        E0_keV=E0,
        Ne=ne,
        Ne_brem=50,
        thickness_ang=1.0e4,
        composition={"Mo": 1, "S": 2},
        crystal="mos2",
        hkl_list=[(1, 0, 0)],
        B_ang2=0.5,
        theta_obs_rad=np.deg2rad(90.0),
        seed=7,
    )
    return {
        "case": case,
        "E_grid": E_grid,
        "spec": np.ones(32),
        "brem_wide": brem_wide,
        "E_grid_brem": E_brem,
        "brem": np.interp(E_grid, E_brem, brem_wide),
    }


def test_repair_line_spec_rewrites_spec_and_reinterp_brem_keeps_brem_wide(monkeypatch):
    import numpy as np

    from pyrite.runs import run

    monkeypatch.setattr(
        run.runner,
        "_line_pair_for_case",
        lambda case, E_grid, *, want_coherent, return_characteristic: (
            np.full(E_grid.shape, 5.0),
            None,
            np.full(E_grid.shape, 1.0),
        ),
    )
    results = {"mos2@30": {30.0: _line_record()}}
    r = results["mos2@30"][30.0]
    brem_wide0 = r["brem_wide"].copy()
    n = run.repair_line_spec(results, material="mos2", line_ne=999, from_config=False)
    assert n == 1
    assert np.all(r["spec"] == 5.0)
    assert np.all(r["spec_characteristic"] == 1.0)
    assert r["case"]["Ne"] == 999
    np.testing.assert_array_equal(r["brem_wide"], brem_wide0)  # brem untouched
    np.testing.assert_allclose(r["brem"], np.interp(r["E_grid"], r["E_grid_brem"], r["brem_wide"]))


def test_repair_line_spec_refreshes_both_spec_and_spec_coherent(monkeypatch):
    """A coherent/both checkpoint (record carries `spec_coherent`) relines BOTH
    arrays onto the new grid from one re-transport, so `spec_coherent` never
    goes stale relative to `spec`/`E_grid`; an incoherent record only gets
    `spec` and grows no `spec_coherent`."""
    import numpy as np

    from pyrite.runs import run

    def fake_pair(case, E_grid, *, want_coherent, return_characteristic):
        spec = np.full(E_grid.shape, 5.0)
        return (
            spec,
            (np.full(E_grid.shape, 9.0) if want_coherent else None),
            np.full(E_grid.shape, 1.0),
        )

    monkeypatch.setattr(run.runner, "_line_pair_for_case", fake_pair)

    both = _line_record()
    both["spec_coherent"] = np.zeros(32)  # stale placeholder to be overwritten
    incoh = _line_record()
    results = {"both@30": {30.0: both}, "incoh@30": {30.0: incoh}}

    n = run.repair_line_spec(results, material="mos2", line_ne=999, from_config=False)

    assert n == 2
    assert np.all(both["spec"] == 5.0)
    assert np.all(both["spec_coherent"] == 9.0)
    assert both["spec"].shape == both["spec_coherent"].shape == both["E_grid"].shape
    assert np.all(incoh["spec"] == 5.0)
    assert "spec_coherent" not in incoh


def test_repair_line_spec_persists_profile_and_explicit_bounds(monkeypatch):
    from pyrite.runs import run

    monkeypatch.setattr(
        run.runner,
        "_line_pair_for_case",
        lambda _case, grid, *, want_coherent, return_characteristic: (
            np.ones(np.asarray(grid).shape),
            None,
            np.ones(np.asarray(grid).shape),
        ),
    )
    results = {"mos2@30": {30.0: _line_record()}}
    record = results["mos2@30"][30.0]

    n = run.repair_line_spec(
        results,
        material="mos2",
        line_ne=60,
        line_start_eV=500.0,
        line_stop_eV=1100.0,
        line_step_eV=200.0,
        fidelity="survey",
        from_config=False,
    )

    assert n == 1
    np.testing.assert_array_equal(record["E_grid"], [500.0, 700.0, 900.0])
    assert record["case"]["Ne"] == 60
    assert record["case"]["line_profile"] == "survey"


def test_repair_line_spec_skips_at_target(monkeypatch):
    import numpy as np

    from pyrite.runs import run

    monkeypatch.setattr(
        run.runner,
        "_line_pair_for_case",
        lambda case, E_grid, *, want_coherent, return_characteristic: (
            np.full(E_grid.shape, 5.0),
            None,
            np.full(E_grid.shape, 1.0),
        ),
    )
    results = {"mos2@30": {30.0: _line_record(ne=200)}}
    # same Ne, same grid, finite spec -> nothing to redo
    n = run.repair_line_spec(results, material="mos2", line_ne=200, from_config=False)
    assert n == 0


def test_repair_line_spec_max_seconds_stops_early(monkeypatch):
    import numpy as np

    from pyrite.runs import run

    monkeypatch.setattr(
        run.runner,
        "_line_pair_for_case",
        lambda case, E_grid, *, want_coherent, return_characteristic: (
            np.full(E_grid.shape, 5.0),
            None,
            np.full(E_grid.shape, 1.0),
        ),
    )
    results = {"mos2@30": {30.0: _line_record(ne=200)}}  # 1 stale record (line_ne bump)
    n = run.repair_line_spec(
        results,
        material="mos2",
        line_ne=999,
        from_config=False,
        max_seconds=0.0,
        status=(s := {}),
    )
    assert n == 0
    assert s["complete"] is False


def test_reline_checkpoints_forwards_flags(monkeypatch, tmp_path):
    from pyrite.checkpoints import recompute as reline

    calls = []
    monkeypatch.setattr(
        reline,
        "reline_checkpoint",
        lambda path, material, **kw: calls.append((material, kw)) or {},
    )
    (tmp_path / "mos2.pkl").write_bytes(b"x")
    reline.reline_checkpoints(
        materials=["mos2"],
        checkpoint_dir=str(tmp_path),
        line_ne=40000,
        line_step_eV=5.0,
        redo_all=True,
    )
    assert calls == [
        (
            "mos2",
            {
                "line_ne": 40000,
                "line_step_eV": 5.0,
                "from_config": True,
                "redo_all": True,
                "save_every": 100,
                "max_seconds": None,
                "status": {},
            },
        )
    ]


def test_reline_profile_defaults_match_for_explicit_materials_and_all(monkeypatch, tmp_path):
    from pyrite.checkpoints import recompute as reline

    for material in ("hopg", "hbn"):
        (tmp_path / f"{material}.pkl").write_bytes(b"")
    calls = []
    monkeypatch.setattr(
        "pyrite.checkpoints.recompute.reline_checkpoint",
        lambda path, material, **kwargs: calls.append((material, kwargs)) or {},
    )

    reline.reline_checkpoints(materials=["hopg", "hbn"], checkpoint_dir=tmp_path, fidelity="survey")
    explicit = calls.copy()
    calls.clear()
    reline.reline_checkpoints(checkpoint_dir=tmp_path, fidelity="survey")

    assert {name: kwargs for name, kwargs in calls} == {name: kwargs for name, kwargs in explicit}
    assert {kwargs["line_ne"] for _, kwargs in calls} == {60}
    assert {kwargs["fidelity"] for _, kwargs in calls} == {"survey"}


def test_reline_resolves_variant_material_profile_and_fidelity_from_metadata(monkeypatch, tmp_path):
    import json

    from pyrite.checkpoints import recompute as reline

    stem = "hopg@sub_100keV-deadbeef0000"
    directory = tmp_path / stem
    directory.mkdir()
    (directory / "line.pkl").touch()
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "dataset_identity": {
                    "material": "hopg",
                    "fidelity": "survey",
                    "catalog_profile": "sub_100keV",
                }
            }
        )
    )
    calls = []
    monkeypatch.setattr(
        reline,
        "reline_checkpoint",
        lambda path, material, **kwargs: calls.append((path, material, kwargs)) or {},
    )

    reline.reline_checkpoints(
        materials=[stem],
        checkpoint_dir=tmp_path,
        require_identity=True,
    )

    assert calls[0][1] == "hopg"
    assert calls[0][2]["fidelity"] == "survey"
    assert calls[0][2]["catalog_profile"] == "sub_100keV"


def test_recompute_legacy_ambiguous_stem_requires_explicit_profile(tmp_path):
    from pyrite.checkpoints.recompute_defaults import dataset_context

    path = tmp_path / "old-derived-stem"

    with pytest.raises(ValueError, match="no usable dataset identity"):
        dataset_context(path, require_identity=True)

    resolved = dataset_context(path, catalog_profile="standard", require_identity=True)
    assert resolved.material == "old-derived-stem"
    assert resolved.catalog_profile == "standard"


def test_recompute_rejects_profile_mismatch(tmp_path):
    import json

    from pyrite.checkpoints.recompute_defaults import dataset_context

    path = tmp_path / "hopg@survey"
    path.mkdir()
    (path / "meta.json").write_text(
        json.dumps(
            {
                "dataset_identity": {
                    "material": "hopg",
                    "fidelity": "survey",
                    "catalog_profile": "survey",
                }
            }
        )
    )

    with pytest.raises(ValueError, match="not requested"):
        dataset_context(path, catalog_profile="standard", require_identity=True)


def test_recomputed_checkpoint_publishes_cas_before_component_and_manifest(monkeypatch, tmp_path):
    import json

    import pyrite.checkpoints.persistence as run

    checkpoint = tmp_path / "hopg@survey"
    checkpoint.mkdir()
    identity = {
        "material": "hopg",
        "fidelity": "survey",
        "catalog_profile": "survey",
        "parameter_sha256": "f" * 64,
    }
    (checkpoint / "meta.json").write_text(json.dumps({"dataset_identity": identity}))
    case = {"crystal": "hopg", "name": "case", "E0_keV": 30.0}
    results = {
        "case": {
            30.0: {
                "case": case,
                "E_grid": [1.0],
                "spec": [2.0],
                "brem": [3.0],
                "eta": 1.0,
            }
        }
    }
    events = []
    monkeypatch.setattr(
        run._checkpoint_store,
        "cas_save",
        lambda material, key, root, payload: events.append(("cas", material, key, root)),
    )
    monkeypatch.setattr(
        run,
        "_checkpoint_components_save",
        lambda path, value, components: events.append(("component", components)),
    )
    monkeypatch.setattr(
        run,
        "_manifest_save",
        lambda path, value, dataset_identity: events.append(("meta", dataset_identity)),
    )

    def write_cases(path, cases, key_fn, dataset_identity):
        events.append(("cases", [key_fn(item) for item in cases], dataset_identity))

    monkeypatch.setattr(run, "_write_case_manifest", write_cases)

    run._save_recomputed_checkpoint(checkpoint, results, components=("line",))

    assert [event[0] for event in events] == ["cas", "component", "meta", "cases"]
    assert events[0][1] == "hopg"
    assert len(events[0][2]) == 64
    assert events[-1][2] == identity
