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

from cxr_mc import _checkpoint_io
from cxr_mc.montecarlo import runner
from cxr_mc.run import (
    _checkpoint_save,
    _load_checkpoint_cached,
    _manifest_save,
    cases_from_results,
    checkpoint_manifest,
    checkpoint_path_for,
    load_checkpoint,
    repair_brem_wide,
    run_sweep,
)

# ---------------------------------------------------------------------------
# The load_checkpoint cache is module-global (functools.lru_cache) -- clear it
# around every test so a checkpoint loaded in one test doesn't sit cached (by
# path/mtime) and leak into an unrelated test.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_checkpoint_cache():
    _load_checkpoint_cached.cache_clear()
    yield
    _load_checkpoint_cached.cache_clear()


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


def _fake_out(case):
    E = np.arange(*case["E_grid"])
    Eb = np.arange(*case["E_grid_brem"])
    return dict(
        E_grid=E,
        spec=np.ones_like(E) * 0.1,
        brem=np.ones_like(E) * 0.01,
        E_grid_brem=Eb,
        brem_wide=np.ones_like(Eb) * 0.001,
        eta=0.05,
    )


def _stub_run_cases(cases, max_workers=None, progress=False, callback=None, should_stop=None):
    """Replaces run_cases: fires the callback immediately, no real MC."""
    for i, case in enumerate(cases):
        if callback is not None:
            callback(i, case, _fake_out(case))


def test_transport_case_forwards_finite_footprint_to_both_trajectories(monkeypatch):
    case = _fake_case("finite", 30.0)
    case.update(crystal_width_mm=0.1, crystal_height_mm=0.2)
    captured = []

    def _transport(*args, **kwargs):
        captured.append(kwargs)
        return {"transport": len(captured)}

    monkeypatch.setattr(runner, "simulate_trajectories", _transport)

    runner._transport_case(case)

    assert len(captured) == 2
    assert all(call["crystal_width_mm"] == 0.1 for call in captured)
    assert all(call["crystal_height_mm"] == 0.2 for call in captured)


def test_brem_for_case_forwards_finite_footprint(monkeypatch):
    case = _fake_case("finite", 30.0)
    case.update(crystal_width_mm=0.1, crystal_height_mm=0.2)
    captured = []

    def _transport(*args, **kwargs):
        captured.append(kwargs)
        return {"transport": True}

    monkeypatch.setattr(runner, "simulate_trajectories", _transport)
    monkeypatch.setattr(runner, "_brem_wide_from_segments", lambda *args: np.array([0.0]))

    runner._brem_for_case(case, np.array([100.0]))

    assert len(captured) == 1
    assert captured[0]["crystal_width_mm"] == pytest.approx(0.1)
    assert captured[0]["crystal_height_mm"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# checkpoint_path_for
# ---------------------------------------------------------------------------


def test_checkpoint_path_for_includes_material_and_dir():
    p = checkpoint_path_for("hopg", checkpoint_dir="ckpts")
    assert "hopg" in p
    assert "ckpts" in p


# ---------------------------------------------------------------------------
# _checkpoint_save
# ---------------------------------------------------------------------------


def test_checkpoint_save_roundtrips(tmp_path):
    rec = {"cfg_a": {30.0: {"case": {"crystal": "hopg"}, "spec": np.array([1.0])}}}
    ckpt = tmp_path / "hopg.pkl"
    _checkpoint_save(str(ckpt), rec)
    assert set(_checkpoint_io.load(str(ckpt))) == {"cfg_a"}


def test_checkpoint_save_is_gzip_compressed(tmp_path):
    """TODO P2 #8: checkpoints are written gzip-compressed, not as plain pickles."""
    ckpt = tmp_path / "hopg.pkl"
    _checkpoint_save(str(ckpt), {"cfg_a": {30.0: {"case": {}, "spec": np.ones(1000)}}})
    assert ckpt.read_bytes()[:2] == b"\x1f\x8b"


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
    monkeypatch.setattr("cxr_mc.run.run_cases", _stub_run_cases)
    cases = [
        _fake_case("cfg_a", 30.0),
        _fake_case("cfg_a", 45.0),
        _fake_case("cfg_b", 30.0),
    ]
    run_sweep(cases, {}, checkpoint_dir=str(tmp_path), progress=False)

    meta_path = tmp_path / "hopg.meta.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        manifest = json.load(f)
    assert manifest["n_records"] == 3
    assert manifest["energies_keV"] == [30.0, 45.0]
    assert manifest["sweep"]["crystal"] == ["hopg"]


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
    monkeypatch.setattr("cxr_mc.run.run_cases", _stub_run_cases)
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
    monkeypatch.setattr("cxr_mc.run.run_cases", _stub_run_cases)
    run_sweep([_fake_case("cfg_a", 30.0)], {}, checkpoint_dir=str(tmp_path), progress=False)
    ckpt = tmp_path / "hopg.pkl"
    assert ckpt.exists()
    saved = _checkpoint_io.load(str(ckpt))
    assert "cfg_a" in saved


def test_run_sweep_resume_skips_cached_cases(tmp_path, monkeypatch):
    existing = {"cfg_a": {30.0: {"case": _fake_case("cfg_a", 30.0), "spec": np.array([1.0])}}}
    with open(tmp_path / "hopg.pkl", "wb") as f:
        pickle.dump(existing, f)

    ran = []

    def _tracking_run_cases(
        cases, max_workers=None, progress=False, callback=None, should_stop=None
    ):
        ran.extend(c["name"] for c in cases)
        _stub_run_cases(cases, callback=callback)

    monkeypatch.setattr("cxr_mc.run.run_cases", _tracking_run_cases)
    cases = [_fake_case("cfg_a", 30.0), _fake_case("cfg_b", 30.0)]
    results = {}
    run_sweep(cases, results, checkpoint_dir=str(tmp_path), resume=True, progress=False)
    assert ran == ["cfg_b"]  # cfg_a was cached
    assert "cfg_a" in results and "cfg_b" in results


def test_run_sweep_reports_initial_and_per_case_progress(tmp_path, monkeypatch):
    monkeypatch.setattr("cxr_mc.run.run_cases", _stub_run_cases)
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
    monkeypatch.setattr("cxr_mc.run.run_cases", _stub_run_cases)
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
    monkeypatch.setattr("cxr_mc.run.run_cases", _stub_run_cases)
    progress = []

    run_sweep(
        [cached_case, _fake_case("cfg_b", 30.0)],
        {},
        checkpoint_dir=str(tmp_path),
        progress=False,
        on_progress=lambda completed, total, cached: progress.append((completed, total, cached)),
    )

    assert progress == [(0, 2, 1), (1, 2, 1)]


def test_run_sweep_on_chunk_fires_per_group(tmp_path, monkeypatch):
    monkeypatch.setattr("cxr_mc.run.run_cases", _stub_run_cases)
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
    monkeypatch.setattr("cxr_mc.run.run_cases", _stub_run_cases)
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
    monkeypatch.setattr("cxr_mc.run.run_cases", _stub_run_cases)
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
    from cxr_mc import run as run_mod

    clock = {"t": 0.0}

    def fake_run_cases(todo, max_workers=None, progress=True, callback=None, should_stop=None):
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
    from cxr_mc import run as run_mod

    clock = {"t": 0.0}

    def fake_run_cases(todo, max_workers=None, progress=True, callback=None, should_stop=None):
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
        max_seconds=None,
    )
    assert complete is True
    assert len(results) == 4


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
    monkeypatch.setattr("cxr_mc.montecarlo._brem_for_case", _spy)
    n = repair_brem_wide({"cfg_a": {30.0: record}}, only_nonfinite=True, progress=False)
    assert n == 1
    assert seen["abs_layers"] == case["abs_layers"]  # full stack reached the runner
    assert np.allclose(record["brem_wide"], 0.002)  # repaired in place


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

    monkeypatch.setattr("cxr_mc.montecarlo._brem_for_case", _spy)

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

    monkeypatch.setattr("cxr_mc.montecarlo._brem_for_case", _spy)
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


def test_repair_brem_wide_retune_skips_record_already_at_target():
    record = _finite_record(_fake_case("cfg_a", 30.0))
    n = repair_brem_wide({"cfg_a": {30.0: record}}, progress=False, ne_brem=5, brem_step_eV=50.0)
    assert n == 0  # Ne_brem and spacing already match; nothing recomputed


def test_rebrem_checkpoints_enumerates_pkls_and_passes_params(monkeypatch, tmp_path):
    from cxr_mc.rebrem import rebrem_checkpoints

    (tmp_path / "MoS2.pkl").write_bytes(b"")
    (tmp_path / "W_grooved.pkl").write_bytes(b"")
    (tmp_path / "MoS2.slim.pkl").write_bytes(b"")  # transfer copy: excluded
    calls: list[tuple[str, dict[str, Any]]] = []

    def _spy(path, **kw):
        calls.append((path, kw))
        return {}

    monkeypatch.setattr("cxr_mc.run.repair_checkpoint", _spy)
    out = rebrem_checkpoints(checkpoint_dir=str(tmp_path), ne_brem=1000, brem_step_eV=25.0)
    assert sorted(out) == ["MoS2", "W_grooved"]
    assert all(kw["ne_brem"] == 1000 and kw["brem_step_eV"] == 25.0 for _, kw in calls)
    assert all(kw["only_nonfinite"] is True for _, kw in calls)  # resumable default

    calls.clear()
    rebrem_checkpoints(
        materials=["MoS2"], checkpoint_dir=str(tmp_path), ne_brem=1000, redo_all=True
    )
    assert [Path(p).name for p, _ in calls] == ["MoS2.pkl"]
    assert calls[0][1]["only_nonfinite"] is False


def test_rebrem_cli_requires_materials_xor_all(monkeypatch):
    """`cxr rebrem` refuses no-selection and materials+--all; accepts either alone."""
    import argparse

    from cxr_mc import rebrem

    ap = argparse.ArgumentParser()
    rebrem.add_subparser(ap.add_subparsers(dest="command"))
    seen: list[Any] = []
    monkeypatch.setattr(rebrem, "rebrem_checkpoints", lambda **kw: seen.append(kw) or {})

    with pytest.raises(SystemExit):
        args = ap.parse_args(["rebrem", "--ne-brem", "1000"])  # neither
        args.func(args)
    with pytest.raises(SystemExit):
        args = ap.parse_args(["rebrem", "MoS2", "--all"])  # both
        args.func(args)
    assert seen == []

    args = ap.parse_args(["rebrem", "MoS2"])
    args.func(args)
    assert seen[-1]["materials"] == ["MoS2"]
    args = ap.parse_args(["rebrem", "-a"])
    args.func(args)
    assert seen[-1]["materials"] is None
