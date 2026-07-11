"""Tests for run.py: checkpoint resume, group-streaming, and brem repair.

Monte Carlo work is replaced with a stub — these tests cover the orchestration
layer (resume/skip, checkpoint writes, on_chunk dispatch), not the physics.
"""

import pickle
from typing import Any

import numpy as np

from cxr_mc import _checkpoint_io
from cxr_mc.run import (
    _checkpoint_save,
    cases_from_results,
    checkpoint_path_for,
    load_checkpoint,
    repair_brem_wide,
    run_sweep,
)

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


def _stub_run_cases(cases, max_workers=None, progress=False, callback=None):
    """Replaces run_cases: fires the callback immediately, no real MC."""
    for i, case in enumerate(cases):
        if callback is not None:
            callback(i, case, _fake_out(case))


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

    def _tracking_run_cases(cases, max_workers=None, progress=False, callback=None):
        ran.extend(c["name"] for c in cases)
        _stub_run_cases(cases, callback=callback)

    monkeypatch.setattr("cxr_mc.run.run_cases", _tracking_run_cases)
    cases = [_fake_case("cfg_a", 30.0), _fake_case("cfg_b", 30.0)]
    results = {}
    run_sweep(cases, results, checkpoint_dir=str(tmp_path), resume=True, progress=False)
    assert ran == ["cfg_b"]  # cfg_a was cached
    assert "cfg_a" in results and "cfg_b" in results


def test_run_sweep_on_chunk_fires_per_group(tmp_path, monkeypatch):
    monkeypatch.setattr("cxr_mc.run.run_cases", _stub_run_cases)
    # same (crystal, thickness, tilt) -> one group; different tilt -> another
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
