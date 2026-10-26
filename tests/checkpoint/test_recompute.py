"""Checkpoint component repair, resume, and recompute persistence contracts.

Scorers are stubbed: these tests exercise repair orchestration, not physics.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from pyrite.checkpoints import persistence, recompute
from pyrite.checkpoints.recompute import repair_brem_wide
from pyrite.materials import CATALOG
from pyrite.montecarlo import runner
from pyrite.runs.run import (
    _load_checkpoint_cached,
    _material_analysis_cache,
)


@pytest.fixture(autouse=True)
def _clear_checkpoint_cache():
    _load_checkpoint_cached.cache_clear()
    _material_analysis_cache.clear()
    yield
    _load_checkpoint_cached.cache_clear()
    _material_analysis_cache.clear()


def _fake_case(
    name: str, E0_keV: float, tilt_deg: float = 0.0, crystal: str = "hopg"
) -> dict[str, Any]:
    """Minimal case dict that satisfies store_result and run_sweep internals."""
    E_grid = (100.0, 200.0, 10.0)
    E_brem = (0.0, 500.0, 50.0)
    return dict(
        name=name,
        crystal=crystal,
        composition=CATALOG.crystal(crystal).composition,
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


def _repair_component(component, results, **kwargs):
    if component == "brem":
        return repair_brem_wide(results, ne_brem=999, progress=False, **kwargs)
    return recompute.repair_cxr_spec(
        results,
        "hopg",
        cxr_ne=999,
        from_config=False,
        **kwargs,
    )


@pytest.mark.parametrize("component", ["brem", "line"])
def test_budgeted_repair_resumes_saved_records_with_the_same_callbacks(
    component,
    monkeypatch,
    tmp_path,
):
    clock = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    scored = []

    def score(case, grid, **_kwargs):
        scored.append(case["name"])
        clock["t"] += 1.0
        return np.full_like(grid, 2.0)

    monkeypatch.setattr("pyrite.montecarlo._brem_for_case", score)
    monkeypatch.setattr(
        runner,
        "_cxr_pair_for_case",
        lambda case, grid, **kwargs: (score(case, grid), None, np.full_like(grid, 3.0)),
    )
    records = {
        name: {30.0: _finite_record(_fake_case(name, 30.0))}
        for name in ("cached", "first", "second")
    }
    count_key = "Ne_brem" if component == "brem" else "Ne"
    records["cached"][30.0]["case"][count_key] = 999
    checkpoint = tmp_path / "hopg.pkl"
    saved = []
    callbacks = []

    def save(results):
        persistence._checkpoint_save(checkpoint, results)
        saved.append(list(scored))

    status = {}
    assert (
        _repair_component(
            component,
            records,
            max_seconds=1.0,
            status=status,
            save_every=1,
            save_cb=save,
            on_progress=lambda *args: callbacks.append(args),
        )
        == 1
    )
    assert status == {"complete": False}
    assert scored == ["first"]
    assert saved == [["first"], ["first"]]  # periodic and final saves
    assert callbacks == [(0, 2, 1), (1, 2, 1)]
    assert records["second"][30.0]["case"][count_key] != 999

    restored = persistence._checkpoint_load(checkpoint)
    callbacks.clear()
    assert (
        _repair_component(
            component,
            restored,
            max_seconds=1.0,
            status=status,
            save_every=1,
            save_cb=save,
            on_progress=lambda *args: callbacks.append(args),
        )
        == 1
    )
    assert status == {"complete": True}
    assert scored == ["first", "second"]
    assert callbacks == [(0, 1, 2), (1, 1, 2)]
    assert len(saved) == 4

    callbacks.clear()
    assert (
        _repair_component(
            component,
            restored,
            status=status,
            save_cb=save,
            on_progress=lambda *args: callbacks.append(args),
        )
        == 0
    )
    assert callbacks == [(0, 0, 3)]
    assert len(saved) == 4
    assert status == {"complete": True}


@pytest.mark.parametrize("component", ["brem", "line"])
def test_invalid_later_grid_fails_before_any_record_mutation(component, monkeypatch):
    records = {
        name: {30.0: _finite_record(_fake_case(name, 30.0))} for name in ("first", "invalid")
    }
    invalid = records["invalid"][30.0]
    if component == "brem":
        invalid["case"]["E_grid_brem"] = np.array([100.0, 50.0])
    else:
        invalid["E_grid"] = np.array([100.0, 50.0])
    before = pickle.dumps(records)
    monkeypatch.setattr("pyrite.montecarlo._brem_for_case", lambda *_args: pytest.fail("scored"))
    monkeypatch.setattr(runner, "_cxr_pair_for_case", lambda *_args, **_kw: pytest.fail("scored"))
    with pytest.raises(ValueError, match="strictly increasing"):
        _repair_component(
            component,
            records,
            save_cb=lambda _r: pytest.fail("saved"),
            on_progress=lambda *_args: pytest.fail("reported progress"),
        )
    assert pickle.dumps(records) == before


@pytest.mark.parametrize("component", ["brem", "line"])
@pytest.mark.parametrize("selectors", [False, True], ids=["record", "selectors"])
@pytest.mark.parametrize("at_target", [False, True], ids=["retune", "cached"])
def test_axial_repair_refuses_before_any_record_or_callback_mutation(
    component, selectors, at_target, monkeypatch
):
    records = {name: {30.0: _finite_record(_fake_case(name, 30.0))} for name in ("first", "axial")}
    case = records["axial"][30.0]["case"]
    if selectors:
        case.update(channeling_model="axial", channeling_axial={"axis_uvw": [0, 0, 1]})
    else:
        case["channeling"] = {"model": "axial", "axis_uvw": [0, 0, 1]}
    if at_target:
        case["Ne_brem" if component == "brem" else "Ne"] = 999
    before = pickle.dumps(records)
    monkeypatch.setattr("pyrite.montecarlo._brem_for_case", lambda *_a: pytest.fail("scored"))
    monkeypatch.setattr(runner, "_cxr_pair_for_case", lambda *_a, **_kw: pytest.fail("scored"))
    with pytest.raises(ValueError, match="#623"):
        _repair_component(
            component,
            records,
            save_cb=lambda _r: pytest.fail("saved"),
            on_progress=lambda *_a: pytest.fail("reported progress"),
        )
    assert pickle.dumps(records) == before


@pytest.mark.parametrize("component", ["brem", "line"])
@pytest.mark.parametrize("json_output", [False, True], ids=["human", "json"])
def test_axial_recompute_cli_refusal_preserves_output_contract(component, json_output, monkeypatch):
    from pyrite.cli.commands import recompute as cli
    from pyrite.montecarlo.case import refuse_axial_channeling

    def unsupported(**_kwargs):
        refuse_axial_channeling({"channeling": {"model": "axial"}})

    function = "rebrem_checkpoints" if component == "brem" else "recxr_checkpoints"
    monkeypatch.setattr(recompute, function, unsupported)
    command = cli.brem_command if component == "brem" else cli.cxr_command
    options = ["hopg", *(["--output", "json"] if json_output else [])]
    result = CliRunner().invoke(command, options, catch_exceptions=False)
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    if json_output:
        assert "#623" in json.dumps(json.loads(result.stdout))
        assert result.stderr == ""
    else:
        assert result.stdout == ""
        assert "#623" in result.stderr


@pytest.mark.parametrize(
    "nodes",
    [
        [1.0, np.nan],
        [1.0, np.inf],
        [2.0, 1.0],
        [1.0, 1.0],
    ],
)
def test_invalid_explicit_brem_nodes_fail_without_mutation(nodes):
    records = {"case": {30.0: _finite_record(_fake_case("case", 30.0))}}
    before = pickle.dumps(records)
    with pytest.raises(ValueError, match="finite, strictly increasing"):
        repair_brem_wide(records, ne_brem=999, brem_nodes=nodes, progress=False)
    assert pickle.dumps(records) == before


@pytest.mark.parametrize("component", ["brem", "line"])
@pytest.mark.parametrize("step", [0.0, -1.0, np.inf, np.nan])
def test_invalid_uniform_step_fails_without_mutation(component, step):
    records = {"case": {30.0: _finite_record(_fake_case("case", 30.0))}}
    before = pickle.dumps(records)
    key = "brem_step_eV" if component == "brem" else "line_step_eV"
    with pytest.raises(ValueError, match="step > 0"):
        _repair_component(component, records, **{key: step})
    assert pickle.dumps(records) == before


def test_explicit_brem_bounds_can_replace_an_empty_stored_grid(monkeypatch):
    record = _finite_record(_fake_case("case", 30.0))
    record["case"]["E_grid_brem"] = np.array([])
    record["E_grid_brem"] = np.array([])
    record["brem_wide"] = np.array([])
    monkeypatch.setattr(
        "pyrite.montecarlo._brem_for_case",
        lambda _case, grid: np.ones_like(grid),
    )
    assert (
        repair_brem_wide(
            {"case": {30.0: record}},
            progress=False,
            brem_start_eV=50.0,
            brem_stop_eV=250.0,
            brem_step_eV=50.0,
        )
        == 1
    )
    np.testing.assert_array_equal(record["E_grid_brem"], [50.0, 100.0, 150.0, 200.0])
    assert record["case"]["E_grid_brem"] == (50.0, 250.0, 50.0)


def test_step_only_brem_request_preserves_historical_skip_on_graded_grid():
    record = _finite_record(_fake_case("case", 30.0))
    nodes = np.array([0.0, 50.0, 300.0, 1000.0])
    record["case"]["E_grid_brem"] = nodes
    record["E_grid_brem"] = nodes
    record["brem_wide"] = np.ones_like(nodes)
    records = {"case": {30.0: record}}
    before = pickle.dumps(records)
    assert repair_brem_wide(records, brem_step_eV=50.0, progress=False) == 0
    assert pickle.dumps(records) == before


@pytest.mark.parametrize("component", ["brem", "line"])
@pytest.mark.parametrize(
    "values, brem_repairs, line_repairs",
    [
        (None, 1, 1),
        ([], 1, 0),
        ([0.0], 1, 0),
        ([np.nan], 1, 1),
        ([np.inf], 1, 1),
        ([1.0], 0, 0),
    ],
)
def test_repair_selection_preserves_component_specific_finiteness(
    component,
    values,
    brem_repairs,
    line_repairs,
    monkeypatch,
):
    record = _finite_record(_fake_case("case", 30.0))
    key = "brem_wide" if component == "brem" else "spec"
    record[key] = values
    monkeypatch.setattr(
        "pyrite.montecarlo._brem_for_case",
        lambda _case, grid: np.ones_like(grid),
    )
    monkeypatch.setattr(
        runner,
        "_cxr_pair_for_case",
        lambda _case, grid, **_kwargs: (np.ones_like(grid), None, np.zeros_like(grid)),
    )
    records = {"case": {30.0: record}}
    if component == "brem":
        assert repair_brem_wide(records, progress=False) == brem_repairs
    else:
        assert recompute.repair_cxr_spec(records, "hopg", from_config=False) == line_repairs


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
    assert np.allclose(record["brem_wide"], 0.002)


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
    assert freed["n"] == 3


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
    assert n == 0


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

    # Compared key by key: a graded profile continuum (hopg/survey is refined,
    # so its grid carries two step sizes) resolves to a ``brem_nodes`` array,
    # and a dict holding one cannot be compared with ``==``.
    by_name = {name: kwargs for name, kwargs in calls}
    explicit_by_name = {name: kwargs for name, kwargs in explicit}
    assert by_name.keys() == explicit_by_name.keys()
    for name, got in by_name.items():
        want = explicit_by_name[name]
        assert got.keys() == want.keys()
        for key in got:
            if isinstance(got[key], np.ndarray) or isinstance(want[key], np.ndarray):
                assert np.array_equal(np.asarray(got[key]), np.asarray(want[key])), (name, key)
            else:
                assert got[key] == want[key], (name, key)
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

    neither = runner.invoke(brem_command, ["--brem-trials", "1000"], catch_exceptions=False)
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
    from pyrite.cli.commands.recompute import cxr_command

    seen: list[Any] = []
    monkeypatch.setattr(recompute, "recxr_checkpoints", lambda **kw: seen.append(kw) or {})
    runner = CliRunner()

    neither = runner.invoke(cxr_command, [], catch_exceptions=False)
    both = runner.invoke(cxr_command, ["MoS2", "--all"], catch_exceptions=False)
    assert neither.exit_code == 2
    assert both.exit_code == 2
    assert "needs material name(s)" in neither.stderr
    assert "--all does not take material names" in both.stderr
    assert seen == []

    material = runner.invoke(cxr_command, ["MoS2"], catch_exceptions=False)
    assert material.exit_code == 0
    assert material.stderr == ""
    assert seen[-1]["materials"] == ["MoS2"]

    all_materials = runner.invoke(cxr_command, ["-a"], catch_exceptions=False)
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
        "_cxr_pair_for_case",
        lambda case, E_grid, *, want_coherent, return_characteristic: (
            np.full(E_grid.shape, 5.0),
            None,
            np.full(E_grid.shape, 1.0),
        ),
    )
    results = {"mos2@30": {30.0: _line_record()}}
    r = results["mos2@30"][30.0]
    brem_wide0 = r["brem_wide"].copy()
    n = run.repair_cxr_spec(results, material="mos2", cxr_ne=999, from_config=False)
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

    monkeypatch.setattr(run.runner, "_cxr_pair_for_case", fake_pair)

    both = _line_record()
    both["spec_coherent"] = np.zeros(32)  # stale placeholder to be overwritten
    incoh = _line_record()
    results = {"both@30": {30.0: both}, "incoh@30": {30.0: incoh}}

    n = run.repair_cxr_spec(results, material="mos2", cxr_ne=999, from_config=False)

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
        "_cxr_pair_for_case",
        lambda _case, grid, *, want_coherent, return_characteristic: (
            np.ones(np.asarray(grid).shape),
            None,
            np.ones(np.asarray(grid).shape),
        ),
    )
    results = {"mos2@30": {30.0: _line_record()}}
    record = results["mos2@30"][30.0]

    n = run.repair_cxr_spec(
        results,
        material="mos2",
        cxr_ne=60,
        cxr_start_eV=500.0,
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
        "_cxr_pair_for_case",
        lambda case, E_grid, *, want_coherent, return_characteristic: (
            np.full(E_grid.shape, 5.0),
            None,
            np.full(E_grid.shape, 1.0),
        ),
    )
    results = {"mos2@30": {30.0: _line_record(ne=200)}}
    # same Ne, same grid, finite spec -> nothing to redo
    n = run.repair_cxr_spec(results, material="mos2", cxr_ne=200, from_config=False)
    assert n == 0


def test_repair_line_spec_max_seconds_stops_early(monkeypatch):
    import numpy as np

    from pyrite.runs import run

    monkeypatch.setattr(
        run.runner,
        "_cxr_pair_for_case",
        lambda case, E_grid, *, want_coherent, return_characteristic: (
            np.full(E_grid.shape, 5.0),
            None,
            np.full(E_grid.shape, 1.0),
        ),
    )
    results = {"mos2@30": {30.0: _line_record(ne=200)}}  # 1 stale record (cxr_ne bump)
    n = run.repair_cxr_spec(
        results,
        material="mos2",
        cxr_ne=999,
        from_config=False,
        max_seconds=0.0,
        status=(s := {}),
    )
    assert n == 0
    assert s["complete"] is False


def test_repair_line_spec_from_config_targets_the_grid_build_cases_would_use(monkeypatch):
    """The bespoke-grid reline workflow (issue #135).

    ``from_config=True`` rebuilds each record's target from the material's
    CURRENT configuration, so the relined grid must equal the one
    :func:`build_cases` resolves at that record's E0 -- both for an energy an
    opt-in stored ``E_grid_line_by_energy`` row covers and for one that falls
    through to automatic resolution (issue #101; bundled profiles carry no
    stored rows since ADR-0013, so the stored branch is exercised here through
    an injected sweep rather than the real catalog). Regression: the call
    passed the record's energy grid where the crystal parameters belong and
    left the returned ``(grid, policy)`` pair unpacked, so this path always
    raised.
    """
    from dataclasses import replace

    import numpy as np

    from pyrite._energy_grid_encoding import decode_energy_grid
    from pyrite.campaign import config as campaign_config
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.sweep import build_cases
    from pyrite.runs import run

    stored_keV, auto_keV = 30.0, 37.0
    stored_grid = np.linspace(10.0, 2000.0, 500)
    base_sweep = material_sweep("mos2")
    sweep = replace(
        base_sweep,
        energy_bins=replace(base_sweep.energy_bins, cxr_by_energy={stored_keV: stored_grid}),
    )
    cxr_by_energy = sweep.energy_bins.cxr_by_energy
    assert stored_keV in cxr_by_energy and auto_keV not in cxr_by_energy

    # what building this material from config would put on each case
    expected = {}
    for E0 in (stored_keV, auto_keV):
        cases = build_cases(replace(sweep, beam=replace(sweep.beam, energy_keV=E0)))
        expected[E0] = decode_energy_grid(cases[0]["E_grid_line"])
    np.testing.assert_allclose(expected[stored_keV], np.asarray(cxr_by_energy[stored_keV], float))

    # repair_cxr_spec rebuilds the material's sweep from the catalog itself;
    # inject the stored-row sweep in its place rather than editing the catalog.
    monkeypatch.setattr(campaign_config, "material_sweep", lambda *a, **k: sweep)
    monkeypatch.setattr(
        run.runner,
        "_cxr_pair_for_case",
        lambda case, E_grid, *, want_coherent, return_characteristic: (
            np.full(E_grid.shape, 5.0),
            None,
            np.full(E_grid.shape, 1.0),
        ),
    )
    records = {E0: _line_record(E0=E0) for E0 in (stored_keV, auto_keV)}
    results = {"mos2@30": {stored_keV: records[stored_keV], auto_keV: records[auto_keV]}}

    n = run.repair_cxr_spec(results, material="mos2")

    assert n == 2
    for E0, r in records.items():
        np.testing.assert_allclose(r["E_grid"], expected[E0])
        assert r["spec"].shape == r["E_grid"].shape
        assert r["brem"].shape == r["E_grid"].shape
        # explicit coordinates outrank any automatic case-local policy
        assert "line_grid_policy" not in r["case"]


def test_reline_checkpoints_forwards_flags(monkeypatch, tmp_path):
    from pyrite.checkpoints import recompute as reline

    calls = []
    monkeypatch.setattr(
        reline,
        "recxr_checkpoint",
        lambda path, material, **kw: calls.append((material, kw)) or {},
    )
    (tmp_path / "mos2.pkl").write_bytes(b"x")
    reline.recxr_checkpoints(
        materials=["mos2"],
        checkpoint_dir=str(tmp_path),
        cxr_ne=40000,
        line_step_eV=5.0,
        redo_all=True,
    )
    assert calls == [
        (
            "mos2",
            {
                "cxr_ne": 40000,
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
        "pyrite.checkpoints.recompute.recxr_checkpoint",
        lambda path, material, **kwargs: calls.append((material, kwargs)) or {},
    )

    reline.recxr_checkpoints(materials=["hopg", "hbn"], checkpoint_dir=tmp_path, fidelity="survey")
    explicit = calls.copy()
    calls.clear()
    reline.recxr_checkpoints(checkpoint_dir=tmp_path, fidelity="survey")

    assert {name: kwargs for name, kwargs in calls} == {name: kwargs for name, kwargs in explicit}
    assert {kwargs["cxr_ne"] for _, kwargs in calls} == {60}
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
        "recxr_checkpoint",
        lambda path, material, **kwargs: calls.append((path, material, kwargs)) or {},
    )

    reline.recxr_checkpoints(
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


def _graded_record(brem_wide=None):
    Eb = np.arange(0.0, 500.0, 50.0)
    return dict(
        E_grid=np.arange(100.0, 200.0, 10.0),
        spec=np.ones(10),
        brem=np.ones(10) * 0.01,
        E_grid_brem=Eb,
        brem_wide=np.full_like(Eb, 0.001) if brem_wide is None else brem_wide,
        eta=0.05,
        scale=1.0,
        case=_fake_case("cfg_a", 30.0),
    )


def test_repair_brem_wide_retunes_onto_explicit_graded_nodes(monkeypatch):
    """A graded target has no (start, stop, step); retune onto the coordinates."""
    nodes = np.geomspace(50.0, 30000.0, 17)
    record = _graded_record()
    monkeypatch.setattr(
        "pyrite.montecarlo._brem_for_case",
        lambda c, E_brem: np.full(np.asarray(E_brem, float).shape, 0.002),
    )

    n = repair_brem_wide(
        {"cfg_a": {30.0: record}}, only_nonfinite=True, progress=False, brem_nodes=nodes
    )

    assert n == 1
    assert np.array_equal(record["E_grid_brem"], nodes)
    # the case carries the exact coordinates, not a uniform triple
    assert np.array_equal(np.asarray(record["case"]["E_grid_brem"]), nodes)
    assert not isinstance(record["case"]["E_grid_brem"], tuple)


def test_repair_brem_wide_skips_a_record_already_on_the_graded_target(monkeypatch):
    nodes = np.geomspace(50.0, 30000.0, 17)
    record = _graded_record(brem_wide=np.full(nodes.shape, 0.001))
    record["E_grid_brem"] = nodes
    record["case"]["E_grid_brem"] = nodes
    monkeypatch.setattr(
        "pyrite.montecarlo._brem_for_case",
        lambda c, E_brem: pytest.fail("already at target; must not recompute"),
    )

    n = repair_brem_wide(
        {"cfg_a": {30.0: record}}, only_nonfinite=True, progress=False, brem_nodes=nodes
    )

    assert n == 0


def test_repair_brem_wide_refuses_graded_nodes_mixed_with_a_uniform_triple():
    with pytest.raises(ValueError, match="cannot be combined"):
        repair_brem_wide(
            {"cfg_a": {30.0: _graded_record()}},
            progress=False,
            brem_nodes=np.geomspace(50.0, 30000.0, 17),
            brem_step_eV=25.0,
        )


def test_repair_brem_wide_rejects_a_degenerate_node_array():
    with pytest.raises(ValueError, match="at least two nodes"):
        repair_brem_wide(
            {"cfg_a": {30.0: _graded_record()}}, progress=False, brem_nodes=np.array([50.0])
        )


def test_rebrem_retunes_onto_a_graded_profile_grid(monkeypatch, tmp_path):
    """A graded profile continuum must reach the retune, not be silently dropped.

    ``uniform_bounds`` raises on a nonuniform grid, and that raise used to land
    in the same ``except`` branch that handles derived stems with no catalog
    row -- so the profile grid was discarded and the record kept its own (#153).
    """
    from pyrite.checkpoints import recompute as rebrem

    (tmp_path / "hopg.pkl").write_bytes(b"")
    nodes = np.geomspace(50.0, 30000.0, 21)

    class _Bins:
        brem = nodes

    class _Detector:
        energy_bins = _Bins()

    class _Sweep:
        energy_bins = _Bins()

    monkeypatch.setattr("pyrite.checkpoints.recompute_defaults.sweep", lambda *a, **k: _Sweep())
    calls = []
    monkeypatch.setattr(
        "pyrite.checkpoints.recompute.repair_checkpoint",
        lambda path, **kwargs: calls.append(kwargs) or {},
    )

    rebrem.rebrem_checkpoints(materials=["hopg"], checkpoint_dir=tmp_path, fidelity="survey")

    assert len(calls) == 1
    assert np.array_equal(calls[0]["brem_nodes"], nodes)
    # a graded target carries no uniform triple alongside it
    assert calls[0]["brem_start_eV"] is None
    assert calls[0]["brem_step_eV"] is None


def test_rebrem_keeps_the_uniform_triple_for_a_uniform_profile_grid(monkeypatch, tmp_path):
    """The uniform path is untouched: no brem_nodes, start/step resolved as before."""
    from pyrite.checkpoints import recompute as rebrem

    (tmp_path / "hopg.pkl").write_bytes(b"")

    class _Bins:
        brem = np.arange(50.0, 1000.0, 25.0)

    class _Detector:
        energy_bins = _Bins()

    class _Sweep:
        energy_bins = _Bins()

    monkeypatch.setattr("pyrite.checkpoints.recompute_defaults.sweep", lambda *a, **k: _Sweep())
    calls = []
    monkeypatch.setattr(
        "pyrite.checkpoints.recompute.repair_checkpoint",
        lambda path, **kwargs: calls.append(kwargs) or {},
    )

    rebrem.rebrem_checkpoints(materials=["hopg"], checkpoint_dir=tmp_path, fidelity="survey")

    assert calls[0]["brem_nodes"] is None
    assert calls[0]["brem_start_eV"] == 50.0
    assert calls[0]["brem_step_eV"] == 25.0
