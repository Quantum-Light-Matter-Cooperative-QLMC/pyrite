"""Tests for the checkpoint slim-export (P2 #5): results.slim_results +
slim.slim_checkpoint. All synthetic + pure-numpy, so CPU-only and fast."""

import pickle
from typing import Any

import numpy as np
import pytest

from cxr_mc import _checkpoint_io
from cxr_mc.config import default_settings, material_sweep
from cxr_mc.results import slim_results
from cxr_mc.slim import slim_checkpoint
from cxr_mc.sweep import build_cases


def _record(tilt_deg: float, E0: float) -> dict[str, Any]:
    E = np.linspace(500.0, 1200.0, 200)
    Eb = np.linspace(0.0, 30000.0, 400)  # the wide-brem grid: the largest field
    return dict(
        E_grid=E,
        spec=np.exp(-((E - 900.0) ** 2) / 50.0),
        brem=np.full_like(E, 0.01),
        E_grid_brem=Eb,
        brem_wide=np.full_like(Eb, 0.001),
        E_pk=900.0,
        fwhm=30.0,
        eta=2.0,
        scale=1.0,
        case={"name": f"t{tilt_deg}", "E0_keV": E0, "tilt_deg": tilt_deg, "crystal": "hopg"},
    )


def _results():
    return {
        f"t{tilt}": {25.0: _record(tilt, 25.0), 30.0: _record(tilt, 30.0)} for tilt in (0.0, 30.0)
    }


def test_drop_wide_brem_removes_largest_fields_only():
    res = _results()
    slim = slim_results(res, drop_wide_brem=True)
    for by_E in slim.values():
        for rec in by_E.values():
            assert "brem_wide" not in rec and "E_grid_brem" not in rec
            assert {"spec", "brem", "E_grid", "case"} <= set(rec)
    assert "brem_wide" in res["t0.0"][25.0]  # original untouched


def test_downcast_to_float32_preserves_values():
    res = _results()
    slim = slim_results(res, downcast=True)
    rec = slim["t0.0"][25.0]
    assert rec["spec"].dtype == np.float32 and rec["brem_wide"].dtype == np.float32
    assert res["t0.0"][25.0]["spec"].dtype == np.float64  # original untouched
    assert np.allclose(rec["spec"], res["t0.0"][25.0]["spec"], rtol=1e-5)


def test_constraint_filters_records():
    res = _results()
    assert set(slim_results(res, tilt_deg=0.0)) == {"t0.0"}
    by_energy = slim_results(res, E0_keV=25.0)
    for by_E in by_energy.values():
        assert set(by_E) == {25.0}


def test_fields_allowlist_keeps_only_requested_plus_case():
    res = _results()
    rec = slim_results(res, fields=["spec"])["t0.0"][25.0]
    assert set(rec) == {"spec", "case"}


def test_no_args_keeps_structure_but_copies_records():
    res = _results()
    slim = slim_results(res)
    assert set(slim) == set(res)
    assert all(set(slim[n]) == set(res[n]) for n in res)
    assert slim["t0.0"][25.0] is not res["t0.0"][25.0]  # fresh dict, safe to mutate


def test_slim_checkpoint_roundtrip_is_smaller(tmp_path):
    res = _results()
    src = tmp_path / "hopg.pkl"
    with open(src, "wb") as f:
        pickle.dump(res, f)
    out = tmp_path / "hopg.slim.pkl"
    slim_checkpoint(str(src), str(out), drop_wide_brem=True, downcast=True)
    assert out.exists() and out.stat().st_size < src.stat().st_size
    reloaded = _checkpoint_io.load(str(out))
    assert set(reloaded) == set(res)
    assert "brem_wide" not in reloaded["t0.0"][25.0]
    assert reloaded["t0.0"][25.0]["spec"].dtype == np.float32


def test_slim_checkpoint_default_out_path(tmp_path):
    src = tmp_path / "hopg.pkl"
    with open(src, "wb") as f:
        pickle.dump(_results(), f)
    slim_checkpoint(str(src))
    assert (tmp_path / "hopg.slim.pkl").exists()


def test_slim_checkpoint_compresslevel_9_is_lossless_and_no_larger(tmp_path):
    """No trimming flag + --compresslevel 9 is a pure recompress: same content,
    same or smaller bytes than the level-6 default -- what `cxr remote pull
    --level9` relies on."""
    res = _results()
    src = tmp_path / "hopg.pkl"
    with open(src, "wb") as f:
        pickle.dump(res, f)
    level6 = tmp_path / "hopg.level6.pkl"
    level9 = tmp_path / "hopg.level9.pkl"
    slim_checkpoint(str(src), str(level6))
    slim_checkpoint(str(src), str(level9), compresslevel=9)
    assert level9.stat().st_size <= level6.stat().st_size
    reloaded = _checkpoint_io.load(str(level9))
    assert reloaded.keys() == res.keys()
    for name, by_E in res.items():
        for E0, record in by_E.items():
            for key, value in record.items():
                got = reloaded[name][E0][key]
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(got, value)
                else:
                    assert got == value


# ---- grid filtering (checkpoint lifecycle: grid-filtered pull) ----------------
def _grid_config_names(material="hopg"):
    """The real current-grid config names for a material, straight from the same
    build_cases path slim_results(grid=) rebuilds."""
    s = default_settings()
    cases = build_cases(material_sweep(material), s.n_electrons, s.n_electrons_brem)
    return sorted({c["name"] for c in cases})


def _grid_plus_stale(material="hopg", n_keep=2):
    """A checkpoint holding a couple of current-grid configs plus stale configs
    whose names are NOT in the current grid."""
    keep = _grid_config_names(material)[:n_keep]
    res = {n: {25.0: _record(0.0, 25.0), 30.0: _record(0.0, 30.0)} for n in keep}
    res["stale_old_config_a"] = {30.0: _record(0.0, 30.0)}
    res["stale_old_config_b"] = {30.0: _record(0.0, 30.0)}
    return res, set(keep)


def test_grid_keeps_only_current_grid_names():
    res, keep = _grid_plus_stale()
    slim = slim_results(res, grid="hopg")
    assert set(slim) == keep  # stale_* dropped, current grid kept
    assert set(res) == keep | {"stale_old_config_a", "stale_old_config_b"}  # input untouched


def test_grid_is_lossless_per_record():
    res, keep = _grid_plus_stale()
    slim = slim_results(res, grid="hopg")
    for name in keep:
        assert set(slim[name]) == {25.0, 30.0}  # every energy of a kept config survives
        assert "brem_wide" in slim[name][25.0]  # no per-record trimming from --grid alone


def test_grid_composes_with_downcast_and_drop_wide_brem():
    res, keep = _grid_plus_stale()
    slim = slim_results(res, grid="hopg", drop_wide_brem=True, downcast=True)
    assert set(slim) == keep
    rec = slim[sorted(keep)[0]][25.0]
    assert "brem_wide" not in rec and rec["spec"].dtype == np.float32


def test_grid_composes_with_value_constraints():
    res, keep = _grid_plus_stale()
    slim = slim_results(res, grid="hopg", E0_keV=25.0)
    assert set(slim) == keep
    for by_E in slim.values():
        assert set(by_E) == {25.0}  # value constraint applied after grid narrowing


def test_slim_checkpoint_grid_roundtrips(tmp_path):
    res, keep = _grid_plus_stale()
    src = tmp_path / "hopg.pkl"
    with open(src, "wb") as f:
        pickle.dump(res, f)
    out = tmp_path / "hopg.grid.pkl"
    slim_checkpoint(str(src), str(out), grid=True)
    reloaded = _checkpoint_io.load(str(out))
    assert set(reloaded) == keep


def test_slim_checkpoint_grid_rejects_quick_stem(tmp_path):
    src = tmp_path / "hopg_quick.pkl"
    with open(src, "wb") as f:
        pickle.dump(_results(), f)
    with pytest.raises(SystemExit, match="quick"):
        slim_checkpoint(str(src), grid=True)


def test_pct_smaller_never_reports_negative_zero():
    """A slim that lands a hair LARGER (tiny checkpoint + float32 pickling
    overhead) must report '0% smaller', not the '-0%' float-format artifact."""
    from cxr_mc.slim import _pct_smaller

    assert str(_pct_smaller(100_004, 100_008)) == "0"  # -0.004% -> 0, not -0
    assert _pct_smaller(100, 37) == 63
    assert _pct_smaller(0, 0) == 0


def test_slim_checkpoint_grid_rejects_unknown_material_before_load(tmp_path):
    """An unknown stem with --grid must exit with a clean one-line message naming
    the stem, not a raw ValueError traceback -- and must do so BEFORE loading the
    (potentially gigabyte-scale) pickle, which the junk bytes pin down."""
    src = tmp_path / "notamaterial.pkl"
    src.write_bytes(b"not a pickle")  # unpickling this would raise, so a load = fail
    with pytest.raises(SystemExit, match="notamaterial"):
        slim_checkpoint(str(src), grid=True)


def test_project_dataset_keeps_only_that_datasets_keys():
    from cxr_mc.results import project_dataset

    rec = {
        "case": {"E0_keV": 30.0},
        "spec": [1.0],
        "E_grid": [1.0],
        "brem_wide": [2.0],
        "brem": [3.0],
        "E_grid_brem": [4.0],
    }
    results = {"n": {30.0: rec}}
    brem = project_dataset(results, "brem")["n"][30.0]
    line = project_dataset(results, "line")["n"][30.0]
    assert set(brem) == {"case", "brem_wide", "brem", "E_grid_brem"}
    assert set(line) == {"case", "spec", "E_grid"}
