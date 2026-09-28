"""Electron-aligned segment blocks for over-budget line sums (#192)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import electron_blocks as blocks
from pyrite.montecarlo.spectrum.lines._attribution import merge_line_attribution


def _segments(ids):
    ids = np.asarray(ids)
    return {
        "electron_id": ids,
        "L_ang": np.arange(ids.size, dtype=float),
        "r_mid": np.arange(3 * ids.size, dtype=float).reshape(ids.size, 3),
        "Ne": 99,
    }


@pytest.mark.parametrize("ids", [[0, 0, 1, 1, 1, 2, 3, 3, 4, 5], [3, 0, 1, 3, 0, 2, 1, 5, 4, 1]])
def test_blocks_partition_rows_without_splitting_electrons(ids):
    segments = _segments(ids)
    parts = list(blocks.iter_electron_blocks(segments, 3))
    assert 1 < len(parts) <= 3
    rows = np.concatenate([part["L_ang"] for part in parts])
    assert sorted(rows) == list(segments["L_ang"])
    owners = [set(part["electron_id"].tolist()) for part in parts]
    assert all(a.isdisjoint(b) for i, a in enumerate(owners) for b in owners[i + 1 :])
    for part in parts:
        assert part["Ne"] == 99
        np.testing.assert_array_equal(part["r_mid"][:, 0], 3 * part["L_ang"])
        assert np.all(np.diff(part["L_ang"]) > 0)  # original order kept


def test_one_block_is_the_input_itself():
    segments = _segments([0, 1, 2])
    assert list(blocks.iter_electron_blocks(segments, 1)) == [segments]


def test_single_electron_is_never_split():
    parts = list(blocks.iter_electron_blocks(_segments([7] * 5), 4))
    assert len(parts) == 1 and parts[0]["L_ang"].size == 5


def test_block_count_follows_headroom():
    per = blocks.LINE_DEVICE_BYTES_PER_SEGMENT
    assert blocks.electron_block_count(1_000, None) == 1
    assert blocks.electron_block_count(1_000, 1_000 * per) == 1
    assert blocks.electron_block_count(1_000, 1_000 * per // 3) == 4
    assert blocks.electron_block_count(1_000, 0) == blocks.MAX_ELECTRON_BLOCKS
    assert blocks.electron_block_count(10**12, 1) == blocks.MAX_ELECTRON_BLOCKS


def test_audit_restore_undoes_partial_accumulation():
    audit = {"start_eV": 1.0, "line_mass": 2.0, "collect": [("a",)]}
    saved = blocks.snapshot_audit(audit)
    audit["line_mass"] = 5.0
    audit["mass_above"] = 1.0
    audit["collect"].append(("b",))
    blocks.restore_audit(audit, saved)
    assert audit == {"start_eV": 1.0, "line_mass": 2.0, "collect": [("a",)]}
    blocks.restore_audit(None, blocks.snapshot_audit(None))


def test_audit_restore_truncates_appended_attribution():
    audit = {"start_eV": 1.0, "attribute": {"top": 2}}
    saved = blocks.snapshot_audit(audit)
    audit.setdefault("attribution", []).append({"mass": 1.0})
    blocks.restore_audit(audit, saved)
    assert audit["attribution"] == []
    assert audit["attribute"] == {"top": 2}


@pytest.fixture(scope="module")
def small_transport():
    sweep = material_sweep("hbn")
    sweep = replace(sweep, beam=replace(sweep.beam, energy_keV=100.0))
    case = build_cases(sweep, n_electrons=6, n_electrons_brem=6)[0]
    return case, runner._transport_case(case, transport_core="lockstep")


_LINES_ONCE = runner._lines_for_segments_once


def _once(case, tp, segments, audit=None):
    return _LINES_ONCE(
        segments,
        tp["E_grid"],
        case,
        tp["n_hat"],
        case.get("abs_layers"),
        tp.get("groove"),
        coherent=False,
        Ne=tp["Ne_lines"],
        truncation_audit=audit,
    )


def test_electron_blocks_add_to_the_whole_line_sum(small_transport):
    case, tp = small_transport
    whole = np.asarray(_once(case, tp, tp["segs"]))
    assert whole.sum() > 0.0
    split = sum(
        np.asarray(_once(case, tp, part)) for part in blocks.iter_electron_blocks(tp["segs"], 3)
    )
    np.testing.assert_allclose(split, whole, rtol=1e-10, atol=1e-12 * whole.max())


def test_block_oom_doubles_blocks_and_restores_the_audit(monkeypatch, small_transport):
    case, tp = small_transport
    expected = np.asarray(_once(case, tp, tp["segs"]))
    calls = []

    class FakeOOM(MemoryError):
        pass

    def once(segments, **kwargs):
        calls.append(segments["L_ang"].size)
        kwargs["truncation_audit"]["line_mass"] = kwargs["truncation_audit"].get("line_mass", 0) + 1
        if len(calls) == 1:
            raise FakeOOM
        return _once(case, tp, segments)

    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(runner, "device_headroom_bytes", lambda: 1)
    monkeypatch.setattr(runner, "electron_block_count", lambda n, headroom: 2)
    monkeypatch.setattr(runner, "_is_gpu_oom", lambda error: isinstance(error, FakeOOM))
    monkeypatch.setattr(runner, "_lines_for_segments_once", once)
    audit = {"start_eV": 50.0}
    spec = runner._lines_for_segments(
        tp["segs"],
        tp["E_grid"],
        case,
        tp["n_hat"],
        None,
        None,
        coherent=False,
        Ne=tp["Ne_lines"],
        truncation_audit=audit,
    )
    np.testing.assert_allclose(spec, expected, rtol=1e-10, atol=1e-12 * expected.max())
    n_blocks = len(calls) - 1
    assert 2 < n_blocks <= 4  # first block failed at 2, retried at 4
    assert audit["line_mass"] == n_blocks  # the failed attempt was undone


def test_cpu_and_coherent_sums_are_never_split(monkeypatch, small_transport):
    case, tp = small_transport
    seen = []
    monkeypatch.setattr(
        runner, "_lines_for_segments_once", lambda segments, **kw: seen.append(segments)
    )
    monkeypatch.setattr(runner, "electron_block_count", lambda n, headroom: 4)
    runner._lines_for_segments(
        tp["segs"], tp["E_grid"], case, tp["n_hat"], None, None, coherent=False
    )
    monkeypatch.setattr(runner._RESOURCE_POLICY, "gpu", True)
    monkeypatch.setattr(runner, "device_headroom_bytes", lambda: 1)
    runner._lines_for_segments(
        tp["segs"], tp["E_grid"], case, tp["n_hat"], None, None, coherent=True
    )
    assert seen == [tp["segs"], tp["segs"]]


def test_line_attribution_accounts_for_the_whole_line_mass(small_transport):
    """#201 diagnostic: per-electron masses sum to the audited line mass, the
    per-electron tails to the audited upper-edge bound, and the kept lines are
    the heaviest; the spectrum is unchanged."""
    case, tp = small_transport
    grid = np.asarray(tp["E_grid"])
    stop = float(grid[-1])
    audit = {
        "start_eV": float(grid[0]),
        "stop_eV": stop,
        "attribute": {"top": 5, "stop_eV": stop},
    }
    spec = np.asarray(_once(case, tp, tp["segs"], audit))
    np.testing.assert_array_equal(spec, np.asarray(_once(case, tp, tp["segs"])))
    merged = merge_line_attribution(audit["attribution"], 5)
    np.testing.assert_allclose(merged["electron_mass"].sum(), float(audit["line_mass"]), rtol=1e-9)
    np.testing.assert_allclose(merged["electron_tail"].sum(), float(audit["mass_above"]), rtol=1e-9)
    assert set(merged["electron_ids"]) <= set(range(tp["Ne_lines"]))
    for rank in ("mass", "tail"):
        lines = merged[f"lines_by_{rank}"]
        assert lines[rank].size == 5 and np.all(np.diff(lines[rank]) <= 0)
        np.testing.assert_allclose(lines["mass"], lines["weight"] * lines["width_eV"], rtol=1e-12)
        assert np.all(lines["A2"] > 0) and np.all(lines["T_abs"] <= 1.0)
