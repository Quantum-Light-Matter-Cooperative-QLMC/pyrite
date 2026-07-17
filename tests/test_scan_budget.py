"""``scan.py --max-minutes`` budget: exit 75 on incomplete work, paused progress state."""

import argparse
import json

import pytest

from cxr_mc import scan


def _args(material, max_minutes, progress_file=None):
    ap = scan._build_parser(argparse.ArgumentParser())
    argv = [material]
    if max_minutes is not None:
        argv += ["--max-minutes", str(max_minutes)]
    if progress_file is not None:
        argv += ["--progress-file", str(progress_file)]
    return ap.parse_args(argv)


def _fake_cases():
    return [
        {
            "name": "cfg0",
            "E0_keV": 30,
            "crystal": "hopg",
            "thickness_ang": 1e4,
            "tilt_deg": 0.0,
            "beam_uvw": (0, 0, 1),
            "hkl_list": [(0, 0, 2)],
        }
    ]


def _stub_cases(monkeypatch):
    # Focus these tests on the budget wiring, not the penetration watchdog
    # (which needs real transport-shaped case dicts): make it a passthrough.
    monkeypatch.setattr(scan, "build_cases", lambda *a, **kw: _fake_cases())
    monkeypatch.setattr(scan, "gate_cases_by_penetration", lambda cases, **kw: (cases, []))


def test_scan_budget_incomplete_exits_75(monkeypatch, tmp_path):
    monkeypatch.setattr(scan, "run_sweep", lambda *a, **kw: False)
    _stub_cases(monkeypatch)
    args = _args("hopg", 5.0)
    args.checkpoint_dir = str(tmp_path)
    with pytest.raises(SystemExit) as exc:
        scan.run(args)
    assert exc.value.code == 75


def test_scan_budget_complete_exits_normally(monkeypatch, tmp_path):
    monkeypatch.setattr(scan, "run_sweep", lambda *a, **kw: True)
    _stub_cases(monkeypatch)
    args = _args("hopg", 5.0)
    args.checkpoint_dir = str(tmp_path)
    scan.run(args)  # no SystemExit


def test_scan_budget_writes_paused_progress_state(monkeypatch, tmp_path):
    monkeypatch.setattr(scan, "run_sweep", lambda *a, **kw: False)
    _stub_cases(monkeypatch)
    progress = tmp_path / "hopg.json"
    args = _args("hopg", 5.0, progress_file=progress)
    args.checkpoint_dir = str(tmp_path)
    with pytest.raises(SystemExit):
        scan.run(args)
    record = json.loads(progress.read_text())
    assert record["state"] == "paused"


def test_scan_no_max_minutes_passes_none_max_seconds(monkeypatch, tmp_path):
    captured = {}

    def _fake_run_sweep(*args, **kwargs):
        captured["max_seconds"] = kwargs.get("max_seconds")
        return True

    monkeypatch.setattr(scan, "run_sweep", _fake_run_sweep)
    _stub_cases(monkeypatch)
    args = _args("hopg", None)
    args.checkpoint_dir = str(tmp_path)
    scan.run(args)
    assert captured["max_seconds"] is None


def test_scan_all_threads_one_deadline_across_materials(monkeypatch, tmp_path):
    """--all shares ONE deadline: each material gets what's left, clamped >= 0.

    A 5-minute budget with a fake clock that burns 200 s per material must hand
    the second material less time than the first, and still CALL the third with
    max_seconds=0.0 (so its fully-cached cases can resolve) instead of skipping.
    """
    clock = {"t": 0.0}
    seen = []

    def _fake_run_sweep(*args, **kwargs):
        seen.append(kwargs["max_seconds"])
        clock["t"] += 200.0
        return True

    monkeypatch.setattr(scan.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(scan, "run_sweep", _fake_run_sweep)
    monkeypatch.setattr(scan, "load_all_materials", lambda *a, **kw: ["hopg", "hbn", "mose2"])
    _stub_cases(monkeypatch)

    ap = scan._build_parser(argparse.ArgumentParser())
    args = ap.parse_args(["--all", "--max-minutes", "5"])
    args.checkpoint_dir = str(tmp_path)
    scan.run(args)  # all complete -> no SystemExit

    assert seen == [300.0, 100.0, 0.0]
    assert seen[1] <= seen[0]  # the second material only gets what's left
    assert seen[2] == 0.0  # deadline elapsed: still called, with zero budget
