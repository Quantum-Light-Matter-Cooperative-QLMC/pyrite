"""``cxr scan --max-minutes`` budget: exit 75 on incomplete work, paused progress state."""

import json

from click.testing import CliRunner

from cxr_mc import scan


def _invoke(material=None, max_minutes=None, progress_file=None, *extra):
    argv = [material]
    if material is None:
        argv = []
    if max_minutes is not None:
        argv += ["--max-minutes", str(max_minutes)]
    if progress_file is not None:
        argv += ["--progress-file", str(progress_file)]
    argv += list(extra)
    return CliRunner().invoke(scan.command, argv, catch_exceptions=False)


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
    result = _invoke("hopg", 5.0, None, "--checkpoint-dir", str(tmp_path))
    assert result.exit_code == 75


def test_scan_budget_complete_exits_normally(monkeypatch, tmp_path):
    monkeypatch.setattr(scan, "run_sweep", lambda *a, **kw: True)
    _stub_cases(monkeypatch)
    result = _invoke("hopg", 5.0, None, "--checkpoint-dir", str(tmp_path))
    assert result.exit_code == 0


def test_scan_budget_writes_paused_progress_state(monkeypatch, tmp_path):
    monkeypatch.setattr(scan, "run_sweep", lambda *a, **kw: False)
    _stub_cases(monkeypatch)
    progress = tmp_path / "hopg.json"
    result = _invoke(
        "hopg",
        5.0,
        progress,
        "--checkpoint-dir",
        str(tmp_path),
    )
    assert result.exit_code == 75
    record = json.loads(progress.read_text())
    assert record["state"] == "paused"


def test_scan_progress_records_compute_cost_when_progress_file_given(monkeypatch, tmp_path):
    """item 6: _run_material wires sweep.case_cost through run_sweep's on_cost
    callback into the remote progress JSON's done_cost/total_cost fields, so
    `cxr remote attach` can render a compute-weighted bar (presentation.py's
    _overall_progress_line(..., use_cost=True))."""

    def _fake_run_sweep(*args, **kwargs):
        assert kwargs["case_cost_fn"] is not None
        kwargs["on_cost"](2.5, 5.0)
        return True

    monkeypatch.setattr(scan, "run_sweep", _fake_run_sweep)
    _stub_cases(monkeypatch)
    progress = tmp_path / "hopg.json"
    result = _invoke("hopg", None, progress, "--checkpoint-dir", str(tmp_path))
    assert result.exit_code == 0
    record = json.loads(progress.read_text())
    assert record["done_cost"] == 2.5
    assert record["total_cost"] == 5.0


def test_scan_progress_omits_cost_fields_without_progress_file(monkeypatch, tmp_path):
    def _fake_run_sweep(*args, **kwargs):
        assert kwargs["case_cost_fn"] is None
        assert kwargs["on_cost"] is None
        return True

    monkeypatch.setattr(scan, "run_sweep", _fake_run_sweep)
    _stub_cases(monkeypatch)
    result = _invoke("hopg", None, None, "--checkpoint-dir", str(tmp_path))
    assert result.exit_code == 0


def test_write_progress_record_omits_cost_fields_when_either_is_none(tmp_path):
    path = tmp_path / "hopg.json"
    scan._write_progress_record(
        path,
        material="hopg",
        total_cases=2,
        cached_cases=0,
        completed_new_cases=1,
        state="running",
        done_cost=3.0,
        total_cost=None,
    )
    record = json.loads(path.read_text())
    assert "done_cost" not in record
    assert "total_cost" not in record


def test_scan_no_max_minutes_passes_none_max_seconds(monkeypatch, tmp_path):
    captured = {}

    def _fake_run_sweep(*args, **kwargs):
        captured["max_seconds"] = kwargs.get("max_seconds")
        return True

    monkeypatch.setattr(scan, "run_sweep", _fake_run_sweep)
    _stub_cases(monkeypatch)
    result = _invoke("hopg", None, None, "--checkpoint-dir", str(tmp_path))
    assert result.exit_code == 0
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

    result = _invoke(
        None,
        5.0,
        None,
        "--all",
        "--checkpoint-dir",
        str(tmp_path),
    )
    assert result.exit_code == 0

    assert seen == [300.0, 100.0, 0.0]
    assert seen[1] <= seen[0]  # the second material only gets what's left
    assert seen[2] == 0.0  # deadline elapsed: still called, with zero budget
