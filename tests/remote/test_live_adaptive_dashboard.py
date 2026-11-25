"""Live dashboard telemetry and exact checkpoint pull regressions."""

import json
import subprocess
from types import SimpleNamespace

from pyrite._adaptive_progress import adaptive_snapshot
from pyrite.console.dashboard import format_job_status, parse_progress_records
from pyrite.remote import lifecycle, scripts, viewer


def _record(**extra):
    return {
        "material": "hopg",
        "state": "running",
        "activity": "computing",
        "total_cases": 4,
        "cached_cases": 0,
        "completed_new_cases": 1,
        **extra,
    }


def test_adaptive_dashboard_detail_and_invalid_fields():
    record = _record(
        adaptive_progress={
            "populations": {
                "cxr": {
                    "n_electrons": 1000,
                    "relative_se": 0.02,
                    "target_rse": 0.01,
                    "variance_of_variance": 0.3,
                    "effective_sample_size": 120,
                    "passes": {"relative_se": False, "variance_sampling": False},
                    "tail_slope": float("nan"),
                    "shift_over_se": "\x1b[31mBAD",
                }
            },
            "block_histories": 512,
        }
    )
    sections = {"PROGRESS": json.dumps(record)}
    assert "ADAPTIVE STOPPING" not in format_job_status(sections, 0)
    verbose = format_job_status(sections, 1)
    assert "RSE 2.000% / 1.000%" in verbose
    assert "VOV 30.000%" in verbose and "ESS 120" in verbose
    assert "last block" not in verbose
    detailed = format_job_status(sections, 2)
    assert "last block 512" in detailed and "BAD" not in detailed
    assert "ADAPTIVE STOPPING" not in format_job_status({"PROGRESS": json.dumps(_record())}, 2)


def test_actual_monitor_snapshot_uses_population_specific_policy():
    def scalar(target):
        return SimpleNamespace(
            settings=SimpleNamespace(
                target_rse=target,
                min_histories=100,
                max_histories=1000,
                max_variance_of_variance=0.1,
                max_history_share=0.05,
                min_effective_histories=20,
            ),
            checks=2,
            projected=800,
            next_point=lambda: 500,
            stop_reason=None,
        )

    cxr, brem = scalar(0.01), scalar(0.05)
    monitor = SimpleNamespace(
        cxr=cxr,
        brem=brem,
        settings=SimpleNamespace(decision_growth=0.1, stability_checks=3, stability_fraction=1),
        record=lambda: {"statistics": {"cxr": {"n_electrons": 300}, "brem": {"n_electrons": 200}}},
    )
    plan = SimpleNamespace(floor=128, budget_bytes=2**30, rows_per_electron=4)
    snapshot = adaptive_snapshot(monitor, plan, 256)
    assert snapshot["populations"]["cxr"]["target_rse"] == 0.01
    assert snapshot["populations"]["brem"]["target_rse"] == 0.05
    assert snapshot["block_histories"] == 256


def test_pull_prefers_actual_stems_over_reservation(monkeypatch):
    actual = "hopg@cleo_sub-123456789abc"
    record = _record(checkpoint_stems=[actual, "hbn@cleo_sub-123456789abc", "../bad"])
    assert parse_progress_records(json.dumps(record))["hopg"]["checkpoint_stems"] == [actual]
    monkeypatch.setattr(lifecycle, "resolve_reported_stems", lambda stems: (stems, []))

    def unexpected(*_args):
        raise AssertionError("must not use predicted reservation")

    monkeypatch.setattr(lifecycle, "resolve_job_stems", unexpected)
    calls = []
    monkeypatch.setattr(lifecycle, "pull", lambda stems, **kw: calls.append((stems, kw)))
    viewer._pull_attached_progress("job-1", {"PROGRESS": json.dumps(record)})
    assert calls == [([actual], {"no_sync": True, "_expand_surveys": False})]


def test_written_probe_requires_index(monkeypatch, tmp_path):
    monkeypatch.setattr(scripts.config, "remote_output_path", lambda *_args: str(tmp_path))
    (tmp_path / "hopg").mkdir()
    (tmp_path / "hbn").mkdir()
    (tmp_path / "hbn" / "line.h5").touch()
    command = scripts._written_stems_command(["hopg", "hbn"])
    assert subprocess.check_output(["sh", "-c", command], text=True).splitlines() == [
        "unwritten hopg",
        "written hbn",
    ]
