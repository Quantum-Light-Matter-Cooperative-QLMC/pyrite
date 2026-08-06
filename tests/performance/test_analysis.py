from __future__ import annotations

import csv
import json

import pytest

from cxr_mc import performance_analysis
from cxr_mc.cli import performance, profile
from tests.helpers.cli import assert_clean_result, invoke


def _record(elapsed, event, **updates):
    record = {
        "schema": "cxr.performance.v1",
        "timestamp": f"2026-07-28T00:00:{elapsed:02.0f}Z",
        "event": event,
        "profile": "baseline",
        "session_id": "session123",
        "material": "hopg",
        "elapsed_seconds": elapsed,
        "host": "node1",
        "parameter_sha256": "abc",
        "engine": "gpu-pipeline",
        "effective_workers": 2,
        "worker_memory_budget_mib": 6144,
        "spec_chunk": 4000,
        "brem_chunk": 2000,
        "completed_new_cases": elapsed // 5,
        "done_cost": elapsed * 2,
        "io_read_bytes": elapsed * 10,
        "io_write_bytes": elapsed * 20,
        "swap_in_bytes": 0,
        "swap_out_bytes": 0,
        "process_cpu_user_seconds": elapsed,
        "process_cpu_system_seconds": elapsed / 2,
        "cpu_affinity_count": 4,
        "process_cpu_percent": 200,
        "cpu_percent": 15,
        "gpu_percent": 10,
        "phase": "transport_wait",
        "in_flight_case_count": 1,
        "transport_seconds_total": elapsed,
        "spectrum_seconds_total": elapsed * 2,
        "driver_wait_seconds_total": elapsed,
        "checkpoint_seconds_total": 0,
        "timed_case_count": max(1, elapsed // 5),
        "gpu_feed_wait_fraction": 1 / 3,
        "gpu_oom_retry_count_total": 0,
        "child_process_rss_max_bytes": 100,
        "process_rss_bytes": 200,
        "memory_percent": 20,
        "memory_available_bytes": 1000,
        "swap_used_bytes": 0,
        "vram_used_mib": 300,
        "vram_percent": 30,
        "cupy_pool_reserved_mib": 200,
        "cupy_pool_peak_mib": 250,
    }
    record.update(updates)
    return record


def _write_profile(tmp_path, records):
    path = tmp_path / "performance-profiles" / "baseline" / "job-7" / "hopg.ndjson"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def test_analyze_writes_intervals_sessions_summary_and_timeline(tmp_path):
    _write_profile(
        tmp_path,
        [_record(0, "start"), _record(5, "tick"), _record(10, "done")],
    )

    result = performance_analysis.analyze_performance_profile(
        "baseline", tmp_path / "performance-profiles"
    )

    assert result["sessions"] == 1
    assert result["intervals"] == 2
    analysis = tmp_path / "performance-profiles" / "baseline" / "analysis"
    sessions = list(csv.DictReader((analysis / "sessions.csv").open()))
    intervals = list(csv.DictReader((analysis / "intervals.csv").open()))
    assert sessions[0]["cost_rate"] == "2.0"
    assert sessions[0]["complete"] == "True"
    assert sessions[0]["successful"] == "True"
    assert intervals[0]["cost_rate"] == "2.0"
    assert intervals[0]["worker_cpu_efficiency"] == "0.75"
    assert intervals[0]["host_core_occupancy"] == "0.5"
    assert "CPU transport supply" in (analysis / "summary.md").read_text()
    assert len(list((analysis / "timelines").glob("*.png"))) == 1


def test_analyze_marks_incomplete_session(tmp_path):
    _write_profile(tmp_path, [_record(0, "start"), _record(5, "tick")])

    result = performance_analysis.analyze_performance_profile(
        "baseline", tmp_path / "performance-profiles"
    )

    assert result["incomplete_sessions"] == 1
    sessions = list(
        csv.DictReader(
            (tmp_path / "performance-profiles" / "baseline" / "analysis" / "sessions.csv").open()
        )
    )
    assert sessions[0]["complete"] == "False"
    assert sessions[0]["successful"] == "False"
    assert "expected one terminal record" in sessions[0]["warnings"]


def test_analyze_excludes_failed_terminal_from_bottleneck_summary(tmp_path):
    _write_profile(
        tmp_path,
        [
            _record(0, "start"),
            _record(5, "tick"),
            _record(
                10,
                "failed",
                gpu_oom_retry_count_total=3,
                gpu_feed_wait_fraction=0.0,
            ),
        ],
    )

    result = performance_analysis.analyze_performance_profile(
        "baseline", tmp_path / "performance-profiles"
    )

    analysis = tmp_path / "performance-profiles" / "baseline" / "analysis"
    sessions = list(csv.DictReader((analysis / "sessions.csv").open()))
    assert sessions[0]["complete"] == "True"
    assert sessions[0]["successful"] == "False"
    assert result["incomplete_sessions"] == 0
    assert result["unsuccessful_sessions"] == 1
    summary = (analysis / "summary.md").read_text()
    assert "Sessions: 1 (0 valid successful)" in summary
    assert "Primary bottleneck: unknown" in summary
    assert "Primary bottleneck: GPU memory limit" not in summary


def test_analyze_marks_sampling_gap_and_counter_discontinuity(tmp_path):
    _write_profile(
        tmp_path,
        [
            _record(0, "start"),
            _record(5, "tick"),
            _record(20, "done", io_read_bytes=1, gpu_percent=None),
        ],
    )

    performance_analysis.analyze_performance_profile("baseline", tmp_path / "performance-profiles")

    intervals = list(
        csv.DictReader(
            (tmp_path / "performance-profiles" / "baseline" / "analysis" / "intervals.csv").open()
        )
    )
    assert intervals[1]["sampling_gap"] == "True"
    assert intervals[1]["counter_discontinuity"] == "True"
    assert intervals[1]["read_rate"] == ""
    assert intervals[1]["gpu_percent"] == ""


def test_analyze_does_not_call_preexisting_idle_swap_memory_pressure(tmp_path):
    idle_swap = {
        "swap_used_bytes": 16 * 1024**2,
        "swap_in_bytes": 70_828_032,
        "swap_out_bytes": 871_804_928,
        "gpu_feed_wait_fraction": 0.01,
        "gpu_percent": 5,
        "transport_seconds_total": 2,
        "spectrum_seconds_total": 8,
    }
    _write_profile(
        tmp_path,
        [
            _record(0, "start", **idle_swap),
            _record(5, "tick", **idle_swap),
            _record(10, "done", **idle_swap),
        ],
    )

    performance_analysis.analyze_performance_profile("baseline", tmp_path / "performance-profiles")

    summary = (
        tmp_path / "performance-profiles" / "baseline" / "analysis" / "summary.md"
    ).read_text()
    assert "spectrum launch/synchronization or host work" in summary
    assert "Primary bottleneck: memory pressure" not in summary


def test_analyze_calls_growing_swap_io_memory_pressure(tmp_path):
    _write_profile(
        tmp_path,
        [
            _record(0, "start", swap_used_bytes=16 * 1024**2),
            _record(5, "tick", swap_used_bytes=16 * 1024**2, swap_out_bytes=4096),
            _record(10, "done", swap_used_bytes=16 * 1024**2, swap_out_bytes=8192),
        ],
    )

    performance_analysis.analyze_performance_profile("baseline", tmp_path / "performance-profiles")

    summary = (
        tmp_path / "performance-profiles" / "baseline" / "analysis" / "summary.md"
    ).read_text()
    assert "Primary bottleneck: memory pressure" in summary


def test_analyze_rejects_schema_mismatch_without_creating_analysis(tmp_path):
    source = _write_profile(tmp_path, [{**_record(0, "start"), "schema": "future"}])

    with pytest.raises(performance_analysis.PerformanceAnalysisError, match="expected schema"):
        performance_analysis.analyze_performance_profile(
            "baseline", tmp_path / "performance-profiles"
        )

    assert source.exists()
    assert not (source.parents[1] / "analysis").exists()


def test_profile_analyze_cli(tmp_path):
    _write_profile(
        tmp_path,
        [_record(0, "start"), _record(5, "tick"), _record(10, "done")],
    )

    result = invoke(
        profile.command,
        [
            "analyze",
            "baseline",
            "--performance-dir",
            str(tmp_path / "performance-profiles"),
        ],
    )

    assert result.exit_code == 0
    assert result.stderr.count("is deprecated") == 1
    assert "cxr performance analyze NAME" in result.stderr
    assert "analyzed 1 sessions (2 intervals)" in result.stdout


def test_performance_analyze_cli(tmp_path):
    _write_profile(
        tmp_path,
        [_record(0, "start"), _record(5, "tick"), _record(10, "done")],
    )

    result = invoke(
        performance.command,
        [
            "analyze",
            "baseline",
            "--performance-dir",
            str(tmp_path / "performance-profiles"),
        ],
    )

    assert_clean_result(result)
    assert "analyzed 1 sessions (2 intervals)" in result.stdout
