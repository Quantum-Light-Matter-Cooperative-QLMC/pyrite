import json
from types import SimpleNamespace

import numpy as np

from cxr_mc.energy_grid.encoding import encode_energy_grid
from cxr_mc.montecarlo import runner
from cxr_mc.perf import performance_profile


def test_gpu_metrics_include_activity_clocks_and_pstate(monkeypatch):
    monkeypatch.setattr(
        performance_profile.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="75, 20, 1000, 2000, 125, 60, 1500, 7000, P2\n",
        ),
    )

    metrics = performance_profile._gpu_metrics()

    assert metrics["gpu_percent"] == 75.0
    assert metrics["gpu_memory_percent"] == 20.0
    assert metrics["gpu_sm_clock_mhz"] == 1500.0
    assert metrics["gpu_memory_clock_mhz"] == 7000.0
    assert metrics["gpu_pstate"] == "P2"


def test_performance_logger_writes_append_only_samples_and_latest(monkeypatch, tmp_path):
    monkeypatch.setattr(
        performance_profile,
        "_gpu_metrics",
        lambda: {
            "gpu_percent": 75.0,
            "vram_used_mib": 1000.0,
            "vram_total_mib": 2000.0,
            "vram_percent": 50.0,
            "gpu_power_watts": 125.0,
            "gpu_temperature_c": 60.0,
        },
    )
    path = tmp_path / "profile" / "hopg.ndjson"
    latest = tmp_path / "profile" / "hopg.latest.json"
    logger = performance_profile.PerformanceLogger(
        path,
        profile="baseline",
        material="hopg",
        static={"effective_workers": 3, "spec_chunk": 4000},
        context=lambda: {
            "completed_new_cases": 2,
            "current": {"energy_keV": 100.0},
            "state": "running",
        },
        latest_path=latest,
        interval_seconds=60,
    )

    logger.start()
    logger.close("done")

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == ["start", "done"]
    assert records[-1]["schema"] == "cxr.performance.v1"
    assert records[-1]["gpu_percent"] == 75.0
    assert records[-1]["effective_workers"] == 3
    assert records[-1]["current"]["energy_keV"] == 100.0
    assert records[-1]["cpu_affinity_count"] >= 1
    assert records[-1]["cpu_iowait_percent"] is not None
    assert records[-1]["swap_total_bytes"] >= 0
    assert records[-1]["process_cpu_user_seconds"] >= 0
    assert records[-1]["process_context_switches"] >= 0
    assert records[-1]["child_process_count"] >= 0
    assert records[-1]["child_process_rss_max_bytes"] >= 0
    assert json.loads(latest.read_text()) == records[-1]


def test_runtime_plan_reports_effective_workers_and_chunks(monkeypatch):
    monkeypatch.setattr(runner, "_GPU", False)
    monkeypatch.setattr(runner, "_mem_worker_cap", lambda: 8)
    cases = [
        {
            "E_grid": encode_energy_grid(np.arange(100.0, 200.0, 10.0)),
            "E_grid_brem": encode_energy_grid(np.arange(100.0, 300.0, 10.0)),
            "Ne": 50,
            "Ne_brem": 10,
        }
    ] * 4

    plan = runner.runtime_plan(cases, max_workers=3)

    assert plan["engine"] == "cpu-pool"
    assert plan["effective_workers"] == 3
    assert plan["line_grid_bins"] == 10
    assert plan["brem_grid_bins"] == 20
    assert plan["spec_chunk"] > 0
    assert plan["brem_chunk"] > 0


def test_timing_aggregate_exposes_phase_totals_and_strips_private_metrics():
    aggregate = runner._TimingAgg()
    out = {
        "_t_transport": 2.0,
        "_t_spectrum": 3.0,
        "_gpu_oom_retries": 1,
        "_cupy_pool_used_mib": 4.0,
        "_cupy_pool_reserved_mib": 5.0,
        "_cupy_pool_peak_mib": 6.0,
    }

    metrics = aggregate.collect(out, wait_seconds=0.5)

    assert out == {}
    assert metrics == {
        "timed_case_count": 1,
        "transport_seconds": 2.0,
        "spectrum_seconds": 3.0,
        "driver_wait_seconds": 0.5,
        "transport_seconds_total": 2.0,
        "spectrum_seconds_total": 3.0,
        "driver_wait_seconds_total": 0.5,
        "gpu_oom_retry_count": 1,
        "cupy_pool_used_mib": 4.0,
        "cupy_pool_reserved_mib": 5.0,
        "cupy_pool_peak_mib": 6.0,
    }


def test_timing_aggregate_emits_backend_neutral_allocator_identity():
    aggregate = runner._TimingAgg()
    out = {
        "_allocator_used_mib": 1.0,
        "_allocator_reserved_mib": 2.0,
        "_allocator_peak_mib": 3.0,
        "_backend": "sycl",
        "_backend_vendor": "intel",
        "_backend_device": "Arc",
        "_backend_fallback_reason": "accelerator_oom_retries_exhausted",
    }

    metrics = aggregate.collect(out)

    assert out == {}
    assert metrics["allocator_reserved_mib"] == 2.0
    assert metrics["backend"] == "sycl"
    assert metrics["backend_vendor"] == "intel"
    assert metrics["backend_device"] == "Arc"
    assert metrics["backend_fallback_reason"] == "accelerator_oom_retries_exhausted"


def test_run_cases_profile_callbacks_report_timing_and_activity(monkeypatch):
    def fake_run_case(_case, record_timing=False):
        assert record_timing
        return {
            "value": 1,
            "_t_transport": 2.0,
            "_t_spectrum": 3.0,
        }

    monkeypatch.setattr(runner, "run_case", fake_run_case)
    timings = []
    activities = []

    result = runner.run_cases(
        [{"crystal": "hopg"}],
        max_workers=0,
        progress=False,
        on_timing=timings.append,
        on_activity=activities.append,
    )

    assert result == [{"value": 1}]
    assert timings[0]["transport_seconds_total"] == 2.0
    assert timings[0]["spectrum_seconds_total"] == 3.0
    assert [activity["phase"] for activity in activities] == ["serial_case", "idle"]
