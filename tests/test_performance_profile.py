import json

import numpy as np

from cxr_mc import performance_profile
from cxr_mc._energy_grid import encode_energy_grid
from cxr_mc.montecarlo import runner


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
