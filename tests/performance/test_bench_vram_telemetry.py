"""Sampled device-wide VRAM peak for pilot resource forecasts; no CUDA required."""

import runpy
from pathlib import Path
from types import SimpleNamespace

HARNESS = Path(__file__).resolve().parents[2] / "checks/geant4_performance/pyrite_bench.py"
NS = runpy.run_path(str(HARNESS))


def test_sampler_keeps_highest_used_and_labels_upper_bound():
    free = iter([900, 400, 700] + [800] * 10_000)
    runtime = SimpleNamespace(memGetInfo=lambda: (next(free), 1000))
    cp = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
    sampler = NS["DevicePeakSampler"](cp, interval_s=0.001)
    sampler.sample()
    sampler.sample()
    record = sampler.finish()
    assert record["peak_device_used_bytes"] == 600
    assert record["samples"] >= 4
    assert "upper bound" in record["method"]
    assert not sampler.thread.is_alive()
