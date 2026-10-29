"""Reject incomplete replica evidence before pooling benchmark uncertainty."""

import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest


def _write_sample(root, name, seed, scores):
    path = root / name
    np.savez_compressed(path.with_suffix(".npz"), **scores)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "pyrite.equal-precision.sample.v1",
                "engine": "geant4",
                "manifest_sha256": "frozen",
                "scene": {"id": "si"},
                "histories": 2,
                "seed": seed,
                "raw_scores_sha256": hashlib.sha256(
                    path.with_suffix(".npz").read_bytes()
                ).hexdigest(),
                "timings_s": {"fresh_process_with_logging": 1.0},
            }
        )
    )
    path.with_suffix(".wall.json").write_text(json.dumps({"wall_s": 1.1}))


def _analyze(root, monkeypatch):
    target = root / "summary"
    monkeypatch.setattr(sys, "argv", ["analyze", str(root), "--output", str(target)])
    script = Path(__file__).resolve().parents[2] / "checks/equal_precision/analyze.py"
    namespace = runpy.run_path(str(script))
    namespace["main"].__globals__["plot"] = lambda *args: None
    namespace["main"]()
    return json.loads((target / "summary.json").read_text())


@pytest.mark.parametrize("incomplete_first", [False, True])
def test_inconsistent_replica_observables_rejected(tmp_path, monkeypatch, incomplete_first):
    complete = {"photons_ge10kev": [0.0, 1.0], "transmitted": [1.0, 0.0]}
    incomplete = {"photons_ge10kev": [0.0, 1.0]}
    samples = [incomplete, complete] if incomplete_first else [complete, incomplete]
    for seed, scores in enumerate(samples):
        _write_sample(tmp_path, f"sample-{seed}", seed, scores)

    with pytest.raises(ValueError, match="observable fields differ"):
        _analyze(tmp_path, monkeypatch)
    assert not (tmp_path / "summary" / "summary.json").exists()


@pytest.mark.parametrize("scores", [{}, {"transmitted": [1.0, 0.0]}])
def test_missing_headline_observable_rejected_before_summary(tmp_path, monkeypatch, scores):
    _write_sample(tmp_path, "sample", 1, scores)
    with pytest.raises(ValueError, match="missing required observable.*photons_ge10kev"):
        _analyze(tmp_path, monkeypatch)
    assert not (tmp_path / "summary" / "summary.json").exists()


def test_complete_replica_observables_use_all_incident_primaries(tmp_path, monkeypatch):
    _write_sample(
        tmp_path,
        "sample-1",
        1,
        {"photons_ge10kev": [0.0, 1.0], "transmitted": [1.0, 0.0]},
    )
    _write_sample(
        tmp_path,
        "sample-2",
        2,
        {"photons_ge10kev": [2.0, 0.0], "transmitted": [1.0, 1.0]},
    )
    result = _analyze(tmp_path, monkeypatch)["results"][0]
    assert result["replicas"] == 2
    for moments in result["observables"].values():
        assert moments["n"] == 4
        assert moments["mean"] == 0.75
    photons = result["observables"]["photons_ge10kev"]
    assert photons["se"] == pytest.approx(np.sqrt(11 / 48))
    assert photons["replica_se"] == pytest.approx(0.25)

    table = (tmp_path / "summary" / "table.md").read_text(encoding="utf-8")
    assert "≥10 keV" in table


@pytest.mark.parametrize("suffix", [".failed", ".failed.json"])
def test_kernel_diagnostic_excludes_late_failed_attempt(tmp_path, monkeypatch, suffix):
    # A timed-out child can leave an audit after its launcher records failure.
    # Failed evidence must be skipped before attempting to parse or rank it.
    (tmp_path / "sample-audit.json").write_text("late invalid audit")
    (tmp_path / f"sample{suffix}").write_text("failed")
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["analyze", str(tmp_path), "--output", str(output)])
    script = (
        Path(__file__).resolve().parents[2] / "checks/equal_precision/analyze_kernel_diagnostic.py"
    )
    runpy.run_path(str(script), run_name="__main__")
    result = json.loads(output.read_text())
    assert result["groups"] == result["samples"] == []
