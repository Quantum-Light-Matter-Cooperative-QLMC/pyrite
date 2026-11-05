"""Statistical reporting for the #278/#286 replica ladder; no transport runs."""

import math
from types import SimpleNamespace

import pytest

from checks.electron_count_accuracy import plan, summarize


def run(n, seed, value, *, centroid=True):
    obs = dict.fromkeys(
        (
            "yield",
            "timepix3_counts",
            "eaglexo_counts",
            "continuum_yield",
            "continuum_centroid_eV",
            "timepix3_continuum_counts",
            "eaglexo_continuum_counts",
        ),
        value,
    )
    if centroid:
        obs["centroid_eV"] = 1000 * value
    return {"energy_keV": 300.0, "n": n, "seed": seed, "observables": obs}


def test_reference_uncertainty_and_single_run_vs_mean_se():
    state = {
        "reference_count": 400,
        "runs": [run(400, 9, 9), run(400, 8, 11), run(100, 1, 8), run(100, 2, 12)],
    }
    table = summarize(state)["300keV"]
    row = table["yield"]["rows"][0]
    # Independent sample SDs: sqrt(8) for the ladder, sqrt(2) for reference.
    assert row["relative_se"] == pytest.approx(math.sqrt(8) / 10)
    assert row["relative_mean_se"] == pytest.approx(0.2)
    assert row["difference_se"] == pytest.approx(math.sqrt(5))
    assert row["bias_vs_reference"] == 0
    assert table["yield"]["reference_relative_se"] == pytest.approx(0.1)
    assert table["centroid_eV"]["population"] == "Ne"


def test_inverse_sqrt_fit_and_legacy_checkpoint():
    state = {
        "reference_count": 1600,
        "runs": [run(1600, 9, 10, centroid=False)],
    }
    for n, delta in [(100, 2), (400, 1)]:
        state["runs"] += [run(n, 1, 10 - delta), run(n, 2, 10 + delta)]
    table = summarize(state)["300keV"]
    assert "centroid_eV" not in table  # Not recoverable from the old scalar record.
    assert table["yield"]["reference_relative_se"] is None
    assert table["yield"]["rows"][0]["difference_se"] is None
    assert table["yield"]["fitted_exponent"] == pytest.approx(-0.5)
    assert table["yield"]["inverse_sqrt_coefficient"] == pytest.approx(math.sqrt(8))


def test_reference_replica_seeds_are_disjoint_from_ladder():
    args = SimpleNamespace(
        energy_kev=[300, 29999],
        reference_count=3200,
        reference_seed=999,
        reference_replicas=3,
        counts=[100, 400, 1600],
        seed0=1000,
        replicas=5,
    )
    planned = plan(args)
    assert len(planned) == len(set(planned)) == 36
    assert [seed for energy, n, seed in planned if energy == 300 and n == 3200] == [999, 998, 997]


def test_resume_replaces_legacy_rows_without_duplicate_replicas(tmp_path, monkeypatch):
    import json
    import sys

    from checks import electron_count_accuracy as diagnostic

    path = tmp_path / "ladder.json"
    legacy = [run(400, 999, 10, centroid=False), run(100, 1000, 9, centroid=False)]
    path.write_text(json.dumps({"reference_count": 400, "runs": legacy}))
    calls = []

    def fake_run(args, energy, n, seed):
        calls.append((n, seed))
        value = run(n, seed, 10)
        value["seconds"] = {}
        for name in diagnostic.LINE + diagnostic.CONTINUUM:
            value["observables"].setdefault(name, 10)
        return value

    monkeypatch.setattr(diagnostic, "one_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "accuracy",
            "--energy-kev",
            "300",
            "--counts",
            "100",
            "--replicas",
            "2",
            "--reference-count",
            "400",
            "--output",
            str(path),
        ],
    )
    diagnostic.main()
    assert calls == [(400, 999), (100, 1000), (100, 1001)]
    saved = json.loads(path.read_text())
    assert len(saved["runs"]) == 3
    assert "centroid_eV" in saved["summary"]["300keV"]
    calls.clear()
    diagnostic.main()
    assert calls == []


def test_markdown_distinguishes_measured_uncertainty_and_projection():
    from checks.electron_count_accuracy import markdown_report

    state = {
        "reference_count": 1600,
        "runs": [run(1600, 9, 10), run(1600, 8, 10)],
    }
    for n, delta in [(100, 2), (400, 1)]:
        state["runs"] += [run(n, 1, 10 - delta), run(n, 2, 10 + delta)]
    report = markdown_report(state)
    assert "single-run RSE | mean RSE" in report
    assert "not established systematic bias" in report
    # Independent analytic result C=sqrt(8), N=(C/0.001)^2.
    assert "Conditional N for budget 0.001: 8000000" in report
    assert "Fitted exponent: -0.5" in report
    assert "Ne_brem: continuum_yield" in report
    assert "Ne: yield" in report
