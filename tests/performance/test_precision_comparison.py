"""Precision comparisons must distinguish bias from unresolved Monte Carlo error."""

import runpy
from pathlib import Path

import numpy as np
import pytest


def compare():
    root = Path(__file__).resolve().parents[2]
    return runpy.run_path(str(root / "checks/equal_precision/analyze_precision.py"))[
        "compare_scores"
    ]


def test_comparison_passes_only_when_entire_interval_fits_margin():
    values = np.tile([0.0, 1.0], 500)
    result = compare()(values, values, margin=0.2, comparisons=1)
    assert result["status"] == "passed"
    assert result["difference_ci"][0] < 0 < result["difference_ci"][1]
    assert result["difference_se_bound"] == pytest.approx(
        2 * np.std(values, ddof=1) / np.sqrt(values.size)
    )


def test_comparison_marks_clear_bias_failed():
    values = np.tile([0.0, 1.0], 500)
    assert compare()(values, values + 1, margin=0.2, comparisons=1)["status"] == "failed"


def test_comparison_does_not_accept_imprecise_or_constant_scores():
    assert compare()([0.0, 1.0], [0.0, 1.0], margin=0.01, comparisons=10)["status"] == "unresolved"
    assert compare()([0.0, 0.0], [0.0, 0.0], margin=0.01, comparisons=1)["status"] == "unresolved"


def test_comparison_rare_primary_floor_is_explicit():
    values = np.zeros(1000)
    values[0] = 1
    assert (
        compare()(values, values, margin=1, comparisons=1, min_nonzero=20)["status"] == "unresolved"
    )


def test_comparison_refuses_nonfinite_scores():
    with pytest.raises(ValueError, match="finite"):
        compare()([0.0, float("nan")], [0.0, 1.0], margin=0.1, comparisons=1)


@pytest.mark.parametrize("mutation", [None, "source", "cutoff"])
def test_analysis_refuses_pooling_changed_precision_sources_or_controls(
    tmp_path, monkeypatch, mutation
):
    import hashlib
    import json
    import sys

    root = Path(__file__).resolve().parents[2]
    for variant in ["fp64", "rad-fp32-24"]:
        for seed in [1, 2]:
            path = tmp_path / f"{variant}-{seed}"
            np.savez_compressed(
                path.with_suffix(".npz"), photons_ge10kev=[0.0, 1.0], transmitted=[0.0, 1.0]
            )
            altered = variant == "rad-fp32-24" and seed == 2
            record = {
                "schema": "pyrite.equal-precision.precision-sample.v1",
                "scene": {"id": "si", "energy_keV": 300},
                "histories": 2,
                "seed": seed,
                "precision_variant": variant,
                "manifest_sha256": "scene-input",
                "precision_source": {
                    "reference_sha256": "reference",
                    "variant_sha256": "changed" if altered and mutation == "source" else variant,
                },
                "max_de_frac": 0.05,
                "cutoff_eV": 2000 if altered and mutation == "cutoff" else 1000,
                "warmup": {
                    "histories": 2,
                    "same_scene_tables_numerics": True,
                    "device_synchronized": True,
                },
                "timings_s": {"transport_and_emission": 1.0},
                "raw_scores_sha256": hashlib.sha256(
                    path.with_suffix(".npz").read_bytes()
                ).hexdigest(),
            }
            path.with_suffix(".json").write_text(json.dumps(record))
    target = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["analyze", str(tmp_path), "--output", str(target)])
    main = runpy.run_path(str(root / "checks/equal_precision/analyze_precision.py"))["main"]
    if mutation is not None:
        with pytest.raises(ValueError, match="replica.*source|replica.*control"):
            main()
    else:
        main()
        summary = json.loads(target.read_text())
        assert sum(r["replicas"] for r in summary["results"]) == 4
        assert summary["qualified_T_epsilon"] is False


def test_analysis_rejects_a_different_frozen_protocol_before_loading_scores(tmp_path, monkeypatch):
    import json
    import sys

    root = Path(__file__).resolve().parents[2]
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes((root / "checks/equal_precision/precision-endpoint.json").read_bytes())
    (tmp_path / "sample.json").write_text(
        json.dumps(
            {
                "schema": "pyrite.equal-precision.precision-sample.v1",
                "precision_protocol_sha256": "another-protocol",
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze",
            str(tmp_path),
            "--protocol",
            str(protocol),
            "--output",
            str(tmp_path / "summary.json"),
        ],
    )
    main = runpy.run_path(str(root / "checks/equal_precision/analyze_precision.py"))["main"]
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        main()
