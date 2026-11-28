"""Statistical equivalence alone cannot qualify the #1041 speed headline."""

import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("version", [2, 3])
def test_missing_numerical_ladder_keeps_equivalent_ratio_qualified(version):
    script = Path(__file__).resolve().parents[2] / "checks/geant4_performance/analyze.py"
    equivalence = runpy.run_path(str(script))["equivalence"]
    scores = np.tile([0.0, 1.0], 5000)
    configs = ["pyrite-cuda", "geant4-32t"]
    samples = [
        dict(
            engine=engine,
            threads=32,
            scene={"id": "si"},
            histories=len(scores),
            scores={"transmitted": scores},
        )
        for engine in ["pyrite", "geant4"]
    ]
    items = [
        dict(
            scene="si",
            config=config,
            set="transport",
            target_rse=0.02,
            histories=len(scores),
            status="planned",
            per_observable_histories={"transmitted": len(scores)},
        )
        for config in configs
    ]
    ratio = dict(
        scene="si",
        set="transport",
        target_rse=0.02,
        pyrite="pyrite-cuda",
        geant4="geant4-32t",
        floor_bound=False,
    )
    ttp = dict(runs=[dict(i, status="met") for i in items], speedup=[ratio])
    spec = dict(
        evaluate_at_target_rse=0.02,
        ci_level=0.9,
        margins_rel={"transmitted": 0.03},
        reference="geant4-32t",
    )
    amendment = dict(skew_shift_over_se=0.1) if version == 3 else None
    result = equivalence(samples, {"items": items}, spec, ttp, amendment)

    assert result["groups"][0]["status"] == "equivalent"
    assert ratio["statistical_gates_passed"]
    assert not ratio["headline"]
    assert ratio["numerical_convergence"]["status"] == "unverified"
    assert ratio["numerical_convergence"]["engines"] == configs


def test_accuracy_references_cannot_enter_timing_comparisons():
    script = Path(__file__).resolve().parents[2] / "checks/geant4_performance/analyze.py"
    compare = runpy.run_path(str(script))["time_to_precision"]
    with pytest.raises(ValueError, match="accuracy-only"):
        compare([{"purpose": "accuracy-only"}], {})
