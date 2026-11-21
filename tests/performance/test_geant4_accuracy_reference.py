"""Keep long accuracy references complete, bounded and separate from timings."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "checks/geant4_performance"


def test_reference_shards_preserve_frozen_counts_and_have_disjoint_seeds():
    make_plan = runpy.run_path(str(HARNESS / "accuracy_reference.py"))["make_plan"]
    source = HARNESS / "precision-plan-v3.json"
    plan = make_plan(source)
    tasks = plan["tasks"]
    frozen = json.loads(source.read_text())
    for scene in ["si-300keV-100um", "si-3MeV-1mm"]:
        expected = next(
            i["histories"]
            for i in frozen["items"]
            if i["scene"] == scene
            and i["config"] == "geant4-32t"
            and i["set"] == "cone"
            and i["target_rse"] == 0.02
        )
        assert sum(t["histories"] for t in tasks if t["scene"] == scene) == expected
    assert all(2 <= t["histories"] <= 1_000_000 for t in tasks)
    assert len({t["seed"] for t in tasks}) == len(tasks)
    assert not {t["seed"] for t in tasks} & {1, 101, 102, 103, 104, 105, 106}
    assert plan["purpose"] == "accuracy-only"
    assert not plan["timing_comparison_eligible"]


def _reference(tmp_path):
    namespace = runpy.run_path(str(HARNESS / "accuracy_reference.py"))
    plan = dict(
        target_rse=1.0,
        threads=8,
        tasks=[dict(index=i, scene="si", histories=2, seed=200001 + i) for i in range(2)],
    )
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    digest = namespace["digest"](path)
    for task, hit in zip(plan["tasks"], [1.0, 2.0], strict=True):
        raw = tmp_path / f"shard-{task['index']:04d}.npz"
        raw.write_bytes(b"retained-primary-scores")
        record = dict(
            purpose="accuracy-only",
            reference_plan_sha256=digest,
            reference_shard=task["index"],
            scene={"id": "si"},
            seed=task["seed"],
            histories=2,
            threads=8,
            executable_sha256="same-binary",
            raw_scores_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            observables={
                "cone_source_photons_ge10kev": dict(
                    n=2,
                    sum=hit,
                    sum_squares=hit**2,
                    nonzero=1,
                )
            },
        )
        raw.with_suffix(".json").write_text(json.dumps(record))
    return namespace["summarize"], path


def test_reference_summary_uses_all_primaries_including_zero_scores(tmp_path):
    summarize, plan = _reference(tmp_path)
    result = summarize(plan, tmp_path)
    row = result["scenes"]["si"]
    assert row["n"] == 4
    assert row["nonzero"] == 2
    assert row["mean"] == 0.75
    assert row["se"] == pytest.approx((11 / 48) ** 0.5)
    assert not result["timing_comparison_eligible"]


@pytest.mark.parametrize("damage", ["missing", "raw", "seed", "binary"])
def test_incomplete_or_mismatched_reference_is_rejected(tmp_path, damage):
    summarize, plan = _reference(tmp_path)
    path = tmp_path / "shard-0001.json"
    if damage == "missing":
        path.unlink()
    elif damage == "raw":
        path.with_suffix(".npz").write_bytes(b"changed")
    else:
        record = json.loads(path.read_text())
        record["seed" if damage == "seed" else "executable_sha256"] = "changed"
        path.write_text(json.dumps(record))
    with pytest.raises((FileNotFoundError, ValueError)):
        summarize(plan, tmp_path)
