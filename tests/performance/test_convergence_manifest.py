"""Frozen numerical planning must retain missing refinements and rare headlines."""

import copy
import hashlib
import json
import math
import runpy
import tarfile
from io import BytesIO
from pathlib import Path
from statistics import NormalDist

import pytest

HERE = Path(__file__).resolve().parents[2] / "checks/geant4_performance"
NS = runpy.run_path(str(HERE / "convergence_manifest.py"))


def test_committed_design_is_reproducible_and_has_no_execution_or_gate_claim():
    frozen = json.loads((HERE / "convergence-design-v1.json").read_text())
    assert NS["make_manifest"](frozen["pilot_files_sha256"]) == frozen
    assert not frozen["headline"]
    assert not frozen["submission_ready"]
    assert frozen["numerical_gate"] == "unmeasured"
    assert frozen["resources"]["confirmation_histories"] is None
    rows = frozen["expected_rows"]
    assert len({r["id"] for r in rows}) == len(rows) == frozen["multiplicity"]
    assert frozen["row_alpha"] == 0.05 / len(rows)
    assert {r["support"] for r in rows} == {"available", "unsupported", "unresolved"}
    assert all(not c["submission_ready"] for c in frozen["configurations"])


def test_common_scales_use_only_prior_inputs_and_rare_members_stay():
    frozen = json.loads((HERE / "convergence-design-v1.json").read_text())
    source = json.loads((HERE / "precision-plan-v3.json").read_text())
    spec = json.loads((HERE / "scenes.json").read_text())
    scales = NS["scales_and_membership"](source, spec)
    assert scales == frozen["scales"]
    for scale in scales:
        assert scale["scale"] == source["cv"][scale["source_cv_key"]]["mean"]
        assert scale["scale"] > 0
    assert sum(s["workload"] == "cone" for s in scales) == 6
    # Existing frozen floors are preserved: stopped Si 3 MeV is diagnostic.
    assert not any(s["scene"] == "si-3MeV-1mm" and s["observable"] == "stopped" for s in scales)
    source["cv"][scales[0]["source_cv_key"]]["mean"] = 0
    with pytest.raises(ValueError, match="scale"):
        NS["scales_and_membership"](source, spec)


def test_family_budgets_include_unsupported_primitives_and_do_not_reassign():
    frozen = json.loads((HERE / "convergence-design-v1.json").read_text())
    for config in frozen["configurations"]:
        subset = [r for r in frozen["expected_rows"] if r["config"] == config["id"]]
        for observable in {r["observable"] for r in subset}:
            nominal = [
                r for r in subset if r["observable"] == observable and r["contrast"] == "nominal:L3"
            ]
            total = nominal[0]["total_allowance"]
            for family in NS["FAMILIES"]:
                ids = {c["id"] for c in config["controls"] if c["family"] == family}
                allocation = sum(r["allowance"] for r in nominal if r["control"] in ids)
                assert allocation == pytest.approx(total / 4) or allocation == 0
            assert sum(r["allowance"] for r in nominal) <= total * (1 + 1e-14)
            assert any(
                r["contrast"] == "nominal:all-L3" and r["observable"] == observable for r in subset
            )
    point = [r for r in frozen["expected_rows"] if r["control"] == "cone_point_approximation"]
    assert point and {r["contrast"] for r in point} == {"nominal:L3"}


def test_algorithm_precision_density_and_straggling_remain_distinct():
    arms = NS["arms"]()
    assert len(arms) == 10
    assert sum(a["engine"] == "pyrite" for a in arms) == 6
    assert {a["precision"] for a in arms if a["algorithm"] == "mixed"} >= {"fp64", "mixed"}
    assert next(a for a in arms if a["engine"] == "penred")["straggling"]
    penred = NS["controls"]("penred", "mixed", 100)
    assert next(c for c in penred if c["id"] == "dsmax_cm")["levels"] == [
        0.001,
        0.0005,
        0.00025,
        0.000125,
    ]
    geant = NS["controls"]("geant4", "wvi", 100)
    assert next(c for c in geant if c["id"] == "msc_range_factor")["levels"] == [
        0.04,
        0.02,
        0.01,
        0.005,
    ]


@pytest.mark.parametrize("variance,allowance,m", [(0.1, 0.001, 100), (1.0, 0.005, 9000)])
def test_confirmation_forecast_has_margin_three_equal_repeats_and_half_budget(
    variance, allowance, m
):
    count = NS["confirmation_count"](variance, allowance, m)
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * m))
    assert count % 3 == 0 and count >= 60000
    assert count >= math.ceil(1.25 * 4 * z**2 * variance / allowance**2)
    assert z * math.sqrt(variance / count) <= allowance / 2


@pytest.mark.parametrize(
    "values",
    [(0, 1, 1), (-1, 1, 1), (1, 0, 1), (math.nan, 1, 1), (1, math.inf, 1), (1, 1, 0), (1, 1, True)],
)
def test_invalid_or_zero_uncertainty_is_not_a_count_forecast(values):
    with pytest.raises(ValueError):
        NS["confirmation_count"](*values)


def _archive(tmp_path):
    record = "evidence/precision/14460/pyrite-scene-101.json"
    raw = record.replace(".json", ".npz")
    path = tmp_path / "precision-14460.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name in (record, raw):
            data = name.encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, BytesIO(data))
    inventory = dict(
        v2_jobs=[14460],
        archives=[dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())],
    )
    return record, raw, inventory


def test_archive_verification_hashes_both_input_records_and_raw_scores(tmp_path):
    record, raw, inventory = _archive(tmp_path)
    result = NS["pilot_files"](tmp_path, dict(pilot=[record]), inventory)
    assert result == {name: hashlib.sha256(name.encode()).hexdigest() for name in (record, raw)}
    broken = copy.deepcopy(inventory)
    broken["archives"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        NS["pilot_files"](tmp_path, dict(pilot=[record]), broken)
    with pytest.raises(ValueError, match="missing"):
        NS["pilot_files"](tmp_path, dict(pilot=[record, "missing-103.json"]), inventory)
    with pytest.raises(ValueError, match="duplicate"):
        NS["pilot_files"](tmp_path, dict(pilot=[record, record]), inventory)


def test_input_hash_list_cannot_omit_a_planning_input():
    frozen = json.loads((HERE / "convergence-design-v1.json").read_text())
    files = frozen["pilot_files_sha256"]
    files.pop(next(iter(files)))
    with pytest.raises(ValueError, match="hashes"):
        NS["make_manifest"](files)


def test_freeze_cannot_be_silently_replaced_or_shrink_multiplicity(tmp_path):
    path = tmp_path / "freeze.json"
    original = dict(multiplicity=10, expected_rows=[dict(id="retained")])
    NS["write_manifest"](path, original)
    before = path.read_bytes()
    NS["write_manifest"](path, original)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="overwrite"):
        NS["write_manifest"](path, dict(multiplicity=9, expected_rows=[]))
    assert json.loads(path.read_text()) == original


def test_check_rejects_removed_unsupported_row(tmp_path, monkeypatch):
    frozen = json.loads((HERE / "convergence-design-v1.json").read_text())
    row = next(r for r in frozen["expected_rows"] if r["support"] == "unsupported")
    frozen["expected_rows"].remove(row)
    frozen["multiplicity"] -= 1
    path = tmp_path / "edited-design.json"
    path.write_text(json.dumps(frozen))
    monkeypatch.setattr("sys.argv", ["convergence_manifest.py", "--check", "--output", str(path)])
    with pytest.raises(ValueError, match="never shrink M"):
        NS["main"]()
