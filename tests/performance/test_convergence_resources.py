"""Pilot forecasts retain frozen membership and never authorize confirmation."""

import copy
import hashlib
import json
import math
import runpy
from pathlib import Path
from statistics import NormalDist

import pytest

from pyrite.perf._convergence_resources import resource_plan


def design():
    return dict(
        schema="pyrite.geant4-performance.convergence-design.v1",
        multiplicity=2,
        row_alpha=0.025,
        family_alpha=0.05,
        eligibility=dict(variance_safety_factor=1.25, repeats=3, min_histories_per_repeat=20000),
        resources=dict(
            confirmation_cpu_core_hours=256,
            confirmation_gpu_hours=8,
            max_cpus=8,
            max_concurrent_cpu_shards=4,
            max_shard_minutes=30,
            gpu_allocations_at_once=1,
        ),
        expected_rows=[
            dict(id="a", allowance=0.01, support="available"),
            dict(id="b", allowance=0.001, support="unsupported"),
        ],
    )


def pilot():
    return dict(
        phase="pilot",
        seeds=[301, 302, 303],
        rows=[
            dict(
                id="a",
                sampling="independent",
                variance=0.02,
                arms=[arm(), arm()],
            )
        ],
    )


def arm():
    return dict(
        histories=20000,
        elapsed_seconds=10.0,
        overhead_seconds=2.0,
        cpus=4,
        gpus=0,
        peak_memory_bytes=1024,
        peak_vram_bytes=0,
        memory_margin_factor=1.5,
        hardware_identity="cpu-node-type",
        runtime_identity="sha256:runtime",
    )


def plan(d=None, p=None):
    return resource_plan(design() if d is None else d, pilot() if p is None else p)


def test_forecast_uses_independent_variance_sum_three_repeats_and_full_cost():
    result = plan()
    row = result["rows"][0]
    z = NormalDist().inv_cdf(1 - 0.05 / 4)
    required = 1.25 * 4 * z**2 * 0.02 / 0.01**2
    assert row["histories_per_arm"] >= max(60000, math.ceil(required))
    assert row["histories_per_arm"] % 3 == 0
    assert z * math.sqrt(0.02 / row["histories_per_arm"]) <= 0.005
    assert len(row["arms"]) == 2
    total_seconds = 2 * (row["histories_per_arm"] * 10 / 20000 + 3 * 2)
    assert result["forecast_cpu_core_hours"] == pytest.approx(total_seconds * 4 / 3600)
    assert all(a["memory_request_bytes"] == 1536 for a in row["arms"])
    assert all(a["peak_memory_bytes"] == 1024 for a in row["arms"])
    assert result["rows"][1]["status"] == "unsupported"
    assert not result["submission_ready"] and not result["headline"]
    assert result["numerical_gate"] == "unmeasured"


def test_paired_population_is_charged_once_and_missing_rows_never_disappear():
    p = pilot()
    p["rows"][0].update(sampling="paired", arms=[arm()])
    result = plan(p=p)
    assert len(result["rows"][0]["arms"]) == 1
    assert len(result["rows"]) == 2
    assert result["multiplicity"] == 2
    assert plan(p=dict(phase="pilot", seeds=[301], rows=[]))["rows"][0]["status"] == "missing"


def test_zero_variance_cannot_forecast_zero_confirmation():
    p = pilot()
    p["rows"][0]["variance"] = 0
    result = plan(p=p)
    assert result["rows"][0]["status"] == "unresolved"
    assert result["forecast_cpu_core_hours"] == 0
    assert not result["cost_complete"]


def test_time_ceiling_splits_each_repeat_and_charges_overhead_every_shard():
    p = pilot()
    for a in p["rows"][0]["arms"]:
        a.update(elapsed_seconds=10000, overhead_seconds=100, gpus=1, peak_vram_bytes=2048)
    result = plan(p=p)
    row = result["rows"][0]
    for a in row["arms"]:
        assert a["shards_per_repeat"] > 1
        assert a["max_shard_seconds"] <= 1800
        assert a["vram_request_bytes"] == 3072
    assert result["forecast_gpu_hours"] > 8
    assert result["budget_status"] == "over-budget"
    assert result["forecast_cpu_core_hours"] > 0  # GPU allocations consume CPU resources too.


def test_global_ceiling_is_checked_even_when_individual_rows_fit():
    d, p = design(), pilot()
    d["expected_rows"][1].update(support="available", allowance=0.01)
    p["rows"].append(copy.deepcopy(p["rows"][0]) | dict(id="b"))
    one_cost = plan()["forecast_cpu_core_hours"]
    d["resources"]["confirmation_cpu_core_hours"] = 1.5 * one_cost
    result = plan(d, p)
    assert all(r["status"] == "forecast" for r in result["rows"])
    assert result["budget_status"] == "over-budget"
    assert result["cost_complete"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d, p: p.update(phase="confirmation"),
        lambda d, p: p.update(seeds=[401]),
        lambda d, p: p["rows"].append(p["rows"][0]),
        lambda d, p: p["rows"][0].update(id="unknown"),
        lambda d, p: p["rows"][0].update(variance=math.nan),
        lambda d, p: p["rows"][0].update(sampling="paired"),
        lambda d, p: p["rows"][0]["arms"][0].update(cpus=9),
        lambda d, p: p["rows"][0]["arms"][0].update(memory_margin_factor=0.5),
        lambda d, p: p["rows"][0]["arms"][0].update(histories=True),
        lambda d, p: d.update(multiplicity=1),
        lambda d, p: d.update(row_alpha=0.05),
    ],
)
def test_malformed_evidence_and_shrunk_multiplicity_fail(mutation):
    d, p = design(), pilot()
    mutation(d, p)
    with pytest.raises(ValueError):
        plan(d, p)


def test_single_primary_over_shard_limit_is_over_budget():
    p = pilot()
    p["rows"][0]["arms"][0].update(overhead_seconds=1800)
    result = plan(p=p)
    assert result["rows"][0]["status"] == "over-budget"
    assert result["budget_status"] == "over-budget"
    assert not result["cost_complete"]


def test_extremely_tight_allowance_is_over_budget_not_an_overflow():
    d = design()
    d["expected_rows"][0]["allowance"] = 1e-200
    assert plan(d=d)["rows"][0]["status"] == "over-budget"


def _pilot_files(tmp_path):

    here = Path(__file__).resolve().parents[2] / "checks/geant4_performance"
    ns = runpy.run_path(str(here / "prepare_convergence_resources.py"))
    d = tmp_path / "design.json"
    d.write_text(json.dumps(design()))
    raw = tmp_path / "pilot-301.npz"
    raw.write_bytes(b"synthetic retention test; not measured evidence")
    p = pilot() | dict(
        schema="pyrite.geant4-performance.convergence-pilot-summary.v1",
        design_sha256=hashlib.sha256(d.read_bytes()).hexdigest(),
        evidence_files_sha256={raw.name: hashlib.sha256(raw.read_bytes()).hexdigest()},
    )
    p["rows"][0]["evidence_files"] = [raw.name]
    summary = tmp_path / "pilot.json"
    summary.write_text(json.dumps(p))
    return ns, d, summary, raw


def test_retention_wrapper_pins_design_summary_and_every_raw_file(tmp_path):

    ns, d, summary, raw = _pilot_files(tmp_path)
    result = ns["make_plan"](d, summary)
    assert result["design_sha256"] == hashlib.sha256(d.read_bytes()).hexdigest()
    assert result["pilot_summary_sha256"] == hashlib.sha256(summary.read_bytes()).hexdigest()
    assert result["evidence_files_sha256"][raw.name] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert result["pilot_row_evidence"] == {"a": [raw.name]}
    raw.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        ns["make_plan"](d, summary)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda p: p.update(design_sha256="0" * 64), "frozen design"),
        (lambda p: p["rows"][0].update(evidence_files=[]), "retained hashed evidence"),
        (lambda p: p["rows"][0].update(evidence_files=["missing"]), "retained hashed evidence"),
        (lambda p: p.update(evidence_files_sha256={"../escape": "0" * 64}), "relative"),
    ],
)
def test_retention_refuses_unpinned_or_unreferenced_inputs(tmp_path, mutation, match):

    ns, d, summary, _ = _pilot_files(tmp_path)
    p = json.loads(summary.read_text())
    mutation(p)
    summary.write_text(json.dumps(p))
    with pytest.raises(ValueError, match=match):
        ns["make_plan"](d, summary)


def test_cli_refuses_to_replace_existing_freeze(tmp_path, monkeypatch):
    ns, d, summary, _ = _pilot_files(tmp_path)
    output = tmp_path / "plan.json"
    args = [
        "prepare_convergence_resources.py",
        "--design",
        str(d),
        "--pilots",
        str(summary),
        "--output",
        str(output),
    ]
    monkeypatch.setattr("sys.argv", args)
    ns["main"]()
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        ns["main"]()
    assert output.read_bytes() == before


def test_full_frozen_design_keeps_all_7074_rows_with_no_pilots():

    here = Path(__file__).resolve().parents[2] / "checks/geant4_performance"
    frozen = json.loads((here / "convergence-design-v1.json").read_text())
    before = copy.deepcopy(frozen)
    result = plan(frozen, dict(phase="pilot", seeds=[301, 302, 303], rows=[]))
    assert frozen == before
    assert len(result["rows"]) == result["multiplicity"] == 7074
    assert [r["id"] for r in result["rows"]] == [r["id"] for r in frozen["expected_rows"]]
    assert result["row_alpha"] == 0.05 / 7074
    assert {r["status"] for r in result["rows"]} == {"missing", "unsupported", "unresolved"}
    assert result["budget_status"] == "incomplete"
    assert result["forecast_cpu_core_hours"] == result["forecast_gpu_hours"] == 0
    assert not result["submission_ready"] and not result["cost_complete"]
