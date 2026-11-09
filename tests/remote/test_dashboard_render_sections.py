"""Behavioral contracts for shared local/remote dashboard section rendering."""

import json

import pytest

from pyrite.console.dashboard import render


def _progress(material, *, state="running", completed=2, **extra):
    return json.dumps(
        {
            "material": material,
            "total_cases": 10,
            "cached_cases": 0,
            "completed_new_cases": completed,
            "state": state,
            **extra,
        }
    )


def test_diagnostic_status_base_output_is_stable():
    sections = {
        "JOB": "diag",
        "STATE": "running",
        "SQUEUE": "job_id=12|state=RUNNING|partition=gpu",
        "META": (
            "job: diag\nkind: line-grid-bounds\nenergies: 40,60\n"
            "slice_minutes: 2\njson_out: /tmp/grid.json"
        ),
    }
    assert render.format_job_status(sections, 0) == (
        "JOB diag\n"
        "  State     RUNNING\n"
        "  Kind      line-grid-bounds\n"
        "  Energies  40,60 keV\n"
        "  Slice     2 min soft / 6 min hard\n"
        "  Output    /tmp/grid.json"
    )

    verbose = render.format_job_status(sections, 2)
    assert "CASE PROGRESS" not in verbose
    assert verbose.index("ALLOCATION") < verbose.index("COMPUTE USAGE")
    assert verbose.index("COMPUTE USAGE") < verbose.index("RECENT LOG")


def test_queued_status_keeps_priority_order_and_pauses_progress():
    sections = {
        "JOB": "queued",
        "META": "job: queued\nmaterials: hopg hbn\nslurm_job_id: 12\nquick: True",
        "STATE": "queued slice 2",
        "SQUEUE": "job_id=12|state=PENDING|partition=gpu",
        "QUEUE": (
            "cohort_partition=gpu|order=priority_desc_job_id_asc\n"
            "job_id=10|priority=200|state=PENDING|partition=gpu|"
            "user=lead|name=first|reason=Priority\n"
            "job_id=12|priority=100|state=PENDING|partition=gpu|"
            "user=me|name=second|reason=Resources"
        ),
        "PROGRESS": _progress("hopg", state="done", completed=10),
    }
    output = render.format_job_status(sections, 1)
    assert "State           PENDING" in output
    assert "Queue position  2/2 pending in gpu" in output
    assert "Queue leader    10 · lead/first · Priority" in output
    assert "Priority snapshot; priority can change" in output
    assert "Ⅱ" in output  # the queued job-wide bar overrides per-material state
    assert "Materials       1/2 complete" in output
    assert output.index("CASE PROGRESS") < output.index("ALLOCATION")


def test_cpu_phase_headline_uses_cpu_but_details_keep_both_phases():
    sections = {
        "JOB": "cpu",
        "META": "job: cpu\nmaterials: hopg\nquick: True\ncpu_only: False",
        "STATE": "profiling CPU hopg [1/1]",
        "SQUEUE": "job_id=41|state=RUNNING",
        "PROGRESS": "\n".join(
            [
                _progress("hopg", state="done", completed=10, phase="primary"),
                _progress("hopg", state="running", completed=2, phase="cpu"),
            ]
        ),
    }
    output = render.format_job_status(sections, 0)
    assert "Phase" in output and "profiling CPU hopg [1/1]" in output
    assert "Materials" in output and "0/1 complete" in output
    assert "20%" in output
    assert "[CPU]" in output
    assert output.count("HOPG") == 2


@pytest.mark.parametrize("detail", [0, 1, 2])
def test_cost_weighted_status_keeps_verbosity_contract(detail):
    sections = {
        "JOB": "cost",
        "META": "job: cost\nmaterials: hopg\nquick: True",
        "STATE": "running hopg",
        "SQUEUE": "job_id=41|state=RUNNING",
        "PROGRESS": _progress(
            "hopg",
            completed=5,
            active_compute_seconds=30.0,
            sample_compute_seconds=20.0,
            sample_new_cost=10.0,
            done_cost=30.0,
            total_cost=100.0,
        ),
    }
    output = render.format_job_status(sections, detail)
    assert " 30%" in output
    assert "ETA ~2m 20s" in output
    assert "CASE PROGRESS" in output
    if detail == 0:
        assert "Progress (compute)" not in output
        assert "Progress (cases)" not in output
        assert "ALLOCATION" not in output
    else:
        assert "Progress (compute)" in output
        assert "Progress (cases)" in output
        assert " 50%" in output
        assert "ALLOCATION" in output
    assert ("COMPUTE USAGE" in output) == (detail == 2)
    assert ("RECENT LOG" in output) == (detail == 2)


@pytest.mark.parametrize(
    ("parallel", "remaining", "total"),
    [(1, 180.0, 200.0), (2, 100.0, 120.0)],
)
def test_time_estimate_schedules_queued_after_started_work(parallel, remaining, total):
    records = {
        "hopg": {
            "material": "hopg",
            "state": "running",
            "total_cases": 10,
            "cached_cases": 0,
            "completed_new_cases": 2,
            "active_compute_seconds": 20.0,
            "sample_compute_seconds": 20.0,
            "sample_new_cases": 2,
        }
    }
    assert render._compute_time_estimate(records, ["hopg", "hbn"], parallel_materials=parallel) == {
        "elapsed": 20.0,
        "remaining": remaining,
        "total": total,
    }


def test_time_estimate_unavailable_and_failed_samples_remain_unavailable():
    assert render._compute_time_estimate({}, ["hopg"]) == {
        "elapsed": None,
        "remaining": None,
        "total": None,
    }
    record = {
        "material": "hopg",
        "state": "failed",
        "total_cases": 10,
        "cached_cases": 0,
        "completed_new_cases": 2,
        "active_compute_seconds": 20.0,
        "measured_new_cases": 2,
    }
    assert render._compute_time_estimate({"hopg": record}) == {
        "elapsed": 20.0,
        "remaining": None,
        "total": None,
    }
