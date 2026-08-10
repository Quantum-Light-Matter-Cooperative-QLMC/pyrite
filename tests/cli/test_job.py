from __future__ import annotations

import json

from pyrite.cli.commands import job
from pyrite.remote import lifecycle, viewer
from tests.helpers.cli import assert_clean_result, invoke


def test_job_list_filters_canonical_kind_and_emits_json(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "jobs_raw",
        lambda: (
            "run-1\t1\tFalse\tscan\thopg\trunning\n"
            "grid-1\t2\tFalse\tenergy-grid\thbn\tdone\n"
            "zhai-1\t3\tFalse\tzhai\tzhai\tdone\n"
        ),
    )

    result = invoke(job.command, ["list", "--kind", "grid", "-o", "json"])

    assert_clean_result(result)
    jobs = json.loads(result.stdout)["payload"]["jobs"]
    assert [(item["job_id"], item["kind"]) for item in jobs] == [("grid-1", "grid")]

    validate = invoke(job.command, ["list", "--kind", "validate", "-o", "json"])

    assert_clean_result(validate)
    jobs = json.loads(validate.stdout)["payload"]["jobs"]
    assert [(item["job_id"], item["kind"]) for item in jobs] == [("zhai-1", "validate")]


def test_job_attach_delegates_to_shared_viewer(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        viewer,
        "attach",
        lambda jobid, detail: seen.update(jobid=jobid, detail=detail),
    )

    result = invoke(job.command, ["attach", "job-7", "-vv"])

    assert_clean_result(result)
    assert seen == {"jobid": "job-7", "detail": 2}


def test_job_stop_by_id_previews_then_stops_with_yes(monkeypatch):
    stopped = []
    monkeypatch.setattr(lifecycle, "_stop_jobid", stopped.append)

    preview = invoke(job.command, ["stop", "job-7"])
    stopped_result = invoke(job.command, ["stop", "job-7", "--yes"])

    assert_clean_result(
        preview,
        stdout="would cancel remote job: job-7\npreview only; re-run with -y/--yes to execute\n",
    )
    assert_clean_result(stopped_result, stdout="would cancel remote job: job-7\n")
    assert stopped == ["job-7"]


def test_job_stop_requires_exactly_one_selector():
    missing = invoke(job.command, ["stop"])
    conflicting = invoke(job.command, ["stop", "job-7", "--all"])

    assert missing.exit_code == 2
    assert conflicting.exit_code == 2
    assert "exactly one of JOBID, --profile NAME, or --all" in missing.stderr
    assert "exactly one of JOBID, --profile NAME, or --all" in conflicting.stderr
