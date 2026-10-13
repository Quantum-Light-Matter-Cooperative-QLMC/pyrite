"""Canonical diagnostic guidance in job status frames."""

from pyrite.console.dashboard import format_job_status


def test_empty_job_status_recommends_canonical_logs_command():
    output = format_job_status(
        {"JOB": "j", "STATE": "running", "META": "job: j\nmaterials: hopg"},
        detail=1,
    )
    assert "No case progress reported yet" in output
    assert "`pyrite job logs`" in output
    assert "pyrite remote logs" not in output
