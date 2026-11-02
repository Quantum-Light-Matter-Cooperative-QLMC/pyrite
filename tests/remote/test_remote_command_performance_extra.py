"""Additional remote performance tests."""

import os
import subprocess
import sys

import pytest

from pyrite.remote import lifecycle, state, viewer
from tests.remote._remote_test_helpers import (
    _remote_main,
)


def test_headless_performance_run_prints_canonical_pull_hint(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: pytest.fail("must not attach"))

    _remote_main(["run", "standard", "-m", "hopg", "--perf", "--headless", "--no-sync"])

    assert "pyrite remote performance pull standard" in capsys.readouterr().out


def test_performance_no_pull_skips_all_artifact_pulls(monkeypatch):
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(
        state,
        "_job_succeeded",
        lambda _jobid: pytest.fail("--no-pull must not query terminal success for pulling"),
    )
    monkeypatch.setattr(
        lifecycle,
        "pull_performance_profile",
        lambda _profile: pytest.fail("--no-pull must not pull performance artifacts"),
    )
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda *_args, **_kwargs: pytest.fail("--no-pull must not pull checkpoints"),
    )

    _remote_main(["run", "standard", "-m", "hopg", "--perf", "--no-pull", "--no-sync"])


def test_performance_run_with_step_warnings_skips_artifact_pull(monkeypatch, capsys):
    """A job that continued past a failed material must not auto-pull.

    Exercises the real ``_job_succeeded`` (only its ``_job_state`` dependency
    is faked) against the terminal state a queue script actually writes when
    some materials fail (``_queue_scripts._queue_script``/
    ``_chunked_queue_tail``), regression-covering #143.
    """
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(
        state,
        "_job_state",
        lambda _jobid: "done with 1 warning(s) [3/3] 2026-09-19T00:00:00+00:00",
    )
    monkeypatch.setattr(state, "_completed_materials", lambda _jobid, _mats: [])
    monkeypatch.setattr(
        lifecycle,
        "pull_performance_profile",
        lambda _profile: pytest.fail(
            "a job with failed steps must not auto-pull performance artifacts"
        ),
    )

    _remote_main(["run", "standard", "-m", "hopg", "--perf", "--no-sync"])

    assert "did not complete successfully" in capsys.readouterr().err


def test_failed_performance_run_skips_artifact_pull(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(state, "_job_succeeded", lambda _jobid: False)
    monkeypatch.setattr(state, "_completed_materials", lambda _jobid, _mats: [])
    monkeypatch.setattr(
        lifecycle,
        "pull_performance_profile",
        lambda _profile: pytest.fail("failed job must not auto-pull performance artifacts"),
    )

    _remote_main(["run", "standard", "-m", "hopg", "--perf", "--no-sync"])

    assert "did not complete successfully" in capsys.readouterr().err


def test_profile_module_writes_stats_and_keeps_inner_exit_code(tmp_path):
    (tmp_path / "exits_two.py").write_text(
        "import sys\nsum(range(1000))\nsys.exit(2)\n", encoding="utf-8"
    )
    prof = tmp_path / "out.prof"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])}

    result = subprocess.run(
        [sys.executable, "-m", "pyrite._entry.profile_module", str(prof), "exits_two"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert prof.stat().st_size > 0
