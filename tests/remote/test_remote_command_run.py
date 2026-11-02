"""Remote run tests."""

import subprocess

import pytest

from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.remote import (  # noqa: F401
    _queue_scripts,
    config,
    lifecycle,
    scripts,
    state,
    transport,
    viewer,
)
from tests.remote._remote_test_helpers import (
    _remote_main,
)


def test_run_hides_raw_command_by_default_and_shows_label(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: None)

    transport._run(["ssh", "-n", "box", "some very secret remote command"])
    assert capsys.readouterr().err == ""

    transport._run(["ssh", "-n", "box", "some very secret remote command"], label="Doing work...")
    assert capsys.readouterr().err == "Doing work...\n"


def test_run_profile_uses_shipped_membership_without_material_option(capsys):
    """A positional profile selects its explicit material membership."""
    _remote_main(["run", "sub_100keV", "--dry-run"])

    out = capsys.readouterr().out
    assert "hopg" in out and "zrte3" in out


def test_run_perf_uses_positional_catalog_profile(capsys):
    _remote_main(["run", "sub_100keV", "--perf", "--dry-run"])

    out = capsys.readouterr().out
    assert "hopg" in out and "zrte3" in out
    assert '_entry.scan "sub_100keV" -m' in out
    assert "--performance-profile sub_100keV" in out


def test_run_rejects_retired_profile_option(capsys):
    result = _remote_main(
        [
            "run",
            "sub_100keV",
            "--profile",
            "standard",
            "--dry-run",
        ]
    )

    assert result == 2
    assert "No such option '--profile'" in capsys.readouterr().err


def test_run_rejects_perf_reps_without_perf(capsys):
    result = _remote_main(
        [
            "run",
            "sub_100keV",
            "--perf-reps",
            "3",
            "--chunk-minutes",
            "0",
            "--dry-run",
        ]
    )

    assert result == 2
    assert "--perf-reps requires --perf" in capsys.readouterr().err


def test_run_perf_reps_default_to_monolithic_mode(capsys):
    result = _remote_main(
        [
            "run",
            "sub_100keV",
            "--perf",
            "--perf-reps",
            "3",
            "--dry-run",
        ]
    )

    assert result is None
    assert "chunk_minutes: 0.0" in capsys.readouterr().out


def test_run_rejects_explicit_nonzero_chunk_for_perf_reps(capsys):
    result = _remote_main(
        [
            "run",
            "sub_100keV",
            "--perf",
            "--perf-reps",
            "3",
            "--chunk-minutes",
            "10",
            "--dry-run",
        ]
    )

    assert result == 2
    assert "--perf-reps requires --chunk-minutes 0" in capsys.readouterr().err


def test_run_nsys_implies_performance_profile(capsys):
    result = _remote_main(["run", "sub_100keV", "-m", "mos2", "--nsys", "--dry-run"])

    assert result is None
    output = capsys.readouterr().out
    assert "performance_profile: sub_100keV" in output
    assert "chunk_minutes: 0.0" in output


def test_run_rejects_nsys_in_chunked_mode(capsys):
    result = _remote_main(
        [
            "run",
            "sub_100keV",
            "-m",
            "mos2",
            "--nsys",
            "--chunk-minutes",
            "10",
            "--dry-run",
        ]
    )

    assert result == 2
    assert "--nsys requires --chunk-minutes 0" in capsys.readouterr().err


def test_run_rejects_nsys_with_multiple_repetitions(capsys):
    result = _remote_main(
        [
            "run",
            "sub_100keV",
            "-m",
            "mos2",
            "--perf-reps",
            "2",
            "--nsys",
            "--chunk-minutes",
            "0",
            "--dry-run",
        ]
    )

    assert result == 2
    assert "--nsys requires --perf-reps 1" in capsys.readouterr().err


def test_run_rejects_nsys_without_explicit_material(capsys):
    result = _remote_main(
        [
            "run",
            "sub_100keV",
            "--perf",
            "--nsys",
            "--chunk-minutes",
            "0",
            "--dry-run",
        ]
    )

    assert result == 2
    assert "--nsys requires one explicit -m/--material" in capsys.readouterr().err


def test_run_performance_runtime_knobs_reach_monolithic_dry_run(capsys):
    result = _remote_main(
        [
            "run",
            "sub_100keV",
            "--material",
            "mos2",
            "--perf",
            "--perf-reps",
            "3",
            "--perf-interval",
            "1",
            "--workers",
            "6",
            "--cxr-chunk",
            "20000",
            "--brem-chunk",
            "10000",
            "--chunk-minutes",
            "0",
            "--dry-run",
        ]
    )

    assert result is None
    output = capsys.readouterr().out
    assert "#SBATCH --cpus-per-task=6" in output
    assert "--perf-interval 1" in output
    assert "performance_repetitions=3" in output
    assert "export PYRITE_MC_CXR_CHUNK=20000" in output
    assert "export PYRITE_MC_BREM_CHUNK=10000" in output


def test_run_nsys_reaches_monolithic_dry_run(capsys):
    result = _remote_main(
        [
            "run",
            "sub_100keV",
            "-m",
            "mos2",
            "-p",
            "--workers",
            "6",
            "--cxr-chunk",
            "20000",
            "--nsys",
            "--chunk-minutes",
            "0",
            "--dry-run",
        ]
    )

    assert result is None
    output = capsys.readouterr().out
    assert "nsys profile" in output
    assert "nsys: True" in output
    assert "export PYRITE_MC_NSYS=1" in output


def test_run_defaults_to_attach_and_pull(monkeypatch):
    events = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda mats, **kwargs: events.append(("run", mats, kwargs)) or "j",
    )
    monkeypatch.setattr(viewer, "attach", lambda jobid: events.append(("attach", jobid)) or True)
    monkeypatch.setattr(state, "_completed_materials", lambda jobid, mats: list(mats))
    monkeypatch.setattr(state, "_materials_needing_pull", lambda jobid, mats: list(mats))
    monkeypatch.setattr(scripts, "_stems", lambda mats, *args, **kwargs: list(mats))
    monkeypatch.setattr(
        lifecycle, "pull", lambda stems, **kwargs: events.append(("pull", stems, kwargs))
    )

    _remote_main(["run", "standard", "-m", "hopg", "--no-sync"])

    assert [event[0] for event in events] == ["run", "attach", "pull"]
    assert events[-1][2]["no_sync"] is True


def test_run_skips_pull_when_successful_remote_scan_used_only_cached_cases(monkeypatch, capsys):
    from pyrite.checkpoints import _checkpoint_store

    pulls = []
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(state, "_completed_materials", lambda _jobid, mats: list(mats))
    monkeypatch.setattr(state, "_materials_needing_pull", lambda _jobid, _mats: [])
    monkeypatch.setattr(_checkpoint_store, "checkpoint_exists", lambda *_args: True)
    monkeypatch.setattr(lifecycle, "pull", lambda stems, **_kwargs: pulls.append(list(stems)))

    _remote_main(["run", "standard", "-m", "hopg", "--no-sync"])

    assert pulls == []
    assert "all successful remote scans used cached cases" in capsys.readouterr().out


def test_run_pulls_cached_remote_checkpoint_missing_locally(monkeypatch):
    from pyrite.checkpoints import _checkpoint_store

    pulls = []
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(state, "_completed_materials", lambda _jobid, mats: list(mats))
    monkeypatch.setattr(state, "_materials_needing_pull", lambda _jobid, _mats: [])
    monkeypatch.setattr(_checkpoint_store, "checkpoint_exists", lambda *_args: False)
    monkeypatch.setattr(lifecycle, "pull", lambda stems, **_kwargs: pulls.append(list(stems)))

    from pyrite.campaign.profiles import named_profile_stem

    _remote_main(["run", "standard", "-m", "hopg", "--no-sync"])

    assert pulls == [[named_profile_stem("hopg")]]


@pytest.mark.parametrize("cache_flag", ["--no-cache", "--recompute"])
@pytest.mark.parametrize("chunk_minutes", [None, "0"])
def test_run_cache_mode_reaches_each_box_side_scan_once(cache_flag, chunk_minutes, capsys):
    args = ["run", "standard", "-m", "hopg", "--dry-run", cache_flag]
    if chunk_minutes is not None:
        args.extend(["--chunk-minutes", chunk_minutes])

    _remote_main(args)

    script = capsys.readouterr().out
    assert script.count(f" {cache_flag}") == 1
    other = "--recompute" if cache_flag == "--no-cache" else "--no-cache"
    assert f" {other}" not in script


def test_run_headless_skips_attach_and_pull(monkeypatch):
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: pytest.fail("must not attach"))
    monkeypatch.setattr(lifecycle, "pull", lambda *_args, **_kwargs: pytest.fail("must not pull"))

    _remote_main(["run", "standard", "-m", "hopg", "--headless", "--no-sync"])


def test_run_no_pull_still_attaches(monkeypatch):
    attached = []
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda jobid: attached.append(jobid) or True)
    monkeypatch.setattr(lifecycle, "pull", lambda *_args, **_kwargs: pytest.fail("must not pull"))

    _remote_main(["run", "standard", "-m", "hopg", "--no-pull", "--no-sync"])

    assert attached == ["j"]


def test_run_perf_reps_attach_but_skip_checkpoint_pull(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle, "start_queue", lambda _mats, **_kwargs: "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: True)
    monkeypatch.setattr(state, "_job_succeeded", lambda _jobid: True)
    monkeypatch.setattr(lifecycle, "pull_performance_profile", lambda _profile: None)
    monkeypatch.setattr(
        state,
        "_completed_materials",
        lambda *_args: pytest.fail("must not resolve profiling checkpoints"),
    )
    monkeypatch.setattr(
        lifecycle,
        "pull",
        lambda *_args, **_kwargs: pytest.fail("must not pull profiling checkpoints"),
    )

    _remote_main(
        [
            "run",
            "sub_100keV",
            "-m",
            "mos2",
            "--perf",
            "--perf-reps",
            "3",
            "--chunk-minutes",
            "0",
            "--no-sync",
        ]
    )

    assert "isolated job-local checkpoints" in capsys.readouterr().err


def test_run_rejects_headless_with_no_pull(capsys):
    assert _remote_main(["run", "standard", "-m", "hopg", "--headless", "--no-pull"]) == 2
    assert "--headless cannot be combined with --no-pull" in capsys.readouterr().err


def test_run_rejects_quick_plus_grid_before_any_work(monkeypatch):
    """--quick checkpoints aren't grid-filterable (pyrite slim rejects _quick stems),
    so run --quick --grid must fail up front -- not run the whole sweep and then
    traceback on the trailing pull."""
    monkeypatch.setattr(
        state, "_live_jobs", lambda: pytest.fail("must reject before the busy check")
    )
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("must reject before syncing"))
    monkeypatch.setattr(
        lifecycle, "start_queue", lambda *a, **kw: pytest.fail("must reject before submitting")
    )
    assert _remote_main(["run", "standard", "-m", "hopg", "--quick", "--grid"]) == 2
