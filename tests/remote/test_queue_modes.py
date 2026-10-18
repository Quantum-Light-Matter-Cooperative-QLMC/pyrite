"""``start_queue`` refuses invalid mode combinations before any remote probe."""

import pytest

from pyrite.remote import config, queue, transport

_PERF = {"performance_profile": "standard"}
_MONOLITHIC = {"chunk_minutes": 0}


@pytest.fixture
def remote_effects(monkeypatch):
    """Fail if validation lets a submission reach busy probes, sync, or staging."""
    effects = []

    def effect(label):
        return lambda *args, **kwargs: effects.append(label)

    monkeypatch.setattr(queue, "_refuse_if_busy", effect("busy"))
    monkeypatch.setattr(queue, "_refuse_if_profile_live", effect("live"))
    monkeypatch.setattr(transport, "sync_code", effect("sync"))
    monkeypatch.setattr(queue, "_stage_job_script", effect("stage"))
    monkeypatch.setattr(queue, "_submit_staged_job", effect("submit"))
    monkeypatch.setattr(config, "remote_gpu_vendor", lambda: "nvidia")
    return effects


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"performance_repetitions": 0, **_PERF}, "integer from 1 to 20"),
        ({"performance_repetitions": True, **_PERF}, "integer from 1 to 20"),
        ({"performance_interval": 0, **_PERF}, "greater than zero"),
        ({"nsys": True}, "require a performance profile"),
        ({"cpu": True, "cpu_only": True, **_PERF}, "cpu and cpu-only modes are mutually exclusive"),
        ({"no_cache": True, "recompute": True}, "no-cache and recompute modes"),
        ({"cpu_only": True, "nsys": True, **_PERF}, "cpu-only mode cannot be combined with nsys"),
        ({"py_spy": True, "nsys": True, **_PERF}, "py-spy cannot be combined with nsys"),
        ({"py_spy": True, **_PERF}, "py-spy requires a monolithic allocation"),
        (
            {"py_spy": True, "performance_repetitions": 2, **_PERF, **_MONOLITHIC},
            "py-spy requires exactly one performance repetition",
        ),
        (
            {"cpu_only": True, "performance_repetitions": 2, **_PERF},
            "cpu-only mode cannot use performance repetitions",
        ),
        ({"cpu_only": True, "performance_interval": 2.0, **_PERF}, "sampling interval"),
        ({"cpu_only": True, "spec_chunk": 4, **_PERF}, "cannot set GPU chunk pins"),
        ({"performance_repetitions": 2, **_PERF}, "repetitions require a monolithic allocation"),
        (
            {"performance_repetitions": 2, "parallel_materials": 2, **_PERF, **_MONOLITHIC},
            "repetitions require one material process per GPU",
        ),
        ({"nsys": True, **_PERF}, "nsys requires a monolithic allocation"),
        (
            {"nsys": True, "performance_repetitions": 2, **_PERF, **_MONOLITHIC},
            "nsys requires exactly one performance repetition",
        ),
        (
            {"nsys": True, "parallel_materials": 2, **_PERF, **_MONOLITHIC},
            "nsys requires one material process per GPU",
        ),
        (
            {"nsys": True, "materials": ["hopg", "mos2"], **_PERF, **_MONOLITHIC},
            "nsys requires exactly one material",
        ),
        ({"cpu": True, **_PERF}, "CPU profiling requires a monolithic allocation"),
        ({"parallel_materials": 2}, "--parallel-materials only applies to a monolithic"),
        ({"overwrite_trajectories": True}, "overwrite-trajectories requires trajectories"),
        (
            {"trajectories": "pyrite-output/trajectories", "cpu_only": True, **_PERF},
            "trajectory capture excludes CPU-only and repeated profiling",
        ),
    ],
)
@pytest.mark.parametrize("dry_run", [False, True])
def test_invalid_mode_combinations_refuse_before_remote_effects(
    remote_effects, options, message, dry_run
):
    options = {"materials": ["hopg"], **options}

    with pytest.raises(SystemExit, match=message):
        queue.start_queue(dry_run=dry_run, no_sync=False, **options)

    assert remote_effects == []


def test_nsys_requires_an_nvidia_target(remote_effects, monkeypatch):
    monkeypatch.setattr(config, "remote_gpu_vendor", lambda: "amd")

    with pytest.raises(SystemExit, match="NVIDIA target"):
        queue.start_queue(["hopg"], nsys=True, **_PERF, **_MONOLITHIC)

    assert remote_effects == []


def test_dry_run_probes_and_submits_nothing(remote_effects, capsys):
    jobid = queue.start_queue(["hopg"], dry_run=True, catalog_profile="sub_100keV")

    assert jobid == "sub_100keV"
    assert remote_effects == []
    assert "# job sub_100keV: hopg" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("options", "pull_row"),
    [
        ({}, "Pull"),
        ({"cpu": True, **_PERF, **_MONOLITHIC}, "Checkpoints"),
        ({"performance_repetitions": 2, **_PERF, **_MONOLITHIC}, "Checkpoints"),
    ],
)
def test_submission_summary_names_the_follow_up_commands(
    remote_effects, monkeypatch, capsys, options, pull_row
):
    monkeypatch.setattr(queue, "_submit_staged_job", lambda *_args, **_kwargs: "48291")

    queue.start_queue(["hopg"], no_sync=True, **options)

    output = capsys.readouterr().out
    assert remote_effects == ["busy", "stage"]
    assert f"{pull_row} " in output
    assert ("Performance " in output) == ("performance_profile" in options)
    assert "Mode " in output
