"""Remote prune tests."""

import os
import subprocess

import pytest

from pyrite import remote
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
from pyrite.remote import cleanup as lifecycle_cleanup
from pyrite.remote import performance as lifecycle_performance
from tests.remote._remote_test_helpers import (
    _make_job,
    _remote_main,
    _run_prune_jobs_command,
)


def test_prune_remote_performance_previews_terminal_jobs(monkeypatch, capsys):
    inventory = [("job-1", "baseline", 2, 80), ("job-2", "keeper", 1, 40)]
    monkeypatch.setattr(lifecycle_performance, "remote_performance_inventory", lambda: inventory)
    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(state, "_job_state", lambda _jobid: "done [1/1]")
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: pytest.fail("preview must not delete"),
    )

    lifecycle.prune_remote_performance(["baseline"])

    output = capsys.readouterr().out
    assert "baseline/job-1 (2 artifact(s), 80 bytes)" in output
    assert "preview only" in output
    assert "keeper" not in output


def test_prune_remote_performance_blocks_live_or_incomplete_jobs(monkeypatch):
    inventory = [("job-1", "baseline", 2, 80)]
    monkeypatch.setattr(lifecycle_performance, "remote_performance_inventory", lambda: inventory)
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job-1", False, ["hopg"])])

    with pytest.raises(SystemExit, match="live job"):
        lifecycle.prune_remote_performance(["baseline"])

    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(state, "_job_state", lambda _jobid: "running hopg")
    with pytest.raises(SystemExit, match="non-terminal job"):
        lifecycle.prune_remote_performance(["baseline"])


def test_prune_remote_performance_revalidates_and_deletes_exact_paths(monkeypatch, capsys):
    inventory = [
        ("job-1", "baseline", 2, 80),
        ("job-2", "baseline", 1, 40),
        ("job-3", "keeper", 1, 20),
    ]
    monkeypatch.setattr(lifecycle_performance, "remote_performance_inventory", lambda: inventory)
    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(state, "_job_state", lambda _jobid: "done [1/1]")
    commands = []
    monkeypatch.setattr(transport, "_ssh_capture", lambda command: commands.append(command) or "")

    lifecycle.prune_remote_performance(["baseline"], yes=True)

    assert len(commands) == 1
    assert f"{config.REMOTE_DIR}/pyrite-output/performance/job-1/baseline" in commands[0]
    assert f"{config.REMOTE_DIR}/pyrite-output/performance/job-2/baseline" in commands[0]
    assert "job-3" not in commands[0]
    assert "deleted 2 remote performance profile path(s)" in capsys.readouterr().out


def test_prune_remote_performance_blocks_changed_inventory(monkeypatch):
    inventories = iter(
        [
            [("job-1", "baseline", 2, 80)],
            [("job-1", "baseline", 3, 120)],
        ]
    )
    monkeypatch.setattr(
        lifecycle_performance, "remote_performance_inventory", lambda: next(inventories)
    )
    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(state, "_job_state", lambda _jobid: "done [1/1]")

    with pytest.raises(SystemExit, match="inventory changed"):
        lifecycle.prune_remote_performance(["baseline"], yes=True)


def test_prune_remote_performance_blocks_changed_job_state(monkeypatch):
    inventory = [("job-1", "baseline", 2, 80)]
    states = iter(["done [1/1]", "FAILED (exit 1)"])
    monkeypatch.setattr(lifecycle_performance, "remote_performance_inventory", lambda: inventory)
    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(state, "_job_state", lambda _jobid: next(states))

    with pytest.raises(SystemExit, match="job state"):
        lifecycle.prune_remote_performance(["baseline"], yes=True)


def test_prune_remote_refuses_while_any_job_is_live(monkeypatch):
    target = type("Target", (), {"stem": "hopg"})()
    monkeypatch.setattr("pyrite.checkpoints.checkpoint_cleanup._targets", lambda *_args: [target])
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: pytest.fail("must not mutate remote checkpoints"),
    )

    with pytest.raises(SystemExit, match="live job"):
        lifecycle.prune_remote()


def test_prune_checkpoint_command_reserves_runs_and_releases():
    command = scripts._prune_checkpoint_stems_command(
        "prune-job",
        ["hopg", "hopg--survey-deadbeef0000"],
        catalog_profile="standard",
        yes=True,
    )

    assert "for stem in hopg hopg--survey-deadbeef0000" in command
    assert "trap release_prune EXIT" in command
    assert "run --no-sync pyrite checkpoint gc --profile standard --yes" in command
    assert command.index("for stem in hopg") < command.index("run --no-sync pyrite checkpoint gc")


def test_prune_remote_dispatches_exact_reserved_stems(monkeypatch, capsys):
    target = type("Target", (), {"stem": "hopg"})()
    commands = []
    monkeypatch.setattr("pyrite.checkpoints.checkpoint_cleanup._targets", lambda *_args: [target])
    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(state, "_reservation_holders", lambda stems: [])
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or "would prune remote",
    )

    lifecycle.prune_remote(catalog_profile="standard")

    assert len(commands) == 1
    assert "for stem in hopg" in commands[0]
    assert "pyrite checkpoint gc --profile standard" in commands[0]
    assert "would prune remote" in capsys.readouterr().out


def test_prune_remote_reserves_previous_profile_identity_for_reclamation(monkeypatch, capsys):
    current = type(
        "Target",
        (),
        {"stem": "hopg@hopg_hbn-aaaaaaaaaaaa"},
    )()
    old = "hopg@hopg_hbn-111111111111"
    commands = []
    monkeypatch.setattr("pyrite.checkpoints.checkpoint_cleanup._targets", lambda *_args: [current])
    monkeypatch.setattr(
        lifecycle_cleanup,
        "_profile_checkpoint_stems",
        lambda *_args, **_kwargs: [current.stem, old],
    )
    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(state, "_reservation_holders", lambda stems: [])
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or "would prune remote",
    )

    lifecycle.prune_remote(catalog_profile="hopg_hbn")

    assert f"for stem in {old} {current.stem}" in commands[0]
    assert "would delete obsolete profile checkpoint: pyrite-output/checkpoints/%s/" in commands[0]
    assert old in commands[0]
    assert "would prune remote" in capsys.readouterr().out


def test_prune_remote_obsolete_only_confirmation_reuses_exact_selection(monkeypatch):
    current = type("Target", (), {"stem": "hopg@hopg_hbn-aaaaaaaaaaaa"})()
    old = "hopg@hopg_hbn-111111111111"
    discoveries = []
    commands = []
    monkeypatch.setattr("pyrite.checkpoints.checkpoint_cleanup._targets", lambda *_args: [current])
    monkeypatch.setattr(
        lifecycle_cleanup,
        "_profile_checkpoint_stems",
        lambda *_args, **_kwargs: discoveries.append(True) or [current.stem, old],
    )
    monkeypatch.setattr(state, "_live_jobs", lambda: [])
    monkeypatch.setattr(state, "_reservation_holders", lambda stems: [])
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or "would delete obsolete profile checkpoint",
    )
    monkeypatch.setattr(lifecycle_cleanup._cli_core, "confirm_destructive", lambda *_args: True)

    lifecycle.prune_remote(catalog_profile="hopg_hbn")

    assert len(discoveries) == 1
    assert len(commands) == 2
    assert all(old in command for command in commands)
    assert "pyrite checkpoint gc --profile hopg_hbn --yes" in commands[1]


def test_prune_jobs_command_keeps_live_and_non_terminal(monkeypatch, tmp_path):
    """Only done/FAILED/cancelled dirs whose latest SLURM id is not live are removed."""
    jobs = tmp_path / "jobs"
    cancelled = _make_job(jobs, "sub_100keV", "cancelled [932] 2026-07-28", [915, 932])
    live = _make_job(jobs, "sub_100keV-7", "running mos2 [4/21] since now", [942, 947])
    crashed = _make_job(jobs, "sub_100keV-6", "running hbn [2/21] since then", [941])
    done = _make_job(jobs, "20260715-113910-abcd", "done [21/21] 2026-07-15", [800])

    command = scripts._prune_job_dirs_command(all_jobs=True, yes=True)
    result = _run_prune_jobs_command(command, jobs, live_ids=["947"])

    assert result.returncode == 0, result.stderr
    assert "PRUNED\tsub_100keV\t" in result.stdout
    assert "PRUNED\t20260715-113910-abcd\t" in result.stdout
    assert "KEPT\tsub_100keV-7\tlive" in result.stdout
    assert not cancelled.exists()
    assert not done.exists()
    assert live.exists()  # live chain never removed
    assert crashed.exists()  # non-terminal state kept


def test_prune_jobs_command_preview_deletes_nothing(tmp_path):
    jobs = tmp_path / "jobs"
    cancelled = _make_job(jobs, "sub_100keV", "cancelled [932] 2026-07-28", [932])

    command = scripts._prune_job_dirs_command(all_jobs=True, yes=False)
    result = _run_prune_jobs_command(command, jobs, live_ids=[])

    assert result.returncode == 0, result.stderr
    assert "WOULD-PRUNE\tsub_100keV\t" in result.stdout
    assert cancelled.exists()  # preview never mutates


def test_prune_jobs_command_profile_scopes_to_family(tmp_path):
    jobs = tmp_path / "jobs"
    _make_job(jobs, "sub_100keV", "cancelled [932]", [932])
    _make_job(jobs, "sub_100keV-2", "cancelled [933]", [933])
    other = _make_job(jobs, "20260715-113910-abcd", "cancelled [800]", [800])

    command = scripts._prune_job_dirs_command(profile="sub_100keV", yes=True)
    result = _run_prune_jobs_command(command, jobs, live_ids=[])

    assert result.returncode == 0, result.stderr
    assert "PRUNED\tsub_100keV\t" in result.stdout
    assert "PRUNED\tsub_100keV-2\t" in result.stdout
    assert other.exists()  # a standard timestamp job is outside the profile family


def test_prune_jobs_command_fail_closed_when_squeue_errors(tmp_path):
    """squeue failure must not delete anything (liveness unverifiable)."""
    jobs = tmp_path / "jobs"
    cancelled = _make_job(jobs, "sub_100keV", "cancelled [932]", [932])
    command = scripts._prune_job_dirs_command(all_jobs=True, yes=True)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    # A squeue that errors: liveness is unverifiable, so nothing may be pruned.
    (bindir / "squeue").write_text("#!/bin/sh\necho boom >&2\nexit 1\n")
    (bindir / "squeue").chmod(0o755)
    command = command.replace(f'JOBS="{remote.REMOTE_DIR}/{remote.JOBS_SUBDIR}"', f'JOBS="{jobs}"')
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        env=dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}"),
    )

    assert result.returncode != 0
    assert cancelled.exists()


def test_prune_job_dirs_reports_preview(monkeypatch, capsys):
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: "WOULD-PRUNE\tsub_100keV\tcancelled [932]\nKEPT\tsub_100keV-7\tlive\n",
    )

    lifecycle.prune_job_dirs(profile="sub_100keV")

    out = capsys.readouterr().out
    assert "jobs/sub_100keV" in out
    assert "cancelled" in out
    assert "kept 1 live job" in out


def test_prune_job_dirs_reports_deletions(monkeypatch, capsys):
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: "PRUNED\tsub_100keV\tcancelled [932]\n"
    )

    lifecycle.prune_job_dirs(all_jobs=True, yes=True)

    out = capsys.readouterr().out
    assert "pruned job directories" in out
    assert "jobs/sub_100keV" in out


def test_prune_job_dirs_requires_exactly_one_selector(monkeypatch):
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: pytest.fail("must not reach ssh")
    )

    with pytest.raises(SystemExit, match="exactly one"):
        lifecycle.prune_job_dirs()
    with pytest.raises(SystemExit, match="exactly one"):
        lifecycle.prune_job_dirs(profile="sub_100keV", all_jobs=True)


def test_prune_jobs_cli_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "prune_job_dirs", lambda **kwargs: calls.append(kwargs))

    _remote_main(["prune-jobs", "--profile", "sub_100keV", "--yes"])

    assert calls == [{"profile": "sub_100keV", "all_jobs": False, "yes": True}]


def test_prune_jobs_cli_rejects_bad_selectors(monkeypatch):
    monkeypatch.setattr(
        lifecycle, "prune_job_dirs", lambda **kwargs: pytest.fail("must not dispatch")
    )

    assert _remote_main(["prune-jobs"]) == 2
    assert _remote_main(["prune-jobs", "--all", "--profile", "sub_100keV"]) == 2
