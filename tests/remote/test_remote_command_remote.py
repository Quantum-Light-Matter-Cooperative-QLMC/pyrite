"""Remote remote tests."""

import os
import subprocess
import sys

import pytest

from pyrite import remote
from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.cli.commands import remote as remote_cli
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
    _FakeLockProc,
    _remote_main,
)


@pytest.mark.parametrize(
    "value",
    [
        "-oProxyCommand=touch-/tmp/pwn",
        "user@host",
        "host name",
        "host;touch",
        "host\nother",
        "",
    ],
)
def test_remote_host_rejects_ssh_option_and_shell_syntax_before_subprocess(monkeypatch, value):
    monkeypatch.setattr(config, "HOST", value)
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("invalid host must not reach ssh"),
    )

    with pytest.raises(SystemExit, match="PYRITE_REMOTE_HOST"):
        transport._ssh_capture(":")


def test_remote_sync_lock_holds_flock_until_stdin_closes(monkeypatch):
    proc = _FakeLockProc("LOCKED\n")
    seen = {}

    def popen(argv, **kwargs):
        seen["argv"] = argv
        return proc

    monkeypatch.setattr(transport.subprocess, "Popen", popen)
    with transport._real_remote_sync_lock():
        assert not proc.stdin.closed
    assert proc.stdin.closed
    assert "flock -x -w" in seen["argv"][-1]
    assert config.SYNC_LOCK_NAME in seen["argv"][-1]


def test_remote_sync_lock_fails_clearly_when_held(monkeypatch):
    proc = _FakeLockProc("", returncode=75)
    monkeypatch.setattr(transport.subprocess, "Popen", lambda *_a, **_k: proc)
    with pytest.raises(SystemExit, match="remote sync lock"):
        with transport._real_remote_sync_lock():
            pytest.fail("body must not run without the lock")
    assert proc.stdin.closed


@pytest.mark.parametrize("value", ["relative/path", "/safe\ninjected", "/safe\0injected", ""])
def test_remote_dir_rejects_nonabsolute_and_control_values(monkeypatch, value):
    monkeypatch.setattr(config, "REMOTE_DIR", value)

    with pytest.raises(SystemExit, match="PYRITE_REMOTE_DIR"):
        remote._queue_script("j", ["hopg"], quick=False, workers=None)


def test_remote_defaults_are_home_relative_and_functional(monkeypatch):
    """Unconfigured defaults must be usable on any box: resolved against the remote $HOME."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYRITE_REMOTE_")}
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pyrite.remote import config; print(config.REMOTE_DIR, config.REMOTE_UV)",
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.split()

    assert out == ["~/pyrite", "~/.local/bin/uv"]


def test_remote_home_expansion_resolves_tilde_over_one_cached_ssh_call(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="/home/lab\n", stderr="")

    config._remote_home.cache_clear()
    monkeypatch.setattr(config, "HOST", "box")
    monkeypatch.setattr(config.subprocess, "run", fake_run)
    monkeypatch.setattr(config, "REMOTE_DIR", "~/pyrite")
    monkeypatch.setattr(config, "REMOTE_UV", "~/.local/bin/uv")
    try:
        assert config.remote_dir() == "/home/lab/pyrite"
        assert config.remote_uv() == "/home/lab/.local/bin/uv"
        assert config.remote_path("jobs") == "/home/lab/pyrite/jobs"
    finally:
        config._remote_home.cache_clear()

    assert len(calls) == 1
    assert calls[0][:4] == ["ssh", "-n", "-o", "BatchMode=yes"]
    assert calls[0][4] == "box"


def test_remote_home_expansion_leaves_absolute_values_without_ssh(monkeypatch):
    monkeypatch.setattr(config, "REMOTE_DIR", "/srv/pyrite")
    monkeypatch.setattr(config, "REMOTE_UV", "uv")

    # The autouse ssh guard would raise if either call reached ssh.
    assert config.remote_dir() == "/srv/pyrite"
    assert config.remote_uv() == "uv"


@pytest.mark.parametrize("stdout, returncode", [("", 0), ("relative\n", 0), ("/home/lab\n", 255)])
def test_remote_home_expansion_fails_closed_when_home_unresolvable(monkeypatch, stdout, returncode):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    config._remote_home.cache_clear()
    monkeypatch.setattr(config, "HOST", "box")
    monkeypatch.setattr(config.subprocess, "run", fake_run)
    monkeypatch.setattr(config, "REMOTE_DIR", "~/pyrite")
    try:
        with pytest.raises(SystemExit, match="cannot resolve '~'"):
            config.remote_dir()
    finally:
        config._remote_home.cache_clear()


def test_remote_scan_entry_accepts_the_performance_flags_queue_scripts_pass():
    from pyrite._entry import scan as entry_scan

    options = {opt for param in entry_scan.command.params for opt in param.opts}
    assert {"--performance-profile", "--performance-dir", "--perf-interval"} <= options


def test_remote_performance_inventory_parses_artifact_totals(monkeypatch):
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: (
            commands.append(command) or "job-2\tbaseline\t3\t120\njob-1\tbaseline\t2\t80\n"
        ),
    )

    inventory = lifecycle.remote_performance_inventory()

    assert inventory == [
        ("job-1", "baseline", 2, 80),
        ("job-2", "baseline", 3, 120),
    ]
    assert '"$JOBS"/*/*/' in commands[0]


def test_remote_performance_inventory_rejects_malformed_output(monkeypatch):
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "job\t../bad\t2\t80\n")

    with pytest.raises(SystemExit, match="remote inventory was malformed"):
        lifecycle.remote_performance_inventory()


def test_remote_dry_run_describes_sbatch_submission(monkeypatch, capsys):
    monkeypatch.setattr(
        lifecycle,
        "_refuse_if_busy",
        lambda *_args: pytest.fail("dry-run must not check busy"),
    )
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("dry-run must not sync"))
    monkeypatch.setattr(
        transport.subprocess, "run", lambda *args, **kwargs: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda _command: pytest.fail("dry-run must not ssh")
    )
    monkeypatch.setattr(
        transport, "_run", lambda *_args, **_kwargs: pytest.fail("dry-run must not ssh")
    )

    remote.start_queue(["hopg"], dry_run=True)

    out = capsys.readouterr().out
    assert "#SBATCH --partition=gpu" in out
    assert "sbatch --parsable" in out
    assert "nohup" not in out and "setsid" not in out


def test_remote_attach_command_is_absent(capsys):
    assert _remote_main(["attach"]) == 2
    assert "No such command 'attach'" in capsys.readouterr().err


def test_remote_run_uses_profile_membership(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle, "start_queue", lambda materials, *args, **kwargs: calls.append(materials)
    )

    _remote_main(["run", "sub_100keV", "--dry-run"])

    assert calls and "hopg" in calls[0] and "zrte3" in calls[0]


def test_remote_run_defers_parallel_materials_default_to_start_queue(monkeypatch):
    """The CLI no longer bakes in a default: --parallel-materials is None
    unless given explicitly, so start_queue (mode-aware) resolves it. The
    monolithic default of 2 is exercised by
    test_chunk_minutes_zero_emits_monolithic_script."""
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "j",
    )

    _remote_main(["run", "standard", "-m", "hopg", "--dry-run", "--chunk-minutes", "0"])

    assert calls[0][1]["parallel_materials"] is None
    assert calls[0][1]["chunk_minutes"] == 0.0


def test_remote_run_accepts_parallel_materials_three_and_four(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "j",
    )

    _remote_main(
        ["run", "sub_100keV", "--parallel-materials", "3", "--chunk-minutes", "0", "--dry-run"]
    )
    _remote_main(
        ["run", "sub_100keV", "--parallel-materials", "4", "--chunk-minutes", "0", "--dry-run"]
    )

    assert [kwargs["parallel_materials"] for _materials, kwargs in calls] == [3, 4]


def test_remote_run_rejects_parallel_materials_above_four(monkeypatch):
    monkeypatch.setattr(
        lifecycle, "start_queue", lambda *_args, **_kwargs: pytest.fail("must reject before start")
    )

    assert (
        _remote_main(["run", "standard", "-m", "hopg", "--parallel-materials", "5", "--dry-run"])
        == 2
    )


def test_remote_run_forwards_parallel_materials(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "start_queue",
        lambda materials, **kwargs: calls.append((materials, kwargs)) or "j",
    )
    monkeypatch.setattr(viewer, "attach", lambda _jobid: False)

    _remote_main(
        [
            "run",
            "standard",
            "-m",
            "hopg",
            "--parallel-materials",
            "3",
            "--chunk-minutes",
            "0",
            "--no-sync",
        ]
    )

    assert calls[0][1]["parallel_materials"] == 3


def test_remote_run_preserves_hyphenated_catalog_material(monkeypatch):
    import pyrite.materials as materials_pkg

    catalog = materials_pkg.CATALOG

    class _CatalogWithComposite:
        def __getattr__(self, name):
            return getattr(catalog, name)

        def profile_materials(self, name):
            membership = catalog.profile_materials(name)
            if name == "standard":
                return (*membership, "mos2-on-sio2-si")
            return membership

    monkeypatch.setattr(materials_pkg, "CATALOG", _CatalogWithComposite())
    calls = []
    monkeypatch.setattr(
        lifecycle, "start_queue", lambda materials, **_kw: calls.append(materials) or "j"
    )
    monkeypatch.setattr(viewer, "attach", lambda _jobid: None)
    monkeypatch.setattr(state, "_completed_materials", lambda _jobid, materials: materials)
    monkeypatch.setattr(lifecycle, "pull", lambda *_args, **_kwargs: None)

    _remote_main(["run", "standard", "-m", "mos2-on-sio2-si", "--no-sync"])

    assert calls == [["mos2-on-sio2-si"]]


def test_remote_run_rejects_unknown_profile_before_busy_or_sync(monkeypatch):
    monkeypatch.setattr(
        state, "_live_jobs", lambda: pytest.fail("must validate before checking busy jobs")
    )
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("must validate before syncing"))

    assert _remote_main(["run", "not_in_catalog"]) == 2


def test_remote_check_refuses_when_a_zhai_job_is_already_live(monkeypatch):
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["zhai"])])
    monkeypatch.setattr(transport, "sync_code", lambda: pytest.fail("must refuse before syncing"))
    with pytest.raises(SystemExit, match="refusing to start"):
        remote_cli.remote_check()


def test_remote_check_no_sync_skips_sync(monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **kw: calls.append(kw) or "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: calls.append("attach") or True)
    monkeypatch.setattr(state, "_job_succeeded", lambda _jobid: True)
    monkeypatch.setattr(lifecycle, "pull_zhai_cache", lambda: calls.append("pull"))

    remote_cli.remote_check(no_sync=True)

    assert calls == [
        {
            "ne": 20_000,
            "ne_brem": 200,
            "ne_supp": 200,
            "tmd_azimuth": 0.0,
            "refresh": False,
            "no_sync": True,
            "dry_run": False,
        },
        "attach",
        "pull",
    ]


@pytest.mark.parametrize(
    ("mode", "dry_run"),
    [("detach", False), ("dry_run", True)],
)
def test_remote_check_non_waiting_modes_skip_attach_and_pull(monkeypatch, mode, dry_run):
    calls = []
    monkeypatch.setattr(lifecycle, "start_zhai_queue", lambda **kw: calls.append(kw) or "j")
    monkeypatch.setattr(viewer, "attach", lambda _jobid: pytest.fail("must not attach"))
    monkeypatch.setattr(lifecycle, "pull_zhai_cache", lambda: pytest.fail("must not pull"))

    assert remote_cli.remote_check(**{mode: True}) == "j"

    assert calls[0]["dry_run"] is dry_run
