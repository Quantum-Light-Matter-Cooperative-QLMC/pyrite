"""Remote clear tests."""

import os
import subprocess

import pytest

from pyrite import remote
from pyrite.cli.commands import _remote_actions
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
from tests.remote._remote_test_helpers import (
    _bash_or_skip,
    _no_live_jobs,
    _remote_main,
)


def test_clear_refuses_when_a_live_job_produces_the_stem(monkeypatch):
    # a live job producing hopg -> clearing hopg must refuse before any ssh
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda *a: pytest.fail("must not ssh when refusing")
    )
    with pytest.raises(SystemExit, match="refusing to clear"):
        remote.clear_remote("hopg", yes=True)


def test_clear_refuses_for_quick_stem_collision(monkeypatch):
    # a live --quick job producing hopg_quick still blocks a clear of hopg
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", True, ["hopg"])])
    with pytest.raises(SystemExit, match="refusing to clear"):
        remote.clear_remote("hopg")


def test_clear_refuses_when_an_ambiguous_submission_holds_a_reservation(monkeypatch):
    """A retained pre-sbatch lock protects a possibly queued job without an ID."""
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(transport, "_ssh_capture", lambda *_args: "RESERVED\thopg\n")

    with pytest.raises(SystemExit, match="reservation"):
        remote.clear_remote("hopg", yes=True)


def test_clear_dry_preview_lists_but_does_not_delete(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(transport, "_ssh_capture", lambda *a: "hopg.pkl\nhopg_quick.pkl\n")
    runs = []
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=False)
    out = capsys.readouterr().out
    assert "would delete" in out and "hopg.pkl" in out and "hopg_quick.pkl" in out
    assert runs == []  # nothing deleted in a dry preview


def test_clear_yes_deletes_existing_files(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or "CLEARED\thopg.pkl\nCLEARED\thopg_quick.pkl\n",
    )
    runs = []
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=True)
    assert runs == []
    assert 'mkdir "$R/$stem"' in commands[0]
    assert 'rm -f "$f"' in commands[0]
    assert "cleared on the box" in capsys.readouterr().out


def test_clear_reports_nothing_when_no_files(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(transport, "_ssh_capture", lambda *a: "\n")
    runs = []
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: runs.append(cmd))
    remote.clear_remote("hopg", yes=True)
    assert "nothing to clear" in capsys.readouterr().out
    assert runs == []  # nothing to delete


def test_clear_multiple_materials_yes_deletes_all_stems(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: (
            commands.append(command)
            or "CLEARED\thopg.pkl\nCLEARED\thbn.pkl\nCLEARED\tdiamond.pkl\n"
        ),
    )
    remote.clear_remote(["hopg", "hbn", "diamond"], yes=True)
    # one reserving delete command covering every stem (material + _quick)
    assert len(commands) == 1
    for stem in ("hopg", "hopg_quick", "hbn", "hbn_quick", "diamond", "diamond_quick"):
        assert stem in commands[0]
    out = capsys.readouterr().out
    assert "cleared on the box" in out and "diamond.pkl" in out


def test_clear_multiple_materials_dry_preview_lists_each(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    commands = []
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: commands.append(command) or "hopg.pkl\nhbn.pkl\n"
    )
    remote.clear_remote(["hopg", "hbn"], yes=False)
    listing = commands[-1]
    for name in ("hopg.pkl", "hopg_quick.pkl", "hbn.pkl", "hbn_quick.pkl"):
        assert name in listing
    assert "would delete" in capsys.readouterr().out


def test_clear_profile_targets_only_profile_stems(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: (
            commands.append(command)
            or "hopg--profile-sub_100keV.pkl\nhbn--profile-sub_100keV.pkl\n"
        ),
    )
    monkeypatch.setattr(
        scripts,
        "_stems",
        lambda materials, quick, **kwargs: [
            f"{material}--full-profile-{kwargs['catalog_profile']}" for material in materials
        ],
    )

    remote.clear_remote(
        ["hopg", "hbn"],
        catalog_profile="sub_100keV",
    )

    listing = commands[-1]
    assert "hopg--full-profile-sub_100keV" in listing
    assert "hbn--full-profile-sub_100keV" in listing
    assert "hopg_quick" not in listing
    assert "would delete" in capsys.readouterr().out


def test_clear_profile_previews_partial_previous_identity_directory(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    old = "hopg@hopg_hbn-111111111111"
    commands = []
    monkeypatch.setattr(
        lifecycle_cleanup,
        "_profile_checkpoint_stems",
        lambda *_args, **_kwargs: [old],
    )
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or f"{old}/\n",
    )

    lifecycle.clear_remote(["hopg", "hbn"], catalog_profile="hopg_hbn")

    assert f"for stem in {old}" in commands[-1]
    assert f"checkpoints/{old}/" in capsys.readouterr().out


def test_clear_profile_cli_uses_profile_membership(monkeypatch):
    calls = []
    monkeypatch.setattr(
        _remote_actions, "_profile_default_materials", lambda profile: ("hopg", "hbn")
    )
    monkeypatch.setattr(
        lifecycle,
        "clear_remote",
        lambda materials, yes, **kwargs: calls.append((materials, yes, kwargs)),
    )

    _remote_main(["rm", "--profile", "sub_100keV", "--yes"])

    assert calls == [(["hopg", "hbn"], True, {"catalog_profile": "sub_100keV"})]


def test_clear_profile_rejects_materials_and_all(capsys):
    assert _remote_main(["rm", "hopg", "--profile", "sub_100keV"]) == 2
    assert "--profile takes no material" in capsys.readouterr().err
    assert _remote_main(["rm", "--all", "--profile", "sub_100keV"]) == 2
    assert "--profile takes no material" in capsys.readouterr().err


def test_clear_rejects_bad_material(monkeypatch):
    monkeypatch.setattr(
        state, "_live_jobs", lambda: pytest.fail("must validate before touching jobs")
    )
    with pytest.raises(SystemExit):
        remote.clear_remote("rm -rf /")


def test_clear_all_refuses_when_a_live_job_is_running(monkeypatch):
    monkeypatch.setattr(state, "_live_jobs", lambda: [("job1", False, ["hopg"])])
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda *a: pytest.fail("must not ssh when refusing")
    )
    with pytest.raises(SystemExit, match="refusing to clear --all"):
        remote.clear_all_remote(yes=True)


def test_clear_all_refuses_when_a_reservation_is_held(monkeypatch):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(state, "_reservation_ledger", lambda: (1000, [("hopg", "job1", 900)]))
    with pytest.raises(SystemExit, match="reservation"):
        remote.clear_all_remote(yes=True)


def test_clear_all_dry_preview_lists_but_does_not_delete(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(state, "_reservation_ledger", lambda: (1000, []))
    commands = []
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: commands.append(command) or "hopg.pkl\nzhai_reproduction/a.pkl\n",
    )
    remote.clear_all_remote(yes=False)
    out = capsys.readouterr().out
    assert "would delete" in out and "hopg.pkl" in out and "zhai_reproduction/a.pkl" in out
    assert len(commands) == 1  # listing only; no delete
    assert "-delete" not in commands[0]


def test_clear_all_yes_deletes_every_pkl(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(state, "_reservation_ledger", lambda: (1000, []))
    commands = []

    def capture(command):
        commands.append(command)
        return "" if "-delete" in command else "hopg.pkl\nhopg_quick.pkl\n"

    monkeypatch.setattr(transport, "_ssh_capture", capture)
    remote.clear_all_remote(yes=True)
    out = capsys.readouterr().out
    assert "cleared on the box: 2 checkpoint file(s)" in out
    assert any("-delete" in c for c in commands)


def test_clear_all_reports_nothing_when_empty(monkeypatch, capsys):
    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(state, "_reservation_ledger", lambda: (1000, []))
    monkeypatch.setattr(transport, "_ssh_capture", lambda *a: "\n")
    remote.clear_all_remote(yes=True)
    assert "nothing to clear" in capsys.readouterr().out


def test_clear_fails_closed_when_squeue_cannot_be_queried(monkeypatch, tmp_path):
    """A scheduler outage cannot be mistaken for an absent job before deletion."""
    bash = _bash_or_skip(tmp_path)
    job = tmp_path / "jobs" / "j"
    job.mkdir(parents=True)
    (job / "meta").write_text("slurm_job_id: 48291\nquick: False\nmaterials: hopg\n")
    commands = tmp_path / "bin"
    commands.mkdir()
    (commands / "squeue").write_text("#!/bin/sh\nexit 7\n")
    (commands / "squeue").chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())

    def local_bash_capture(remote_cmd):
        env = {**os.environ, "PATH": f"{commands}:{os.environ.get('PATH', '')}"}
        result = subprocess.run([bash, "-c", remote_cmd], capture_output=True, env=env)
        if result.returncode != 0:
            raise SystemExit(f"ssh command failed (exit {result.returncode})")
        return result.stdout.decode()

    monkeypatch.setattr(transport, "_ssh_capture", local_bash_capture)
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kw: pytest.fail("must not delete"))

    with pytest.raises(SystemExit, match="ssh command failed"):
        remote.clear_remote("hopg", yes=True)


def test_clear_listing_snippet_exits_zero_when_quick_pkl_missing(monkeypatch, tmp_path, capsys):
    """Regression (checkpoint-lifecycle verification, 2026-07-07): the box-side
    listing loop's last command is `[ -f "$f" ] && echo "$f"`, whose failure
    status leaks out as the loop's -- and hence ssh's -- exit status whenever the
    last stem (<material>_quick.pkl) is missing, making _ssh_capture abort the
    clear before the dry preview prints. Execute the real snippet under bash
    against a checkpoints/ dir holding only hopg.pkl."""
    bash = _bash_or_skip(tmp_path)
    (tmp_path / "pyrite-output" / "checkpoints").mkdir(parents=True)
    (tmp_path / "pyrite-output" / "checkpoints" / "hopg.pkl").write_bytes(b"x")  # no hopg_quick.pkl

    _no_live_jobs(monkeypatch)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())

    def local_bash_capture(remote_cmd):
        # same semantics as _ssh_capture, but run the snippet locally
        r = subprocess.run([bash, "-c", remote_cmd], capture_output=True, encoding="utf-8")
        if r.returncode != 0:
            raise SystemExit(f"ssh command failed (exit {r.returncode})")
        return r.stdout

    monkeypatch.setattr(transport, "_ssh_capture", local_bash_capture)
    monkeypatch.setattr(
        transport, "_run", lambda cmd, **kw: pytest.fail("dry preview must not delete")
    )
    remote.clear_remote("hopg", yes=False)  # must not raise SystemExit
    out = capsys.readouterr().out
    assert "hopg.pkl" in out and "hopg_quick.pkl" not in out
