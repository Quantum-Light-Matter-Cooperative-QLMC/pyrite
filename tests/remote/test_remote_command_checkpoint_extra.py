"""Additional remote checkpoint tests."""

import os
import shutil
import subprocess

import pytest

from pyrite import remote
from pyrite.remote import config, lifecycle, scripts, state, transport
from pyrite.remote import queue as lifecycle_queue
from tests.remote._remote_test_helpers import (
    _bash_or_skip,
    lifecycle_pull,
)


def test_scancel_jobs_command_batches_sentinel_scancel_and_release():
    cmd = scripts._scancel_jobs_command(["job1", "job2"])

    # every STOP sentinel lands before the single scancel
    assert cmd.index(': > "$D/STOP"') < cmd.index("scancel $SIDS")
    assert cmd.count("scancel ") == 1
    # one shared squeue poll loop, then per-job release + terminal state
    assert cmd.count("squeue -h") == 1
    assert 'echo "cancelled [$SID]' in cmd
    assert '"$R"/*' in cmd  # reservation release


def test_scancel_jobs_command_functional(monkeypatch, tmp_path):
    """Run the batched cancel: STOP sentinels, one scancel, terminal states,
    reservation release -- and a job without a scheduler id is skipped, not
    fatal."""
    bash = _bash_or_skip(tmp_path)
    jobs = tmp_path / "jobs"
    for jobid, sid in [("j1", "101"), ("j2", "102")]:
        jobdir = jobs / jobid
        jobdir.mkdir(parents=True)
        (jobdir / "meta").write_text(f"slurm_job_id: {sid}\n")
    (jobs / "j3").mkdir()  # no meta -> skipped
    reservation = jobs / "reservations" / "hopg"
    reservation.mkdir(parents=True)
    (reservation / "jobid").write_text("j1")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    scancel_log = tmp_path / "scancel.args"
    fake_scancel = bin_dir / "scancel"
    fake_scancel.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" >> "{scancel_log.as_posix()}"\n')
    fake_squeue = bin_dir / "squeue"
    fake_squeue.write_text("#!/bin/sh\nexit 0\n")  # nothing live -> poll loop exits at once
    for fake in (fake_scancel, fake_squeue):
        fake.chmod(0o755)
    monkeypatch.setattr(config, "REMOTE_DIR", tmp_path.as_posix())
    monkeypatch.setenv("PATH", f"{bin_dir.as_posix()}:{os.environ['PATH']}")

    command = scripts._scancel_jobs_command(["j1", "j2", "j3"])
    result = subprocess.run(
        [bash, "-c", command], capture_output=True, text=True, env=os.environ.copy()
    )

    assert result.returncode == 0, result.stderr
    assert "skipping j3" in result.stderr
    assert scancel_log.read_text().split() == ["101", "102"]  # one batched scancel
    assert (jobs / "j1" / "STOP").exists() and (jobs / "j2" / "STOP").exists()
    assert (jobs / "j1" / "state").read_text().startswith("cancelled [101]")
    assert not reservation.exists()  # released
    assert "cancelled SLURM job 102 for job j2" in result.stdout


def test_component_pull_projects_transfer_pickle_and_installs_split_store(monkeypatch, tmp_path):
    import numpy as np

    from pyrite.checkpoints import _checkpoint_io, _checkpoint_store

    payload = {
        "cfg": {
            30.0: {
                "case": {"crystal": "hopg", "Ne_brem": 10},
                "E_grid": np.array([1.0, 2.0]),
                "spec": np.array([3.0, 4.0]),
                "E_grid_brem": np.array([1.0, 2.0]),
                "brem_wide": np.array([5.0, 6.0]),
                "brem": np.array([5.0, 6.0]),
            }
        }
    }
    transfer = tmp_path / "transfer.pkl"
    _checkpoint_io.dump(payload, str(transfer))
    transfers = []

    def fake_download(command, destination, **_kwargs):
        transfers.append((command, destination))
        shutil.copyfile(transfer, destination)

    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(transport, "_ssh_download", fake_download)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "")  # no survey siblings

    remote.pull(["hopg"], grid=True, level9=True, no_sync=True)

    assert (tmp_path / "pyrite-output" / "checkpoints" / "hopg" / "line.h5").is_file()
    assert (tmp_path / "pyrite-output" / "checkpoints" / "hopg" / "brem.h5").is_file()
    assert (tmp_path / "pyrite-output" / "checkpoints" / "hopg" / "characteristic.h5").is_file()
    loaded = _checkpoint_store.load("hopg", tmp_path / "pyrite-output" / "checkpoints")
    assert np.array_equal(loaded["cfg"][30.0]["spec"], np.array([3.0, 4.0]))
    assert len(transfers) == 1
    transfer_command, destination = transfers[0]
    assert "/checkpoints/hopg" in transfer_command
    assert "pyrite checkpoint slim" in transfer_command
    assert "pyrite slim" not in transfer_command
    assert "--grid" in transfer_command
    assert f"--compresslevel {_checkpoint_io.MAX_LEVEL}" in transfer_command
    # Streamed straight down the ssh session's stdout: no box temp to trap-clean,
    # so the box's compress pass overlaps the transfer.
    assert transfer_command.endswith("-o -")
    assert "cat " not in transfer_command
    assert destination == tmp_path / "pyrite-output" / "checkpoints" / ".hopg.incoming.pkl"
    syntax = subprocess.run(["bash", "-n", "-c", transfer_command], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr


def test_split_profile_selector_parses_and_rejects_empty_halves():
    assert lifecycle._split_profile_selector("hopg") is None
    assert lifecycle._split_profile_selector("hopg@sub_100keV") == ("hopg", "sub_100keV")
    with pytest.raises(SystemExit, match="invalid MATERIAL@PROFILE"):
        lifecycle._split_profile_selector("hopg@")
    with pytest.raises(SystemExit, match="invalid MATERIAL@PROFILE"):
        lifecycle._split_profile_selector("@sub_100keV")


def test_split_profile_selector_treats_full_at_stem_as_literal():
    """After the @-stem migration a resolved on-disk stem carries an ``@`` too
    (``<material>@<label>-<digest>``). It is an exact checkpoint, not a
    MATERIAL@PROFILE query, so pull must pass it through literally rather than
    try to re-resolve ``<label>-<digest>`` as a catalog profile."""
    from pyrite.campaign.profiles import named_profile_stem

    at_stem = named_profile_stem("hopg", "full", catalog_profile="sub_100keV")
    assert "@" in at_stem and at_stem != "hopg@sub_100keV"  # full stem, has digest tail
    assert lifecycle._split_profile_selector(at_stem) is None
    # a bare selector (no digest tail) still resolves as a profile query
    assert lifecycle._split_profile_selector("hopg@sub_100keV") == ("hopg", "sub_100keV")


def test_orphaned_reservation_jobs_partitions_dead_stale_from_live_and_recent(monkeypatch):
    now = 10_000
    monkeypatch.setattr(
        state,
        "_reservation_ledger",
        lambda: (
            now,
            [
                ("hopg", "dead", now - 600),  # dead + stale -> orphan
                ("hbn", "dead", now - 600),
                ("mose2", "live", now - 600),  # has SLURM allocation -> protected
                ("wse2", "fresh", now - 10),  # dead but mid-submit window -> protected
            ],
        ),
    )
    monkeypatch.setattr(
        state, "_slurm_job_id", lambda jobid: "48291" if jobid in {"live", "fresh"} else None
    )
    monkeypatch.setattr(state, "_slurm_state", lambda sid: "RUNNING" if sid == "48291" else None)

    orphans, protected = remote._orphaned_reservation_jobs(300.0)

    assert orphans == {"dead": ["hbn", "hopg"]}
    assert protected == {"live": ["mose2"], "fresh": ["wse2"]}


def test_reap_reservations_dry_run_previews_without_releasing(monkeypatch, capsys):
    monkeypatch.setattr(
        state,
        "_orphaned_reservation_jobs",
        lambda _min_age: ({"dead": ["hopg", "hbn"]}, {"live": ["mose2"]}),
    )
    runs = []
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: runs.append(cmd))

    remote.reap_reservations(yes=False)

    out = capsys.readouterr().out
    assert "would release dead" in out
    assert "keeping live" in out
    assert "preview only; re-run with -y/--yes to execute" in out
    assert runs == []


def test_reap_reservations_yes_releases_orphans_and_stamps_terminal_state(monkeypatch, capsys):
    monkeypatch.setattr(
        state, "_orphaned_reservation_jobs", lambda _min_age: ({"dead": ["hopg"]}, {})
    )
    runs = []
    monkeypatch.setattr(transport, "_run", lambda cmd, **kw: runs.append(cmd[-1]))

    remote.reap_reservations(yes=True)

    (command,) = runs
    assert 'J="dead"' in command  # releases only the orphan's own locks
    assert "rm -rf" in command
    assert "reaped (orphan reservations released)" in command
    assert "reaped 1 orphaned job(s)" in capsys.readouterr().out


def test_reap_job_command_releases_only_matching_owner(tmp_path):
    bash = _bash_or_skip(tmp_path)
    monkeypatch_dir = tmp_path
    reservations = monkeypatch_dir / "jobs" / "reservations"
    for stem, owner in (("hopg", "dead"), ("mose2", "dead"), ("wse2", "other")):
        d = reservations / stem
        d.mkdir(parents=True)
        (d / "jobid").write_text(f"{owner}\n")
    (monkeypatch_dir / "jobs" / "dead").mkdir(parents=True)

    old = config.REMOTE_DIR
    config.REMOTE_DIR = str(monkeypatch_dir)
    try:
        result = subprocess.run(
            [bash, "-c", scripts._reap_job_command("dead")], capture_output=True, text=True
        )
    finally:
        config.REMOTE_DIR = old

    assert result.returncode == 0, result.stderr
    assert not (reservations / "hopg").exists()
    assert not (reservations / "mose2").exists()
    assert (reservations / "wse2").exists()  # another owner's lock untouched
    assert "reaped" in (monkeypatch_dir / "jobs" / "dead" / "state").read_text()


def test_start_profile_submit_suggests_profile_pull_not_a_stem_wall(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle_queue, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")

    remote.start_queue(["hopg", "hbn", "mose2", "wse2"], no_sync=True, catalog_profile="sub_100keV")

    output = capsys.readouterr().out
    assert "pyrite remote pull --profile sub_100keV" in output
    assert "pyrite remote pull hopg--" not in output  # no hash-stem wall


def test_start_standard_submit_suggests_stem_pull(monkeypatch, capsys):
    monkeypatch.setattr(lifecycle_queue, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")

    from pyrite.campaign.profiles import named_profile_stem

    remote.start_queue(["hopg"], no_sync=True)

    output = capsys.readouterr().out
    assert f"pyrite remote pull {named_profile_stem('hopg')} " in output


def test_interrupted_job_upload_releases_its_checkpoint_reservations(monkeypatch):
    commands = []
    monkeypatch.setattr(lifecycle_queue, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport, "_run", lambda command, **_kwargs: commands.append(command))
    monkeypatch.setattr(
        transport.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: pytest.fail("must not submit"))

    from pyrite.campaign.profiles import named_profile_stem

    with pytest.raises(KeyboardInterrupt):
        remote.start_queue(["hopg"], no_sync=True)

    assert len(commands) == 2
    assert 'J="' in commands[-1][-1]
    assert f'if [ "$(cat "$R/{named_profile_stem("hopg")}/jobid"' in commands[-1][-1]


def test_profile_checkpoint_stems_discovers_previous_profile_identity(monkeypatch):
    old = "hopg@hopg_hbn-111111111111"
    same_label_variant = "hbn@hopg_hbn-222222222222"
    unrelated = "hopg@other-333333333333"
    detector_labelled = "hopg@pixel_a-444444444444"
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda _command: f"{old}\n{same_label_variant}\n{unrelated}\n{detector_labelled}\n",
    )
    identities = {
        old: {"catalog_profile": "hopg_hbn"},
        same_label_variant: {"catalog_profile": "standard", "variant": "hopg_hbn"},
        unrelated: {"catalog_profile": "other"},
        detector_labelled: {"catalog_profile": "hopg_hbn", "detector_id": "pixel_a"},
    }
    monkeypatch.setattr(
        lifecycle_pull,
        "_remote_meta_json",
        lambda stem: ("raw", {"dataset_identity": identities[stem]}),
    )

    assert lifecycle._profile_checkpoint_stems(
        "hopg_hbn",
        current_stems=["hopg@hopg_hbn-aaaaaaaaaaaa"],
    ) == [
        old,
        "hopg@hopg_hbn-aaaaaaaaaaaa",
        detector_labelled,
    ]


def test_stems_predicts_qualified_stem_for_non_standard_catalog_profile(monkeypatch):
    import pyrite.campaign.profiles as profiles_module

    # standard runs adaptive counts by default (#361), so its stem is hashed too.
    assert scripts._stems(["mos2"], False) == [profiles_module.named_profile_stem("mos2")]

    calls = []

    def fake_named_profile_stem(
        material, fidelity, *, catalog_profile="standard", detector_id=None
    ):
        calls.append((material, fidelity, catalog_profile, detector_id))
        return f"{material}--{fidelity}-qualified"

    monkeypatch.setattr(profiles_module, "named_profile_stem", fake_named_profile_stem)

    assert scripts._stems(["mos2"], False, catalog_profile="sub_100keV") == ["mos2--full-qualified"]
    assert calls == [("mos2", "full", "sub_100keV", "default")]


def test_stems_quick_suffix():
    assert remote._stems(["mose2", "wse2"], True) == ["mose2_quick", "wse2_quick"]
    from pyrite.campaign.profiles import named_profile_stem

    assert remote._stems(["mose2"], False) == [named_profile_stem("mose2")]


def test_stem_resolves_survey_profile():
    from pyrite.campaign.profiles import named_profile_stem

    assert remote._stems(["mos2"], False, "survey") == [named_profile_stem("mos2", "survey")]


def test_submit_failure_releases_its_checkpoint_reservations():
    command = remote._submit_slurm_command("j", ["hopg"])

    assert "FAILED (sbatch submission)" in command
    assert '"$R/hopg/jobid"' in command


def test_checkpoint_reservation_rejects_a_second_stager_for_the_same_stem(monkeypatch, tmp_path):
    """The remote ``mkdir`` lock closes the gap between a busy check and sbatch."""
    monkeypatch.setattr(config, "REMOTE_DIR", str(tmp_path))

    first = subprocess.run(
        ["bash", "-c", remote._reserve_checkpoint_stems_command("first", ["hopg"])],
        capture_output=True,
        text=True,
    )
    second = subprocess.run(
        ["bash", "-c", remote._reserve_checkpoint_stems_command("second", ["hopg"])],
        capture_output=True,
        text=True,
    )

    assert first.returncode == 0
    assert second.returncode != 0
    assert "refusing to stage" in second.stderr
    assert (tmp_path / "jobs" / "reservations" / "hopg" / "jobid").read_text().strip() == "first"
