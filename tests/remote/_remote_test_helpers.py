"""Shared helpers for remote command tests."""

import importlib
import io
import json
import os
import shutil
import subprocess

import pytest

from pyrite import remote
from pyrite.cli.commands import remote as cli  # noqa: F401
from pyrite.cli.commands import remote as remote_cli
from pyrite.console import output as _cli_core
from pyrite.console.dashboard import state as dashboard_state
from pyrite.remote import (  # noqa: F401
    _queue_scripts,
    config,
    lifecycle,
    scripts,
    state,
    transport,
    viewer,
)
from pyrite.remote import queue as lifecycle_queue


def _remote_main(argv):
    """Route `pyrite-remote` argv to the object that still owns each spelling.

    `run`, `rebrem`, `reline`, `validate`, and `check` retired as *spellings* at
    0.3.0, but the command objects behind them are still the implementation the
    canonical `pyrite run --remote` / `pyrite checkpoint recompute ... --remote`
    paths invoke. These tests are about that implementation, so they address it
    directly instead of through a group entry that no longer mounts it.
    """
    head, *rest = argv
    owners = {
        "run": remote_cli.start_command,
        "rebrem": remote_cli.rebrem_command,
        "reline": remote_cli.recxr_command,
        "validate": remote_cli.check_command,
        "check": remote_cli.check_command,
    }
    if head in owners:
        return _cli_core.run(owners[head], rest, prog_name=f"pyrite-remote {head}")
    if head in ("status", "logs"):
        # These moved wholesale to the canonical `pyrite job` noun at 0.3.0.
        from pyrite.cli.commands.job import command as job_command

        return _cli_core.run(job_command, [head, *rest], prog_name="pyrite job")
    return remote_cli.main(argv)


lifecycle_pull = importlib.import_module("pyrite.remote.pull")


def _use_amd_target(monkeypatch):
    monkeypatch.setattr(config, "REMOTE_GPU_VENDOR", "amd")
    monkeypatch.setattr(config, "SLURM_PARTITION", "gpu-amd")
    monkeypatch.setattr(config, "SLURM_NODELIST", "qlmc-ace")
    monkeypatch.setattr(config, "SLURM_GRES", "gpu:radeon8060s:1")


class _FakeLockProc:
    def __init__(self, first_line, returncode=0):
        self.stdout = io.StringIO(first_line)
        self.stdin = io.StringIO()
        self.returncode = returncode

    def wait(self):
        return self.returncode


def _quiet_submission(monkeypatch):
    monkeypatch.setattr(lifecycle_queue, "_refuse_if_busy", lambda *_args: None)
    monkeypatch.setattr(transport.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(transport, "_ssh_capture", lambda _command: "48291\n")


def _queue_payload(*rows, partition="gpu"):
    return "\n".join([f"cohort_partition={partition}|order=priority_desc_job_id_asc", *rows])


def _status_output(state, *, squeue_state="RUNNING", sid="48291", progress=""):
    """Fake one round-trip of the shared status command (the marked sections
    _format_job_status / the attach loop read)."""
    return dashboard_state._encode_sections(
        {
            "JOB": "j",
            "META": f"job: j\nmaterials: hopg\nslurm_job_id: {sid}",
            "STATE": state,
            "SQUEUE": f"job_id={sid}|state={squeue_state}",
            "PROGRESS": progress,
        }
    )


class _FakeKeyListener:
    """Scripted stand-in for viewer.KeyListener: one poll() result per call."""

    def __init__(self, polls, *, active=True):
        self._polls = list(polls)
        self.active = active
        self.stopped = False

    def poll(self):
        return self._polls.pop(0) if self._polls else []

    def stop(self):
        self.stopped = True


_PARTIAL_SECTIONS = {
    "META": "materials: hbn hopg\nquick: False\nfidelity: full\ncatalog_profile: high_energy",
    "PROGRESS": (
        '{"material":"hbn","total_cases":4,"cached_cases":1,'
        '"completed_new_cases":1,"state":"running"}'
    ),
}


def _timed_record(
    *,
    material="hopg",
    total=10,
    cached=0,
    completed=0,
    state="running",
    seconds=None,
    measured=None,
    done_cost=None,
    total_cost=None,
    measured_cost=None,
):
    record = {
        "material": material,
        "total_cases": total,
        "cached_cases": cached,
        "completed_new_cases": completed,
        "state": state,
    }
    optional = {
        "active_compute_seconds": seconds,
        "measured_new_cases": measured,
        "done_cost": done_cost,
        "total_cost": total_cost,
        "measured_new_cost": measured_cost,
    }
    record.update({key: value for key, value in optional.items() if value is not None})
    return record


def _no_live_jobs(monkeypatch):
    monkeypatch.setattr(state, "_live_jobs", lambda: [])


def _bash_or_skip(tmp_path):
    """A bash that can reach tmp_path, or skip. Git Bash handles C:/-style posix
    paths; a WSL bash can't, and would hit the snippet's `|| exit 0` and silently
    return an empty listing -- so probe the cd before trusting the test."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    probe = subprocess.run([bash, "-c", f"cd '{tmp_path.as_posix()}'"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("bash cannot reach the pytest tmp dir")
    return bash


def _fake_remote_catalog(listing, meta_by_stem):
    """Route ``transport._ssh_capture`` calls: the checkpoint-dir listing
    command returns ``listing`` verbatim; any other command is treated as a
    ``[ -f ... ] ... stat ... cat ...`` meta.json fetch and matched by which
    stem's path it names."""

    def fake(command):
        if "-mindepth 1 -maxdepth 1 -type d" in command:
            return listing
        for stem, (mtime, meta) in meta_by_stem.items():
            if f"checkpoints/{stem}/meta.json" in command:
                return f"{mtime}\n{json.dumps(meta)}"
        return ""

    return fake


def _run_prune_jobs_command(command, jobs_dir, live_ids):
    """Run a _prune_job_dirs_command against a real tmp jobs tree with a squeue shim."""
    bindir = jobs_dir.parent / "bin"
    bindir.mkdir(exist_ok=True)
    squeue = bindir / "squeue"
    squeue.write_text('#!/bin/sh\nprintf "%s\\n" ' + " ".join(f'"{i}"' for i in live_ids) + "\n")
    squeue.chmod(0o755)
    command = command.replace(
        f'JOBS="{remote.REMOTE_DIR}/{remote.JOBS_SUBDIR}"', f'JOBS="{jobs_dir}"'
    )
    env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
    return subprocess.run(["bash", "-c", command], capture_output=True, text=True, env=env)


def _make_job(jobs_dir, name, state_line, slurm_ids):
    job = jobs_dir / name
    job.mkdir(parents=True)
    (job / "meta").write_text("".join(f"slurm_job_id: {i}\n" for i in slurm_ids))
    (job / "state").write_text(state_line + "\n")
    return job


def _stub_sync_tree(tmp_path, monkeypatch, *, text="VALUE = 1\n"):
    """A minimal two-file sync payload rooted at ``tmp_path``."""
    source = tmp_path / "src" / "pyrite"
    source.mkdir(parents=True)
    (source / "module.py").write_text(text, newline="")
    (source / "table.bin").write_bytes(b"\x00\x01\x02")
    monkeypatch.setattr(config, "LOCAL_ROOT", tmp_path)
    monkeypatch.setattr(config, "SYNC_PATHS", ["src"])
    return source


def _live_job_line(jobid, digest, revision="deadbeef", material="hopg"):
    return f"{jobid}\tFalse\t{material}\t{digest}\t{revision}\n"
