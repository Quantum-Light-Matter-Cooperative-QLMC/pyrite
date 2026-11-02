"""Public client contracts and exact CLI/API submission parity, without SSH."""

import subprocess

import pytest

from pyrite.cli.commands import scan
from pyrite.console import config as store
from pyrite.remote import (
    RemoteError,
    RemoteJob,
    RemoteStatus,
    RemoteTarget,
    client,
    config,
    lifecycle,
    queue,
    state,
    submit,
    targets,
    transport,
)
from tests.helpers.cli import assert_clean_result, invoke


@pytest.fixture
def destination(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.delenv("PYRITE_REMOTE_TARGET", raising=False)
    target = targets.save_target(
        "compute",
        {
            "host": "box",
            "remote_dir": "/srv/pyrite",
            "uv": "uv",
            "gpu_vendor": "amd",
            "partition": "gpu-amd",
        },
    )
    monkeypatch.setattr(queue.scripts, "_new_jobid", lambda: "fixed-job")
    return target


@pytest.mark.parametrize("chunk_minutes", [0.0, 10.0])
def test_exact_cli_api_script_parity(destination, capsys, chunk_minutes):
    handle = submit(
        "standard", target="compute", materials=["hopg"], dry_run=True, chunk_minutes=chunk_minutes
    )
    assert capsys.readouterr() == ("", "")
    explicit = submit(
        "standard",
        target=destination,
        materials=["hopg"],
        dry_run=True,
        chunk_minutes=chunk_minutes,
    )
    assert explicit.script == handle.script
    result = invoke(
        scan.command,
        [
            "standard",
            "-m",
            "hopg",
            "--remote",
            "--target",
            "compute",
            "--dry-run",
            "--chunk-minutes",
            str(chunk_minutes),
        ],
    )
    assert_clean_result(result)
    assert handle.script in result.stdout
    assert "#SBATCH --partition=gpu-amd" in handle.script
    assert "PYRITE_MC_BACKEND=rocm" in handle.script
    assert handle.checkpoint_stems
    assert not capsys.readouterr().out


def test_submission_uses_same_staging_and_retains_target(destination, monkeypatch, capsys):
    staged = []
    monkeypatch.setattr(queue, "_refuse_if_busy", lambda *a: None)
    monkeypatch.setattr(queue, "_stage_job_script", lambda *a: staged.append(a))
    monkeypatch.setattr(queue, "_submit_staged_job", lambda *a, **kw: "123")
    handle = submit("standard", target="compute", materials=["hopg"], no_sync=True)
    assert staged[0][0] == handle.jobid
    assert staged[0][-1] == handle.script
    assert handle.scheduler_id == "123"
    targets.save_target("compute", {"host": "changed"}, edit=True)
    seen = []
    monkeypatch.setattr(
        state, "_job_state", lambda job: seen.append(config.remote_host()) or "done [123]"
    )
    monkeypatch.setattr(state, "_slurm_job_id", lambda job: "123")
    assert handle.status().succeeded
    assert seen == ["box"]
    assert capsys.readouterr() == ("", "")


def test_status_logs_stop(destination, monkeypatch, capsys):
    handle = RemoteJob("job", destination)
    monkeypatch.setattr(state, "_job_state", lambda job: "running [1]")
    monkeypatch.setattr(state, "_slurm_job_id", lambda job: "1")
    assert not handle.status().terminal
    seen = []
    monkeypatch.setattr(
        transport, "_ssh_capture", lambda command: seen.append(command) or "diagnostic\n"
    )
    assert handle.logs(lines=20) == "diagnostic\n"
    assert "tail -n 20" in seen[0]
    monkeypatch.setattr(
        lifecycle, "_stop_jobid", lambda job: seen.append((job, config.remote_host()))
    )
    assert handle.stop().terminal
    assert seen[-1] == ("job", "box")
    assert capsys.readouterr() == ("", "")


def test_wait_and_timeout(destination, monkeypatch):
    snapshots = iter([RemoteStatus("job", "running", "1"), RemoteStatus("job", "done", "2")])
    monkeypatch.setattr(RemoteJob, "status", lambda self: next(snapshots))
    monkeypatch.setattr(client.time, "sleep", lambda seconds: None)
    handle = RemoteJob("job", destination)
    assert handle.wait(poll_interval=0.01).succeeded
    monkeypatch.setattr(RemoteJob, "status", lambda self: RemoteStatus("job", "queued", "1"))
    with pytest.raises(TimeoutError):
        handle.wait(timeout=0)


def test_pull_results_and_partial_failure(destination, monkeypatch, tmp_path, capsys):
    from pyrite.console import outputs

    monkeypatch.setattr(outputs, "output_dir", lambda kind: tmp_path)

    def pull(stems, **kwargs):
        assert stems == ["hopg"]
        assert config.remote_host() == "box"
        kwargs["summary"].update(completed=["hopg"], failed=[], errors={})

    monkeypatch.setattr(lifecycle, "pull", pull)
    handle = RemoteJob("job", destination, checkpoint_stems=("hopg",))
    assert handle.pull(no_sync=True).paths == (tmp_path / "hopg",)

    def partial(stems, **kwargs):
        kwargs["summary"].update(
            completed=["hopg"], failed=["hbn"], errors={"hbn": "transfer failed"}
        )
        raise SystemExit("partial failure")

    monkeypatch.setattr(lifecycle, "pull", partial)
    with pytest.raises(RemoteError, match="partial failure") as error:
        handle.pull()
    assert error.value.result.failed == ("hbn",)
    assert error.value.result.paths == (tmp_path / "hopg",)
    assert capsys.readouterr() == ("", "")


def test_errors_are_library_errors(destination, monkeypatch, capsys):
    with pytest.raises(ValueError, match="unknown remote target"):
        submit(target="absent", materials=["hopg"], dry_run=True)
    with pytest.raises(ValueError, match="unknown profile"):
        submit("absent", target=destination, dry_run=True)
    with pytest.raises(ValueError):
        RemoteTarget(host="bad;host")

    def fail(job):
        raise SystemExit("ssh failed")

    monkeypatch.setattr(state, "_job_state", fail)
    with pytest.raises(RemoteError, match="ssh failed"):
        RemoteJob("job", destination).status()
    assert capsys.readouterr() == ("", "")


def test_quiet_transport_is_local_to_operation(destination, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(transport.subprocess, "run", lambda *a, **kw: seen.append(kw))
    with client._operation(destination):
        transport._run(["ssh", "box", "true"], label="hidden")
    assert capsys.readouterr() == ("", "")
    assert seen[0]["stdout"] == subprocess.DEVNULL
    transport._run(["ssh", "box", "true"], label="shown")
    assert capsys.readouterr().err == "shown\n"
    assert "stdout" not in seen[1]


@pytest.mark.parametrize("method", ["status", "logs", "stop", "wait", "pull"])
def test_dry_run_refuses_lifecycle(destination, method):
    handle = RemoteJob("preview", destination, dry_run=True)
    with pytest.raises(ValueError, match="dry-run"):
        getattr(handle, method)()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"workers": -1},
        {"chunk_minutes": -1},
        {"materials": "hopg"},
        {"parallel_materials": 0},
        {"parallel_materials": 2},
    ],
)
def test_invalid_submission_values(destination, kwargs):
    with pytest.raises((ValueError, TypeError)):
        submit(target=destination, dry_run=True, **kwargs)


def test_unsuccessful_terminal_status():
    assert RemoteStatus("job", "FAILED (scan)", "1").terminal
    assert not RemoteStatus("job", "done with 1 warning(s)", "1").succeeded


def test_explicit_target_provenance_does_not_overwrite_another_host(destination):
    from dataclasses import replace

    with config.use_target(destination):
        targets.remember_job("profile-job")
    with config.use_target(replace(destination, host="other", name=None)):
        targets.remember_job("profile-job")
    with pytest.raises(ValueError, match="multiple targets"):
        targets.job_target("profile-job")


def test_submission_records_target_even_when_ssh_response_is_lost(destination, monkeypatch):
    from pyrite.remote import jobs

    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: (_ for _ in ()).throw(SystemExit("lost response")),
    )
    monkeypatch.setattr(jobs, "_release_if_submission_definitely_failed", lambda *args: None)
    with config.use_target(destination), pytest.raises(SystemExit, match="lost response"):
        jobs._submit_staged_job("profile-job", [])
    assert targets.job_target("profile-job") == destination


def test_missing_implicit_target_is_a_value_error(destination, monkeypatch):
    monkeypatch.setattr(config, "HOST", None)
    monkeypatch.delenv("PYRITE_REMOTE_HOST", raising=False)
    with pytest.raises(ValueError, match="not configured"):
        targets.resolve_target()
