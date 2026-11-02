"""Offline checks for remote app metadata, SLURM scripts, and tunnel selection."""

import subprocess

import pytest
from click.testing import CliRunner

from pyrite.cli.commands import remote_app as remote_cli
from pyrite.console.json import job_kind, remote_jobs
from pyrite.remote import app


def _config(monkeypatch):
    monkeypatch.setattr(app.config, "remote_host", lambda: "login")
    monkeypatch.setattr(app.config, "remote_dir", lambda: "/scratch/pyrite")
    monkeypatch.setattr(app.config, "remote_path", lambda *p: "/scratch/pyrite/" + "/".join(p))
    monkeypatch.setattr(
        app.config, "sbatch_remote_path", lambda *p: "/scratch/pyrite/" + "/".join(p)
    )
    monkeypatch.setattr(app.config, "remote_catalog_path", lambda: "/scratch/pyrite/catalog")
    monkeypatch.setattr(
        app.config, "remote_user_catalog_path", lambda: "/scratch/pyrite/user-catalog"
    )
    monkeypatch.setattr(app.config, "remote_uv", lambda: "uv")
    monkeypatch.setattr(app.config, "slurm_partition", lambda: "cpu")
    monkeypatch.setattr(app.config, "slurm_nodelist", lambda: None)
    monkeypatch.setattr(app.config, "slurm_gres", lambda: "gpu:1")
    monkeypatch.setattr(app.config, "remote_gpu_vendor", lambda: "nvidia")
    monkeypatch.setattr(app.transport, "_check_materials", lambda *_: None)


@pytest.mark.parametrize("name", ["analysis", "viewer", "validation"])
def test_script_cpu_default_is_authenticated_and_loopback(monkeypatch, name):
    _config(monkeypatch)
    material = "hopg" if name != "validation" else None
    argv, notebook, port = remote_cli._invocation(name, material, edit=False, batch=False)
    rendered = app._script(
        "app-123",
        name,
        material,
        argv=argv,
        notebook=notebook,
        base_port=port,
        gpu=False,
        walltime="04:00:00",
        batch=False,
    )
    assert "#SBATCH --partition=cpu" in rendered
    assert "#SBATCH --gres=" not in rendered
    assert "--host 127.0.0.1" in rendered
    assert "--token --token-password-file" in rendered
    assert "--no-token" not in rendered
    assert "url: http://127.0.0.1:%s/?access_token=%s" in rendered
    assert 'chmod 600 "$JOBDIR/meta"' in rendered
    assert '"$PORT"' in rendered
    result = subprocess.run(["bash", "-n"], input=rendered, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["analysis", "viewer", "validation"])
@pytest.mark.parametrize("batch", [False, True])
def test_app_invocation_uses_matching_builder(name, batch):
    material = None if name == "validation" else "wse2"
    argv, notebook, port = remote_cli._invocation(name, material, edit=False, batch=batch)
    assert notebook in argv
    assert isinstance(port, int)
    assert ("export" in argv) is batch
    assert ("--material" in argv) is (name != "validation")
    if name != "validation":
        assert argv[-1] == "wse2"


def test_app_invocation_refuses_unknown_app():
    with pytest.raises(ValueError, match="unknown remote app"):
        remote_cli._invocation("unknown", None, edit=False, batch=False)


def test_gpu_is_opt_in(monkeypatch):
    _config(monkeypatch)
    argv, notebook, port = remote_cli._invocation("viewer", "hopg", edit=True, batch=False)
    script = app._script(
        "app-123",
        "viewer",
        "hopg",
        argv=argv,
        notebook=notebook,
        base_port=port,
        gpu=True,
        walltime="01:20:00",
        batch=False,
    )
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --time=01:20:00" in script
    assert "marimo -q edit" in script


def test_batch_is_html_export_and_no_server(monkeypatch):
    _config(monkeypatch)
    argv, notebook, port = remote_cli._invocation("analysis", "hopg", edit=False, batch=True)
    script = app._script(
        "app-123",
        "analysis",
        "hopg",
        argv=argv,
        notebook=notebook,
        base_port=port,
        gpu=False,
        walltime="01:00:00",
        batch=True,
    )
    assert "marimo export html" in script
    assert '"$JOBDIR/app.html"' in script
    assert "--token-password-file" not in script
    assert 'echo "done $(date -Is)"' in script
    assert subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


def test_invalid_time_rejected_before_remote_work(monkeypatch):
    _config(monkeypatch)
    with pytest.raises(ValueError, match="--time"):
        app._script(
            "app-123",
            "viewer",
            None,
            argv=["python", "-m", "marimo", "run", "a.py"],
            notebook="a.py",
            base_port=2719,
            gpu=False,
            walltime="01:00:00; rm -rf ~",
            batch=False,
        )


def test_node_uses_jump_host_only_when_needed(monkeypatch):
    monkeypatch.setattr(app.config, "remote_host", lambda: "login")
    monkeypatch.setattr(app.config, "ssh_argv", lambda *p: ["ssh", *p])
    local = app._tunnel_argv("login", 2719, 2720)
    remote = app._tunnel_argv("worker-001", 2719, 2720)
    assert "-J" not in local
    assert remote[-3:] == ["-J", "login", "worker-001"]
    assert "127.0.0.1:2719:127.0.0.1:2720" in local
    with pytest.raises(ValueError):
        app._tunnel_argv("evil;touch /tmp/owned", 2719, 2720)


def test_authenticated_metadata_required():
    fields = {
        "node": "compute-01",
        "port": "2718",
        "url": "http://127.0.0.1:2718/?access_token=secret",
        "ready": "yes",
    }
    assert app._ready(fields) == ("compute-01", 2718, fields["url"])
    assert app._ready({**fields, "ready": "no"}) is None
    with pytest.raises(ValueError):
        app._ready({**fields, "url": "http://127.0.0.1:2718/"})


def test_remote_no_token_refused_before_submit(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("must not submit")

    monkeypatch.setattr(app, "submit", fail)
    result = CliRunner().invoke(remote_cli.command, ["analysis", "--no-token"])
    assert result.exit_code != 0
    assert "forbidden" in result.output


def test_job_kind_app_round_trip():
    assert job_kind("app") == "app"
    record = "app-123\t123456\t?\tapp\t-\trunning 2026-11-02T12:00:00+00:00"
    parsed = remote_jobs(record, kind="app")
    assert parsed.payload["jobs"][0]["kind"] == "app"
