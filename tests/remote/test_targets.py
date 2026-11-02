"""Target selection, persistence and CLI isolation; no network."""

from dataclasses import replace

import pytest

from pyrite.cli.commands import job, remote_targets, scan
from pyrite.cli.commands import remote as remote_cli
from pyrite.console import config as store
from pyrite.remote import config, scripts, targets
from tests.helpers.cli import assert_clean_result, invoke


@pytest.fixture
def configured(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.delenv("PYRITE_REMOTE_TARGET", raising=False)
    monkeypatch.setattr(config, "_remote_home", lambda host: "/home/remote")
    targets.save_target(
        "nvidia", {"host": "box", "remote_dir": "/srv/nvidia", "uv": "uv", "partition": "gpu"}
    )
    targets.save_target(
        "amd",
        {
            "host": "box",
            "remote_dir": "/srv/amd",
            "uv": "uv",
            "gpu_vendor": "amd",
            "partition": "gpu-amd",
            "nodelist": "ace",
            "gres": "gpu:amd:1",
        },
    )


def test_named_precedence_and_context_restoration(configured, monkeypatch):
    monkeypatch.setenv("PYRITE_REMOTE_PARTITION", "environment")
    monkeypatch.setenv("PYRITE_REMOTE_TARGET", "nvidia")
    assert targets.resolve_target().partition == "gpu"
    with config.use_target("amd"):
        assert config.remote_dir() == "/srv/amd"
        assert config.slurm_partition() == "gpu-amd"
        with config.use_target(replace(targets.resolve_target("nvidia"), host="other")):
            assert config.remote_host() == "other"
        assert config.remote_host() == "box"
    assert config.remote_dir() == "/srv/nvidia"


def test_legacy_selection(configured, monkeypatch):
    monkeypatch.setattr(config, "HOST", None)
    monkeypatch.setattr(config, "SLURM_PARTITION", None)
    store.set_stored("remote.target", "legacy")
    store.set_stored("remote.partition", "flat")
    monkeypatch.setenv("PYRITE_REMOTE_PARTITION", "environment")
    assert targets.resolve_target().host == "legacy"
    assert targets.resolve_target().partition == "environment"
    monkeypatch.delenv("PYRITE_REMOTE_PARTITION")
    assert targets.resolve_target().partition == "flat"


@pytest.mark.parametrize(
    "values",
    [
        {"host": "-bad"},
        {"host": "box", "partition": "bad\ntext"},
        {"host": "box", "remote_dir": "../relative"},
        {"host": "box", "uv": "uv;bad"},
    ],
)
def test_invalid_target(values):
    with pytest.raises(ValueError):
        targets.RemoteTarget(**values)


def test_unknown_target(configured):
    with pytest.raises(ValueError, match="unknown remote target"):
        targets.resolve_target("absent")


def test_target_crud(configured):
    assert_clean_result(invoke(remote_targets.command, ["use", "amd"]))
    assert targets.resolve_target().name == "amd"
    assert_clean_result(invoke(remote_targets.command, ["edit", "amd", "--partition", "compute"]))
    assert targets.resolve_target().partition == "compute"
    result = invoke(remote_targets.command, ["list", "--output", "json"])
    assert_clean_result(result)
    import json

    assert json.loads(result.stdout)["amd"]["host"] == "box"
    assert_clean_result(invoke(remote_targets.command, ["remove", "amd"]))
    assert "amd" in targets.named_targets()
    assert_clean_result(invoke(remote_targets.command, ["remove", "amd", "--yes"]))
    assert targets.selected_name() is None
    assert "amd" not in targets.named_targets()


def test_run_named_target(configured, monkeypatch):
    seen = []
    monkeypatch.setattr(
        remote_cli, "_cli_start", lambda **kw: seen.append(config.slurm_partition())
    )
    assert_clean_result(
        invoke(scan.command, ["standard", "-m", "hopg", "--remote", "--target", "amd", "--dry-run"])
    )
    assert seen == ["gpu-amd"]
    result = invoke(scan.command, ["standard", "--target", "amd"])
    assert result.exit_code == 2
    assert "--target requires --remote" in result.stderr


def test_remote_commands_and_job_lookup(configured, monkeypatch):
    seen = []
    monkeypatch.setattr(remote_cli, "_cli_sync", lambda **kw: seen.append(config.remote_dir()))
    assert_clean_result(invoke(remote_cli.command, ["sync", "--target", "amd"]))
    with config.use_target("amd"):
        targets.remember_job("testjob")
    targets.default_target("nvidia")
    # job.py imports its action by value.
    monkeypatch.setattr(job, "_cli_status", lambda **kw: seen.append(config.remote_dir()))
    assert_clean_result(invoke(job.command, ["status", "testjob"]))
    assert seen == ["/srv/amd", "/srv/amd"]
    with config.use_target("nvidia"):
        targets.remember_job("testjob")
    result = invoke(job.command, ["status", "testjob"])
    assert result.exit_code == 2
    assert "multiple targets" in result.stderr
    assert_clean_result(invoke(job.command, ["status", "testjob", "--target", "amd"]))


def test_batch_target_and_metadata(configured):
    with config.use_target("amd"):
        script = scripts._slurm_batch_script("job", "true", job_name="test", reservation_stems=[])
        metadata = scripts._write_job_script_command("/srv/amd/jobs/job", "kind: run\n")
    assert "#SBATCH --partition=gpu-amd" in script
    assert "PYRITE_MC_BACKEND=rocm" in script
    assert "target: amd" in metadata


def test_explicit_host_overrides_default_named_target(configured):
    targets.default_target("amd")
    with config.override_remote_host("other"):
        assert config.remote_host() == "other"
        assert config.slurm_partition() == "gpu-amd"
    assert config.remote_host() == "box"


def test_omitted_named_fields_follow_environment_then_flat_keys(configured, monkeypatch):
    monkeypatch.setattr(config, "SLURM_PARTITION", None)
    targets.save_target("minimal", {"host": "box"})
    store.set_stored("remote.partition", "flat")
    assert targets.resolve_target("minimal").partition == "flat"
    monkeypatch.setenv("PYRITE_REMOTE_PARTITION", "environment")
    assert targets.resolve_target("minimal").partition == "environment"
    monkeypatch.setenv("PYRITE_REMOTE_PARTITION", "")
    assert targets.resolve_target("amd").partition == "gpu-amd"
