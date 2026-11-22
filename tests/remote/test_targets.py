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


def _hoffman(**extra):
    values = {
        "host": "h2",
        "remote_dir": "/srv/h2",
        "uv": "uv",
        "partition": "campus",
        "account": "pi_lab",
        "constraint": "A100|H200|RTXPRO6000Bw",
        "max_time": "1-00:00:00",
    }
    targets.save_target("h2", values | extra)


def test_site_policy_fields_round_trip_through_cli(configured):
    assert_clean_result(
        invoke(
            remote_targets.command,
            [
                "add",
                "h2",
                "--host",
                "h2",
                "--partition",
                "campus",
                "--account",
                "pi_lab",
                "--constraint",
                "A100|H200",
                "--max-time",
                "1-00:00:00",
            ],
        )
    )
    target = targets.resolve_target("h2")
    assert (target.account, target.constraint, target.max_time) == (
        "pi_lab",
        "A100|H200",
        "1-00:00:00",
    )
    assert_clean_result(invoke(remote_targets.command, ["edit", "h2", "--constraint", "H200"]))
    assert targets.resolve_target("h2").constraint == "H200"


@pytest.mark.parametrize(
    "values",
    [
        {"host": "box", "account": "pi lab"},
        {"host": "box", "constraint": "A100\n#SBATCH --x"},
        {"host": "box", "constraint": "A100 H200"},
        {"host": "box", "max_time": "24h"},
    ],
)
def test_invalid_site_policy_fields(values):
    with pytest.raises(ValueError):
        targets.RemoteTarget(**values)


def test_batch_script_site_policy(configured):
    _hoffman()
    with config.use_target("h2"):
        unlimited = scripts._slurm_batch_script("job", "true", job_name="t", reservation_stems=[])
        chunked = scripts._slurm_batch_script(
            "job", "true", job_name="t", reservation_stems=[], time_limit="1200", chunked=True
        )
    assert "#SBATCH --account=pi_lab\n" in unlimited
    assert "#SBATCH --constraint=A100|H200|RTXPRO6000Bw\n" in unlimited
    assert "#SBATCH --time=1-00:00:00\n" in unlimited
    assert "UNLIMITED" not in unlimited
    assert "#SBATCH --time=1200\n" in chunked


def test_batch_script_rejects_time_over_cap(configured):
    _hoffman()
    with config.use_target("h2"), pytest.raises(SystemExit, match="--chunk-minutes 480 or less"):
        scripts._slurm_batch_script(
            "job", "true", job_name="t", reservation_stems=[], time_limit="1800", chunked=True
        )


def test_batch_script_without_site_policy_is_unchanged(configured):
    with config.use_target("nvidia"):
        script = scripts._slurm_batch_script("job", "true", job_name="t", reservation_stems=[])
    assert "--account" not in script and "--constraint" not in script
    assert "#SBATCH --partition=gpu\n#SBATCH --nodes=1\n" in script
    assert "#SBATCH --time=UNLIMITED\n" in script


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("30", 1800),
        ("30:15", 1815),
        ("2:00:00", 7200),
        ("1-00:00:00", 86400),
        ("1-12", 129600),
        ("1-0:30", 88200),
        ("UNLIMITED", None),
    ],
)
def test_slurm_time_seconds(value, seconds):
    assert config.slurm_time_seconds(value) == seconds


def test_host_exec_wraps_host_side_python(configured):
    _hoffman(host_exec="srun --time=00:30:00 --cpus-per-task=2 --mem=8G")
    with config.use_target("nvidia"):
        plain = config.host_uv_command()
    assert plain.startswith("PYRITE_DATA_DIR=") and plain.endswith(' "uv"')
    with config.use_target("h2"):
        wrapped = config.host_uv_command("PYRITE_REMOTE_DIR=/srv/h2")
        gc = scripts._prune_checkpoint_stems_command("gcjob", ["hopg"], catalog_profile="standard")
    assert wrapped.startswith(
        "srun --time=00:30:00 --cpus-per-task=2 --mem=8G env PYRITE_DATA_DIR="
    )
    assert wrapped.endswith(' PYRITE_REMOTE_DIR=/srv/h2 "uv"')
    assert "srun --time=00:30:00 --cpus-per-task=2 --mem=8G env " in gc


@pytest.mark.parametrize(
    "value", ["srun; rm -rf ~", "srun $(id)", "srun 'quoted'", "srun --x  --y", "srun\n"]
)
def test_host_exec_rejects_shell_text(value):
    with pytest.raises(ValueError):
        targets.RemoteTarget(host="box", host_exec=value)


def test_add_existing_and_edit_missing_name_the_other_command(configured):
    result = invoke(remote_targets.command, ["add", "amd", "--host", "box"])
    assert result.exit_code == 2
    assert "pyrite remote target edit amd" in result.stderr
    result = invoke(remote_targets.command, ["edit", "absent", "--partition", "gpu"])
    assert result.exit_code == 2
    assert "pyrite remote target add absent" in result.stderr
