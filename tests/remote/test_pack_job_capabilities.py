"""Pack capability is an explicit job contract, not a script heuristic."""

import shlex
import shutil
import subprocess

import pytest

from pyrite.remote import _environment_pack as pack
from pyrite.remote import config, jobs, scripts, trajectories, transport


@pytest.fixture
def pack_target(monkeypatch):
    identity = pack.PackIdentity("a" * 64, "nvidia", "linux-64-cuda", "b" * 64)
    monkeypatch.setattr(pack, "pack_identity", lambda: identity)
    target = config.RemoteTarget(
        host="box", remote_dir="/shared/pyrite", pixi="pixi", env_transfer="pack"
    )
    with config.use_target(target):
        yield


def test_stage_refuses_undeclared_pack_before_any_transfer(pack_target, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported job must not stage anything")

    monkeypatch.setattr(pack, "stage_pack", unexpected)
    monkeypatch.setattr(transport, "_run", unexpected)
    with pytest.raises(SystemExit, match="does not yet support env_transfer=pack"):
        jobs._stage_job_script("job", [], "upload", "PYRITE_PACK_PREFIX=pretend")
    with pytest.raises(SystemExit, match="does not yet support env_transfer=pack"):
        trajectories.submit_score({})


def test_declared_capability_does_not_inspect_script(pack_target, monkeypatch):
    calls = []
    monkeypatch.setattr(pack, "stage_pack", lambda: calls.append("pack"))
    monkeypatch.setattr(transport, "_run", lambda *a, **k: calls.append("transport"))
    monkeypatch.setattr(jobs, "is_quiet", lambda: True)
    jobs._stage_job_script("job", [], "upload", "echo payload", supports_pack=True)
    assert calls == ["pack", "transport", "transport"]


@pytest.mark.parametrize(
    ("builder", "args"),
    [
        (scripts._queue_script, ("job", ["hopg"], True, 1)),
        (scripts._chunked_queue_script, ("job", ["hopg"], True, 1, 10)),
        (scripts._zhai_queue_script, ("job", 10, 10, 10, 0, False)),
        (scripts._rebrem_queue_script, ("job", ["hopg"], 10, 1, False)),
        (scripts._rebrem_chunked_queue_script, ("job", ["hopg"], 10, 1, False, 10)),
        (scripts._recxr_queue_script, ("job", ["hopg"], 10, 1, False)),
        (scripts._recxr_chunked_queue_script, ("job", ["hopg"], 10, 1, False, 10)),
    ],
)
def test_supported_queue_payloads_use_pack(pack_target, builder, args):
    script = builder(*args)
    assert "PYRITE_PACK_PREFIX=" in script
    assert "run --as-is" not in script
    assert "install --locked" not in script
    if shutil.which("bash"):
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)


def test_host_pack_commands_keep_host_exec_and_never_run_pixi(pack_target, monkeypatch):
    monkeypatch.setattr(config, "host_exec", lambda: "srun --time=00:30:00")
    for command in (
        config.host_run_command() + " pyrite checkpoint slim stem -o -",
        scripts._prune_checkpoint_stems_command("job", ["hopg"], all_profiles=True),
        trajectories._host_command("inventory", {}),
        trajectories._host_command("export", {}),
    ):
        assert "srun --time=00:30:00 /usr/bin/env" in command
        assert "pixi-unpack" in command
        assert "run --as-is" not in command
        assert "install --locked" not in command
        if shutil.which("bash"):
            subprocess.run(["bash", "-n"], input=command, text=True, check=True)


def test_pack_host_wrapper_keeps_stdout_clean_and_propagates_failure(monkeypatch):
    if not shutil.which("bash"):
        pytest.skip("requires bash")
    monkeypatch.setattr(pack, "unpack_block", lambda: "echo provisioning; set -u")
    launcher = shlex.split(pack.host_pack_launcher())
    result = subprocess.run(
        [*launcher, "bash", "-c", "printf payload; exit 7"],
        text=True,
        capture_output=True,
    )
    assert result.stdout == "payload"
    assert "provisioning" in result.stderr
    assert result.returncode == 7
