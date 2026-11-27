"""Offline pack identity, target policy, staging and shell rendering."""

import hashlib
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from pyrite.energy_grid import convergence_job
from pyrite.remote import _environment_pack as pack
from pyrite.remote import config, queue, scripts, targets

LOCK = b"""environments:
  default:
    packages:
      linux-64:
      - pypi: ./
      - pypi: https://example.org/example-1-py3-none-any.whl
  nvidia:
    packages:
      linux-64-cuda:
      - pypi: ./
packages:
- pypi: ./
  name: pyrite-mc
- pypi: https://example.org/example-1-py3-none-any.whl
  name: example
"""


def test_pack_selection_and_keys():
    digest, platform = pack.lock_selection(LOCK, "default")
    assert digest == hashlib.sha256(LOCK).hexdigest()
    assert platform == "linux-64"
    assert pack.lock_selection(LOCK, "nvidia")[1] == "linux-64-cuda"
    assert pack.lock_selection(LOCK + b"\n", "default")[0] != digest
    identity = pack.PackIdentity(digest, "default", platform, "a" * 64)
    changed = replace(identity, code_digest="b" * 64)
    assert identity.key == changed.key
    assert identity.wheel_key != changed.wheel_key
    assert identity.key != replace(identity, lock_hash="c" * 64).key


def test_nonwheel_dependency_is_not_silently_dropped():
    with pytest.raises(SystemExit, match="non-wheel dependencies: example"):
        pack.lock_selection(LOCK.replace(b"-py3-none-any.whl", b".tar.gz"), "default")


def test_target_mode():
    assert config.RemoteTarget(host="box").env_transfer == "install"
    assert targets.stored_target({"host": "box", "env_transfer": "pack"}).env_transfer == "pack"
    assert "env_transfer" in targets._FIELDS
    with pytest.raises(ValueError, match="env_transfer"):
        config.RemoteTarget(host="box", env_transfer="bad")


def test_pack_rejects_py_spy_before_submission():
    target = config.RemoteTarget(host="box", env_transfer="pack")
    with config.use_target(target):
        with pytest.raises(SystemExit, match="requires env_transfer=install"):
            queue._ProfilingModes(py_spy=True)._validate_py_spy(chunked=False)
        with pytest.raises(ValueError, match="requires env_transfer=install"):
            scripts._queue_script("job", ["hopg"], True, 1, performance_profile="p", py_spy=True)


def test_job_blocks(monkeypatch):
    target = config.RemoteTarget(host="box", remote_dir="/shared/pyrite", pixi="pixi")
    identity = pack.PackIdentity("a" * 64, "nvidia", "linux-64-cuda", "b" * 64)
    monkeypatch.setattr(pack, "pack_identity", lambda: identity)
    with config.use_target(target):
        install = scripts._pixi_install_block(once=True)
        assert "install --locked -e nvidia" in install
        assert config.shell_job_run() == config.shell_remote_run()
    with config.use_target(replace(target, env_transfer="pack")):
        unpack = scripts._pixi_install_block(once=True)
        assert "pixi-unpack --shell bash" in unpack
        assert "install --locked" not in unpack
        assert "linux-64-cuda" in unpack
        assert identity.lock_hash in unpack
        assert "synced code mismatch" in unpack
        assert "$HOME/.pixi/bin" in unpack
        assert config.shell_job_run() == "env"
        assert config.shell_job_python() == '"$PYRITE_PACK_PREFIX/env/bin/python"'
        assert config.shell_remote_run() != "env"
    if shutil.which("bash"):
        subprocess.run(["bash", "-n"], input=unpack, text=True, check=True)


def test_remote_cache_hit_does_not_build_or_upload(monkeypatch, tmp_path):
    target = config.RemoteTarget(host="login", transfer_host="dtn", remote_dir="/shared/pyrite")
    calls = []

    def capture(command, *, host=None):
        calls.append((command, host))
        return "present\n"

    def unexpected(*args, **kwargs):
        pytest.fail("cache hit must not build or transfer")

    monkeypatch.setattr(pack.transport, "_ssh_capture", capture)
    monkeypatch.setattr(pack.transport, "_run", unexpected)
    with config.use_target(target):
        pack._stage_artifact(tmp_path / "pack.tar", "/shared/pack.tar", unexpected)
    assert calls == [('if test -s "/shared/pack.tar"; then echo present; fi', "dtn")]


@pytest.mark.parametrize(
    "command", ["start", "start-windows", "start-precision", "start-bandwidth"]
)
def test_convergence_jobs_run_in_the_unpacked_environment(command, monkeypatch, capsys):
    target = config.RemoteTarget(
        host="box", remote_dir="/shared/pyrite", pixi="pixi", env_transfer="pack"
    )
    identity = pack.PackIdentity("a" * 64, "nvidia", "linux-64-cuda", "b" * 64)
    monkeypatch.setattr(pack, "pack_identity", lambda: identity)
    with config.use_target(target):
        convergence_job.main([command, "--dry-run", "--json-out", "report.json"])
    script = capsys.readouterr().out
    assert "PYRITE_PACK_PREFIX=" in script
    assert "env python -m pyrite.energy_grid." in script
    assert "run --as-is" not in script
    assert "install --locked" not in script


def test_missing_pixi_pack_is_a_clear_error(monkeypatch, tmp_path):
    identity = pack.PackIdentity("a" * 64, "default", "linux-64", "b" * 64)
    monkeypatch.setattr(pack.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="pixi global install pixi-pack"):
        pack._build_dependencies(identity, tmp_path / "dependencies.tar")


def test_missing_build_package_is_a_clear_error(monkeypatch, tmp_path):
    def missing(name):
        raise pack.PackageNotFoundError(name)

    identity = pack.PackIdentity("a" * 64, "default", "linux-64", "b" * 64)
    monkeypatch.setattr(pack, "distribution", missing)
    with pytest.raises(SystemExit, match="Pixi default env"):
        pack._build_wheel(identity, tmp_path / "project-wheel.tar")


def test_local_cache_reuses_completed_archive(tmp_path):
    archive = tmp_path / "cache" / "dependencies.tar"
    calls = []

    def build(output: Path):
        calls.append(output)
        output.write_bytes(b"pack")

    pack._cache_file(archive, build)
    pack._cache_file(archive, build)
    assert archive.read_bytes() == b"pack"
    assert len(calls) == 1
