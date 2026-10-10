import subprocess
import sys
from pathlib import Path

from pyrite import DATA_DIR
from pyrite.console import config as _config
from pyrite.console.config import workspace_root
from pyrite.paths import (
    cache_dir,
    config_dir,
    data_dir,
    migrate_legacy_state,
    state_dir,
    user_data_dir,
)


def test_data_dir_preserves_public_package_constant():
    assert data_dir() == DATA_DIR
    assert data_dir().parent.name == "pyrite"


def test_workspace_root_precedence(monkeypatch, tmp_path):
    config_path = tmp_path / "state" / "config.toml"
    stored = tmp_path / "stored"
    environment = tmp_path / "environment"
    explicit = tmp_path / "explicit"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    _config.set_stored("workspace.root", str(stored))
    monkeypatch.setenv("PYRITE_HOME", str(environment))

    assert workspace_root() == environment.resolve()
    assert workspace_root(explicit) == explicit.resolve()


def test_workspace_root_uses_store_then_cwd(monkeypatch, tmp_path):
    config_path = tmp_path / "state" / "config.toml"
    stored = tmp_path / "stored"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.delenv("PYRITE_HOME", raising=False)
    _config.set_stored("workspace.root", str(stored))

    assert workspace_root() == stored.resolve()

    config_path.unlink()
    monkeypatch.chdir(cwd)
    assert workspace_root() == cwd.resolve()


def test_config_dir_matches_config_store_parent():
    # conftest redirects CONFIG_PATH in-process; read the shipped default fresh.
    probe = "from pyrite.console.config import CONFIG_PATH; print(CONFIG_PATH.parent)"
    parent = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert Path(parent) == config_dir()
    assert config_dir().name == "pyrite"


def test_legacy_state_moves_out_of_config_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("pyrite.paths.config_dir", lambda: tmp_path / "config")
    monkeypatch.setattr("pyrite.paths.state_dir", lambda: tmp_path / "state")
    legacy = tmp_path / "config" / "viewer-default"
    legacy.parent.mkdir()
    legacy.write_text("hopg\n")
    target = tmp_path / "state" / "viewer-default"

    assert migrate_legacy_state(target) == target
    assert target.read_text() == "hopg\n"
    assert not legacy.exists()
    # Elsewhere, or already present: untouched.
    other = tmp_path / "elsewhere" / "viewer-default"
    assert migrate_legacy_state(other) == other


def test_platform_cache_and_data_use_pyrite(monkeypatch, tmp_path):
    monkeypatch.setattr("pyrite.paths.user_cache_path", lambda *args, **kwargs: tmp_path / "cache")
    monkeypatch.setattr("pyrite.paths.user_data_path", lambda *args, **kwargs: tmp_path / "data")
    monkeypatch.setattr("pyrite.paths.user_state_path", lambda *args, **kwargs: tmp_path / "state")
    assert cache_dir() == tmp_path / "cache"
    assert user_data_dir() == tmp_path / "data"
    assert state_dir() == tmp_path / "state"


def test_checkpoint_defaults_follow_workspace_root():
    from pyrite.checkpoints import archive, checkpoint_cleanup
    from pyrite.cli import _completion
    from pyrite.runs import run

    expected = _config.output_dir("checkpoints")
    assert Path(run.DEFAULT_CHECKPOINT_DIR) == expected
    assert Path(archive.DEFAULT_ROOT) == expected
    assert checkpoint_cleanup.output_dir("checkpoints") == expected
    assert _completion._default_checkpoint_root() == expected
