import subprocess
import sys
from pathlib import Path

from pyrite import DATA_DIR
from pyrite.console import config as _config
from pyrite.console.config import workspace_root
from pyrite.paths import cache_dir, data_dir, state_dir, user_data_dir


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


def test_state_dir_matches_config_store_parent():
    # conftest redirects CONFIG_PATH in-process; read the shipped default fresh.
    probe = "from pyrite.console.config import CONFIG_PATH; print(CONFIG_PATH.parent)"
    parent = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert Path(parent) == state_dir()
    assert state_dir().name == "pyrite"


def test_platform_cache_and_data_use_pyrite(monkeypatch, tmp_path):
    monkeypatch.setattr("pyrite.paths.user_cache_path", lambda *args, **kwargs: tmp_path / "cache")
    monkeypatch.setattr("pyrite.paths.user_data_path", lambda *args, **kwargs: tmp_path / "data")
    assert cache_dir() == tmp_path / "cache"
    assert user_data_dir() == tmp_path / "data"


def test_checkpoint_defaults_follow_workspace_root():
    from pyrite.checkpoints import archive, checkpoint_cleanup
    from pyrite.cli import _completion
    from pyrite.runs import run

    expected = workspace_root() / "checkpoints"
    assert Path(run.DEFAULT_CHECKPOINT_DIR) == expected
    assert Path(archive.DEFAULT_ROOT) == expected
    assert Path(checkpoint_cleanup._DEFAULT_CHECKPOINT_DIR) == expected
    assert _completion._ARCHIVE_CHECKPOINT_ROOT == expected
