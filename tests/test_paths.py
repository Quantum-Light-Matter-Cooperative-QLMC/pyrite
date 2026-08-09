from pathlib import Path

from cxr_mc import DATA_DIR
from cxr_mc.cli import _config
from cxr_mc.paths import data_dir, state_dir, workspace_root


def test_data_dir_preserves_public_package_constant():
    assert data_dir() == DATA_DIR
    assert data_dir().parent.name == "cxr_mc"


def test_workspace_root_precedence(monkeypatch, tmp_path):
    config_path = tmp_path / "state" / "config.toml"
    stored = tmp_path / "stored"
    environment = tmp_path / "environment"
    explicit = tmp_path / "explicit"
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    _config.set_stored("workspace.root", str(stored))
    monkeypatch.setenv("CXR_HOME", str(environment))

    assert workspace_root() == environment.resolve()
    assert workspace_root(explicit) == explicit.resolve()


def test_workspace_root_uses_store_then_cwd(monkeypatch, tmp_path):
    config_path = tmp_path / "state" / "config.toml"
    stored = tmp_path / "stored"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setattr(_config, "CONFIG_PATH", config_path)
    monkeypatch.delenv("CXR_HOME", raising=False)
    _config.set_stored("workspace.root", str(stored))

    assert workspace_root() == stored.resolve()

    config_path.unlink()
    monkeypatch.chdir(cwd)
    assert workspace_root() == cwd.resolve()


def test_state_dir_matches_config_store_parent():
    assert state_dir() == _config.CONFIG_PATH.parent


def test_checkpoint_defaults_follow_workspace_root():
    from cxr_mc.checkpoints import archive, checkpoint_cleanup
    from cxr_mc.cli import _completion
    from cxr_mc.runs import run

    expected = workspace_root() / "checkpoints"
    assert Path(run.DEFAULT_CHECKPOINT_DIR) == expected
    assert Path(archive.DEFAULT_ROOT) == expected
    assert Path(checkpoint_cleanup._DEFAULT_CHECKPOINT_DIR) == expected
    assert _completion._ARCHIVE_CHECKPOINT_ROOT == expected
