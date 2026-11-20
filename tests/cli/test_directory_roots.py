"""Coverage for config roots, workspace resolution, and cache/data separation."""

from pathlib import Path

from click.testing import CliRunner

from pyrite import _backend, paths
from pyrite.cli.commands.config import command as config_command
from pyrite.console import config, outputs
from pyrite.xsgen import store


def test_platform_roots_accept_independent_environment_overrides(monkeypatch, tmp_path):
    for name, func in (
        ("PYRITE_CONFIG_DIR", paths.config_dir),
        ("PYRITE_DATA_DIR", paths.user_data_dir),
        ("PYRITE_CACHE_DIR", paths.cache_dir),
        ("PYRITE_STATE_DIR", paths.state_dir),
    ):
        path = tmp_path / name.lower()
        monkeypatch.setenv(name, str(path))
        assert func() == path


def test_workspace_does_not_redirect_shared_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setenv("PYRITE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PYRITE_HOME", str(tmp_path / "first"))
    assert config.fetched_data_root() == tmp_path / "data"
    assert store.user_table_dir() == tmp_path / "data" / "xsgen" / "tables"
    monkeypatch.setenv("PYRITE_HOME", str(tmp_path / "second"))
    assert store.user_table_dir() == tmp_path / "data" / "xsgen" / "tables"


def test_backend_store_environment_and_explicit_precedence(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.delenv("PYRITE_MC_BACKEND", raising=False)
    config.set_stored("backend.default", "cpu")
    assert _backend.select_backend().name == "cpu"
    monkeypatch.setenv("PYRITE_MC_BACKEND", "auto")
    assert config.resolve("backend.default").value == "auto"
    assert _backend.select_backend("cpu").name == "cpu"


def test_home_output_tree_is_not_discovered(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.delenv("PYRITE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    (tmp_path / "pyrite-output").mkdir()
    project = tmp_path / "project" / "src"
    project.mkdir(parents=True)
    monkeypatch.chdir(project)
    assert outputs.output_workspace() == project
    (tmp_path / "project" / "pyrite-output").mkdir()
    assert outputs.output_workspace() == tmp_path / "project"


def test_config_paths_command_reports_effective_locations(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setenv("PYRITE_CACHE_DIR", str(tmp_path / "scratch"))
    monkeypatch.setenv("PYRITE_DATA_DIR", str(tmp_path / "datasets"))
    output = CliRunner().invoke(config_command, ["paths", "-o", "json"])
    assert output.exit_code == 0, output.output
    result = json.loads(output.output)["payload"]["paths"]
    assert result["cache.line-grids"]["path"] == str(tmp_path / "scratch" / "line-grids")
    assert result["xsgen_tables"]["path"] == str(tmp_path / "datasets" / "xsgen" / "tables")


def test_global_line_grid_cache_is_workspace_independent(monkeypatch, tmp_path):
    from pyrite._cxr_grid_policy import _cache_path

    monkeypatch.setenv("PYRITE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PYRITE_HOME", str(tmp_path / "a"))
    first = _cache_path("abcd")
    monkeypatch.setenv("PYRITE_HOME", str(tmp_path / "b"))
    assert _cache_path("abcd") == first


def test_user_catalog_defaults_below_config_root(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr(config, "config_dir", lambda: tmp_path)
    monkeypatch.delenv("PYRITE_USER_CATALOG", raising=False)
    assert config.user_catalog_path() == tmp_path / "catalog"
