"""Coverage for config roots, workspace resolution, and cache/data separation."""

from pathlib import Path

import pytest
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
    monkeypatch.setattr(outputs, "_cwd", lambda: project)
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


def test_discovery_notice_is_cli_only(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.delenv("PYRITE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(outputs, "_discovery_notified", set())
    project = tmp_path / "project"
    (project / "pyrite-output").mkdir(parents=True)
    (project / "sub").mkdir()
    monkeypatch.setattr(outputs, "_cwd", lambda: project / "sub")

    monkeypatch.setattr(outputs, "_migration_enabled", False)
    assert outputs.output_workspace() == project
    assert "discovered" not in capsys.readouterr().err

    monkeypatch.setattr(outputs, "_migration_enabled", True)
    assert outputs.output_workspace() == project
    assert f"discovered at {project}" in capsys.readouterr().err


def test_cli_migrates_legacy_user_catalog_without_config_store(monkeypatch, tmp_path, capsys):
    legacy = tmp_path / "config"
    (legacy / "profiles").mkdir(parents=True)
    (legacy / "profiles" / "x.toml").write_text("")
    (legacy / "energy-grid-artifacts").mkdir()
    (legacy / "config.toml").write_text("")
    monkeypatch.setenv("PYRITE_CONFIG_DIR", str(legacy))
    monkeypatch.setenv("PYRITE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("PYRITE_USER_CATALOG", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATH", legacy / "config.toml")
    monkeypatch.setattr(config, "_catalog_migrated", set())

    monkeypatch.setattr(config, "_catalog_migration_enabled", False)
    with pytest.warns(RuntimeWarning, match="legacy user catalog"):
        assert config.user_catalog_path() == legacy
    assert (legacy / "profiles").is_dir()

    monkeypatch.setattr(config, "_catalog_migration_enabled", True)
    assert config.user_catalog_path() == legacy / "catalog"
    assert (legacy / "catalog" / "profiles" / "x.toml").is_file()
    assert (legacy / "config.toml").is_file()
    assert (tmp_path / "data" / "catalog" / "energy-grid-artifacts").is_dir()
    assert "moved legacy PyRITE catalog entries" in capsys.readouterr().err


def test_explicit_user_catalog_is_never_migrated(monkeypatch, tmp_path):
    legacy = tmp_path / "config"
    (legacy / "profiles").mkdir(parents=True)
    monkeypatch.setenv("PYRITE_CONFIG_DIR", str(legacy))
    monkeypatch.setenv("PYRITE_USER_CATALOG", str(tmp_path / "mine"))
    monkeypatch.setattr(config, "CONFIG_PATH", legacy / "config.toml")
    monkeypatch.setattr(config, "_catalog_migration_enabled", True)
    monkeypatch.setattr(config, "_catalog_migrated", set())
    assert config.user_catalog_path() == tmp_path / "mine"
    assert (legacy / "profiles").is_dir()


def test_macos_shared_root_entries_do_not_collide(monkeypatch, tmp_path):
    """macOS puts config, data and state in one directory; entries stay disjoint."""
    shared = tmp_path / "Application Support" / "pyrite"
    for name in ("PYRITE_CONFIG_DIR", "PYRITE_DATA_DIR", "PYRITE_STATE_DIR"):
        monkeypatch.setenv(name, str(shared))
    monkeypatch.setenv("PYRITE_CACHE_DIR", str(tmp_path / "Caches" / "pyrite"))
    monkeypatch.delenv("PYRITE_USER_CATALOG", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATH", paths.config_dir() / "config.toml")
    entries = [
        config.CONFIG_PATH,
        config.user_catalog_path(),
        config.xsgen_data_root(),
        paths.user_data_dir() / "datasets",
    ]
    tops = [entry.relative_to(shared).parts[0] for entry in entries]
    assert len(set(tops)) == len(tops), tops
    assert config.user_artifact_dir().parent == config.user_catalog_path()


def test_remote_runtime_exports_data_root_alongside_home():
    from pyrite.remote import config as remote_config
    from pyrite.remote import scripts

    root = remote_config.shell_word(remote_config.remote_dir())
    env = remote_config.remote_runtime_env()
    assert f"PYRITE_DATA_DIR={root}" in env and f"PYRITE_HOME={root}" in env
    script = scripts._slurm_batch_script("job1", "echo ok", job_name="t")
    assert f"export PYRITE_DATA_DIR={root}" in script
