"""The deprecated pre-workspace table tier: warning, migration, removal target."""

import json
import warnings

import numpy as np
import pytest

from pyrite.cli.commands import tables as tables_command
from pyrite.console import config
from pyrite.xsgen import store
from pyrite.xsgen.store import ElementTarget, TableRequest
from tests.helpers.cli import assert_clean_result, invoke


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """An explicit workspace, so the legacy directory is a separate tier."""
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr("pyrite.paths.user_data_path", lambda *a, **k: tmp_path / "data")
    monkeypatch.setattr("pyrite.xsgen.store.data_dir", lambda: tmp_path / "packaged")
    monkeypatch.setenv("PYRITE_HOME", str(tmp_path / "ws"))
    monkeypatch.setattr(store, "_WARNED_LEGACY", set())
    return tmp_path


def _legacy_table(z=29):
    request = TableRequest("elsepa", "a" * 64, ElementTarget(z=z), "elastic_dcs", {"muffin": 0})
    stored = store.store(request, {"x": np.arange(3.0)}, root=store.legacy_table_dir())
    return request.key, stored


def test_legacy_tier_is_only_distinct_with_a_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr("pyrite.paths.user_data_path", lambda *a, **k: tmp_path / "data")
    monkeypatch.delenv("PYRITE_HOME", raising=False)
    assert store.legacy_table_dir() == store.user_table_dir()
    report = store.migrate_legacy_tables()
    assert report.active is False and report.copied == ()


def test_resolving_from_the_legacy_tier_warns_once_naming_the_fix(workspace):
    key, _ = _legacy_table()

    with pytest.warns(FutureWarning, match=r"pyrite tables migrate.*") as caught:
        found = store.resolve(key)
    assert found is not None and found.tier == "legacy"
    assert store.LEGACY_TABLE_TIER_REMOVE_IN in str(caught[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert store.resolve(key) is not None  # once per process


def test_migrate_copies_missing_pairs_and_never_touches_the_source(workspace):
    key, legacy = _legacy_table(29)
    present_key, _ = _legacy_table(30)
    selected = store.user_table_dir()
    selected.mkdir(parents=True)
    for suffix in (".npz", ".json"):
        (selected / f"{present_key}{suffix}").write_bytes(b"workspace copy")
    before = {path.name: path.read_bytes() for path in store.legacy_table_dir().iterdir()}

    dry = store.migrate_legacy_tables(dry_run=True)
    assert dry.copied == (key,) and not (selected / f"{key}.json").exists()

    report = store.migrate_legacy_tables()

    assert report.copied == (key,) and report.present == (present_key,)
    assert (selected / f"{key}.npz").read_bytes() == legacy.path.read_bytes()
    assert (selected / f"{present_key}.json").read_bytes() == b"workspace copy"
    after = {path.name: path.read_bytes() for path in store.legacy_table_dir().iterdir()}
    assert after == before
    assert store.resolve(key).tier == "user"
    assert not [p for p in selected.iterdir() if p.name.startswith(".")]


def test_migrate_command_reports_text_and_json(workspace):
    key, _ = _legacy_table()

    dry = invoke(tables_command.command, ["migrate", "--dry-run"])
    assert_clean_result(
        dry,
        stdout=(
            f"would copy 1 table(s) from {store.legacy_table_dir()} to "
            f"{store.user_table_dir()}; 0 already present\n"
        ),
    )

    result = invoke(tables_command.command, ["migrate", "-o", "json"])
    envelope = json.loads(result.stdout)
    assert result.exit_code == 0 and result.stderr == ""
    assert envelope["schema"] == "pyrite.tables.migrate.v1"
    assert envelope["payload"]["copied"] == [key]
    assert envelope["payload"]["dry_run"] is False


def test_migrate_without_a_workspace_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATH", tmp_path / "config.toml")
    monkeypatch.setattr("pyrite.paths.user_data_path", lambda *a, **k: tmp_path / "data")
    monkeypatch.delenv("PYRITE_HOME", raising=False)

    result = invoke(tables_command.command, ["migrate"])

    assert result.exit_code == 0
    assert result.stdout.startswith("nothing to migrate:")
