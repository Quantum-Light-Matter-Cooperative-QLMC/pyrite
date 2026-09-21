"""Contracts for the ``pyrite tables`` command group.

Stream separation, exit status, and the JSON envelope are the automation
contract: ``pyrite tables path`` is meant to be usable as
``cd "$(pyrite tables path)"``, and ``-o json`` must emit exactly one envelope
with no prose mixed in.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyrite.cli.commands import tables as tables_command
from pyrite.console import config as _config
from pyrite.xsgen.store import ElementTarget, TableRequest, store, user_table_dir
from tests.helpers.cli import assert_clean_result, invoke


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Redirect the config store, the user tables, and the packaged tier."""
    monkeypatch.setattr(_config, "CONFIG_PATH", tmp_path / "pyrite" / "config.toml")
    monkeypatch.setattr("pyrite.paths.user_data_path", lambda *a, **k: tmp_path / "data")
    monkeypatch.setattr("pyrite.paths.user_cache_path", lambda *a, **k: tmp_path / "cache")
    monkeypatch.setattr("pyrite.xsgen.store.data_dir", lambda: tmp_path / "packaged")
    for code in ("ELSEPA", "SBETHE", "BREMSLIB"):
        monkeypatch.delenv(f"PYRITE_XSGEN_{code}_SOURCE", raising=False)
    return tmp_path


@pytest.fixture
def elsepa_tree(tmp_path):
    root = tmp_path / "elsepa-2020"
    (root / "database").mkdir(parents=True)
    for name in ("elscata.f", "elsepa2020.f", "radial.f"):
        (root / name).write_text("      PROGRAM x\n", encoding="utf-8")
    return root


def _store_one(quantity: str = "elastic_dcs", z: int = 29) -> TableRequest:
    request = TableRequest(
        code="elsepa",
        code_version="a" * 64,
        target=ElementTarget(z=z),
        quantity=quantity,
        model={"muffin": 1},
    )
    store(request, {"theta_deg": np.linspace(0.0, 180.0, 4)})
    return request


# --- path -----------------------------------------------------------------


def test_path_prints_one_bare_line_for_shell_substitution(isolated):
    result = invoke(tables_command.command, ["path"])
    assert_clean_result(result, stdout=f"{user_table_dir()}\n")


def test_path_json_reports_both_tiers(isolated):
    result = invoke(tables_command.command, ["path", "-o", "json"])
    assert result.exit_code == 0
    envelope = json.loads(result.stdout)
    assert envelope["ok"] is True
    assert envelope["payload"]["user"] == str(user_table_dir())
    assert envelope["payload"]["packaged"].endswith("xsgen/tables")


# --- list -----------------------------------------------------------------


def test_list_of_an_empty_store_is_a_header_and_nothing_else(isolated):
    result = invoke(tables_command.command, ["list"])
    assert_clean_result(result, stdout="KEY\tCODE\tQUANTITY\tTARGET\tTIER\n")


def test_list_reports_a_stored_table(isolated):
    request = _store_one()
    result = invoke(tables_command.command, ["list"])

    assert result.exit_code == 0
    rows = result.stdout.splitlines()
    assert rows[0].split("\t") == ["KEY", "CODE", "QUANTITY", "TARGET", "TIER"]
    assert rows[1].split("\t") == [request.key, "elsepa", "elastic_dcs", "Z=29", "user"]


def test_list_json_carries_one_envelope(isolated):
    _store_one()
    result = invoke(tables_command.command, ["list", "-o", "json"])

    envelope = json.loads(result.stdout)
    assert envelope["schema"] == "pyrite.tables.list.v1"
    assert envelope["ok"] is True
    assert len(envelope["payload"]["tables"]) == 1


# --- show -----------------------------------------------------------------


def test_show_prints_the_provenance_manifest(isolated):
    request = _store_one()
    result = invoke(tables_command.command, ["show", request.key])

    assert result.exit_code == 0
    manifest = json.loads(result.stdout)
    assert manifest["key"] == request.key
    assert manifest["deck_hash"] == request.deck_hash
    assert manifest["modifications"], "CC BY requires the adaptation note"


def test_show_accepts_an_unambiguous_key_prefix(isolated):
    request = _store_one()
    result = invoke(tables_command.command, ["show", request.key[:8]])
    assert json.loads(result.stdout)["key"] == request.key


def test_show_refuses_an_ambiguous_prefix_rather_than_guessing(isolated):
    """The empty prefix matches everything; picking one would be a wrong answer."""
    _store_one(quantity="elastic_dcs")
    _store_one(quantity="transport_xs")

    result = invoke(tables_command.command, ["show", ""])
    assert result.exit_code != 0
    assert "ambiguous" in result.stderr
    assert result.stdout == ""


def test_show_of_an_unknown_key_fails_with_a_pointer(isolated):
    result = invoke(tables_command.command, ["show", "deadbeef"])
    assert result.exit_code != 0
    assert "pyrite tables list" in result.stderr
    assert result.stdout == ""


# --- sources --------------------------------------------------------------


def test_sources_list_reports_every_code_even_when_one_is_missing(isolated, elsepa_tree, tmp_path):
    invoke(tables_command.command, ["sources", "set", "elsepa", str(elsepa_tree)])
    result = invoke(tables_command.command, ["sources", "list"])

    assert result.exit_code == 0, "a missing tree is a reported status, not a failure"
    rows = {line.split("\t")[0]: line for line in result.stdout.splitlines()[1:] if "\t" in line}
    assert set(rows) >= {"elsepa", "sbethe", "bremslib"}
    assert "ready" in rows["elsepa"]
    assert "missing" in rows["sbethe"]


def test_sources_set_stores_a_resolved_path(isolated, elsepa_tree, monkeypatch):
    monkeypatch.chdir(elsepa_tree.parent)
    result = invoke(tables_command.command, ["sources", "set", "elsepa", "elsepa-2020"])

    assert result.exit_code == 0
    # Resolved, not as typed: the stored value outlives this working directory.
    assert result.stdout == f"xsgen.elsepa_source = {elsepa_tree.resolve()}\n"
    assert _config.resolve("xsgen.elsepa_source").value == str(elsepa_tree.resolve())


def test_sources_set_rejects_a_directory_that_does_not_hold_the_code(isolated, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = invoke(tables_command.command, ["sources", "set", "elsepa", str(empty)])

    assert result.exit_code == 2, "a bad argument value is a usage error"
    assert "does not look like a elsepa tree" in result.stderr
    assert "elscata.f" in result.stderr


def test_sources_set_rejects_an_unknown_code_by_naming_the_known_ones(isolated, tmp_path):
    result = invoke(tables_command.command, ["sources", "set", "penelope", str(tmp_path)])

    assert result.exit_code == 2
    assert "unknown code" in result.stderr
    assert "elsepa" in result.stderr and "sbethe" in result.stderr


def test_config_set_accepts_an_xsgen_source_key(isolated, elsepa_tree):
    """The config store and `tables sources set` are two doors to one value."""
    from pyrite.cli.commands import config as config_command

    result = invoke(config_command.command, ["set", "xsgen.elsepa_source", str(elsepa_tree)])
    assert result.exit_code == 0
    assert _config.resolve("xsgen.elsepa_source").value == str(elsepa_tree.resolve())


def test_config_set_rejects_a_bad_xsgen_source_without_mentioning_profiles(isolated, tmp_path):
    """Regression: the validator used to fall through to the profile check.

    Every key it did not name explicitly was validated against the catalog's
    profile names, so a path value was rejected as an "unknown profile".
    """
    from pyrite.cli.commands import config as config_command

    result = invoke(config_command.command, ["set", "xsgen.elsepa_source", str(tmp_path)])

    assert result.exit_code == 2
    assert "does not look like a elsepa tree" in result.stderr
    # "profile.current" still appears in the usage line's key list, which is
    # correct; what must not appear is the profile *validator's* verdict.
    assert "unknown profile" not in result.stderr
