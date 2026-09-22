"""Code-tree resolution, its precedence, and its failure message.

D9 forbids a silent fallback to a surrogate model, so "no tree" must be a
loud, actionable error rather than a degraded result. The precedence exists so
a user testing a patched upstream is never silently served the vendored copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyrite.xsgen._errors import SourceUnavailableError
from pyrite.xsgen.sources import (
    code_names,
    code_spec,
    iter_sources,
    missing_data_dirs,
    resolve_source,
    source_digest,
    vendored_root,
)


@pytest.fixture
def elsepa_tree(tmp_path: Path) -> Path:
    """Build a directory that satisfies ELSEPA's marker and digest sources."""
    root = tmp_path / "elsepa-2020"
    (root / "database").mkdir(parents=True)
    for name in code_spec("elsepa").digest_sources:
        (root / name).write_text(f"      PROGRAM {name}\n", encoding="utf-8")
    return root


def test_every_code_declares_what_resolution_needs():
    assert set(code_names()) == {"elsepa", "sbethe", "bremslib"}
    for name in code_names():
        spec = code_spec(name)
        assert spec.marker and spec.config_key.startswith("xsgen.")
        assert spec.upstream


def test_bremslib_declares_no_buildable_program():
    """BremsLib is read, not run (D7): PyRITE never compiles its GPL-3 source."""
    assert code_spec("bremslib").programs == {}
    assert code_spec("bremslib").digest_sources == ()


def test_an_unknown_code_names_the_known_ones():
    with pytest.raises(KeyError, match="elsepa"):
        code_spec("penelope")


def test_an_explicit_path_wins(elsepa_tree, monkeypatch):
    monkeypatch.setenv("PYRITE_XSGEN_ELSEPA_SOURCE", str(elsepa_tree.parent / "ignored"))
    resolved = resolve_source("elsepa", elsepa_tree)
    assert resolved.root == elsepa_tree.resolve()
    assert resolved.origin == "explicit path"


def test_an_explicit_path_that_does_not_hold_the_code_raises(tmp_path):
    """Falling through to another tier would silently ignore what was asked for."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SourceUnavailableError) as excinfo:
        resolve_source("elsepa", empty)
    assert str(empty) in str(excinfo.value)


def test_a_configured_path_beats_the_vendored_tree(elsepa_tree, monkeypatch, tmp_path):
    vendored = tmp_path / "vendored" / "xsgen" / "elsepa"
    (vendored / "database").mkdir(parents=True)
    (vendored / "elscata.f").write_text("vendored\n", encoding="utf-8")
    monkeypatch.setattr("pyrite.xsgen.sources.data_dir", lambda: tmp_path / "vendored")
    monkeypatch.setenv("PYRITE_XSGEN_ELSEPA_SOURCE", str(elsepa_tree))

    resolved = resolve_source("elsepa")
    assert resolved.root == elsepa_tree.resolve()
    assert resolved.origin == "PYRITE_XSGEN_ELSEPA_SOURCE"


def test_the_vendored_tree_is_used_when_nothing_is_configured(monkeypatch, tmp_path):
    """Zero-config offline generation, which the remote cluster workflow needs."""
    vendored = tmp_path / "vendored" / "xsgen" / "elsepa"
    (vendored / "database").mkdir(parents=True)
    (vendored / "elscata.f").write_text("vendored\n", encoding="utf-8")
    monkeypatch.setattr("pyrite.xsgen.sources.data_dir", lambda: tmp_path / "vendored")
    monkeypatch.delenv("PYRITE_XSGEN_ELSEPA_SOURCE", raising=False)
    monkeypatch.setattr("pyrite.console.config._read_store", dict)

    assert resolve_source("elsepa").origin == "vendored"


def test_the_sibling_checkout_is_the_last_resort(elsepa_tree, monkeypatch, tmp_path):
    """The documented layout: the code tree beside the PyRITE checkout.

    The built-in default is ``../elsepa-2020`` relative to the working
    directory, so this exercises the real path an unconfigured developer hits
    before anything is vendored.
    """
    checkout = tmp_path / "pyrite"
    checkout.mkdir()
    monkeypatch.setattr("pyrite.xsgen.sources.data_dir", lambda: tmp_path / "absent")
    monkeypatch.delenv("PYRITE_XSGEN_ELSEPA_SOURCE", raising=False)
    monkeypatch.setattr("pyrite.console.config._read_store", dict)
    monkeypatch.chdir(checkout)

    resolved = resolve_source("elsepa")
    assert resolved.origin == "sibling checkout"
    assert resolved.root == elsepa_tree.resolve()


def test_an_unresolvable_tree_names_every_tier_the_fix_and_the_upstream(monkeypatch, tmp_path):
    monkeypatch.setattr("pyrite.xsgen.sources.data_dir", lambda: tmp_path / "absent")
    monkeypatch.delenv("PYRITE_XSGEN_SBETHE_SOURCE", raising=False)
    monkeypatch.setattr("pyrite.console.config._read_store", dict)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SourceUnavailableError) as excinfo:
        resolve_source("sbethe")
    message = str(excinfo.value)
    assert "sbethe.f" in message
    assert "vendored" in message and "sibling checkout" in message
    assert "pyrite tables sources set sbethe" in message
    assert "10.17632/7zw25f428t.2" in message


def test_the_digest_changes_when_a_source_is_patched(elsepa_tree):
    before = source_digest(resolve_source("elsepa", elsepa_tree))
    (elsepa_tree / "radial.f").write_text("      PROGRAM patched\n", encoding="utf-8")
    assert source_digest(resolve_source("elsepa", elsepa_tree)) != before


def test_the_digest_ignores_files_the_programs_do_not_compile(elsepa_tree):
    """A note in the tree is not a different code."""
    before = source_digest(resolve_source("elsepa", elsepa_tree))
    (elsepa_tree / "notes.txt").write_text("scratch\n", encoding="utf-8")
    (elsepa_tree / "database" / "z_001.dat").write_text("h\n", encoding="utf-8")
    assert source_digest(resolve_source("elsepa", elsepa_tree)) == before


def test_a_missing_declared_source_is_reported_not_hashed_as_absent(elsepa_tree):
    (elsepa_tree / "radial.f").unlink()
    with pytest.raises(SourceUnavailableError, match="radial.f"):
        source_digest(resolve_source("elsepa", elsepa_tree))


def test_missing_data_directories_are_reported_separately(elsepa_tree):
    """A missing tree is configured; a missing database is fetched."""
    assert missing_data_dirs(resolve_source("elsepa", elsepa_tree)) == ()
    (elsepa_tree / "database").rmdir()
    assert missing_data_dirs(resolve_source("elsepa", elsepa_tree)) == ("database",)


def test_fetched_data_directory_fills_a_tree_that_only_ships_source(
    monkeypatch,
    tmp_path,
):
    source_root = tmp_path / "packaged" / "xsgen" / "sbethe"
    source_root.mkdir(parents=True)
    (source_root / "sbethe.f").write_text("      PROGRAM sbethe\n")
    fetched = tmp_path / "user" / "xsgen" / "reference-data" / "sbethe" / "sdbase"
    fetched.mkdir(parents=True)
    monkeypatch.setattr("pyrite.xsgen.sources.data_dir", lambda: tmp_path / "packaged")
    monkeypatch.setattr("pyrite.xsgen.sources.user_data_dir", lambda: tmp_path / "user")
    monkeypatch.delenv("PYRITE_XSGEN_SBETHE_SOURCE", raising=False)
    monkeypatch.setattr("pyrite.console.config._read_store", dict)

    resolved = resolve_source("sbethe")

    assert resolved.origin == "vendored"
    assert resolved.data_dirs == {"sdbase": fetched}
    assert missing_data_dirs(resolved) == ()


def test_vendored_root_is_reportable_before_anything_is_vendored():
    assert vendored_root("elsepa").name == "elsepa"
    assert vendored_root("elsepa").parent.name == "xsgen"


def test_iter_sources_reports_one_failure_without_hiding_the_others(elsepa_tree, monkeypatch):
    monkeypatch.setattr("pyrite.xsgen.sources.data_dir", lambda: elsepa_tree / "absent")
    monkeypatch.setenv("PYRITE_XSGEN_ELSEPA_SOURCE", str(elsepa_tree))
    monkeypatch.setenv("PYRITE_XSGEN_SBETHE_SOURCE", str(elsepa_tree / "nope"))

    report = dict(iter_sources(["elsepa", "sbethe"]))
    assert not isinstance(report["elsepa"], str)
    assert isinstance(report["sbethe"], str)
