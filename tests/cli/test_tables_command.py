"""Contracts for the ``pyrite tables`` command group.

Stream separation, exit status, and the JSON envelope are the automation
contract: ``pyrite tables path`` is meant to be usable as
``cd "$(pyrite tables path)"``, and ``-o json`` must emit exactly one envelope
with no prose mixed in.
"""

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
    # Default sources are ``../<tree>`` relative to the cwd; a real sibling
    # checkout beside the developer's repo must not leak in.
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
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


# --- generate and fetch --------------------------------------------------


def test_fetch_reports_the_installed_sbethe_database(isolated, monkeypatch, tmp_path):
    from pyrite.xsgen.fetch import FetchResult

    destination = tmp_path / "data" / "xsgen" / "reference-data" / "sbethe" / "sdbase"
    monkeypatch.setattr(
        "pyrite.xsgen.fetch.fetch_sbethe",
        lambda archive=None: FetchResult("sbethe", destination, "a" * 64, 599, True),
    )

    result = invoke(tables_command.command, ["fetch", "sbethe"])

    assert_clean_result(result, stdout=f"installed: {destination} (599 files)\n")


def test_fetch_reports_the_installed_bremslib_tables(isolated, monkeypatch, tmp_path):
    from pyrite.xsgen.fetch import FetchResult

    destination = tmp_path / "data" / "xsgen" / "tables"
    archive = tmp_path / "bremslib-tables.zip"
    archive.write_bytes(b"zip")
    seen = []

    def fetch(given=None):
        seen.append(given)
        return FetchResult("bremslib", destination, "c" * 64, 24, True)

    monkeypatch.setattr("pyrite.xsgen.fetch.fetch_bremslib", fetch)

    result = invoke(tables_command.command, ["fetch", "bremslib", "--archive", str(archive)])

    assert_clean_result(result, stdout=f"installed: {destination} (24 tables)\n")
    assert seen == [str(archive)]


def test_fetch_reports_a_missing_release_as_a_cli_error(isolated, monkeypatch):
    from pyrite.xsgen import DataFetchError

    def fetch(given=None):
        raise DataFetchError("this PyRITE build pins no BremsLib table release")

    monkeypatch.setattr("pyrite.xsgen.fetch.fetch_bremslib", fetch)

    result = invoke(tables_command.command, ["fetch", "bremslib"])

    assert result.exit_code == 1
    assert "pins no BremsLib table release" in result.stderr
    assert result.stdout == ""


def test_generate_reports_a_new_elsepa_table(isolated, monkeypatch, tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "data" / "xsgen" / "tables" / f"{'a' * 64}.npz"
    table = SimpleNamespace(
        key="a" * 64,
        path=path,
        tier="user",
        manifest={"manifest_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        "pyrite.xsgen.elsepa.generate_element",
        lambda *a, **k: SimpleNamespace(table=table, generated=True),
    )

    result = invoke(
        tables_command.command,
        ["generate", "--code", "elsepa", "--element", "79", "--energy", "1000"],
    )

    assert_clean_result(result, stdout=f"generated: {'a' * 64}\npath: {path}\n")


def _fake_elsepa_material_tables(monkeypatch, tmp_path):
    from types import SimpleNamespace

    results = tuple(
        SimpleNamespace(
            table=SimpleNamespace(
                key=char * 64,
                path=tmp_path / f"{char * 64}.npz",
                tier="user",
                manifest={"manifest_sha256": "f" * 64},
            ),
            generated=generated,
        )
        for char, generated in (("a", True), ("b", False))
    )
    calls = []

    def fake(material, **kwargs):
        calls.append((material, kwargs))
        return results

    monkeypatch.setattr("pyrite.xsgen.elsepa.catalog.generate_catalog", fake)
    return results, calls


def test_generate_elsepa_material_reports_every_table(isolated, monkeypatch, tmp_path):
    results, calls = _fake_elsepa_material_tables(monkeypatch, tmp_path)

    result = invoke(
        tables_command.command, ["generate", "--code", "elsepa", "--material", "silicon"]
    )

    assert_clean_result(
        result,
        stdout=(
            f"generated: {'a' * 64}\npath: {results[0].table.path}\n"
            f"reused: {'b' * 64}\npath: {results[1].table.path}\n"
        ),
    )
    assert calls == [("silicon", {"overwrite": False, "keep_on_failure": False})]


def test_generate_elsepa_material_json_is_one_envelope(isolated, monkeypatch, tmp_path):
    _fake_elsepa_material_tables(monkeypatch, tmp_path)

    result = invoke(
        tables_command.command,
        ["generate", "--code", "elsepa", "--material", "silicon", "-o", "json"],
    )

    assert result.exit_code == 0, result.stderr
    envelope = json.loads(result.stdout)
    assert envelope["schema"] == "pyrite.tables.generate-material.v1"
    assert envelope["payload"]["material"] == "silicon"
    assert [row["generated"] for row in envelope["payload"]["tables"]] == [True, False]


def test_generate_elsepa_material_rejects_element_and_energy(isolated):
    result = invoke(
        tables_command.command,
        ["generate", "--code", "elsepa", "--material", "silicon", "--element", "14"],
    )

    assert result.exit_code != 0
    assert "--code elsepa does not accept --element" in result.stderr


def test_generate_requires_an_explicit_energy(isolated):
    """ELSEPA and SBETHE need disjoint options, so the check moved into the body."""
    result = invoke(
        tables_command.command,
        ["generate", "--code", "elsepa", "--element", "79"],
    )

    assert result.exit_code != 0
    assert "--code elsepa requires --energy" in result.stderr


def test_generate_rejects_options_belonging_to_the_other_code(isolated):
    """Ignoring them would silently generate a table for a different target."""
    result = invoke(
        tables_command.command,
        ["generate", "--code", "elsepa", "--element", "79", "--energy", "1000", "--density", "1.0"],
    )

    assert result.exit_code != 0
    assert "--code elsepa does not accept --density" in result.stderr


def test_generate_reports_a_new_sbethe_table(isolated, monkeypatch, tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "table.npz"
    table = SimpleNamespace(
        key="c" * 64,
        path=path,
        tier="user",
        manifest={"manifest_sha256": "d" * 64},
    )
    seen = {}

    def fake_generate(name, composition, **kwargs):
        seen["name"] = name
        seen["composition"] = composition
        seen.update(kwargs)
        return SimpleNamespace(table=table, generated=True)

    monkeypatch.setattr("pyrite.xsgen.sbethe.generate_material", fake_generate)

    result = invoke(
        tables_command.command,
        [
            "generate",
            "--code",
            "sbethe",
            "--name",
            "water",
            "--element-count",
            "1:2",
            "--element-count",
            "8:1",
            "--density",
            "1.0",
            "--mean-excitation",
            "75.0",
        ],
    )

    assert_clean_result(result, stdout=f"generated: {'c' * 64}\npath: {path}\n")
    assert seen["name"] == "water"
    assert seen["composition"] == {1: 2.0, 8: 1.0}
    assert seen["density_g_cm3"] == 1.0
    assert seen["mean_excitation_eV"] == 75.0
    assert seen["band_gap_eV"] is None


def test_generate_resolves_a_catalog_sbethe_material(isolated, monkeypatch, tmp_path):
    from types import SimpleNamespace

    table = SimpleNamespace(
        key="c" * 64,
        path=tmp_path / "table.npz",
        tier="user",
        manifest={"manifest_sha256": "d" * 64},
    )
    seen = {}

    def fake_generate(name, composition, **kwargs):
        seen.update(name=name, composition=composition, **kwargs)
        return SimpleNamespace(table=table, generated=True)

    monkeypatch.setattr("pyrite.xsgen.sbethe.generate_material", fake_generate)

    result = invoke(
        tables_command.command,
        ["generate", "--code", "sbethe", "--material", "silicon"],
    )

    assert_clean_result(result, stdout=f"generated: {'c' * 64}\npath: {table.path}\n")
    assert seen["name"] == "silicon"
    assert set(seen["composition"]) == {14}
    assert seen["composition"][14] > 0.0
    assert seen["density_g_cm3"] == pytest.approx(2.33, rel=0.02)
    assert seen["mean_excitation_eV"] == pytest.approx(173.0)


def test_generate_rejects_catalog_and_manual_sbethe_inputs_together(isolated):
    result = invoke(
        tables_command.command,
        ["generate", "--code", "sbethe", "--material", "silicon", "--density", "2.33"],
    )

    assert result.exit_code != 0
    assert "--code sbethe does not accept --density" in result.stderr


def test_generate_rejects_a_malformed_element_count(isolated):
    result = invoke(
        tables_command.command,
        [
            "generate",
            "--code",
            "sbethe",
            "--name",
            "water",
            "--element-count",
            "hydrogen",
            "--density",
            "1.0",
            "--mean-excitation",
            "75.0",
        ],
    )

    assert result.exit_code != 0
    assert "--element-count expects Z:N" in result.stderr


def test_generate_reports_a_new_bremslib_table(isolated, monkeypatch, tmp_path):
    """BremsLib is read, not run, so the only required option is the element."""
    from types import SimpleNamespace

    seen: dict[str, object] = {}
    path = tmp_path / "data" / "xsgen" / "tables" / f"{'c' * 64}.npz"
    table = SimpleNamespace(
        key="c" * 64,
        path=path,
        tier="user",
        manifest={"manifest_sha256": "d" * 64},
    )

    def fake(z, **kwargs):
        seen["z"] = z
        seen.update(kwargs)
        return SimpleNamespace(table=table, generated=True)

    monkeypatch.setattr("pyrite.xsgen.bremslib.generate_element", fake)

    result = invoke(
        tables_command.command,
        ["generate", "--code", "bremslib", "--element", "79", "--t1-max", "2.5"],
    )

    assert_clean_result(result, stdout=f"generated: {'c' * 64}\npath: {path}\n")
    assert seen["z"] == 79
    assert seen["t1_max_MeV"] == 2.5


def test_generate_defaults_bremslib_to_the_complete_library_range(isolated, monkeypatch):
    from types import SimpleNamespace

    from pyrite.xsgen.bremslib.read import COMPLETE_T1_MAX_MEV

    seen: dict[str, object] = {}

    def fake(z, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            table=SimpleNamespace(key="c" * 64, path="t.npz", tier="user", manifest={}),
            generated=False,
        )

    monkeypatch.setattr("pyrite.xsgen.bremslib.generate_element", fake)
    invoke(tables_command.command, ["generate", "--code", "bremslib", "--element", "79"])

    assert seen["t1_max_MeV"] == COMPLETE_T1_MAX_MEV


def test_generate_requires_an_element_for_bremslib(isolated):
    result = invoke(tables_command.command, ["generate", "--code", "bremslib"])

    assert result.exit_code != 0
    assert "--code bremslib requires --element" in result.stderr


def test_generate_rejects_an_energy_grid_for_bremslib(isolated):
    """The library's energies are its own grid, so --energy chooses nothing."""
    result = invoke(
        tables_command.command,
        ["generate", "--code", "bremslib", "--element", "79", "--energy", "1000"],
    )

    assert result.exit_code != 0
    assert "--code bremslib does not accept --energy" in result.stderr


def test_generate_rejects_an_energy_bound_for_a_code_that_runs(isolated):
    result = invoke(
        tables_command.command,
        ["generate", "--code", "elsepa", "--element", "79", "--energy", "1000", "--t1-max", "2"],
    )

    assert result.exit_code != 0
    assert "--code elsepa does not accept --t1-max" in result.stderr


def test_generate_rejects_keep_on_failure_for_bremslib(isolated):
    """Nothing is run, so there is no scratch directory the flag could keep."""
    result = invoke(
        tables_command.command,
        ["generate", "--code", "bremslib", "--element", "79", "--keep-on-failure"],
    )

    assert result.exit_code != 0
    assert "--code bremslib does not accept --keep-on-failure" in result.stderr


def test_generate_json_is_one_machine_readable_envelope(isolated, monkeypatch, tmp_path):
    from types import SimpleNamespace

    path = tmp_path / "table.npz"
    table = SimpleNamespace(
        key="a" * 64,
        path=path,
        tier="user",
        manifest={"manifest_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        "pyrite.xsgen.elsepa.generate_element",
        lambda *a, **k: SimpleNamespace(table=table, generated=False),
    )

    result = invoke(
        tables_command.command,
        [
            "generate",
            "--code",
            "elsepa",
            "--element",
            "79",
            "--energy",
            "1000",
            "-o",
            "json",
        ],
    )

    envelope = json.loads(result.stdout)
    assert envelope["schema"] == "pyrite.tables.generate.v1"
    assert envelope["payload"]["generated"] is False
    assert envelope["payload"]["manifest_sha256"] == "b" * 64
    assert result.stderr == ""


def test_fetch_json_is_one_machine_readable_envelope(isolated, monkeypatch, tmp_path):
    from pyrite.xsgen.fetch import FetchResult

    destination = tmp_path / "sdbase"
    monkeypatch.setattr(
        "pyrite.xsgen.fetch.fetch_sbethe",
        lambda archive=None: FetchResult("sbethe", destination, "b" * 64, 599, False),
    )

    result = invoke(tables_command.command, ["fetch", "sbethe", "-o", "json"])

    envelope = json.loads(result.stdout)
    assert envelope["schema"] == "pyrite.tables.fetch.v1"
    assert envelope["payload"]["installed"] is False
    assert envelope["payload"]["path"] == str(destination)
    assert result.stderr == ""


def test_fetch_rejects_a_code_without_downloadable_data(isolated):
    result = invoke(tables_command.command, ["fetch", "penelope"])

    assert result.exit_code == 2
    assert "sbethe" in result.stderr
    assert result.stdout == ""


# --- sources --------------------------------------------------------------


def test_sources_list_distinguishes_missing_tree_from_missing_database(
    isolated,
    elsepa_tree,
    tmp_path,
):
    invoke(tables_command.command, ["sources", "set", "elsepa", str(elsepa_tree)])
    result = invoke(tables_command.command, ["sources", "list"])

    assert result.exit_code == 0, "a missing tree is a reported status, not a failure"
    rows = {line.split("\t")[0]: line for line in result.stdout.splitlines()[1:] if "\t" in line}
    assert set(rows) >= {"elsepa", "sbethe", "bremslib"}
    assert "ready" in rows["elsepa"]
    assert "incomplete" in rows["sbethe"]
    assert "missing data directories: sdbase" in result.stdout
    assert "missing" in rows["bremslib"]


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


def test_the_cli_projectile_list_matches_the_deck():
    """The CLI spells the list out to keep group construction free of xsgen imports.

    That duplication is only safe if something pins the two together.
    """
    from pyrite.xsgen.sbethe.deck import PROJECTILES

    assert tables_command._PROJECTILES == tuple(sorted(PROJECTILES))


# --- verify ---------------------------------------------------------------


def _pin(monkeypatch, request: TableRequest, *, digest: str | None = None):
    from pyrite.xsgen import verify

    stored = store(request, {"theta_deg": np.linspace(0.0, 180.0, 4)}, overwrite=True)
    row = ("elsepa", "Z=29", request.key, digest or stored.digest)
    monkeypatch.setattr(verify, "pinned_tables", lambda codes=verify.CODES: [row])
    return stored


def test_verify_passes_when_the_pinned_manifest_digest_matches(isolated, monkeypatch):
    _pin(monkeypatch, _store_one())

    result = invoke(tables_command.command, ["verify", "--require", "bremslib,elsepa"])

    assert_clean_result(result, stdout="1/1 pinned tables and datasets verified\n")


def test_verify_exits_one_naming_the_fix_when_a_pinned_table_is_missing(isolated, monkeypatch):
    from pyrite.xsgen import verify

    monkeypatch.setattr(
        verify,
        "pinned_tables",
        lambda codes=verify.CODES: [("bremslib", "Z=79", "f" * 64, "0" * 64)],
    )

    result = invoke(tables_command.command, ["verify", "--require", "bremslib,elsepa"])

    assert result.exit_code == 1
    assert result.stdout == "0/1 pinned tables and datasets verified\n"
    assert "missing: bremslib Z=79" in result.stderr
    assert "pyrite tables fetch bremslib --archive PATH" in result.stderr
    assert "Traceback" not in result.output


def test_verify_flags_a_manifest_that_is_not_the_pinned_one(isolated, monkeypatch):
    _pin(monkeypatch, _store_one(), digest="1" * 64)

    result = invoke(tables_command.command, ["verify", "--require", "bremslib,elsepa"])

    assert result.exit_code == 1
    assert "mismatch: elsepa Z=29" in result.stderr


def test_verify_flags_an_edited_manifest_as_corrupt(isolated, monkeypatch):
    stored = _pin(monkeypatch, _store_one())
    manifest = stored.path.with_suffix(".json")
    body = json.loads(manifest.read_text(encoding="utf-8"))
    body["target_label"] = "tampered"
    manifest.write_text(json.dumps(body), encoding="utf-8")

    result = invoke(tables_command.command, ["verify", "--require", "bremslib,elsepa"])

    assert result.exit_code == 1
    assert "corrupt: elsepa Z=29" in result.stderr


def test_verify_json_is_one_envelope_and_still_exits_one(isolated, monkeypatch):
    from pyrite.xsgen import verify

    monkeypatch.setattr(
        verify, "pinned_tables", lambda codes=verify.CODES: [("elsepa", "Z=6", "e" * 64, "0" * 64)]
    )

    result = invoke(
        tables_command.command, ["verify", "--require", "bremslib,elsepa", "-o", "json"]
    )

    assert result.exit_code == 1
    envelope = json.loads(result.stdout)
    assert envelope["schema"] == "pyrite.tables.verify.v1"
    assert envelope["ok"] is False
    assert envelope["payload"]["failed"][0]["status"] == "missing"
    assert result.stderr == ""


def test_verify_rejects_an_unknown_code_as_a_usage_error(isolated):
    result = invoke(tables_command.command, ["verify", "--require", "sbethe"])

    assert result.exit_code == 2


# --- fetched datasets (eedl, eadl) -------------------------------------------


@pytest.fixture
def small_dataset(monkeypatch):
    """Swap the 25 MB EEDL pin for a few bytes with the same install shape."""
    import hashlib
    from dataclasses import replace

    from pyrite import datasets

    body = b"eedl record\r\n"
    fake = replace(datasets.EEDL, sha256=hashlib.sha256(body).hexdigest(), download_sha256=None)
    monkeypatch.setitem(datasets.DATASETS, "eedl", fake)
    return body


def test_fetch_installs_a_dataset_from_a_local_file(isolated, small_dataset, tmp_path):
    source = tmp_path / "EEDL2025.ALL"
    source.write_bytes(small_dataset)

    result = invoke(tables_command.command, ["fetch", "eedl", "--archive", str(source)])

    installed = isolated / "data" / "datasets" / "eedl" / "EEDL.endf"
    assert_clean_result(result, stdout=f"installed: {installed}\n")
    assert installed.read_bytes() == small_dataset
    again = invoke(
        tables_command.command, ["fetch", "eedl", "--archive", str(source), "-o", "json"]
    )
    envelope = json.loads(again.stdout)
    assert envelope["payload"]["installed"] is False
    assert envelope["payload"]["file_count"] == 1


def test_fetch_refuses_a_dataset_that_does_not_match_its_pin(isolated, small_dataset, tmp_path):
    source = tmp_path / "EEDL2025.ALL"
    source.write_bytes(b"something else")

    result = invoke(tables_command.command, ["fetch", "eedl", "--archive", str(source)])

    assert result.exit_code == 1
    assert "SHA-256 mismatch" in result.stderr and "nothing was installed" in result.stderr
    assert not (isolated / "data" / "datasets" / "eedl" / "EEDL.endf").exists()


def test_verify_reports_a_missing_dataset_with_its_fetch_command(isolated, small_dataset):
    result = invoke(tables_command.command, ["verify", "--require", "eedl"])

    assert result.exit_code == 1
    assert result.stdout == "0/1 pinned tables and datasets verified\n"
    assert "missing: eedl EEDL.endf" in result.stderr
    assert "`pyrite tables fetch eedl`" in result.stderr


def test_verify_passes_an_installed_dataset(isolated, small_dataset, tmp_path):
    source = tmp_path / "EEDL2025.ALL"
    source.write_bytes(small_dataset)
    invoke(tables_command.command, ["fetch", "eedl", "--archive", str(source)])

    result = invoke(tables_command.command, ["verify", "--require", "eedl"])

    assert_clean_result(result, stdout="1/1 pinned tables and datasets verified\n")
