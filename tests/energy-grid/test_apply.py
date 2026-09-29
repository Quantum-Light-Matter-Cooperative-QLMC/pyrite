import json as _json
import tomllib

import pytest

from pyrite.energy_grid import apply
from tests.helpers.energy_grid_catalog import BASE_TOML, COMBINED


def test_apply_rewrites_only_owned_blocks_and_reparses():
    new_text, skipped = apply.apply_bounds(BASE_TOML, COMBINED)
    assert skipped == []
    assert "stop = 2700.0, num = 897" in new_text
    assert 'display_name = "HOPG"' in new_text  # untouched line preserved
    tomllib.loads(new_text)  # still valid TOML


def test_apply_skips_manual_line_unless_forced():
    # Line-grid provenance now lives inline (decision 3): a row's own
    # source = "manual" is what apply_bounds consults, not a sidecar.
    manual_toml = BASE_TOML.replace(
        'num = 864, endpoint = true } }, source = "derived" }',
        'num = 864, endpoint = true } }, source = "manual" }',
    )

    new_text, skipped = apply.apply_bounds(manual_toml, COMBINED)
    assert "hopg:30" in skipped
    assert "stop = 2600.0, num = 864" in new_text  # original 30 keV row kept

    forced, skipped2 = apply.apply_bounds(manual_toml, COMBINED, force=True)
    assert skipped2 == []
    assert "stop = 2700.0, num = 897" in forced


BASE_TOML_NO_GRID = """schema_version = 1

[profiles.standard]
energy_keV = { values = [30.0, 100.0] }

[energy_grids.hopg]
line_by_energy = [
  { energy_keV = 30.0, grid = { linspace = { start = 10.0, stop = 2600.0, num = 864, endpoint = true } }, source = "derived" },
]

[materials.hfs2]
display_name = "HfS2"

[materials.hopg]
display_name = "HOPG"
"""

COMBINED_NO_GRID = {
    "hfs2": {
        "line_rows": [
            {"energy_keV": 30.0, "start_eV": 10.0, "stop_eV": 2800.0, "num": 930},
        ],
        "brem": {"stop_eV": 140000.0, "step_eV": 25.0, "raw_eV": 133000.0},
    }
}


def test_apply_inserts_new_line_block_and_leaves_brem_unstored():
    new_text, skipped = apply.apply_bounds(BASE_TOML_NO_GRID, COMBINED_NO_GRID)
    assert skipped == []
    assert "stop = 2800.0, num = 930" in new_text
    assert "[energy_grids.hfs2]" in new_text
    # The derived brem band is a diagnostic; no uniform override is written.
    assert "E_grid_brem" not in new_text
    assert 'display_name = "HfS2"' in new_text  # untouched line preserved
    assert 'display_name = "HOPG"' in new_text  # neighboring section untouched
    tomllib.loads(new_text)  # still valid TOML


def test_new_material_rows_are_not_seeded_from_shared_grid(tmp_path, monkeypatch):
    shared = BASE_TOML_NO_GRID.replace("energy_grids.hopg", "energy_grids.standard")
    # A shared manual row must neither block nor seed a different material.
    shared = shared.replace('source = "derived"', 'source = "manual"')
    new_text, skipped = apply.apply_bounds(shared, COMBINED_NO_GRID)
    assert skipped == []
    rows = tomllib.loads(new_text)["energy_grids"]["hfs2"]["line_by_energy"]
    assert len(rows) == 1
    assert rows[0]["grid"]["linspace"]["stop"] == 2800.0

    toml_path = tmp_path / "materials.toml"
    json_path = tmp_path / "combined.json"
    toml_path.write_text(shared)
    json_path.write_text(_json.dumps(COMBINED_NO_GRID))
    monkeypatch.setattr(apply, "load_material_catalog", lambda *args, **kwargs: None)
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *args, **kwargs: False)
    refs = apply.add_file(json_path, catalog_path=toml_path)
    from pyrite import _energy_grid_artifacts as artifacts

    stored = artifacts.load_artifact(tmp_path / "energy-grid-artifacts", refs["hfs2"])
    assert stored.identity["line_rows"][0]["stop_eV"] == 2800.0


def test_show_does_not_report_another_materials_rows(tmp_path):
    shared = BASE_TOML_NO_GRID.replace("energy_grids.hopg", "energy_grids.standard")
    path = tmp_path / "materials.toml"
    path.write_text(shared)
    output = apply.show("hfs2", "line", catalog_path=path)
    assert "line grid @" not in output
    assert "no stored per-energy rows" in output


def test_add_file_writes_deduplicated_artifact_and_only_repoints_profile(
    tmp_path, monkeypatch, capsys
):
    from pyrite import _energy_grid_artifacts as artifacts

    toml_path = tmp_path / "materials.toml"
    json_path = tmp_path / "combined.json"
    toml_path.write_text(BASE_TOML)
    json_path.write_text(_json.dumps(COMBINED))
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *args, **kwargs: False)
    monkeypatch.setattr(apply, "load_material_catalog", lambda path, **kwargs: None)

    refs = apply.add_file(json_path, catalog_path=toml_path)
    digest = refs["hopg"]
    updated = toml_path.read_text()
    parsed = tomllib.loads(updated)

    assert parsed["profiles"]["standard"]["energy_grid_refs"] == {"hopg": digest}
    assert parsed["profiles"]["standard"]["energy_keV"] == {"values": [30.0, 100.0]}
    assert parsed["profiles"]["standard"]["overrides"]["hopg"]["E_grid_brem"] == {
        "arange": {"start": 0.0, "stop": 136500.0, "step": 25.0}
    }
    assert parsed["energy_grids"] == tomllib.loads(BASE_TOML)["energy_grids"]
    stored = artifacts.load_artifact(tmp_path / "energy-grid-artifacts", digest)
    assert stored.identity["brem_grid"]["stop_eV"] == 140000.0
    assert [row["stop_eV"] for row in stored.identity["line_rows"]] == [2700.0, 4600.0]

    refs_again = apply.add_file(json_path, catalog_path=toml_path)
    assert refs_again == refs
    assert artifacts.inventory_artifacts(tmp_path / "energy-grid-artifacts") == (digest,)
    assert "added hopg ->" in capsys.readouterr().out


def test_add_file_keeps_named_profile_beam_energies(tmp_path, monkeypatch):
    from pyrite import _energy_grid_artifacts as artifacts

    toml_path = tmp_path / "materials.toml"
    json_path = tmp_path / "combined.json"
    named = BASE_TOML.replace(
        "[energy_grids.hopg]",
        "[profiles.hopg_hbn]\n"
        'materials = ["hopg"]\n'
        "energy_keV = { values = [30.0, 35.0, 40.0] }\n\n"
        "[energy_grids.hopg]",
    )
    toml_path.write_text(named)
    combined = _json.loads(_json.dumps(COMBINED))
    combined["hopg"]["line_rows"] = [
        {"energy_keV": 35.0, "start_eV": 10.0, "stop_eV": 2900.0, "num": 964}
    ]
    json_path.write_text(_json.dumps(combined))
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *args, **kwargs: False)
    monkeypatch.setattr(apply, "load_material_catalog", lambda path, **kwargs: None)

    digest = apply.add_file(json_path, catalog_path=toml_path, profile="hopg_hbn")["hopg"]
    stored = artifacts.load_artifact(tmp_path / "energy-grid-artifacts", digest)

    assert stored.identity["beam_energies_keV"] == [30.0, 35.0, 40.0]
    assert [row["energy_keV"] for row in stored.identity["line_rows"]] == [30.0, 35.0, 100.0]


def test_add_file_dry_run_writes_no_artifact_or_catalog(tmp_path, monkeypatch, capsys):
    toml_path = tmp_path / "materials.toml"
    json_path = tmp_path / "combined.json"
    toml_path.write_text(BASE_TOML)
    json_path.write_text(_json.dumps(COMBINED))
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *args, **kwargs: False)

    refs = apply.add_file(json_path, catalog_path=toml_path, dry_run=True)

    assert set(refs) == {"hopg"}
    assert toml_path.read_text() == BASE_TOML
    assert not (tmp_path / "energy-grid-artifacts").exists()
    assert "+energy_grid_refs" in capsys.readouterr().out


def test_add_file_preserves_manual_row_from_referenced_artifact(tmp_path, monkeypatch):
    from pyrite import _energy_grid_artifacts as artifacts

    toml_path = tmp_path / "materials.toml"
    json_path = tmp_path / "combined.json"
    toml_path.write_text(BASE_TOML)
    json_path.write_text(_json.dumps(COMBINED))
    monkeypatch.setattr(apply, "load_material_catalog", lambda path, **kwargs: None)
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        apply._provenance, "is_manual_line", lambda material, energy, **kwargs: False
    )
    first_ref = apply.add_file(json_path, catalog_path=toml_path)["hopg"]

    changed = _json.loads(_json.dumps(COMBINED))
    changed["hopg"]["line_rows"][0]["stop_eV"] = 2900.0
    json_path.write_text(_json.dumps(changed))
    monkeypatch.setattr(
        apply._provenance,
        "is_manual_line",
        lambda material, energy, **kwargs: material == "hopg" and energy == 30.0,
    )
    second_ref = apply.add_file(json_path, catalog_path=toml_path)["hopg"]

    assert second_ref == first_ref
    stored = artifacts.load_artifact(tmp_path / "energy-grid-artifacts", second_ref)
    rows = {row["energy_keV"]: row for row in stored.identity["line_rows"]}
    assert rows[30.0]["stop_eV"] == 2700.0


def test_remove_line_rows_repoints_artifact_without_deleting_legacy_rows(tmp_path, monkeypatch):
    from pyrite import _energy_grid_artifacts as artifacts

    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    monkeypatch.setattr(apply, "load_material_catalog", lambda path: None)

    removed, digest = apply.remove_line_rows(
        "hopg", [30], catalog_path=toml_path, profile="standard"
    )
    parsed = tomllib.loads(toml_path.read_text())
    stored = artifacts.load_artifact(tmp_path / "energy-grid-artifacts", digest)

    assert removed == [30.0]
    assert parsed["profiles"]["standard"]["energy_grid_refs"]["hopg"] == digest
    assert [row["energy_keV"] for row in parsed["energy_grids"]["hopg"]["line_by_energy"]] == [
        30.0,
        100.0,
    ]
    assert [row["energy_keV"] for row in stored.identity["line_rows"]] == [100.0]


def test_remove_line_rows_fails_closed_when_catalog_changed_after_preview(tmp_path):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    preview = toml_path.read_text()
    toml_path.write_text(BASE_TOML + "\n# concurrent edit\n")

    with pytest.raises(ValueError, match="changed after preview"):
        apply.remove_line_rows("hopg", [30], catalog_path=toml_path, expected_original=preview)

    assert not (tmp_path / "energy-grid-artifacts").exists()


def test_set_line_artifact_repoints_ref_and_preserves_legacy_payload(tmp_path, monkeypatch):
    from pyrite import _energy_grid_artifacts as artifacts

    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    original = tomllib.loads(BASE_TOML)
    stamped = []
    monkeypatch.setattr(apply, "load_material_catalog", lambda path: None)
    monkeypatch.setattr(
        apply._provenance,
        "set_line",
        lambda material, energy, source, note=None, **kwargs: stamped.append(
            (material, energy, source, note, kwargs.get("profile"))
        ),
    )

    digest = apply.set_line_artifact(
        "hopg", 30, 2800, profile="standard", note="reviewed", catalog_path=toml_path
    )

    updated = tomllib.loads(toml_path.read_text())
    stored = artifacts.load_artifact(tmp_path / "energy-grid-artifacts", digest)
    row = next(row for row in stored.identity["line_rows"] if row["energy_keV"] == 30.0)
    assert updated["profiles"]["standard"]["energy_grid_refs"] == {"hopg": digest}
    assert updated["energy_grids"] == original["energy_grids"]
    assert (
        updated["profiles"]["standard"]["energy_keV"]
        == original["profiles"]["standard"]["energy_keV"]
    )
    assert row == {"energy_keV": 30.0, "start_eV": 10.0, "stop_eV": 2800.0, "num": 864}
    assert stamped == [("hopg", 30.0, "manual", "reviewed", "standard")]


def test_set_brem_artifact_repoints_ref_and_preserves_profile_override(tmp_path, monkeypatch):
    from pyrite import _energy_grid_artifacts as artifacts

    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    original = tomllib.loads(BASE_TOML)
    stamped = []
    monkeypatch.setattr(apply, "load_material_catalog", lambda path: None)
    monkeypatch.setattr(
        apply._provenance,
        "set_brem",
        lambda material, source, note=None, **kwargs: stamped.append(
            (material, source, note, kwargs.get("profile"))
        ),
    )

    digest = apply.set_brem_artifact(
        "hopg", 150000, profile="standard", note="reviewed", catalog_path=toml_path
    )

    updated = tomllib.loads(toml_path.read_text())
    stored = artifacts.load_artifact(tmp_path / "energy-grid-artifacts", digest)
    assert updated["profiles"]["standard"]["energy_grid_refs"] == {"hopg": digest}
    assert updated["energy_grids"] == original["energy_grids"]
    assert (
        updated["profiles"]["standard"]["overrides"]
        == original["profiles"]["standard"]["overrides"]
    )
    assert stored.identity["brem_grid"] == {
        "start_eV": 0.0,
        "stop_eV": 150000.0,
        "step_eV": 25.0,
    }
    assert stamped == [("hopg", "manual", "reviewed", "standard")]


@pytest.mark.parametrize(
    ("setter", "call"),
    [
        ("set_line", lambda: apply.set_line_artifact("hopg", 30, 2800)),
        ("set_brem", lambda: apply.set_brem_artifact("hopg", 150000)),
    ],
)
def test_artifact_setters_restore_catalog_and_provenance_when_stamping_fails(
    tmp_path, monkeypatch, setter, call
):
    toml_path = tmp_path / "materials.toml"
    provenance_path = tmp_path / "line_grid_provenance.toml"
    toml_path.write_text(BASE_TOML)
    provenance_path.write_text("# existing provenance\n")
    monkeypatch.setattr(apply, "_CATALOG_PATH", toml_path)
    monkeypatch.setattr(apply._provenance, "PROVENANCE_PATH", provenance_path)
    monkeypatch.setattr(apply, "load_material_catalog", lambda path: None)
    monkeypatch.setattr(
        apply._provenance,
        setter,
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("provenance write failed")),
    )

    with pytest.raises(OSError, match="provenance write failed"):
        call()

    assert toml_path.read_text() == BASE_TOML
    assert provenance_path.read_text() == "# existing provenance\n"


def test_validated_line_row_requires_bandwidth_and_resolution_together():
    row = {"energy_keV": 30.0, "start_eV": 10.0, "bandwidth": {"stop_eV": 100.0}}
    with pytest.raises(ValueError, match="bandwidth and resolution together"):
        apply._validated_line_row(row, field="hopg.line_rows[0]")


def test_validated_line_row_flattens_separate_derived_criteria():
    row = {
        "energy_keV": 30.0,
        "start_eV": 10.0,
        "bandwidth": {"stop_eV": 100.0, "coverage": 0.95},
        "resolution": {"num": 91, "observable_class": "intrinsic_source"},
    }
    assert apply._validated_line_row(row, field="hopg.line_rows[0]") == {
        "energy_keV": 30.0,
        "start_eV": 10.0,
        "stop_eV": 100.0,
        "num": 91,
    }
