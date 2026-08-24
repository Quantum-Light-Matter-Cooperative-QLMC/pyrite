import json as _json
import tomllib

import pytest

from pyrite.energy_grid import apply
from tests.helpers.energy_grid_catalog import BASE_TOML, COMBINED


class _NoManual:
    def is_manual_brem(self, *a):
        return False


def test_apply_rewrites_only_owned_blocks_and_reparses():
    new_text, skipped = apply.apply_bounds(BASE_TOML, COMBINED, provenance_mod=_NoManual())
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

    new_text, skipped = apply.apply_bounds(manual_toml, COMBINED, provenance_mod=_NoManual())
    assert "hopg:30" in skipped
    assert "stop = 2600.0, num = 864" in new_text  # original 30 keV row kept

    forced, skipped2 = apply.apply_bounds(
        manual_toml, COMBINED, force=True, provenance_mod=_NoManual()
    )
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


def test_apply_inserts_new_line_and_brem_blocks_when_absent():
    new_text, skipped = apply.apply_bounds(
        BASE_TOML_NO_GRID, COMBINED_NO_GRID, provenance_mod=_NoManual()
    )
    assert skipped == []
    assert "stop = 2800.0, num = 930" in new_text
    assert "[energy_grids.hfs2]" in new_text
    assert "stop = 140000.0, step = 25.0" in new_text
    assert 'display_name = "HfS2"' in new_text  # untouched line preserved
    assert 'display_name = "HOPG"' in new_text  # neighboring section untouched
    tomllib.loads(new_text)  # still valid TOML


def test_apply_file_writes_and_validates(tmp_path, monkeypatch, capsys):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    json_path = tmp_path / "combined.json"
    json_path.write_text(_json.dumps(COMBINED))
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    stamped = []
    monkeypatch.setattr(
        apply._provenance, "set_brem", lambda m, s, note=None: stamped.append((m, s))
    )
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *a: False)
    # The BASE_TOML fixture is a deliberately minimal single-material stub, so
    # stub the full-catalog re-parse (production validates the real catalog).
    monkeypatch.setattr(apply, "load_material_catalog", lambda p: None)
    apply.apply_file(json_path, slurm_id="458", date="2026-07-22")
    text = toml_path.read_text()
    assert "stop = 2700.0, num = 897" in text
    # line rows stamp source inline; apply_file no longer touches a
    # provenance sidecar for them (decision 3).
    assert 'source = "derived"' in text
    assert ("hopg", "derived job 458 (2026-07-22)") in stamped
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "warning: material catalog changed; golden is now stale; run `pyrite-dev regen-golden`\n"
    )


def test_add_file_writes_deduplicated_artifact_and_only_repoints_profile(
    tmp_path, monkeypatch, capsys
):
    from pyrite.energy_grid import artifacts

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
    from pyrite.energy_grid import artifacts

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
    from pyrite.energy_grid import artifacts

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
    from pyrite.energy_grid import artifacts

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
    from pyrite.energy_grid import artifacts

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
    from pyrite.energy_grid import artifacts

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
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
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


def test_apply_validation_failure_leaves_catalog_and_provenance_unchanged(tmp_path, monkeypatch):
    toml_path = tmp_path / "materials.toml"
    provenance_path = tmp_path / "line_grid_provenance.toml"
    json_path = tmp_path / "combined.json"
    toml_path.write_text(BASE_TOML)
    provenance_path.write_text("# existing provenance\n")
    json_path.write_text(_json.dumps(COMBINED))
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    monkeypatch.setattr(apply._provenance, "PROVENANCE_PATH", provenance_path)
    monkeypatch.setattr(
        apply,
        "load_material_catalog",
        lambda path: (_ for _ in ()).throw(ValueError("invalid candidate")),
    )

    with pytest.raises(ValueError, match="invalid candidate"):
        apply.apply_file(json_path, slurm_id="458", date="2026-07-22")

    assert toml_path.read_text() == BASE_TOML
    assert provenance_path.read_text() == "# existing provenance\n"


def test_apply_provenance_failure_rolls_back_both_files(tmp_path, monkeypatch):
    toml_path = tmp_path / "materials.toml"
    provenance_path = tmp_path / "line_grid_provenance.toml"
    json_path = tmp_path / "combined.json"
    toml_path.write_text(BASE_TOML)
    provenance_path.write_text("# existing provenance\n")
    json_path.write_text(_json.dumps(COMBINED))
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    monkeypatch.setattr(apply._provenance, "PROVENANCE_PATH", provenance_path)
    monkeypatch.setattr(apply, "load_material_catalog", lambda path: None)
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        apply._provenance,
        "set_brem",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("provenance write failed")),
    )

    with pytest.raises(OSError, match="provenance write failed"):
        apply.apply_file(json_path)

    assert toml_path.read_text() == BASE_TOML
    assert provenance_path.read_text() == "# existing provenance\n"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("energy_keV", 0.0, "energy_keV"),
        ("start_eV", float("nan"), "start_eV"),
        ("stop_eV", -1.0, "stop_eV"),
        ("num", 0, "num"),
    ],
)
def test_apply_rejects_invalid_line_domains_before_write(
    tmp_path, monkeypatch, field, value, match
):
    toml_path = tmp_path / "materials.toml"
    json_path = tmp_path / "combined.json"
    combined = _json.loads(_json.dumps(COMBINED))
    combined["hopg"]["line_rows"][0][field] = value
    toml_path.write_text(BASE_TOML)
    json_path.write_text(_json.dumps(combined))
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)

    with pytest.raises(ValueError, match=match):
        apply.apply_file(json_path)

    assert toml_path.read_text() == BASE_TOML


def test_set_line_grid_stamps_manual_inline_and_autocomputes_num(tmp_path, monkeypatch, capsys):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    calls = []
    monkeypatch.setattr(
        apply._provenance,
        "set_line",
        lambda m, e, s, note=None: calls.append((m, float(e), s, note)),
    )
    monkeypatch.setattr(apply, "load_material_catalog", lambda path: None)
    apply.set_line_grid("hopg", 30.0, 3000.0, note="widen tail")
    text = toml_path.read_text()
    assert "stop = 3000.0" in text
    assert 'source = "manual"' in text  # inline catalog source, decision 3
    assert calls == [("hopg", 30.0, "manual", "widen tail")]  # note stays sidecar-only
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "golden is now stale" in captured.err


def test_set_line_grid_rejects_invalid_domain_without_writes(tmp_path, monkeypatch):
    toml_path = tmp_path / "materials.toml"
    provenance_path = tmp_path / "line_grid_provenance.toml"
    toml_path.write_text(BASE_TOML)
    provenance_path.write_text("# existing provenance\n")
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    monkeypatch.setattr(apply._provenance, "PROVENANCE_PATH", provenance_path)

    with pytest.raises(ValueError, match="stop must be greater than start"):
        apply.set_line_grid("hopg", 30.0, 5.0)

    assert toml_path.read_text() == BASE_TOML
    assert provenance_path.read_text() == "# existing provenance\n"


def test_set_brem_rejects_invalid_step_without_writes(tmp_path, monkeypatch):
    toml_path = tmp_path / "materials.toml"
    provenance_path = tmp_path / "line_grid_provenance.toml"
    toml_path.write_text(BASE_TOML)
    provenance_path.write_text("# existing provenance\n")
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    monkeypatch.setattr(apply._provenance, "PROVENANCE_PATH", provenance_path)

    with pytest.raises(ValueError, match="step"):
        apply.set_brem_grid("hopg", 140000.0, step_eV=0)

    assert toml_path.read_text() == BASE_TOML
    assert provenance_path.read_text() == "# existing provenance\n"


def test_delete_line_grid_removes_material_entry_and_falls_back_to_shared_default(
    tmp_path, monkeypatch
):
    toml_path = tmp_path / "materials.toml"
    text = (
        BASE_TOML
        + "\n[energy_grids.standard]\n"
        + "line_by_energy = [\n"
        + '  { energy_keV = 30.0, grid = { values = [1.0, 2.0] }, source = "derived" },\n'
        + '  { energy_keV = 100.0, grid = { values = [3.0, 4.0] }, source = "derived" },\n'
        + "]\n"
    )
    toml_path.write_text(text)
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    monkeypatch.setattr(apply, "load_material_catalog", lambda path: None)

    deleted = apply.delete_line_grid("hopg", [30.0, 100.0])

    assert deleted == [30.0, 100.0]
    new_text = toml_path.read_text()
    assert "[energy_grids.hopg]" not in new_text
    assert "[energy_grids.standard]" in new_text  # untouched


def test_delete_line_grid_dry_run_prints_diff_and_writes_nothing(tmp_path, monkeypatch, capsys):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    monkeypatch.setattr(apply, "load_material_catalog", lambda path: None)

    deleted = apply.delete_line_grid("hopg", [100.0], dry_run=True)

    assert deleted == [100.0]
    assert toml_path.read_text() == BASE_TOML
    assert "-  { energy_keV = 100.0" in capsys.readouterr().out


def test_delete_line_grid_fails_closed_when_catalog_changed_after_preview(tmp_path, monkeypatch):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    monkeypatch.setattr(apply, "load_material_catalog", lambda path: None)
    toml_path.write_text(BASE_TOML + "\n# concurrent edit\n")

    with pytest.raises(ValueError, match="catalog changed after preview"):
        apply.delete_line_grid("hopg", [100.0], expected_original=BASE_TOML)

    assert toml_path.read_text().endswith("# concurrent edit\n")


def test_delete_line_grid_unknown_energy_errors_without_writes(tmp_path, monkeypatch):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    monkeypatch.setattr(apply, "load_material_catalog", lambda path: None)

    with pytest.raises(ValueError, match=r"no line-grid row at \[50\.0\] keV"):
        apply.delete_line_grid("hopg", [50.0])

    assert toml_path.read_text() == BASE_TOML


def test_delete_line_grid_unknown_material_errors_without_writes(tmp_path, monkeypatch):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)

    with pytest.raises(ValueError, match="no energy_grids entry for material"):
        apply.delete_line_grid("ghost", [30.0])

    assert toml_path.read_text() == BASE_TOML
