import json as _json
import tomllib

import pytest

from cxr_mc.line_grid import apply

BASE_TOML = """schema_version = 1

[profiles.standard]
energy_keV = { values = [30.0, 100.0] }

[materials.hopg]
label = "HOPG"
profile = "standard"
E_grid_line_by_energy = [
  { energy_keV = 30.0, grid = { linspace = { start = 10.0, stop = 2600.0, num = 864, endpoint = true } } },
  { energy_keV = 100.0, grid = { linspace = { start = 50.0, stop = 4600.0, num = 1518, endpoint = true } } },
]
E_grid_brem = { arange = { start = 0.0, stop = 136500.0, step = 25.0 } }
"""

COMBINED = {
    "hopg": {
        "line_rows": [
            {"energy_keV": 30.0, "start_eV": 10.0, "stop_eV": 2700.0, "num": 897},
            {"energy_keV": 100.0, "start_eV": 50.0, "stop_eV": 4600.0, "num": 1518},
        ],
        "brem": {"stop_eV": 140000.0, "step_eV": 25.0, "raw_eV": 133000.0},
    }
}


class _NoManual:
    def is_manual_line(self, *a):
        return False

    def is_manual_brem(self, *a):
        return False


def test_apply_rewrites_only_owned_blocks_and_reparses():
    new_text, skipped = apply.apply_bounds(BASE_TOML, COMBINED, provenance_mod=_NoManual())
    assert skipped == []
    assert "stop = 2700.0, num = 897" in new_text
    assert 'label = "HOPG"' in new_text  # untouched line preserved
    tomllib.loads(new_text)  # still valid TOML


def test_apply_skips_manual_line_unless_forced():
    class ManualHopg30:
        def is_manual_line(self, material, energy):
            return material == "hopg" and float(energy) == 30.0

        def is_manual_brem(self, *a):
            return False

    new_text, skipped = apply.apply_bounds(BASE_TOML, COMBINED, provenance_mod=ManualHopg30())
    assert "hopg:30" in skipped
    assert "stop = 2600.0, num = 864" in new_text  # original 30 keV row kept
    forced, skipped2 = apply.apply_bounds(
        BASE_TOML, COMBINED, force=True, provenance_mod=ManualHopg30()
    )
    assert skipped2 == []
    assert "stop = 2700.0, num = 897" in forced


BASE_TOML_NO_GRID = """schema_version = 1

[profiles.standard]
energy_keV = { values = [30.0, 100.0] }

[materials.hfs2]
label = "HfS2"
profile = "standard"

[materials.hopg]
label = "HOPG"
profile = "standard"
E_grid_line_by_energy = [
  { energy_keV = 30.0, grid = { linspace = { start = 10.0, stop = 2600.0, num = 864, endpoint = true } } },
]
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
    assert "E_grid_brem = { arange = { start = 0.0, stop = 140000.0, step = 25.0 } }" in new_text
    assert 'label = "HfS2"' in new_text  # untouched line preserved
    assert 'label = "HOPG"' in new_text  # neighboring section untouched
    tomllib.loads(new_text)  # still valid TOML


def test_apply_file_writes_stamps_provenance_and_validates(
    tmp_path, monkeypatch, capsys
):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    json_path = tmp_path / "combined.json"
    json_path.write_text(_json.dumps(COMBINED))
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    stamped = []
    monkeypatch.setattr(
        apply._provenance, "set_line", lambda m, e, s, note=None: stamped.append((m, float(e), s))
    )
    monkeypatch.setattr(apply._provenance, "set_brem", lambda m, s, note=None: None)
    monkeypatch.setattr(apply._provenance, "is_manual_line", lambda *a: False)
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *a: False)
    # The BASE_TOML fixture is a deliberately minimal single-material stub, so
    # stub the full-catalog re-parse (production validates the real catalog).
    monkeypatch.setattr(apply, "load_material_catalog", lambda p: None)
    apply.apply_file(json_path, slurm_id="458", date="2026-07-22")
    assert "stop = 2700.0, num = 897" in toml_path.read_text()
    assert ("hopg", 30.0, "derived job 458 (2026-07-22)") in stamped
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "warning: material catalog changed; golden is now stale; "
        "run `cxr line-grid regen-golden`\n"
    )


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
    monkeypatch.setattr(apply._provenance, "is_manual_line", lambda *args: False)
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *args: False)
    monkeypatch.setattr(
        apply._provenance,
        "set_line",
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


def test_set_line_grid_stamps_manual_and_autocomputes_num(
    tmp_path, monkeypatch, capsys
):
    toml_path = tmp_path / "materials.toml"
    toml_path.write_text(BASE_TOML)
    monkeypatch.setattr(apply, "_MATERIALS_TOML", toml_path)
    calls = []
    monkeypatch.setattr(
        apply._provenance,
        "set_line",
        lambda m, e, s, note=None: calls.append((m, float(e), s, note)),
    )
    monkeypatch.setattr(apply._provenance, "is_manual_line", lambda *a: False)
    monkeypatch.setattr(apply._provenance, "is_manual_brem", lambda *a: False)
    monkeypatch.setattr(apply, "load_material_catalog", lambda path: None)
    apply.set_line_grid("hopg", 30.0, 3000.0, note="widen tail")
    text = toml_path.read_text()
    assert "stop = 3000.0" in text
    assert calls == [("hopg", 30.0, "manual", "widen tail")]
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
