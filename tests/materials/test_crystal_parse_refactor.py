"""Regression coverage for crystal-row validation and assembly."""

from pathlib import Path

from pyrite.materials import _parse
from pyrite.materials._catalog_decode import _Errors
from pyrite.materials._schema import CrystalInfo


def _row() -> dict[str, object]:
    return {
        "cif": "cifs/example.cif",
        "validation_id": "example",
        "formula": "C",
        "B_ang2": 0.0,
        "surface_hkl": [0, 0, 1],
    }


def _parse_rows(rows, tmp_path, monkeypatch):
    """Isolate CIF I/O while exercising real catalog decoding and CrystalSpec."""
    cif = tmp_path / "example.cif"
    monkeypatch.setattr(_parse, "_cif_path", lambda *_args, **_kwargs: cif)
    monkeypatch.setattr(
        _parse,
        "_parse_info",
        lambda _key, _cif, mosaic, _path, _errors: CrystalInfo({}, (), 1.0, (), mosaic),
    )
    errors = _Errors()
    result = _parse._parse_crystals(rows, errors, source=tmp_path / "catalog.toml")
    return result, errors.items


def test_crystal_metadata_and_fiber_families(tmp_path: Path, monkeypatch) -> None:
    row = {
        **_row(),
        "full_name": " Graphite ",
        "phase": " hexagonal ",
        "cod_id": 123,
        "mp_id": "mp-42",
        "mosaic_fwhm_deg": 0.4,
        "hkl_families": [[0, 0, 2], [0, 0, 4]],
        "hkl_reason": " basal reflections ",
        "texture": "fiber",
        "reference_hkl": [0, 0, 2],
        "layers_per_cell": 2,
    }
    result, errors = _parse_rows({"example": row}, tmp_path, monkeypatch)
    assert errors == []
    crystal = result["example"]
    assert (crystal.full_name, crystal.phase) == ("Graphite", "hexagonal")
    assert (crystal.cod_id, crystal.mp_id) == (123, "mp-42")
    assert crystal.B_ang2 == 0.0
    assert crystal.info.mosaic_fwhm_deg == 0.4
    assert crystal.hkl_families == ((0, 0, 2), (0, 0, 4))
    assert crystal.hkl_reason == "basal reflections"
    assert crystal.texture == "fiber"
    assert crystal.reference_hkl == (0, 0, 2)
    assert crystal.layers_per_cell == 2
    assert row["hkl_families"] == [[0, 0, 2], [0, 0, 4]]


def test_crystal_errors_accumulate_in_original_order(tmp_path: Path, monkeypatch) -> None:
    row = {
        **_row(),
        "validation_id": "",
        "formula": " C ",
        "cod_id": True,
        "mp_id": "invalid",
        "B_ang2": -1,
        "mosaic_fwhm_deg": 0,
        "hkl_families": [[0, 0, -2], [0, 0, 2], [0, 0, 2], [0, 0, 0], [1, 0, 0]],
        "texture": "fiber",
        "reference_hkl": [0, 1, 0],
        "layers_per_cell": False,
    }
    result, errors = _parse_rows({"example": row}, tmp_path, monkeypatch)
    prefix = "crystals.example"
    assert errors == [
        f"{prefix}.validation_id: must be a nonempty string",
        f"{prefix}.formula: must be an ASCII chemical formula",
        f"{prefix}.cod_id: must be a positive integer",
        f"{prefix}.mp_id: must match 'mp-<digits>'",
        f"{prefix}.B_ang2: must be finite and nonnegative",
        f"{prefix}.mosaic_fwhm_deg: must be finite and positive",
        f"{prefix}.hkl_families[0]: must use the positive representative",
        f"{prefix}.hkl_families[2]: duplicate family",
        f"{prefix}.hkl_families[3]: must be nonzero",
        f"{prefix}.hkl_reason: is required when hkl_families is pinned",
        f"{prefix}.hkl_families: fiber families must be parallel to surface_hkl",
        f"{prefix}.reference_hkl: must be a positive multiple of surface_hkl",
        f"{prefix}.layers_per_cell: must be a positive integer",
    ]
    crystal = result["example"]
    assert crystal.formula == "C"
    assert crystal.texture == "single-crystal"
    assert crystal.hkl_families == ((0, 0, 2), (1, 0, 0))


def test_crystal_family_provenance_for_absent_and_invalid_pins(tmp_path: Path, monkeypatch) -> None:
    rows = {
        "absent": {**_row(), "hkl_reason": " orphan "},
        "invalid": {**_row(), "hkl_families": [], "hkl_reason": ""},
    }
    result, errors = _parse_rows(rows, tmp_path, monkeypatch)
    assert errors == [
        "crystals.absent.hkl_reason: requires hkl_families",
        "crystals.invalid.hkl_families: must be a nonempty array",
        "crystals.invalid.hkl_reason: is required when hkl_families is pinned",
    ]
    assert result["absent"].hkl_reason == "orphan"
    assert result["invalid"].hkl_reason == ""
    assert result["invalid"].hkl_families == ()
