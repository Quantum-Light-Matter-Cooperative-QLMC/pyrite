"""One-object-per-file catalog directories: layout rules, round trip, write-back."""

import shutil

import pytest

from pyrite._catalog_layout import (
    CatalogLayoutError,
    bundled_catalog,
    read_raw,
    read_sources,
    read_text,
    split_text,
    write_text,
)
from pyrite.materials import MaterialConfigError, load_material_catalog

_MINIMAL = {
    "catalog.toml": "schema_version = 1\n",
    "profiles/standard.toml": 'materials = ["a"]\n',
    "materials/a.toml": 'crystal = "x"\n\n[validation]\nid = "v"\n',
    "beams/b.toml": '# keep me\nrep_rate_hz = 1.0\n\n[longitudinal]\nkind = "gaussian"\n',
}


def _write_tree(root, files):
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_directory_assembles_tables_keyed_by_file_stem(tmp_path):
    raw = read_raw(_write_tree(tmp_path, _MINIMAL))

    assert raw["schema_version"] == 1
    assert raw["materials"] == {"a": {"crystal": "x", "validation": {"id": "v"}}}
    assert raw["beams"]["b"]["longitudinal"] == {"kind": "gaussian"}


def test_text_round_trip_is_byte_exact(tmp_path):
    root = _write_tree(tmp_path, _MINIMAL)

    text = read_text(root)

    assert "[materials.a.validation]" in text
    assert "[beams.b.longitudinal]" in text
    assert split_text(text) == _MINIMAL


def test_bundled_catalog_text_round_trip_is_byte_exact():
    files = {relative: content.decode() for relative, content in read_sources(bundled_catalog())}

    assert split_text(read_text(bundled_catalog())) == files


def test_bundled_directory_matches_its_single_file_assembly(tmp_path):
    single = tmp_path / "materials.toml"
    single.write_text(read_text(bundled_catalog()))
    (tmp_path / "energy-grid-artifacts").mkdir()

    directory = load_material_catalog(bundled_catalog())
    flat = load_material_catalog(single)

    assert set(directory.profile_names) == set(flat.profile_names)
    for profile in directory.profile_names:
        assert directory.profile_materials(profile) == flat.profile_materials(profile)
    assert set(directory.materials) == set(flat.materials)


def test_write_text_touches_only_changed_objects(tmp_path):
    root = _write_tree(tmp_path, _MINIMAL)
    untouched = (root / "materials/a.toml").stat().st_mtime_ns
    text = read_text(root)

    text = text.replace("rep_rate_hz = 1.0", "rep_rate_hz = 2.0")
    text += "\n[detectors.new]\nobservation_angle_deg = 90.0\n"
    text = text.replace('[profiles.standard]\nmaterials = ["a"]\n', "")
    write_text(root, text)

    assert (root / "beams/b.toml").read_text().startswith("# keep me\nrep_rate_hz = 2.0")
    assert (root / "detectors/new.toml").read_text() == "observation_angle_deg = 90.0\n"
    assert not (root / "profiles/standard.toml").exists()
    assert (root / "materials/a.toml").stat().st_mtime_ns == untouched


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"notes.txt": "x"}, "unexpected catalog entry"),
        ({"beams/readme.md": "x"}, "expected a <name>.toml object file"),
        ({"beams/-bad.toml": "x = 1\n"}, "object name must use"),
        ({"beams/B.toml": "x = 1\n"}, "only by case"),
        ({"catalog.toml": "schema_version = 1\n[beams.c]\n"}, "belongs in beams/<name>.toml"),
        ({"beams/broken.toml": "x = \n"}, "broken.toml"),
    ],
)
def test_layout_violations_name_the_offending_file(tmp_path, extra, message):
    root = _write_tree(tmp_path, {**_MINIMAL, **extra})

    with pytest.raises(CatalogLayoutError, match=message):
        read_raw(root)


def test_missing_manifest_is_a_catalog_error(tmp_path):
    root = _write_tree(tmp_path, {k: v for k, v in _MINIMAL.items() if k != "catalog.toml"})

    with pytest.raises(MaterialConfigError, match="missing catalog manifest"):
        load_material_catalog(root)


def test_hidden_files_and_artifact_store_are_ignored(tmp_path):
    root = _write_tree(tmp_path, {**_MINIMAL, ".DS_Store": "", "beams/.b.toml.swp": ""})
    (root / "energy-grid-artifacts").mkdir()

    assert set(read_raw(root)["beams"]) == {"b"}


def test_copied_bundled_directory_validates(tmp_path):
    root = tmp_path / "catalog"
    shutil.copytree(bundled_catalog(), root)

    catalog = load_material_catalog(root)

    assert "standard" in catalog.profile_names
