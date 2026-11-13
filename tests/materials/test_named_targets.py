"""Scalar target references resolve to the existing typed geometry and identities."""

import pytest
import tomlkit

from pyrite._catalog_layout import bundled_catalog, read_text
from pyrite.geometry import Slab, Stack, load_target
from pyrite.materials import MaterialConfigError
from pyrite.materials.catalog import load_material_catalog

SLAB = {
    "kind": "slab",
    "material": "silicon",
    "thickness_ang": 20000.0,
    "tilt_deg": 30.0,
    "tilt_azim_deg": 0.0,
}


@pytest.fixture
def document():
    doc = tomlkit.parse(read_text(bundled_catalog()))
    doc["profiles"]["standard"]["materials"] = ["silicon"]
    doc["profiles"]["standard"].pop("overrides", None)
    return doc


def load(document, tmp_path, *, profile="standard"):
    path = tmp_path / "catalog.toml"
    path.write_text(tomlkit.dumps(document))
    return load_material_catalog(path, profile=profile)


def test_scalar_slab_loader_and_profile_values(document, tmp_path):
    document["targets"] = {"film": SLAB}
    document["profiles"]["standard"]["targets"] = {"silicon": "film"}
    catalog = load(document, tmp_path)
    target = load_target("film", catalog=catalog)
    assert isinstance(target, Slab)
    assert target.material == catalog.materials["silicon"].crystal_key
    assert float(target.thickness_ang) == 20000.0
    assert catalog.materials["silicon"].scan.thickness_ang.tolist() == [20000.0]
    assert catalog.profile_target_refs["standard"] == {"silicon": "film"}
    with pytest.raises(TypeError):
        catalog.named_targets["film"]["thickness_ang"] = 1
    with pytest.raises(ValueError):
        catalog.materials["silicon"].scan.thickness_ang.flags.writeable = True
    with pytest.raises(KeyError, match="unknown target preset"):
        load_target("missing", catalog=catalog)


def test_stack_loader_preserves_layer_namespace_and_order(document, tmp_path):
    document["targets"] = {
        "supported": {
            "kind": "stack",
            "layers": [
                {"material": "silicon", "thickness_ang": 20000.0},
                {"material": "silicon", "thickness_ang": 30000.0},
                {"material": "sio2", "thickness_ang": 40000.0},
            ],
        }
    }
    document["profiles"]["standard"]["targets"] = {"silicon": "supported"}
    catalog = load(document, tmp_path)
    target = load_target("supported", catalog=catalog)
    assert isinstance(target, Stack)
    assert [layer.material for layer in target.layers] == ["silicon", "silicon", "sio2"]
    assert [float(layer.thickness_ang) for layer in target.layers] == [20000, 30000, 40000]
    assert [layer.material for layer in catalog.materials["silicon"].stack] == ["silicon", "sio2"]


@pytest.mark.parametrize(
    "change",
    [
        {"thickness_ang": [100.0]},
        {"thickness_ang": float("nan")},
        {"thickness_ang": float("inf")},
        {"thickness_ang": True},
        {"tilt_deg": 0},
        {"tilt_deg": float("inf")},
        {"tilt_azim_deg": 90},
        {"crystal_width_mm": 1},
        {"crystal_width_mm": -1, "crystal_height_mm": 1},
        {"material": "sio2"},
        {"kind": "mesh"},
        {"cut": {}},
        {"mosaic": 1},
        {"groove_spacing_ang": 100},
    ],
)
def test_invalid_unused_definitions_rejected(document, tmp_path, change):
    document["targets"] = {"bad": {**SLAB, **change}}
    with pytest.raises(MaterialConfigError, match="targets.bad"):
        load(document, tmp_path)


@pytest.mark.parametrize("thickness", [[100.0], float("nan"), float("inf"), -1, True])
def test_stack_layer_requires_finite_positive_scalar(document, tmp_path, thickness):
    document["targets"] = {
        "bad": {
            "kind": "stack",
            "layers": [
                {"material": "silicon", "thickness_ang": 100},
                {"material": "silicon", "thickness_ang": thickness},
            ],
        }
    }
    with pytest.raises(MaterialConfigError, match=r"targets.bad.layers\[1\].thickness_ang"):
        load(document, tmp_path)


@pytest.mark.parametrize("refs", [{"silicon": "missing"}, {"hopg": "film"}, [], {"silicon": []}])
def test_invalid_references_rejected(document, tmp_path, refs):
    document["targets"] = {"film": SLAB}
    document["profiles"]["standard"]["targets"] = refs
    with pytest.raises(MaterialConfigError, match="profiles.standard.targets"):
        load(document, tmp_path)


def test_membership_and_geometry_conflict(document, tmp_path):
    document["targets"] = {"film": SLAB}
    row = document["profiles"]["standard"]
    row["targets"] = {"silicon": "film"}
    row["materials"] = ["hopg"]
    with pytest.raises(MaterialConfigError, match="outside profile membership"):
        load(document, tmp_path)
    row["materials"] = ["silicon"]
    row["overrides"] = {"silicon": {"thickness_ang": 100}}
    with pytest.raises(MaterialConfigError, match="conflicts with geometry overrides"):
        load(document, tmp_path)


@pytest.mark.parametrize(
    "field,value", [("materials", 1), ("overrides", []), ("overrides", {"silicon": 4})]
)
def test_malformed_profile_aggregates_catalog_error(document, tmp_path, field, value):
    document["targets"] = {"film": SLAB}
    row = document["profiles"]["standard"]
    row["targets"] = {"silicon": "film"}
    row[field] = value
    with pytest.raises(MaterialConfigError):
        load(document, tmp_path)


def test_named_inline_and_renamed_have_same_identity(document, tmp_path, monkeypatch):
    from pyrite.campaign import config
    from pyrite.campaign.profiles import named_profile_identity

    row = document["profiles"]["standard"]
    row["materials"] = ["silicon"]
    row.pop("thickness_layers", None)
    row["thickness_ang"] = 20000.0
    row["tilt_deg"] = 30.0
    row["tilt_azim_deg"] = 0.0
    inline = load(document, tmp_path)
    monkeypatch.setattr(config, "CATALOG", inline)
    before = named_profile_identity("silicon")
    document["targets"] = {"film": {**SLAB, "description": "label only"}}
    row["targets"] = {"silicon": "film"}
    named = load(document, tmp_path)
    monkeypatch.setattr(config, "CATALOG", named)
    after = named_profile_identity("silicon")
    assert before == after
    assert config.material_sweep("silicon").target == load_target("film", catalog=named)
    document["targets"]["other"] = document["targets"].pop("film")
    row["targets"]["silicon"] = "other"
    renamed = load(document, tmp_path)
    monkeypatch.setattr(config, "CATALOG", renamed)
    assert named_profile_identity("silicon") == before
    document["targets"]["other"]["thickness_ang"] = 30000
    changed = load(document, tmp_path)
    monkeypatch.setattr(config, "CATALOG", changed)
    assert named_profile_identity("silicon") != before


def test_reference_lifecycle_and_invalid_delete_rollback(document, tmp_path):
    from pyrite.campaign.target_presets import delete_target, rename_target
    from pyrite.catalog_edit import CatalogDraft

    document["targets"] = {"film": SLAB, "spare": SLAB}
    document["profiles"]["standard"]["targets"] = {"silicon": "film"}
    load(document, tmp_path)
    draft = CatalogDraft.open(tmp_path / "catalog.toml")
    rename_target(draft.document, "film", "renamed")
    draft.save()
    assert draft.document["profiles"]["standard"]["targets"]["silicon"] == "renamed"
    before = draft.path.read_bytes()
    with pytest.raises(ValueError, match="cannot delete referenced"):
        delete_target(draft.document, "renamed")
    assert draft.path.read_bytes() == before
    delete_target(draft.document, "spare")
    draft.save()
    # Even hand edits cannot save a dangling reference.
    del draft.document["targets"]["renamed"]
    before = draft.path.read_bytes()
    with pytest.raises(MaterialConfigError, match="unknown target preset"):
        draft.save()
    assert draft.path.read_bytes() == before


def test_multiple_profiles_and_directory_layer_readonly(tmp_path, monkeypatch):
    from copy import deepcopy

    from pyrite._catalog_layout import write_text
    from tests.helpers.user_catalog import copy_full_catalog

    path = copy_full_catalog(tmp_path / "catalog")
    doc = tomlkit.parse(read_text(path))
    doc["targets"] = {"film": SLAB}
    doc["profiles"]["survey"] = deepcopy(doc["profiles"]["standard"])
    for name in ("standard", "survey"):
        doc["profiles"][name]["materials"] = ["silicon"]
        doc["profiles"][name].pop("overrides", None)
        doc["profiles"][name]["targets"] = {"silicon": "film"}
    write_text(path, tomlkit.dumps(doc))
    assert (path / "targets" / "film.toml").is_file()
    first = load_material_catalog(path)
    second = load_material_catalog(path, profile="survey")
    assert first.materials["silicon"].target_fields == second.materials["silicon"].target_fields
    from pyrite import _catalog_layout as layout

    monkeypatch.setattr(layout, "bundled_catalog", lambda: path)
    monkeypatch.setattr(layout, "user_layer", lambda: tmp_path / "user")
    with pytest.raises(layout.BundledProfileError):
        layout.require_writable_object(path, "targets", "film")
    layered = tomlkit.parse(layout.read_text(path))
    layered["targets"]["copy"] = SLAB
    layout.write_text(path, tomlkit.dumps(layered))
    assert (tmp_path / "user" / "targets" / "copy.toml").is_file()
    assert load_material_catalog(path).named_targets["copy"] == first.named_targets["film"]


def test_stack_named_and_inline_identity_match(document, tmp_path, monkeypatch):
    from pyrite.campaign import config
    from pyrite.campaign.profiles import named_profile_identity

    layers = [{"material": "sio2", "thickness_ang": 30000.0}]
    row = document["profiles"]["standard"]
    row.pop("thickness_layers", None)
    row.update({"thickness_ang": 20000.0, "tilt_deg": 30.0, "tilt_azim_deg": 0.0})
    document["materials"]["silicon"]["stack"] = layers
    monkeypatch.setattr(config, "CATALOG", load(document, tmp_path))
    inline = named_profile_identity("silicon")
    inline_target = config.material_sweep("silicon").target
    document["materials"]["silicon"].pop("stack")
    document["targets"] = {
        "supported": {
            "kind": "stack",
            "layers": [{"material": "silicon", "thickness_ang": 20000.0}, *layers],
        }
    }
    row["targets"] = {"silicon": "supported"}
    monkeypatch.setattr(config, "CATALOG", load(document, tmp_path))
    assert named_profile_identity("silicon") == inline
    assert config.material_sweep("silicon").target == inline_target


def test_unused_presets_do_not_change_identity(document, tmp_path, monkeypatch):
    from pyrite.campaign import config
    from pyrite.campaign.profiles import named_profile_identity

    monkeypatch.setattr(config, "CATALOG", load(document, tmp_path))
    before = named_profile_identity("silicon")
    document["targets"] = {"film": SLAB}
    document["filters"] = {
        "window": {
            "material": "silicon",
            "thickness_mm": 0.1,
            "size_mm": [20.0, 10.0],
            "distance_mm": 50.0,
        }
    }
    monkeypatch.setattr(config, "CATALOG", load(document, tmp_path))
    assert named_profile_identity("silicon") == before


def test_remote_catalog_layer_transfer_covers_preset_tables():
    from pyrite._catalog_layout import LAYERED_TABLES

    assert {"filters", "targets"} <= set(LAYERED_TABLES)


def test_cut_and_unbounded_footprint_lower_to_the_slab_api(document, tmp_path):
    """#624: the admitted axial geometry is expressible from a named target."""
    from pyrite.geometry import CrystalCut

    document["targets"] = {
        "fibre": {
            **SLAB,
            "thickness_ang": 3000.0,
            "tilt_deg": 0.0,
            "allow_normal_incidence": True,
            "footprint": "unbounded",
            "cut": {"surface_hkl": [0, 0, 1]},
        }
    }
    catalog = load(document, tmp_path)
    target = load_target("fibre", catalog=catalog)
    assert target == Slab(
        "silicon",
        thickness_ang=3000.0,
        tilt_deg=0.0,
        width=None,
        height=None,
        cut=CrystalCut(surface_hkl=(0, 0, 1)),
        allow_normal_incidence=True,
    )
    plain = load_target("plain", catalog=load(_with_plain(document), tmp_path))
    assert plain.footprint is not None and plain._crystal_cut() is None


def _with_plain(document):
    document["targets"] = {"plain": SLAB}
    return document


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"footprint": "infinite"}, "must be 'unbounded'"),
        (
            {"footprint": "unbounded", "crystal_width_mm": 1, "crystal_height_mm": 1},
            "excludes crystal_width_mm",
        ),
        ({"cut": [0, 0, 1]}, "must be a table"),
        ({"cut": {"surface_hkl": [0, 0, 0]}}, "must be nonzero"),
        ({"cut": {"surface_hkl": [0, 0, 1], "roll_deg": 1.0}}, "roll_deg"),
    ],
)
def test_invalid_cut_and_footprint_rejected(document, tmp_path, change, message):
    document["targets"] = {"bad": {**SLAB, **change}}
    with pytest.raises(MaterialConfigError, match=message):
        load(document, tmp_path)


def test_a_stack_has_no_cut(document, tmp_path):
    document["targets"] = {
        "bad": {
            "kind": "stack",
            "cut": {"surface_hkl": [0, 0, 1]},
            "layers": [
                {"material": "silicon", "thickness_ang": 100},
                {"material": "sio2", "thickness_ang": 100},
            ],
        }
    }
    with pytest.raises(MaterialConfigError, match="targets.bad"):
        load(document, tmp_path)
