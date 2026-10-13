"""Authoring units, deterministic registration, and catalog ownership (#486)."""

from dataclasses import replace

import pytest
from scipy.constants import Avogadro

from pyrite import _catalog_layout as layout
from pyrite.materials import build_crystal, build_medium, load_material_catalog, save_material
from tests.helpers.material_authoring import preserve_catalog_registry  # noqa: F401


@pytest.mark.parametrize(
    "recipe",
    [
        {"formula": "SiO2"},
        {"atom_counts": {"O": 2, "Si": 1}},
        {"atomic_fractions": {"Si": 1 / 3, "O": 2 / 3}},
        {"mass_fractions": {"Si": 28.085 / 60.083, "O": 31.998 / 60.083}},
    ],
)
def test_composition_representations_are_equivalent(recipe):
    expected = build_medium("silica", formula="SiO2", density_g_cm3=2.2)
    built = build_medium("different-name", density_g_cm3=2.2, **recipe)
    assert built.composition == expected.composition
    assert built.identity == expected.identity
    assert built.density_g_cm3 == pytest.approx(2.2)
    assert dict(built.composition)["O"] == pytest.approx(2 * dict(built.composition)["Si"])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"formula": "H2O"}, "unsupported.*H"),
        ({"formula": "Si(OH)2"}, "invalid formula"),
        ({"formula": "Si0"}, "positive"),
        ({"formula": "Si", "density_g_cm3": -1}, "positive"),
        ({"formula": "Si", "density_g_cm3": float("nan")}, "positive"),
        ({"mass_fractions": {"Si": 0.8}}, "sum to 1"),
        ({"atomic_fractions": {}}, "nonempty"),
        ({"atomic_fractions": {"Si": 1, "O": 0}}, "positive"),
        ({"formula": "Si", "atom_counts": {"Si": 1}}, "exactly one"),
    ],
)
def test_bad_compositions_fail_before_writes(kwargs, match):
    options = {"density_g_cm3": 2.2, **kwargs}
    with pytest.raises(ValueError, match=match):
        build_medium("mine", **options)


def test_fractional_formula_and_elemental_density():
    built = build_medium("alloy", formula="Si0.5Ge0.5", density_g_cm3=3)
    assert dict(built.composition)["Si"] == dict(built.composition)["Ge"]
    silicon = build_medium("amorphous-si", formula="Si", density_g_cm3=2.33)
    assert silicon.composition[0][1] == pytest.approx(2.33 * Avogadro * 1e-24 / 28.085)


def test_save_only_user_layer_and_fresh_load(_isolate_user_catalog):
    before = layout.read_sources(layout.bundled_catalog())
    old = load_material_catalog()
    built = build_medium("my-silica", formula="SiO2", density_g_cm3=2.2)
    assert not (_isolate_user_catalog / "media" / "my-silica.toml").exists()
    loaded = save_material(built)
    assert loaded.media[built.name] == built.medium
    assert built.name not in old.media
    assert load_material_catalog().media[built.name] == built.medium
    assert layout.object_sources(layout.bundled_catalog(), "media")[built.name] == "user"
    assert layout.read_sources(layout.bundled_catalog())[: len(before)] == before
    with pytest.raises(ValueError, match="already exists"):
        save_material(build_medium("MY-SILICA", formula="Si", density_g_cm3=2.33))


def test_bundled_names_cannot_be_shadowed():
    with pytest.raises(ValueError, match="already exists"):
        save_material(build_medium("SILICON", formula="Si", density_g_cm3=2.33))


def test_existing_crystal_recipe_reuses_configuration():
    built = build_crystal("my-silicon", crystal="silicon")
    loaded = save_material(built)
    assert loaded.materials[built.name].crystal_key == "silicon"
    assert loaded.crystals["silicon"].composition == built.composition
    with pytest.raises(ValueError, match="cannot be overridden"):
        build_crystal("another", crystal="silicon", B_ang2=0)


def _new_cif(name, source):
    return build_crystal(
        name,
        cif=source,
        formula="Si",
        phase="diamond",
        surface_hkl=(1, 1, 1),
        B_ang2=0.46,
        photon_energies_eV=[100, 200, 300],
    )


def test_cif_is_captured_and_copied(tmp_path, _isolate_user_catalog):
    source = tmp_path / "input.cif"
    source.write_bytes((layout.bundled_catalog().parent / "cifs" / "silicon.cif").read_bytes())
    built = _new_cif("my-phase", source)
    source.unlink()
    loaded = save_material(built)
    spec = loaded.crystals[built.name]
    assert spec.cif.is_relative_to(_isolate_user_catalog / "cifs")
    assert spec.cif.read_bytes() == built.cif_bytes
    assert spec.phase == "diamond"
    assert spec.cut == (1, 1, 1)
    assert spec.composition == built.composition
    assert load_material_catalog().crystals[built.name].composition == built.composition


def test_invalid_cif_or_density_and_symlinks(tmp_path):
    source = layout.bundled_catalog().parent / "cifs" / "silicon.cif"
    link = tmp_path / "linked.cif"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        _new_cif("linked", link)
    with pytest.raises(ValueError, match="density disagrees"):
        build_crystal("bad-density", crystal="silicon", density_g_cm3=10)
    with pytest.raises(ValueError, match="stoichiometry|elements disagree"):
        build_crystal(
            "bad-formula",
            cif=source,
            formula="SiO2",
            phase="diamond",
            surface_hkl=(1, 1, 1),
            B_ang2=0.46,
            photon_energies_eV=[100],
        )


def test_external_catalog_is_independent(tmp_path, _isolate_user_catalog):
    import shutil

    external = tmp_path / "external"
    shutil.copytree(layout.bundled_catalog(), external)
    built = build_medium("external-only", formula="C", density_g_cm3=2)
    loaded = save_material(built, catalog=external)
    assert built.name in loaded.media
    assert (external / "media" / f"{built.name}.toml").is_file()
    assert not (_isolate_user_catalog / "media" / f"{built.name}.toml").exists()
    assert built.name not in load_material_catalog().media


def test_failed_publication_rolls_back(tmp_path, monkeypatch, _isolate_user_catalog):
    from pyrite.materials import authoring

    source = layout.bundled_catalog().parent / "cifs" / "silicon.cif"
    built = _new_cif("rollback-phase", source)
    original = authoring._exclusive_write

    def fail(path, data, created):
        if path.parent.name == "materials":
            raise OSError("injected publication failure")
        original(path, data, created)

    monkeypatch.setattr(authoring, "_exclusive_write", fail)
    with pytest.raises(OSError, match="publication failure"):
        save_material(built)
    assert not (_isolate_user_catalog / "crystals" / f"{built.name}.toml").exists()
    assert not list((_isolate_user_catalog / "cifs").glob("*.cif"))
    assert built.name not in load_material_catalog().materials


def test_invalid_definition_rolls_back_without_records(_isolate_user_catalog):
    built = build_medium("invalid", formula="Si", density_g_cm3=2.33)
    with pytest.raises(ValueError):
        save_material(replace(built, record_toml="[composition]\nSi = -1\n"))
    assert not (_isolate_user_catalog / "media" / "invalid.toml").exists()


def test_saved_crystal_refreshes_loaded_simulation_bindings():
    import pyrite.materials as materials
    from pyrite.campaign import config, geometry
    from pyrite.materials import crystal

    previous = materials.CATALOG
    source = layout.bundled_catalog().parent / "cifs" / "silicon.cif"
    built = _new_cif("fresh-crystal", source)
    saved = save_material(built)
    assert materials.CATALOG is saved
    assert geometry.CATALOG is saved
    assert "fresh-crystal" in config.MATERIALS
    assert "fresh-crystal" in crystal.CRYSTALS
    assert "fresh-crystal" not in previous.materials
    assert geometry.crystal_params("fresh-crystal")["composition"] == list(built.composition)


def test_cif_traversal_rejected_before_asset_write(tmp_path, _isolate_user_catalog):
    built = _new_cif("safe-crystal", layout.bundled_catalog().parent / "cifs" / "silicon.cif")
    outside = tmp_path / "outside.cif"
    changed = built.record_toml.replace('cif = "cifs/', f'cif = "{tmp_path}/')
    with pytest.raises(ValueError, match="changed since build|managed content hash"):
        save_material(replace(built, record_toml=changed))
    assert not outside.exists()
    assert not list(tmp_path.glob("*.cif"))


def test_single_file_catalog_save_is_independent(tmp_path):
    external = tmp_path / "complete.toml"
    external.write_text(layout.read_text(layout.bundled_catalog()))
    built = build_medium("single-file", formula="C", density_g_cm3=2)
    saved = save_material(built, catalog=external)
    assert saved.media[built.name] == built.medium
    assert layout.object_sources(external, "media")[built.name] == "catalog"
    assert built.name not in load_material_catalog().media


def test_shared_layout_cannot_shadow_bundled_cif(tmp_path, _isolate_user_catalog):
    source = layout.bundled_catalog().parent / "cifs" / "silicon.cif"
    directory = _isolate_user_catalog / "cifs"
    directory.mkdir()
    (directory / "silicon.cif").write_text("invalid user shadow")
    # A bundled crystal reads its own asset owner, not a user asset of the same name.
    spec = load_material_catalog().crystals["silicon"]
    assert spec.cif == source


def test_overlapping_authoring_fails_closed(_isolate_user_catalog):
    lock = _isolate_user_catalog / ".material-authoring.lock"
    lock.mkdir()
    built = build_medium("busy", formula="C", density_g_cm3=2)
    with pytest.raises(ValueError, match="already in progress"):
        save_material(built)
    assert not (_isolate_user_catalog / "media" / "busy.toml").exists()
    lock.rmdir()
    save_material(built)
    assert not lock.exists()


def test_missing_layer_table_reports_registered_medium_name(monkeypatch):
    from pyrite.xsgen._errors import TableNotFoundError
    from pyrite.xsgen.sbethe import catalog as tables

    built = build_medium("registered-silica", formula="SiO2", density_g_cm3=2.2)
    save_material(built)
    monkeypatch.setattr(tables, "resolve", lambda key: None)
    with pytest.raises(TableNotFoundError, match="--material registered-silica"):
        tables.resolve_composition_table("custom-crystal:layer-1", built.composition)


def test_partial_occupancy_is_rejected(tmp_path):
    source = tmp_path / "partial.cif"
    source.write_text("""data_partial
_cell_length_a 5.43
_cell_length_b 5.43
_cell_length_c 5.43
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si1 Si 0 0 0 0.5
""")
    with pytest.raises(ValueError, match="full occupancy"):
        _new_cif("partial", source)


def test_single_file_post_publication_failure_restores_catalog(tmp_path, monkeypatch):
    from pyrite.materials import authoring

    external = tmp_path / "complete.toml"
    original = layout.read_text(layout.bundled_catalog())
    external.write_text(original)
    built = build_medium("rollback-single", formula="C", density_g_cm3=2)
    load = authoring.load_material_catalog

    def fail_reload(path=None, **kwargs):
        if path == external:
            raise OSError("injected reload failure")
        return load(path, **kwargs)

    monkeypatch.setattr(authoring, "load_material_catalog", fail_reload)
    with pytest.raises(OSError, match="reload failure"):
        save_material(built, catalog=external)
    assert external.read_text() == original
    assert not (tmp_path / ".material-authoring.lock").exists()


def test_registered_medium_lowers_into_real_layered_case():
    import pyrite as pr
    from pyrite import api

    built = build_medium("layer-silica", formula="SiO2", density_g_cm3=2.2)
    save_material(built)
    save_material(build_crystal("layer-film", crystal="silicon"))
    target = pr.Stack(
        layers=(pr.Layer("layer-film", 100), pr.Layer(built.name, 100)),
        tilt_deg=30,
        footprint=None,
    )
    scene = pr.Scene(
        beam=pr.Beam(energy_keV=30, transverse=pr.Spot(), longitudinal=pr.PointBunch()),
        target=target,
        detector=pr.Detector(),
        physics=pr.Physics(inelastic_model="continuous", brem_source="none"),
    )
    case = api.build_case(
        scene,
        pr.Numerics(
            cxr_trials=1,
            brem_trials=1,
            energy_bins=pr.EnergyBins(cxr=[1000, 1050], brem=[1000, 1050]),
        ),
    )
    assert tuple(case["abs_layers"][1][2]) == built.composition
    assert case["layer_radiators"][1] is None
