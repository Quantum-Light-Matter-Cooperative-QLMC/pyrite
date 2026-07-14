"""Focused tests for the immutable TOML material catalog."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest


def _write_catalog(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "materials.toml"
    path.write_text(text)
    return path


def _minimal_catalog(*, crystal: str = "mos2", material_rows: str = "") -> str:
    return f"""
schema_version = 1
[profiles.base]
thickness_ang = {{ logspace = {{ start = 2.0, stop = 3.0, num = 2 }} }}
energy_keV = {{ values = [25.0, 30.0] }}
tilt_deg = {{ linspace = {{ start = 0.0, stop = 80.0, num = 3, endpoint = false }} }}
tilt_azim_deg = 0.0
E_grid_line = {{ arange = {{ start = 50.0, stop = 60.0, step = 2.0 }} }}
E_grid_brem = 0.0
[crystals.{crystal}]
cif = "cifs/{crystal}.cif"
validation_id = "test-fixture"
B_ang2 = 0.6
beam_uvw = [0, 0, 2]
layers_per_cell = 2
E_grid = {{ values = [100.0, 200.0] }}
[media.sio2]
composition = {{ Si = 0.02205, O = 0.04410 }}
{material_rows}
"""


def test_packaged_catalog_exposes_frozen_ordered_public_api():
    from cxr_mc.materials import CATALOG, MaterialCatalog

    assert isinstance(CATALOG, MaterialCatalog)
    assert len(CATALOG.crystals) == 21
    assert len(CATALOG.materials) == 21
    assert CATALOG.material_keys == tuple(CATALOG.materials)
    assert CATALOG.crystal("hbn") is CATALOG.crystals["hbn"]
    assert CATALOG.material("mote2") is CATALOG.materials["mote2"]

    with pytest.raises(TypeError):
        CATALOG.materials["new"] = CATALOG.material("mote2")  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        CATALOG.material("mote2").label = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError):
        CATALOG.material("mote2").scan.tilt_deg[0] = 1.0
    with pytest.raises(TypeError):
        CATALOG.crystal("hbn").lattice["c"] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError):
        CATALOG.crystal("hbn").basis[0][1][0] = 1.0

    np.testing.assert_array_equal(
        CATALOG.material("hbn").scan.thickness_ang,
        np.concatenate([np.logspace(2, 5, 6), np.logspace(5, 6, 2, endpoint=False)]),
    )


def test_grid_descriptors_profile_overrides_and_layer_count_conversion(tmp_path):
    from cxr_mc.materials import load_material_catalog

    path = _write_catalog(
        tmp_path,
        _minimal_catalog(
            material_rows="""
[materials.sample]
label = "sample"
profile = "base"
crystal = "mos2"
thickness_layers = { values = [3, 4] }
tilt_azim_deg = { logspace = { start = 0.0, stop = 2.0, num = 3, base = 2.0 } }
stack = [{ material = "sio2", thickness_ang = 2850.0, azimuth_deg = 12.0 }]
""",
        ),
    )

    catalog = load_material_catalog(path)
    scan = catalog.material("sample").scan
    np.testing.assert_array_equal(scan.energy_keV, [25.0, 30.0])
    np.testing.assert_array_equal(scan.E_grid_line, np.arange(50.0, 60.0, 2.0))
    np.testing.assert_array_equal(scan.tilt_deg, np.linspace(0.0, 80.0, 3, endpoint=False))
    np.testing.assert_array_equal(scan.tilt_azim_deg, np.logspace(0.0, 2.0, 3, base=2.0))
    np.testing.assert_allclose(scan.thickness_ang, np.array([3.0, 4.0]) * 12.294 / 2.0)
    assert catalog.material("sample").stack[0].azimuth_deg == 12.0


def test_pinned_hkls_add_negatives_and_require_positive_representatives(tmp_path):
    from cxr_mc.materials import MaterialConfigError, load_material_catalog

    valid = _minimal_catalog(
        material_rows="""
[materials.mos2]
label = "MoS2"
profile = "base"
""",
    ).replace(
        "layers_per_cell = 2",
        'layers_per_cell = 2\nhkl_families = [[0, 0, 2]]\nhkl_reason = "basal cut"',
    )
    catalog = load_material_catalog(_write_catalog(tmp_path, valid))
    assert catalog.crystal("mos2").hkl_list == ((0, 0, 2), (0, 0, -2))

    invalid = valid.replace("[[0, 0, 2]]", "[[0, 0, -2]]").replace(
        'hkl_reason = "basal cut"', ""
    )
    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, invalid))
    assert "crystals.mos2.hkl_families[0]" in str(caught.value)
    assert "crystals.mos2.hkl_reason" in str(caught.value)


def test_semantic_errors_accumulate_with_paths(tmp_path):
    from cxr_mc.materials import MaterialConfigError, load_material_catalog

    path = _write_catalog(
        tmp_path,
        """
schema_version = 2
surprise = true
[profiles.bad]
thickness_ang = { values = [] }
energy_keV = { arange = { start = 10.0, stop = 20.0, step = 0.0 } }
tilt_deg = nan
[crystals.bad]
cif = "../secrets.cif"
validation_id = ""
B_ang2 = -1.0
beam_uvw = [0, 0, 0]
hkl_families = [[0, 0, 0]]
[media.bad]
composition = { Xe = -1.0 }
[materials.bad]
profile = "missing"
crystal = "missing"
substrate = "missing"
stack = [{ material = "missing", thickness_ang = -2.0 }]
""",
    )

    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(path)
    message = str(caught.value)
    for expected in (
        "schema_version",
        "catalog.surprise",
        "profiles.bad.thickness_ang",
        "crystals.bad.cif",
        "crystals.bad.beam_uvw",
        "media.bad.composition.Xe",
        "materials.bad.label",
        "materials.bad.profile",
        "materials.bad.crystal",
        "materials.bad.stack[0].material",
    ):
        assert expected in message


def test_duplicate_toml_definition_is_material_config_error(tmp_path):
    from cxr_mc.materials import MaterialConfigError, load_material_catalog

    path = _write_catalog(tmp_path, "schema_version=1\nschema_version=1\n")
    with pytest.raises(MaterialConfigError, match="Cannot overwrite a value"):
        load_material_catalog(path)


def test_runnable_crystal_must_use_supported_transport_elements(tmp_path):
    from cxr_mc.materials import MaterialConfigError, load_material_catalog

    text = _minimal_catalog(
        crystal="lif",
        material_rows="""
[materials.lif]
label = "LiF"
profile = "base"
""",
    )
    with pytest.raises(MaterialConfigError, match="unsupported transport elements.*F.*Li"):
        load_material_catalog(_write_catalog(tmp_path, text))


def test_missing_mott_tables_warn_without_rejecting_catalog(tmp_path, caplog):
    from cxr_mc.materials import load_material_catalog

    text = _minimal_catalog(
        crystal="ws2",
        material_rows="""
[materials.ws2]
label = "WS2"
profile = "base"
""",
    )
    catalog = load_material_catalog(_write_catalog(tmp_path, text))

    assert catalog.material_keys == ("ws2",)
    assert "no Mott transport table for W" in caplog.text


@pytest.fixture(scope="module")
def legacy_catalog_golden():
    from cxr_mc.materials import CRYSTALS
    from cxr_mc.materials.registry import CRYSTAL_PARAMS, MATERIAL_CONFIGS

    return CRYSTALS, CRYSTAL_PARAMS, MATERIAL_CONFIGS


def _sorted_basis(basis):
    return sorted(((element, tuple(position)) for element, position in basis), key=lambda x: x)


def test_packaged_catalog_matches_all_legacy_crystal_and_scan_values(legacy_catalog_golden):
    from cxr_mc.materials import CATALOG

    legacy_crystals, legacy_crystal_params, legacy_materials = legacy_catalog_golden
    assert tuple(CATALOG.crystals) == tuple(legacy_crystals)
    assert CATALOG.material_keys == tuple(legacy_materials)

    for key, legacy in legacy_crystals.items():
        actual = CATALOG.crystal(key)
        old_lattice = legacy["lattice"]
        if old_lattice["system"] == "cubic":
            expected = (old_lattice["a"],) * 3 + (90.0, 90.0, 90.0)
        elif old_lattice["system"] == "hexagonal":
            expected = (old_lattice["a"], old_lattice["a"], old_lattice["c"], 90.0, 90.0, 120.0)
        elif old_lattice["system"] == "orthorhombic":
            expected = (
                old_lattice["a"], old_lattice["b"], old_lattice["c"], 90.0, 90.0, 90.0
            )
        else:
            expected = tuple(old_lattice[name] for name in ("a", "b", "c", "alpha", "beta", "gamma"))
        np.testing.assert_allclose(
            tuple(actual.lattice[name] for name in ("a", "b", "c", "alpha", "beta", "gamma")),
            expected,
            rtol=0.0,
            atol=1e-8,
        )
        assert actual.V_cell == pytest.approx(legacy["V_cell"], rel=2e-15)
        for (actual_el, actual_pos), (old_el, old_pos) in zip(
            _sorted_basis(actual.basis), _sorted_basis(legacy["basis"]), strict=True
        ):
            assert actual_el == old_el
            np.testing.assert_allclose(actual_pos, old_pos, rtol=0.0, atol=1e-8)
        assert actual.mosaic_fwhm_deg == legacy["mosaic_fwhm_deg"]
        expected_counts = {}
        for element, _ in legacy["basis"]:
            expected_counts[element] = expected_counts.get(element, 0) + 1
        assert dict(actual.composition) == pytest.approx(
            {element: count / legacy["V_cell"] for element, count in expected_counts.items()}
        )

    assert set(legacy_crystal_params) == set(CATALOG.crystals) - {"lif"}
    for key, legacy in legacy_crystal_params.items():
        actual = CATALOG.crystal(key)
        assert actual.B_ang2 == legacy["B_ang2"]
        assert actual.beam_uvw == legacy["beam_uvw"]
        np.testing.assert_array_equal(actual.E_grid, legacy["E_grid"])
        assert list(actual.hkl_list) == legacy.get("hkl_list", [])
        assert actual.hkl_reason == legacy.get("hkl_list_reason")

    for key, legacy in legacy_materials.items():
        actual = CATALOG.material(key)
        assert actual.label == legacy["label"]
        assert actual.crystal_key == legacy.get("crystal", key)
        for grid_key in (
            "thickness_ang", "energy_keV", "tilt_deg", "tilt_azim_deg", "E_grid_line", "E_grid_brem"
        ):
            np.testing.assert_array_equal(getattr(actual.scan, grid_key), np.atleast_1d(legacy[grid_key]))
        assert actual.substrate == legacy.get("substrate")
        assert [(layer.material, layer.thickness_ang, layer.beam_uvw, layer.azimuth_deg) for layer in actual.stack] == [
            (layer.material, layer.thickness_ang, layer.beam_uvw, layer.azimuth_deg)
            for layer in legacy.get("stack", ())
        ]


def test_catalog_cif_data_preserves_representative_crystal_physics(monkeypatch):
    from cxr_mc.materials import CATALOG
    from cxr_mc.materials import crystal as crystal_module

    for key, hkl in (("diamond", (1, 1, 1)), ("hbn", (0, 0, 2)), ("mose2", (1, 0, 0))):
        expected_S, expected_g = crystal_module.structure_factor(key, hkl, 1000.0, B_ang2=0.6)
        spec = CATALOG.crystal(key)
        monkeypatch.setitem(
            crystal_module.CRYSTALS,
            key,
            {"lattice": spec.lattice, "basis": spec.basis, "V_cell": spec.V_cell},
        )
        actual_S, actual_g = crystal_module.structure_factor(key, hkl, 1000.0, B_ang2=0.6)
        assert actual_g == pytest.approx(expected_g, rel=2e-8)
        assert actual_S == pytest.approx(expected_S, rel=2e-7, abs=2e-7)

    for key in ("diamond", "mose2"):
        monkeypatch.undo()
        expected = crystal_module.dominant_reflections(key, n_families=2, B_ang2=0.6)
        spec = CATALOG.crystal(key)
        monkeypatch.setitem(
            crystal_module.CRYSTALS,
            key,
            {"lattice": spec.lattice, "basis": spec.basis, "V_cell": spec.V_cell},
        )
        assert crystal_module.dominant_reflections(key, n_families=2, B_ang2=0.6) == expected
