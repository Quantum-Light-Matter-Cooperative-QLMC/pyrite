"""Focused tests for the immutable TOML material catalog."""

import hashlib
import json
import tomllib
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


PER_BEAM_ENTRIES = """{ energy_keV = 25.0, grid = { linspace = { start = 10.0, stop = 58.0, num = 17, endpoint = true } } },
  { energy_keV = 30.0, grid = { values = [20.0, 23.0, 26.0] } }"""


def _catalog_with_per_beam_line_grids() -> str:
    return _minimal_catalog(
        material_rows="""
[materials.sample]
label = "sample"
profile = "base"
crystal = "mos2"
"""
    ).replace(
        "E_grid_line = { arange = { start = 50.0, stop = 60.0, step = 2.0 } }",
        f"E_grid_line_by_energy = [\n  {PER_BEAM_ENTRIES},\n]",
    )


def test_per_beam_line_grids_are_exact_read_only_and_projected(tmp_path, monkeypatch):
    from cxr_mc import config
    from cxr_mc.materials import load_material_catalog

    catalog = load_material_catalog(_write_catalog(tmp_path, _catalog_with_per_beam_line_grids()))
    scan = catalog.material("sample").scan

    assert scan.E_grid_line is None
    assert tuple(scan.E_grid_line_by_energy) == (25.0, 30.0)
    np.testing.assert_array_equal(scan.E_grid_line_by_energy[25.0], np.linspace(10.0, 58.0, 17))
    with pytest.raises(TypeError):
        scan.E_grid_line_by_energy[25.0] = np.array([1.0])
    with pytest.raises(ValueError):
        scan.E_grid_line_by_energy[25.0][0] = 1.0

    monkeypatch.setattr(config, "CATALOG", catalog)
    grid = config.material_grid("sample")
    sweep = config.material_sweep("sample")
    trajectory = config.trajectory_sweep("sample")
    assert grid["E_grid_line"] is None
    assert grid["E_grid_line_by_energy"] is scan.E_grid_line_by_energy
    assert sweep.E_grid_line is None
    assert sweep.E_grid_line_by_energy is scan.E_grid_line_by_energy
    assert trajectory.E_grid_line is None
    assert trajectory.E_grid_line_by_energy is scan.E_grid_line_by_energy


def test_fixed_material_line_grid_overrides_profile_mapping(tmp_path):
    from cxr_mc.materials import load_material_catalog

    text = _catalog_with_per_beam_line_grids().replace(
        'profile = "base"',
        'profile = "base"\nE_grid_line = { values = [75.0, 78.0] }',
        1,
    )
    scan = load_material_catalog(_write_catalog(tmp_path, text)).material("sample").scan
    np.testing.assert_array_equal(scan.E_grid_line, [75.0, 78.0])
    assert scan.E_grid_line_by_energy is None


def test_material_scan_overrides_apply_bespoke_line_and_brem_grids(tmp_path):
    # The key enabling fact for bespoke per-material grids: a [materials.X] block
    # may override BOTH E_grid_line_by_energy and E_grid_brem on top of its
    # profile, and the resolved ScanSpec carries the material's own values -- no
    # catalog schema change is needed to give each material a bespoke grid.
    from cxr_mc.materials import load_material_catalog

    material_line = (
        "{ energy_keV = 25.0, grid = { values = [11.0, 14.0] } },\n  "
        "{ energy_keV = 30.0, grid = { values = [40.0, 44.0] } }"
    )
    text = _catalog_with_per_beam_line_grids().replace(
        'profile = "base"',
        'profile = "base"\n'
        f"E_grid_line_by_energy = [\n  {material_line},\n]\n"
        "E_grid_brem = { values = [7.0, 8.0, 9.0] }",
        1,
    )
    scan = load_material_catalog(_write_catalog(tmp_path, text)).material("sample").scan

    # brem override wins over the profile's E_grid_brem = 0.0
    np.testing.assert_array_equal(scan.E_grid_brem, [7.0, 8.0, 9.0])
    # line-by-energy override wins over the profile's per-beam mapping
    np.testing.assert_array_equal(scan.E_grid_line_by_energy[25.0], [11.0, 14.0])
    np.testing.assert_array_equal(scan.E_grid_line_by_energy[30.0], [40.0, 44.0])


@pytest.mark.parametrize(
    ("replacement", "error_path"),
    [
        (
            "{ energy_keV = 25.0, grid = 50.0 },\n  { energy_keV = 25.0, grid = 60.0 }",
            "E_grid_line_by_energy[1].energy_keV",
        ),
        ("{ energy_keV = 25.0, grid = 50.0 }", "E_grid_line_by_energy"),
        (
            "{ energy_keV = 25.0, grid = 50.0 },\n  "
            "{ energy_keV = 30.0, grid = 60.0 },\n  "
            "{ energy_keV = 40.0, grid = 70.0 }",
            "E_grid_line_by_energy[2].energy_keV",
        ),
    ],
)
def test_per_beam_line_grid_keys_match_beam_energies(tmp_path, replacement, error_path):
    from cxr_mc.materials import MaterialConfigError, load_material_catalog

    text = _catalog_with_per_beam_line_grids().replace(PER_BEAM_ENTRIES, replacement)
    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))
    assert error_path in str(caught.value)


def test_per_beam_line_grid_duplicate_is_reported_after_invalid_grid(tmp_path):
    from cxr_mc.materials import MaterialConfigError, load_material_catalog

    replacement = (
        "{ energy_keV = 25.0, grid = { values = [] } },\n  "
        "{ energy_keV = 25.0, grid = 60.0 },\n  "
        "{ energy_keV = 30.0, grid = 70.0 }"
    )
    text = _catalog_with_per_beam_line_grids().replace(PER_BEAM_ENTRIES, replacement)

    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))

    assert any(
        "E_grid_line_by_energy[0].grid: grid must be nonempty" in error
        for error in caught.value.errors
    )
    assert any(
        "E_grid_line_by_energy[1].energy_keV: duplicates beam energy 25" in error
        for error in caught.value.errors
    )


def test_bundled_crystal_validation_ids_are_ledgered():
    from cxr_mc import DATA_DIR

    with (DATA_DIR / "materials.toml").open("rb") as stream:
        raw = tomllib.load(stream)
    validation_ids = {
        row["validation_id"].strip()
        for row in raw["crystals"].values()
        if isinstance(row.get("validation_id"), str) and row["validation_id"].strip()
    }
    ledger = (Path(__file__).parents[1] / "docs" / "physics-validation-ledger.md").read_text()

    missing = sorted(
        validation_id for validation_id in validation_ids if f"| `{validation_id}` |" not in ledger
    )
    assert not missing, f"catalog validation IDs missing from ledger: {missing}"


def test_packaged_catalog_exposes_frozen_ordered_public_api():
    from cxr_mc.materials import CATALOG, MaterialCatalog

    assert isinstance(CATALOG, MaterialCatalog)
    assert len(CATALOG.crystals) == 48
    assert len(CATALOG.materials) == 49
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
        [
            1000.0,
            5000.0,
            10000.0,
            40000.0,
            100000.0,
            200000.0,
            500000.0,
            1000000.0,
            10000000.0,
        ],
    )


def test_standard_profile_uses_requested_angles_energies_and_line_grids():
    from cxr_mc.materials import CATALOG

    expected_bounds = {
        30.0: (10.0, 2500.0),
        50.0: (10.0, 3000.0),
        100.0: (50.0, 9000.0),
        150.0: (50.0, 11800.0),
        200.0: (50.0, 13600.0),
        250.0: (50.0, 14800.0),
        300.0: (50.0, 16400.0),
    }
    # silicon carries the standard profile unmodified (no per-material overrides),
    # so it exposes the profile's requested angles, energies, and line grids. The
    # bespoke crystals (hopg/diamond/wse2/mose2) override the line grids and are
    # covered by test_material_scan_overrides_apply_bespoke_line_and_brem_grids.
    scan = CATALOG.material("silicon").scan
    np.testing.assert_array_equal(scan.energy_keV, list(expected_bounds))
    np.testing.assert_array_equal(scan.tilt_deg, [5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0])
    np.testing.assert_array_equal(
        scan.tilt_azim_deg, [95.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0]
    )
    assert scan.E_grid_line is None
    assert tuple(scan.E_grid_line_by_energy) == tuple(expected_bounds)
    for energy, (start, stop) in expected_bounds.items():
        grid = scan.E_grid_line_by_energy[energy]
        assert grid[0] == start
        assert grid[-1] == stop
        spacing = np.diff(grid)
        assert np.all(spacing == pytest.approx(spacing[0]))
        assert spacing[0] == pytest.approx(3.0, abs=0.002)


def test_exposed_arrays_cannot_have_writes_reenabled():
    from cxr_mc.materials import CATALOG

    arrays = []
    for crystal in CATALOG.crystals.values():
        if crystal.E_grid is not None:
            arrays.append(crystal.E_grid)
        arrays.extend(position for _, position in crystal.basis)
    for material in CATALOG.materials.values():
        arrays.extend(
            (
                material.scan.thickness_ang,
                material.scan.energy_keV,
                material.scan.tilt_deg,
                material.scan.tilt_azim_deg,
                material.scan.E_grid_brem,
            )
        )
        if material.scan.E_grid_line is not None:
            arrays.append(material.scan.E_grid_line)
        if material.scan.E_grid_line_by_energy is not None:
            arrays.extend(material.scan.E_grid_line_by_energy.values())
        if material.scan.thickness_layers is not None:
            arrays.append(material.scan.thickness_layers)

    for values in arrays:
        try:
            with pytest.raises(ValueError):
                values.flags.writeable = True
        finally:
            values.flags.writeable = False
        with pytest.raises(ValueError):
            values[0] = -1.0


@pytest.mark.parametrize("version", ["true", "1.0"])
def test_schema_version_requires_integer_one(tmp_path, version):
    from cxr_mc.materials import MaterialConfigError, load_material_catalog

    text = _minimal_catalog().replace("schema_version = 1", f"schema_version = {version}")
    with pytest.raises(MaterialConfigError, match="schema_version"):
        load_material_catalog(_write_catalog(tmp_path, text))


@pytest.mark.parametrize(
    ("valid", "invalid"),
    [
        ("energy_keV = { values = [25.0, 30.0] }", "energy_keV = { values = [true] }"),
        (
            "energy_keV = { values = [25.0, 30.0] }",
            'energy_keV = { values = ["twenty-five"] }',
        ),
        (
            "E_grid_line = { arange = { start = 50.0, stop = 60.0, step = 2.0 } }",
            "E_grid_line = { arange = { start = true, stop = 60.0, step = 2.0 } }",
        ),
        (
            "E_grid_line = { arange = { start = 50.0, stop = 60.0, step = 2.0 } }",
            "E_grid_line = { arange = { start = 50.0, stop = 60.0, step = false } }",
        ),
        (
            "tilt_deg = { linspace = { start = 0.0, stop = 80.0, num = 3, endpoint = false } }",
            "tilt_deg = { linspace = { start = 0.0, stop = 80.0, num = 2.5, endpoint = false } }",
        ),
        (
            "tilt_deg = { linspace = { start = 0.0, stop = 80.0, num = 3, endpoint = false } }",
            "tilt_deg = { linspace = { start = 0.0, stop = 80.0, num = inf, endpoint = false } }",
        ),
        (
            "tilt_deg = { linspace = { start = 0.0, stop = 80.0, num = 3, endpoint = false } }",
            "tilt_deg = { linspace = { start = 0.0, stop = 80.0, num = 3, endpoint = 1 } }",
        ),
        (
            "tilt_azim_deg = 0.0",
            "tilt_azim_deg = { logspace = { start = 0.0, stop = 2.0, num = 3, base = false } }",
        ),
    ],
)
def test_grid_descriptor_types_are_strict_and_errors_are_wrapped(tmp_path, valid, invalid):
    from cxr_mc.materials import MaterialConfigError, load_material_catalog

    text = _minimal_catalog().replace(valid, invalid)
    with pytest.raises(MaterialConfigError, match="profiles.base"):
        load_material_catalog(_write_catalog(tmp_path, text))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tilt_deg", "-0.1"),
        ("tilt_deg", "90.0"),
        ("tilt_azim_deg", "-0.1"),
        ("tilt_azim_deg", "360.1"),
    ],
)
def test_scan_angles_stay_in_physical_domains(tmp_path, field, value):
    from cxr_mc.materials import MaterialConfigError, load_material_catalog

    material_rows = f"""
[materials.mos2]
label = "MoS2"
profile = "base"
{field} = {value}
"""
    with pytest.raises(MaterialConfigError, match=rf"materials\.mos2\.scan\.{field}"):
        load_material_catalog(
            _write_catalog(tmp_path, _minimal_catalog(material_rows=material_rows))
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


def test_catalog_scalar_and_logspace_energy_grids_reach_runner_exactly(tmp_path, monkeypatch):
    from cxr_mc import config
    from cxr_mc import sweep as sweep_module
    from cxr_mc.materials import load_material_catalog
    from cxr_mc.montecarlo import runner

    text = _minimal_catalog(
        material_rows="""
[materials.sample]
label = "sample"
profile = "base"
crystal = "mos2"
""",
    )
    text = (
        text.replace(
            "E_grid_line = { arange = { start = 50.0, stop = 60.0, step = 2.0 } }",
            "E_grid_line = 75.0",
        )
        .replace(
            "E_grid_brem = 0.0",
            "E_grid_brem = { logspace = { start = 1.0, stop = 3.0, num = 3 } }",
        )
        # the base fixture's tilt_deg spans 0 deg, which build_cases now rejects
        # (the azim-rework banned-angle guard); pin legal emission angles so this
        # test exercises the grid passthrough, not the guard.
        .replace(
            "tilt_deg = { linspace = { start = 0.0, stop = 80.0, num = 3, endpoint = false } }",
            "tilt_deg = { values = [5.0, 45.0] }",
        )
    )
    catalog = load_material_catalog(_write_catalog(tmp_path, text))
    monkeypatch.setattr(config, "CATALOG", catalog)
    monkeypatch.setattr(sweep_module, "CATALOG", catalog)

    case = sweep_module.build_cases(
        config.material_sweep("sample"), n_electrons=1, n_electrons_brem=1
    )[0]
    monkeypatch.setattr(runner, "simulate_trajectories", lambda *args, **kwargs: {})

    transport = runner._transport_case(case)

    np.testing.assert_array_equal(transport["E_grid"], [75.0])
    np.testing.assert_array_equal(transport["E_brem"], [10.0, 100.0, 1000.0])


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

    invalid = valid.replace("[[0, 0, 2]]", "[[0, 0, -2]]").replace('hkl_reason = "basal cut"', "")
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


def test_crystal_requires_exactly_one_orientation(tmp_path):
    from cxr_mc.materials import MaterialConfigError, load_material_catalog

    material = """
[materials.mos2]
label = "MoS2"
profile = "base"
"""
    neither = _minimal_catalog(material_rows=material).replace("beam_uvw = [0, 0, 2]\n", "")
    with pytest.raises(
        MaterialConfigError, match="requires exactly one of beam_uvw or surface_hkl"
    ):
        load_material_catalog(_write_catalog(tmp_path, neither))

    both = _minimal_catalog(material_rows=material).replace(
        "beam_uvw = [0, 0, 2]", "beam_uvw = [0, 0, 2]\nsurface_hkl = [0, 0, 1]"
    )
    with pytest.raises(
        MaterialConfigError, match="requires exactly one of beam_uvw or surface_hkl"
    ):
        load_material_catalog(_write_catalog(tmp_path, both))


def test_crystal_accepts_reciprocal_surface_orientation(tmp_path):
    from cxr_mc.materials import load_material_catalog

    text = _minimal_catalog(
        material_rows="""
[materials.mos2]
label = "MoS2"
profile = "base"
""",
    ).replace("beam_uvw = [0, 0, 2]", "surface_hkl = [2, 0, -1]")
    spec = load_material_catalog(_write_catalog(tmp_path, text)).crystal("mos2")

    assert spec.beam_uvw is None
    assert spec.surface_hkl == (2, 0, -1)


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


def _array_fingerprint(values):
    array = np.asarray(values, dtype="<f8")
    return {
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


@pytest.fixture(scope="module")
def serialized_catalog_golden():
    path = Path(__file__).parent / "data" / "material_catalog_golden.json"
    return json.loads(path.read_text())


def test_packaged_catalog_matches_independent_serialized_golden(serialized_catalog_golden):
    from cxr_mc.materials import CATALOG

    golden = serialized_catalog_golden
    assert tuple(CATALOG.crystals) == tuple(golden["crystal_keys"])
    assert tuple(key for key, spec in CATALOG.crystals.items() if spec.E_grid is not None) == tuple(
        golden["configured_crystal_keys"]
    )
    assert CATALOG.material_keys == tuple(golden["material_keys"])

    for key, expected in golden["crystals"].items():
        actual = CATALOG.crystal(key)
        assert dict(actual.lattice) == expected["lattice"]
        assert actual.V_cell == pytest.approx(expected["V_cell"], rel=2e-15)
        assert actual.mosaic_fwhm_deg == expected["mosaic_fwhm_deg"]
        assert [element for element, _ in actual.composition] == [
            item[0] for item in expected["composition"]
        ]
        np.testing.assert_allclose(
            [density for _, density in actual.composition],
            [item[1] for item in expected["composition"]],
            rtol=2e-15,
        )
        assert [element for element, _ in actual.basis] == [site[0] for site in expected["basis"]]
        np.testing.assert_allclose(
            np.asarray([position for _, position in actual.basis]),
            np.asarray([site[1] for site in expected["basis"]]),
            rtol=0.0,
            atol=1e-12,
        )
        config = expected["config"]
        assert actual.B_ang2 == config["B_ang2"]
        beam_uvw = list(actual.beam_uvw) if actual.beam_uvw is not None else None
        surface_hkl = list(actual.surface_hkl) if actual.surface_hkl is not None else None
        assert beam_uvw == config["beam_uvw"]
        assert surface_hkl == config["surface_hkl"]
        assert list(map(list, actual.hkl_list)) == config["hkl_list"]
        assert actual.hkl_reason == config["hkl_reason"]
        assert actual.layers_per_cell == config["layers_per_cell"]
        if config["E_grid"] is None:
            assert actual.E_grid is None
        else:
            assert _array_fingerprint(actual.E_grid) == config["E_grid"]

    assert [element for element, _ in CATALOG.media["sio2"].composition] == [
        item[0] for item in golden["media"]["sio2"]
    ]
    np.testing.assert_allclose(
        [density for _, density in CATALOG.media["sio2"].composition],
        [item[1] for item in golden["media"]["sio2"]],
        rtol=0.0,
        atol=0.0,
    )
    for key, expected in golden["materials"].items():
        actual = CATALOG.material(key)
        assert actual.label == expected["label"]
        assert actual.profile == expected["profile"]
        assert actual.crystal_key == expected["crystal_key"]
        assert actual.substrate == expected["substrate"]
        assert [
            {
                "material": layer.material,
                "thickness_ang": layer.thickness_ang,
                "beam_uvw": list(layer.beam_uvw) if layer.beam_uvw else None,
                "azimuth_deg": layer.azimuth_deg,
            }
            for layer in actual.stack
        ] == expected["stack"]
        scan_golden = expected["scan"]
        for grid_name in (
            "thickness_ang",
            "energy_keV",
            "tilt_deg",
            "tilt_azim_deg",
            "E_grid_brem",
        ):
            fingerprint = scan_golden[grid_name]
            assert _array_fingerprint(getattr(actual.scan, grid_name)) == fingerprint
        assert actual.scan.E_grid_line is None
        expected_line_grids = scan_golden["E_grid_line_by_energy"]
        assert tuple(map(str, actual.scan.E_grid_line_by_energy)) == tuple(expected_line_grids)
        for energy, fingerprint in expected_line_grids.items():
            assert (
                _array_fingerprint(actual.scan.E_grid_line_by_energy[float(energy)]) == fingerprint
            )

    special = golden["special_grids"]
    np.testing.assert_array_equal(
        CATALOG.material("hbn").scan.thickness_ang, special["hbn_thickness_ang"]
    )
    np.testing.assert_array_equal(CATALOG.material("hbn").scan.tilt_deg, special["hbn_tilt_deg"])
    mote2_grid = special["mote2_E_grid_descriptor"]
    np.testing.assert_array_equal(
        CATALOG.crystal("mote2").E_grid,
        np.arange(mote2_grid["start"], mote2_grid["stop"], mote2_grid["step"]),
    )
    np.testing.assert_array_equal(
        CATALOG.material("mote2").scan.tilt_deg, special["mote2_tilt_deg"]
    )
    np.testing.assert_array_equal(
        CATALOG.material("mote2").scan.tilt_azim_deg, special["mote2_tilt_azim_deg"]
    )


def test_catalog_resolves_stack_layers_to_serialized_physical_data(serialized_catalog_golden):
    from cxr_mc.materials import CATALOG

    expected_stacks = serialized_catalog_golden["resolved_stacks"]
    assert tuple(expected_stacks) == tuple(
        key for key, material in CATALOG.materials.items() if material.stack
    )
    for material_key, expected in expected_stacks.items():
        layers = CATALOG.resolve_stack(material_key, expected["film_thickness_ang"])
        actual = [
            {
                "key": layer["key"],
                "z_top_ang": layer["z_top_ang"],
                "z_bottom_ang": layer["z_bottom_ang"],
                "composition": [list(item) for item in layer["composition"]],
                "beam_uvw": list(layer["beam_uvw"]) if layer["beam_uvw"] else None,
                "azimuth_deg": layer["azimuth_deg"],
            }
            for layer in layers
        ]
        assert actual == expected["layers"]


def test_resolved_stack_inherits_surface_and_direct_override_clears_it(tmp_path):
    from cxr_mc.materials import load_material_catalog

    text = _minimal_catalog(
        material_rows="""
[materials.sample]
label = "sample"
profile = "base"
crystal = "mos2"
stack = [
  { material = "mos2", thickness_ang = 10.0 },
  { material = "mos2", thickness_ang = 20.0, beam_uvw = [0, 1, 0] },
]
""",
    ).replace("beam_uvw = [0, 0, 2]", "surface_hkl = [2, 0, -1]")
    layers = load_material_catalog(_write_catalog(tmp_path, text)).resolve_stack("sample", 5.0)

    assert [(layer["beam_uvw"], layer["surface_hkl"]) for layer in layers] == [
        (None, (2, 0, -1)),
        (None, (2, 0, -1)),
        ((0, 1, 0), None),
    ]


def test_catalog_matches_serialized_physics_for_every_crystal(serialized_catalog_golden):
    from cxr_mc.materials import CATALOG
    from cxr_mc.materials import crystal as crystal_module

    for key, expected in serialized_catalog_golden["crystals"].items():
        spec = CATALOG.crystal(key)
        entry = crystal_module.CRYSTALS[key]
        assert entry["lattice"] is spec.lattice
        assert entry["basis"] is spec.basis
        assert entry["V_cell"] == spec.V_cell
        assert entry["mosaic_fwhm_deg"] == spec.mosaic_fwhm_deg

        physics = expected["physics"]
        hkl = tuple(physics["hkl"])
        structure, g_mag = crystal_module.structure_factor(key, hkl, 1000.0, B_ang2=spec.B_ang2)
        assert g_mag == pytest.approx(physics["g_mag"], rel=2e-12)
        assert structure.real == pytest.approx(physics["structure_factor"][0], rel=2e-12, abs=2e-12)
        assert structure.imag == pytest.approx(physics["structure_factor"][1], rel=2e-12, abs=2e-12)
        assert [
            list(hkl)
            for hkl in crystal_module.dominant_reflections(key, n_families=2, B_ang2=spec.B_ang2)
        ] == physics["dominant_reflections"]
