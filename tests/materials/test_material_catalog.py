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


def _minimal_catalog(
    *, crystal: str = "mos2", material_rows: str = "", profile_extra: str = ""
) -> str:
    return f"""
schema_version = 1
[profiles.standard]
thickness_ang = {{ logspace = {{ start = 2.0, stop = 3.0, num = 2 }} }}
energy_keV = {{ values = [25.0, 30.0] }}
tilt_deg = {{ linspace = {{ start = 5.0, stop = 80.0, num = 3, endpoint = false }} }}
tilt_azim_deg = 0.0
E_grid_line = {{ arange = {{ start = 50.0, stop = 60.0, step = 2.0 }} }}
E_grid_brem = 0.0
{profile_extra}
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


PER_BEAM_ENTRIES = (
    "{ energy_keV = 25.0, grid = { linspace = { start = 10.0, stop = 58.0, num = 17, "
    'endpoint = true } }, source = "derived" },\n'
    '  { energy_keV = 30.0, grid = { values = [20.0, 23.0, 26.0] }, source = "derived" }'
)

_NO_FLAT_LINE_GRID = "E_grid_line = { arange = { start = 50.0, stop = 60.0, step = 2.0 } }\n"


def _catalog_with_per_beam_line_grids(*, entries: str = PER_BEAM_ENTRIES) -> str:
    """A catalog where material "sample" resolves its line grid from its own
    ``[energy_grids.sample]`` store entry (decision 3,
    docs/adr/0005-energy-grid-schema-decisions.md), not a profile-level grid."""
    base = _minimal_catalog(
        material_rows="""
[materials.sample]
label = "sample"
crystal = "mos2"
"""
    ).replace(_NO_FLAT_LINE_GRID, "")
    return base + f"\n[energy_grids.sample]\nline_by_energy = [\n  {entries},\n]\n"


def test_per_beam_line_grids_are_exact_read_only_and_projected(tmp_path, monkeypatch):
    from pyrite.campaign import config
    from pyrite.materials import load_material_catalog

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
    assert sweep.detector.energy_bins.line is None
    assert sweep.detector.energy_bins.line_by_energy is scan.E_grid_line_by_energy
    assert trajectory.detector.energy_bins.line is None
    assert trajectory.detector.energy_bins.line_by_energy is scan.E_grid_line_by_energy


def test_fixed_material_line_grid_overrides_profile_mapping(tmp_path):
    from pyrite.materials import load_material_catalog

    text = (
        _catalog_with_per_beam_line_grids()
        + "\n[profiles.standard.overrides.sample]\n"
        + "E_grid_line = { values = [75.0, 78.0] }\n"
    )
    scan = load_material_catalog(_write_catalog(tmp_path, text)).material("sample").scan
    np.testing.assert_array_equal(scan.E_grid_line, [75.0, 78.0])
    assert scan.E_grid_line_by_energy is None


def test_material_scan_overrides_apply_bespoke_line_and_brem_grids(tmp_path):
    # A material's own [energy_grids.<material>] entry is authoritative and
    # replaces the shared default wholesale (decision 3); its E_grid_brem
    # override still lives under [profiles.standard.overrides.<material>], a
    # separate axis untouched by the line-grid store.
    from pyrite.materials import load_material_catalog

    material_line = (
        '{ energy_keV = 25.0, grid = { values = [11.0, 14.0] }, source = "derived" },\n  '
        '{ energy_keV = 30.0, grid = { values = [40.0, 44.0] }, source = "derived" }'
    )
    text = (
        _catalog_with_per_beam_line_grids(entries=material_line)
        + "\n[profiles.standard.overrides.sample]\n"
        + "E_grid_brem = { values = [7.0, 8.0, 9.0] }\n"
    )
    scan = load_material_catalog(_write_catalog(tmp_path, text)).material("sample").scan

    # brem override wins over the profile's E_grid_brem = 0.0
    np.testing.assert_array_equal(scan.E_grid_brem, [7.0, 8.0, 9.0])
    np.testing.assert_array_equal(scan.E_grid_line_by_energy[25.0], [11.0, 14.0])
    np.testing.assert_array_equal(scan.E_grid_line_by_energy[30.0], [40.0, 44.0])


@pytest.mark.parametrize(
    ("replacement", "error_path"),
    [
        (
            "{ energy_keV = 25.0, grid = 50.0 },\n  { energy_keV = 25.0, grid = 60.0 }",
            "energy_grids.sample.line_by_energy[1].energy_keV",
        ),
        ("{ energy_keV = 25.0, grid = 50.0 }", "materials.sample.scan"),
    ],
)
def test_per_beam_line_grid_keys_match_beam_energies(tmp_path, replacement, error_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = _catalog_with_per_beam_line_grids().replace(PER_BEAM_ENTRIES, replacement)
    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))
    assert error_path in str(caught.value)


def test_per_beam_line_grid_duplicate_is_reported_after_invalid_grid(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    replacement = (
        '{ energy_keV = 25.0, grid = { values = [] }, source = "derived" },\n  '
        '{ energy_keV = 25.0, grid = 60.0, source = "derived" },\n  '
        '{ energy_keV = 30.0, grid = 70.0, source = "derived" }'
    )
    text = _catalog_with_per_beam_line_grids().replace(PER_BEAM_ENTRIES, replacement)

    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))

    assert any(
        "energy_grids.sample.line_by_energy[0].grid: grid must be nonempty" in error
        for error in caught.value.errors
    )
    assert any(
        "energy_grids.sample.line_by_energy[1].energy_keV: duplicates beam energy 25" in error
        for error in caught.value.errors
    )


def test_energy_grids_row_requires_a_valid_source(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    replacement = PER_BEAM_ENTRIES.replace('source = "derived"', 'source = "bogus"', 1)
    text = _catalog_with_per_beam_line_grids().replace(PER_BEAM_ENTRIES, replacement)

    with pytest.raises(
        MaterialConfigError, match=r"energy_grids\.sample\.line_by_energy\[0\]\.source"
    ):
        load_material_catalog(_write_catalog(tmp_path, text))


def test_energy_grids_store_may_hold_more_energies_than_a_material_needs(tmp_path):
    """Decision 3: the shared store may be a superset of what any one
    profile currently needs -- pruning a profile's energy_keV must never
    force deleting store rows. An unused row is simply not an error."""
    from pyrite.materials import load_material_catalog

    extra = (
        PER_BEAM_ENTRIES
        + ',\n  { energy_keV = 40.0, grid = { values = [1.0, 2.0] }, source = "derived" }'
    )
    text = _catalog_with_per_beam_line_grids(entries=extra)
    scan = load_material_catalog(_write_catalog(tmp_path, text)).material("sample").scan

    assert tuple(scan.E_grid_line_by_energy) == (25.0, 30.0)


def _catalog_with_artifact_ref(digest: str) -> str:
    text = _catalog_with_per_beam_line_grids().replace(_NO_FLAT_LINE_GRID, "")
    return text.replace(
        "[profiles.standard]\n",
        f'[profiles.standard]\nenergy_grid_refs = {{ sample = "{digest}" }}\n',
        1,
    )


def test_profile_artifact_ref_resolves_immutable_grid_and_brem(tmp_path):
    from pyrite.energy_grid import artifacts
    from pyrite.materials import load_material_catalog

    identity = artifacts.artifact_identity(
        "sample",
        [
            {"energy_keV": 25, "start_eV": 11, "stop_eV": 59, "num": 5},
            {"energy_keV": 30, "start_eV": 12, "stop_eV": 72, "num": 4},
            {"energy_keV": 40, "start_eV": 20, "stop_eV": 80, "num": 3},
        ],
        {"start_eV": 0, "stop_eV": 100, "step_eV": 20},
        [25, 30, 40],
    )
    stored = artifacts.write_artifact(tmp_path / "energy-grid-artifacts", identity)
    catalog = load_material_catalog(
        _write_catalog(tmp_path, _catalog_with_artifact_ref(stored.digest))
    )
    scan = catalog.material("sample").scan

    assert scan.energy_keV.tolist() == [25.0, 30.0, 40.0]
    assert scan.E_grid_line_by_energy is not None
    assert scan.E_grid_line_by_energy[25.0].tolist() == [11.0, 23.0, 35.0, 47.0, 59.0]
    assert scan.E_grid_brem.tolist() == [0.0, 20.0, 40.0, 60.0, 80.0]
    assert catalog.profile_energy_grid_ref("standard", "sample") == stored.digest
    assert catalog.resolved_energy_grid_refs == {"sample": stored.digest}


def test_profile_artifact_ref_matches_equivalent_legacy_resolution(tmp_path):
    from pyrite.energy_grid import artifacts
    from pyrite.materials import load_material_catalog

    legacy_text = (
        _catalog_with_per_beam_line_grids()
        + "\n[profiles.standard.overrides.sample]\n"
        + "E_grid_brem = { arange = { start = 0.0, stop = 100.0, step = 20.0 } }\n"
    )
    legacy = load_material_catalog(_write_catalog(tmp_path, legacy_text)).material("sample").scan
    identity = artifacts.artifact_identity(
        "sample",
        [
            {"energy_keV": 25, "start_eV": 10, "stop_eV": 58, "num": 17},
            {"energy_keV": 30, "start_eV": 20, "stop_eV": 26, "num": 3},
        ],
        {"start_eV": 0, "stop_eV": 100, "step_eV": 20},
        [25, 30],
    )
    stored = artifacts.write_artifact(tmp_path / "energy-grid-artifacts", identity)
    artifact_text = legacy_text.replace(
        "[profiles.standard]\n",
        f'[profiles.standard]\nenergy_grid_refs = {{ sample = "{stored.digest}" }}\n',
        1,
    )
    resolved = (
        load_material_catalog(_write_catalog(tmp_path, artifact_text)).material("sample").scan
    )

    np.testing.assert_array_equal(resolved.energy_keV, legacy.energy_keV)
    np.testing.assert_array_equal(resolved.E_grid_brem, legacy.E_grid_brem)
    assert resolved.E_grid_line_by_energy is not None
    assert legacy.E_grid_line_by_energy is not None
    for energy in legacy.E_grid_line_by_energy:
        np.testing.assert_array_equal(
            resolved.E_grid_line_by_energy[energy], legacy.E_grid_line_by_energy[energy]
        )


def test_profile_artifact_ref_rejects_missing_or_wrong_material(tmp_path):
    from pyrite.energy_grid import artifacts
    from pyrite.materials import MaterialConfigError, load_material_catalog

    missing = "a" * 64
    with pytest.raises(MaterialConfigError, match="artifact .* is missing"):
        load_material_catalog(_write_catalog(tmp_path, _catalog_with_artifact_ref(missing)))

    identity = artifacts.artifact_identity(
        "other",
        [
            {"energy_keV": 25, "start_eV": 10, "stop_eV": 60, "num": 6},
            {"energy_keV": 30, "start_eV": 10, "stop_eV": 70, "num": 7},
        ],
        {"stop_eV": 100, "step_eV": 20},
        [25, 30],
    )
    stored = artifacts.write_artifact(tmp_path / "energy-grid-artifacts", identity)
    with pytest.raises(MaterialConfigError, match="does not match ref key"):
        load_material_catalog(_write_catalog(tmp_path, _catalog_with_artifact_ref(stored.digest)))


def _catalog_with_default_store_only(*, entries: str = PER_BEAM_ENTRIES) -> str:
    """Material "sample" has no ``[energy_grids.sample]`` entry of its own; it
    resolves against the shared default keyed by its profile's name
    (``standard`` in Phase 1) -- the fallback ~30 bundled materials.toml
    materials that never diverged from the profile's own derived bounds use."""
    base = _minimal_catalog(
        material_rows="""
[materials.sample]
label = "sample"
crystal = "mos2"
"""
    ).replace(_NO_FLAT_LINE_GRID, "")
    return base + f"\n[energy_grids.standard]\nline_by_energy = [\n  {entries},\n]\n"


def test_material_without_own_store_entry_falls_back_to_shared_default(tmp_path):
    from pyrite.materials import load_material_catalog

    scan = (
        load_material_catalog(_write_catalog(tmp_path, _catalog_with_default_store_only()))
        .material("sample")
        .scan
    )
    assert tuple(scan.E_grid_line_by_energy) == (25.0, 30.0)
    np.testing.assert_array_equal(scan.E_grid_line_by_energy[25.0], np.linspace(10.0, 58.0, 17))


def test_missing_beam_energy_errors_without_default_store_coverage(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    only_25 = '{ energy_keV = 25.0, grid = { values = [1.0, 2.0] }, source = "derived" }'
    text = _catalog_with_default_store_only(entries=only_25)

    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))

    assert any("requires E_grid_line" in error for error in caught.value.errors)


def test_material_config_error_groups_identical_messages_across_materials(tmp_path):
    # A profile-wide setting invalid for every material (no E_grid_line, no
    # material store entry for the profile's beam energies) must not repeat
    # one near-duplicate line per material (TODO.md Bugs #3).
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = _minimal_catalog(
        material_rows="""
[materials.sample_a]
label = "sample_a"
crystal = "mos2"

[materials.sample_b]
label = "sample_b"
crystal = "mos2"
"""
    ).replace(_NO_FLAT_LINE_GRID, "")

    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))

    assert len(caught.value.errors) == 2
    message = str(caught.value)
    assert message.count("requires E_grid_line") == 1
    assert "2 paths (materials.sample_a.scan, materials.sample_b.scan)" in message
    assert (
        "run `pyrite material energy-grid derive "
        "--energy 25.0,30.0 --material sample_a,sample_b`"
    ) in message


def test_named_profile_scan_defaults_apply_only_to_members(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = _minimal_catalog(
        material_rows="""
[materials.sample_a]
label = "sample_a"
crystal = "mos2"

[materials.sample_b]
label = "sample_b"
crystal = "mos2"

[profiles.narrowed]
thickness_ang = 1000.0
energy_keV = 35.0
tilt_deg = 5.0
tilt_azim_deg = 95.0
E_grid_brem = 0.0
materials = ["sample_a"]
"""
    )

    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text), profile="narrowed")

    message = str(caught.value)
    assert "materials.sample_a.scan" in message
    assert "materials.sample_b.scan" not in message
    assert (
        "run `pyrite material energy-grid derive "
        "--energy 35.0 --material sample_a`"
    ) in message


def _catalog_with_two_profiles(tmp_path: Path) -> Path:
    """``narrowed`` restricts membership to "mos2"; ``standard`` (no
    ``materials`` row) allows every catalog material -- Phase 3 scan/submit
    ``--profile`` intersection semantics read this membership."""
    text = _minimal_catalog(
        material_rows="""
[materials.mos2]
label = "mos2"
crystal = "mos2"

[profiles.narrowed]
thickness_ang = { logspace = { start = 2.0, stop = 3.0, num = 2 } }
energy_keV = { values = [25.0, 30.0] }
tilt_deg = { linspace = { start = 5.0, stop = 80.0, num = 3, endpoint = false } }
tilt_azim_deg = 0.0
E_grid_line = { arange = { start = 50.0, stop = 60.0, step = 2.0 } }
E_grid_brem = 0.0
materials = ["mos2"]
"""
    )
    return _write_catalog(tmp_path, text)


def test_profile_names_and_memberships_are_exposed(tmp_path):
    from pyrite.materials import load_material_catalog

    catalog = load_material_catalog(_catalog_with_two_profiles(tmp_path))

    assert catalog.profile_names == ("standard", "narrowed")
    assert catalog.profile_memberships == {"narrowed": ("mos2",)}
    assert catalog.profile_materials("standard") is None
    assert catalog.profile_materials("narrowed") == ("mos2",)
    with pytest.raises(KeyError, match="unknown profile"):
        catalog.profile_materials("bogus")


def test_packaged_profiles_have_explicit_membership():
    from pyrite.materials import CATALOG

    assert all(CATALOG.profile_materials(name) is not None for name in CATALOG.profile_names)
    assert CATALOG.profile_materials("standard") == CATALOG.profile_materials("sub_100keV")
    assert CATALOG.profile_materials("high_energy") == ("tise2", "gep", "ges", "rese2")


def test_material_validation_metadata_is_typed_and_validated(tmp_path):
    from pyrite.materials import MaterialConfigError, MaterialValidationSpec, load_material_catalog

    text = _minimal_catalog(
        material_rows="""
[materials.sample]
label = "sample"
crystal = "mos2"
[materials.sample.validation]
crystal_database_match = "unverified"
"""
    )
    material = load_material_catalog(_write_catalog(tmp_path, text)).material("sample")
    assert material.validation == MaterialValidationSpec(crystal_database_match="unverified")

    invalid = text.replace(
        'crystal_database_match = "unverified"', 'crystal_database_match = "maybe"'
    )
    with pytest.raises(MaterialConfigError, match="must be 'verified' or 'unverified'"):
        load_material_catalog(_write_catalog(tmp_path, invalid))


def _catalog_with_standard_beam(beam_block: str) -> str:
    """The minimal catalog plus a ``[profiles.standard.beam]`` distribution
    block (appended after the material rows -- a valid out-of-order TOML
    sub-table of the already-opened ``[profiles.standard]``)."""
    return (
        _minimal_catalog(
            material_rows="""
[materials.mos2]
label = "mos2"
crystal = "mos2"
"""
        )
        + beam_block
    )


def test_profile_beam_block_decodes_and_reaches_material_sweep(tmp_path, monkeypatch):
    from pyrite.campaign import config
    from pyrite.materials import load_material_catalog

    text = _catalog_with_standard_beam(
        "\n[profiles.standard.beam]\n"
        "transverse_fwhm_mm = 0.5\n"
        "bunch_length_fs = 120.0\n"
        'long_shape = "uniform"\n'
        "rep_rate_hz = 1000.0\n"
        "bunch_charge_pc = 2.5\n"
    )
    catalog = load_material_catalog(_write_catalog(tmp_path, text))

    assert catalog.profile_beam("standard") == {
        "transverse_fwhm_mm": 0.5,
        "bunch_length_fs": 120.0,
        "long_shape": "uniform",
        "rep_rate_hz": 1000.0,
        "bunch_charge_pc": 2.5,
    }

    monkeypatch.setattr(config, "CATALOG", catalog)
    sweep = config.material_sweep("mos2")
    # The isotropic alias routes onto both transverse planes; distribution
    # fields apply; energy stays the per-material scan grid (not the beam block).
    assert sweep.beam.transverse_fwhm_x_mm == 0.5
    assert sweep.beam.transverse_fwhm_y_mm == 0.5
    assert sweep.beam.bunch_length_fs == 120.0
    assert sweep.beam.long_shape == "uniform"
    assert sweep.beam.rep_rate_hz == 1000.0
    assert sweep.beam.bunch_charge_pc == 2.5
    np.testing.assert_array_equal(sweep.beam.energy_keV, catalog.material("mos2").scan.energy_keV)


def test_profile_beam_block_rejects_energy_and_bad_values(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = _catalog_with_standard_beam(
        "\n[profiles.standard.beam]\n"
        "energy_keV = [40.0]\n"  # energy is NOT settable in the beam block
        "transverse_fwhm_x_mm = -1.0\n"  # must be finite positive
        'long_shape = "triangular"\n'  # not a known sampling law
        "long_offsets_fs = 5.0\n"  # must be an array
    )
    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))
    joined = "\n".join(caught.value.errors)
    assert "profiles.standard.beam.energy_keV: unknown key" in joined
    assert "profiles.standard.beam.transverse_fwhm_x_mm" in joined
    assert "profiles.standard.beam.long_shape" in joined
    assert "profiles.standard.beam.long_offsets_fs" in joined


def test_profile_beam_offsets_coerced_to_tuple_and_absent_profile_is_none(tmp_path):
    from pyrite.materials import load_material_catalog

    text = _catalog_with_standard_beam(
        "\n[profiles.standard.beam]\nlong_offsets_fs = [-10.0, 0.0, 10.0]\n"
    )
    catalog = load_material_catalog(_write_catalog(tmp_path, text))
    beam = catalog.profile_beam("standard")
    assert isinstance(beam["long_offsets_fs"], tuple)
    assert beam["long_offsets_fs"] == (-10.0, 0.0, 10.0)
    # A profile with no beam block -> None (the standard BeamSpec default applies).
    assert catalog.profile_beam("bogus") is None


def _named_beam_catalog(*, beam_ref: str, beams_block: str) -> str:
    """A minimal catalog whose ``standard`` profile attaches ``beam_ref`` by
    name, plus whatever ``[beams.*]`` rows ``beams_block`` defines."""
    return (
        _minimal_catalog(
            material_rows="""
[materials.mos2]
label = "mos2"
crystal = "mos2"
""",
            profile_extra=f'beam = "{beam_ref}"',
        )
        + beams_block
    )


def test_named_beam_reference_resolves_to_same_payload_as_inline_block(tmp_path):
    from pyrite.materials import load_material_catalog

    ref_text = _named_beam_catalog(
        beam_ref="rf_gun_200fs",
        beams_block=(
            '\n[beams.rf_gun_200fs]\nlabel = "RF gun, 200 fs"\n'
            "transverse_fwhm_mm = 0.5\n"
            "bunch_length_fs = 120.0\n"
            'long_shape = "uniform"\n'
            "rep_rate_hz = 1000.0\n"
            "bunch_charge_pc = 2.5\n"
        ),
    )
    inline_text = _catalog_with_standard_beam(
        "\n[profiles.standard.beam]\n"
        "transverse_fwhm_mm = 0.5\n"
        "bunch_length_fs = 120.0\n"
        'long_shape = "uniform"\n'
        "rep_rate_hz = 1000.0\n"
        "bunch_charge_pc = 2.5\n"
    )
    (tmp_path / "ref").mkdir()
    (tmp_path / "inline").mkdir()
    ref_catalog = load_material_catalog(_write_catalog(tmp_path / "ref", ref_text))
    inline_catalog = load_material_catalog(_write_catalog(tmp_path / "inline", inline_text))

    # Resolved payload is value-identical -- the name and label never leak into
    # the payload config.material_sweep actually reads (decision 3).
    assert ref_catalog.profile_beam("standard") == inline_catalog.profile_beam("standard")
    assert "label" not in ref_catalog.profile_beam("standard")

    # But the beam is independently visible as a named catalog object, label
    # included.
    assert ref_catalog.beam_keys == ("rf_gun_200fs",)
    assert ref_catalog.beams["rf_gun_200fs"]["label"] == "RF gun, 200 fs"


def test_named_beam_and_inline_block_together_is_a_decode_error(tmp_path):
    """Decision 4: a profile cannot spell both ``beam = "NAME"`` and an inline
    ``[profiles.NAME.beam]`` table. TOML's own duplicate-key rule (both
    spellings share the ``beam`` key) rejects this before catalog validation
    ever runs -- no bespoke mutual-exclusion check is needed."""
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = _named_beam_catalog(
        beam_ref="rf_gun_200fs",
        beams_block=(
            "\n[beams.rf_gun_200fs]\nrep_rate_hz = 1000.0\n"
            "\n[profiles.standard.beam]\nrep_rate_hz = 500.0\n"
        ),
    )
    with pytest.raises(MaterialConfigError):
        load_material_catalog(_write_catalog(tmp_path, text))


def test_named_beam_unknown_reference_errors(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = _named_beam_catalog(beam_ref="bogus", beams_block="")
    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))
    assert "profiles.standard.beam: unknown beam 'bogus'" in "\n".join(caught.value.errors)


def test_named_beam_round_trips_negative_alpha_twiss(tmp_path):
    from pyrite.materials import load_material_catalog

    text = _named_beam_catalog(
        beam_ref="diverging",
        beams_block=(
            "\n[beams.diverging]\n"
            "\n[beams.diverging.transverse]\n"
            "normalized_emittance_x_mm_mrad = 0.1\n"
            "beta_twiss_x_m = 0.05\n"
            "alpha_twiss_x = -1.5\n"
        ),
    )
    catalog = load_material_catalog(_write_catalog(tmp_path, text))
    transverse = catalog.profile_beam("standard")["transverse"]
    assert transverse["alpha_twiss_x"] == -1.5


def test_named_beam_renaming_does_not_change_resolved_payload(tmp_path):
    from pyrite.materials import load_material_catalog

    def catalog_for(beam_key: str) -> object:
        text = _named_beam_catalog(
            beam_ref=beam_key,
            beams_block=f"\n[beams.{beam_key}]\nrep_rate_hz = 1000.0\n",
        )
        (tmp_path / beam_key).mkdir()
        return load_material_catalog(_write_catalog(tmp_path / beam_key, text))

    first = catalog_for("lab_gun")
    second = catalog_for("renamed_gun")
    assert first.profile_beam("standard") == second.profile_beam("standard")


def test_profile_detector_decodes_selected_profile_and_reaches_material_sweep(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from pyrite.campaign import config
    from pyrite.detectors import Detector, EnergyBins
    from pyrite.materials import load_material_catalog

    text = (
        _catalog_with_two_profiles(tmp_path).read_text()
        + "\n[profiles.standard.detector]\n"
        + "observation_angle_deg = 91.0\n"
        + "polar_acceptance_deg = 12.0\n"
        + "solid_angle_sr = 0.05\n"
        + "\n[profiles.narrowed.detector]\n"
        + "observation_angle_deg = 119.0\n"
        + "polar_acceptance_deg = 16.6\n"
        + "solid_angle_sr = 0.066\n"
    )
    path = _write_catalog(tmp_path, text)
    catalog = load_material_catalog(path, profile="narrowed")

    assert catalog.profile_detector("standard") == Detector(91.0, 12.0, 0.05)
    selected = Detector(119.0, 16.6, 0.066)
    assert catalog.profile_detector("narrowed") == selected

    monkeypatch.setattr(config, "_catalog", lambda catalog_profile="standard": catalog)
    profiled = config.material_sweep("mos2", catalog_profile="narrowed")
    explicit = config.material_sweep(
        "mos2",
        catalog_profile="narrowed",
        detector=Detector(100.0, 8.0, 0.01),
    )
    assert replace(profiled.detector, energy_bins=EnergyBins()) == selected
    assert replace(explicit.detector, energy_bins=EnergyBins()) == Detector(100.0, 8.0, 0.01)


def test_profile_detector_omission_inherits_standard_then_legacy_fallback(tmp_path):
    from pyrite.detectors import Detector
    from pyrite.materials import load_material_catalog

    fallback = load_material_catalog(
        _write_catalog(
            tmp_path,
            _minimal_catalog(
                material_rows="""
[materials.mos2]
label = "mos2"
crystal = "mos2"
"""
            ),
        )
    )
    assert fallback.profile_detector("standard") == Detector()

    text = (
        _catalog_with_two_profiles(tmp_path).read_text()
        + "\n[profiles.standard.detector]\nobservation_angle_deg = 91.0\n"
    )
    inherited = load_material_catalog(_write_catalog(tmp_path, text), profile="narrowed")
    assert inherited.profile_detector("narrowed") == Detector(91.0)


def test_profile_detector_rejects_bad_fields_with_catalog_path(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = (
        _minimal_catalog(
            material_rows="""
[materials.mos2]
label = "mos2"
crystal = "mos2"
"""
        )
        + "\n[profiles.standard.detector]\n"
        + "observation_angle_deg = 181.0\n"
        + 'qe_curve = "/tmp/machine-local.csv"\n'
    )
    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))

    assert "profiles.standard.detector" in str(caught.value)


def test_profile_longitudinal_policy_decodes_and_reaches_material_sweep(tmp_path, monkeypatch):
    from pyrite.campaign import config
    from pyrite.materials import load_material_catalog

    text = _catalog_with_standard_beam(
        "\n[profiles.standard.beam]\n"
        "bunch_charge_pc = 1.0\n"
        "\n[profiles.standard.beam.longitudinal]\n"
        'kind = "microtrain"\n'
        "envelope_rms_fs = 200.0\n"
        "retained_coherence = 0.9\n"
    )
    catalog = load_material_catalog(_write_catalog(tmp_path, text))

    policy = catalog.profile_beam("standard")["longitudinal"]
    assert policy == {
        "kind": "microtrain",
        "envelope_rms_fs": 200.0,
        "retained_coherence": 0.9,
    }
    monkeypatch.setattr(config, "CATALOG", catalog)
    sweep = config.material_sweep("mos2")
    assert sweep.beam.bunch_charge_pc == 1.0
    assert sweep.beam.longitudinal.kind == "microtrain"
    assert sweep.beam.longitudinal.envelope_rms_fs == 200.0


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ('kind = "compressed"\nenvelope_rms_fs = 200.0', "must be omitted"),
        ('kind = "gaussian"\nenvelope_rms_fs = 200.0\nretained_coherence = 0.8', "does not accept"),
        ('kind = "microtrain"', "must be a finite positive"),
    ],
)
def test_profile_longitudinal_policy_rejects_invalid_combinations(tmp_path, policy, message):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = _catalog_with_standard_beam("\n[profiles.standard.beam.longitudinal]\n" + policy + "\n")
    with pytest.raises(MaterialConfigError, match=message):
        load_material_catalog(_write_catalog(tmp_path, text))


def test_bundled_crystal_validation_ids_are_ledgered():
    from pyrite import DATA_DIR

    with (DATA_DIR / "materials.toml").open("rb") as stream:
        raw = tomllib.load(stream)
    validation_ids = {
        row["validation_id"].strip()
        for row in raw["crystals"].values()
        if isinstance(row.get("validation_id"), str) and row["validation_id"].strip()
    }
    ledger = (
        Path(__file__).parents[2] / "docs" / "validation" / "physics-validation-ledger.md"
    ).read_text()

    missing = sorted(
        validation_id for validation_id in validation_ids if f"| `{validation_id}` |" not in ledger
    )
    assert not missing, f"catalog validation IDs missing from ledger: {missing}"


_ALLOWED_PHASES = frozenset({"1T", "1T'", "2H", "3R", "4H", "6H", "Td"})


def test_every_bundled_crystal_has_a_full_name():
    from pyrite.materials import CATALOG

    missing = sorted(key for key, spec in CATALOG.crystals.items() if not spec.full_name)
    assert not missing, f"crystals missing a full_name: {missing}"


def test_bundled_crystal_phases_use_known_polytype_labels():
    from pyrite.materials import CATALOG

    bad = sorted(
        (key, spec.phase)
        for key, spec in CATALOG.crystals.items()
        if spec.phase is not None and spec.phase not in _ALLOWED_PHASES
    )
    assert not bad, f"crystals with unrecognized phase labels: {bad}"


def test_bundled_crystal_external_ids_are_well_formed():
    from pyrite.materials import CATALOG

    for key, spec in CATALOG.crystals.items():
        if spec.cod_id is not None:
            assert isinstance(spec.cod_id, int) and spec.cod_id > 0, key
        if spec.mp_id is not None:
            assert spec.mp_id.startswith("mp-"), key


def test_same_named_polytypes_keep_separate_checkpoints():
    """full_name is display-only. Polytypes that share a full_name (e.g. SiC 4H
    vs 6H) must still map to distinct crystal keys and distinct per-material
    checkpoint pickles -- otherwise a sweep over one polytype would clobber the
    other's results, since run_sweep names the pickle for ``case['crystal']``."""
    from pyrite.materials import CATALOG
    from pyrite.runs.run import checkpoint_path_for

    # SiC polytypes: identical material name, different phase, different key.
    assert CATALOG.crystal("4h_sic").full_name == CATALOG.crystal("6h_sic").full_name
    assert CATALOG.crystal("4h_sic").phase != CATALOG.crystal("6h_sic").phase
    assert checkpoint_path_for("4h_sic") != checkpoint_path_for("6h_sic")

    # No two crystal keys collide on a checkpoint path, even where full_name repeats.
    paths = [checkpoint_path_for(key) for key in CATALOG.crystals]
    assert len(set(paths)) == len(paths)

    # Any full_name shared by >1 crystal must still resolve to unique checkpoints.
    from collections import defaultdict

    by_name: dict[str, list[str]] = defaultdict(list)
    for key, spec in CATALOG.crystals.items():
        if spec.full_name:
            by_name[spec.full_name].append(key)
    for name, keys in by_name.items():
        checkpoints = {checkpoint_path_for(key) for key in keys}
        assert len(checkpoints) == len(keys), f"{name!r} polytypes share a checkpoint: {keys}"


def test_packaged_catalog_exposes_frozen_ordered_public_api():
    from pyrite.materials import CATALOG, MaterialCatalog

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
    from pyrite.materials import CATALOG

    expected_bounds = {
        30.0: (10.0, 2500.0),
        40.0: (10.0, 2800.0),
        50.0: (10.0, 3000.0),
        60.0: (10.0, 4200.0),
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
    from pyrite.materials import CATALOG

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
    from pyrite.materials import MaterialConfigError, load_material_catalog

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
            "tilt_deg = { linspace = { start = 5.0, stop = 80.0, num = 3, endpoint = false } }",
            "tilt_deg = { linspace = { start = 5.0, stop = 80.0, num = 2.5, endpoint = false } }",
        ),
        (
            "tilt_deg = { linspace = { start = 5.0, stop = 80.0, num = 3, endpoint = false } }",
            "tilt_deg = { linspace = { start = 5.0, stop = 80.0, num = inf, endpoint = false } }",
        ),
        (
            "tilt_deg = { linspace = { start = 5.0, stop = 80.0, num = 3, endpoint = false } }",
            "tilt_deg = { linspace = { start = 5.0, stop = 80.0, num = 3, endpoint = 1 } }",
        ),
        (
            "tilt_azim_deg = 0.0",
            "tilt_azim_deg = { logspace = { start = 0.0, stop = 2.0, num = 3, base = false } }",
        ),
    ],
)
def test_grid_descriptor_types_are_strict_and_errors_are_wrapped(tmp_path, valid, invalid):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = _minimal_catalog().replace(valid, invalid)
    with pytest.raises(MaterialConfigError, match="profiles.standard"):
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
    from pyrite.materials import MaterialConfigError, load_material_catalog

    material_rows = f"""
[materials.mos2]
label = "MoS2"

[profiles.standard.overrides.mos2]
{field} = {value}
"""
    with pytest.raises(MaterialConfigError, match=rf"materials\.mos2\.scan\.{field}"):
        load_material_catalog(
            _write_catalog(tmp_path, _minimal_catalog(material_rows=material_rows))
        )


def test_grid_descriptors_profile_overrides_and_layer_count_conversion(tmp_path):
    from pyrite.materials import load_material_catalog

    path = _write_catalog(
        tmp_path,
        _minimal_catalog(
            material_rows="""
[materials.sample]
label = "sample"
crystal = "mos2"
stack = [{ material = "sio2", thickness_ang = 2850.0, azimuth_deg = 12.0 }]

[profiles.standard.overrides.sample]
thickness_layers = { values = [3, 4] }
tilt_azim_deg = { logspace = { start = 0.0, stop = 2.0, num = 3, base = 2.0 } }
""",
        ),
    )

    catalog = load_material_catalog(path)
    scan = catalog.material("sample").scan
    np.testing.assert_array_equal(scan.energy_keV, [25.0, 30.0])
    np.testing.assert_array_equal(scan.E_grid_line, np.arange(50.0, 60.0, 2.0))
    np.testing.assert_array_equal(scan.tilt_deg, np.linspace(5.0, 80.0, 3, endpoint=False))
    np.testing.assert_array_equal(scan.tilt_azim_deg, np.logspace(0.0, 2.0, 3, base=2.0))
    np.testing.assert_allclose(scan.thickness_ang, np.array([3.0, 4.0]) * 12.294 / 2.0)
    assert catalog.material("sample").stack[0].azimuth_deg == 12.0


def test_catalog_scalar_and_logspace_energy_grids_reach_runner_exactly(tmp_path, monkeypatch):
    from pyrite.campaign import config
    from pyrite.campaign import sweep as sweep_module
    from pyrite.materials import load_material_catalog
    from pyrite.montecarlo import runner

    text = _minimal_catalog(
        material_rows="""
[materials.sample]
label = "sample"
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
            "tilt_deg = { linspace = { start = 5.0, stop = 80.0, num = 3, endpoint = false } }",
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
    from pyrite.materials import MaterialConfigError, load_material_catalog

    valid = _minimal_catalog(
        material_rows="""
[materials.mos2]
label = "MoS2"
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
    from pyrite.materials import MaterialConfigError, load_material_catalog

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
        "profiles.standard",
        "crystals.bad.cif",
        "crystals.bad.beam_uvw",
        "media.bad.composition.Xe",
        "materials.bad.label",
        "materials.bad.crystal",
        "materials.bad.stack[0].material",
    ):
        assert expected in message


def test_crystal_requires_exactly_one_orientation(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    material = """
[materials.mos2]
label = "MoS2"
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
    from pyrite.materials import load_material_catalog

    text = _minimal_catalog(
        material_rows="""
[materials.mos2]
label = "MoS2"
""",
    ).replace("beam_uvw = [0, 0, 2]", "surface_hkl = [2, 0, -1]")
    spec = load_material_catalog(_write_catalog(tmp_path, text)).crystal("mos2")

    assert spec.beam_uvw is None
    assert spec.surface_hkl == (2, 0, -1)


def test_duplicate_toml_definition_is_material_config_error(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    path = _write_catalog(tmp_path, "schema_version=1\nschema_version=1\n")
    with pytest.raises(MaterialConfigError, match="Cannot overwrite a value"):
        load_material_catalog(path)


def test_runnable_crystal_must_use_supported_transport_elements(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = _minimal_catalog(
        crystal="lif",
        material_rows="""
[materials.lif]
label = "LiF"
""",
    )
    with pytest.raises(MaterialConfigError, match="unsupported transport elements.*F.*Li"):
        load_material_catalog(_write_catalog(tmp_path, text))


def test_missing_mott_tables_warn_without_rejecting_catalog(tmp_path, caplog):
    from pyrite.materials import load_material_catalog

    text = _minimal_catalog(
        crystal="ws2",
        material_rows="""
[materials.ws2]
label = "WS2"
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
    path = Path(__file__).parents[1] / "data" / "material_catalog_golden.json"
    return json.loads(path.read_text())


def test_packaged_catalog_matches_independent_serialized_golden(serialized_catalog_golden):
    from pyrite.materials import CATALOG

    golden = serialized_catalog_golden
    assert tuple(CATALOG.crystals) == tuple(golden["crystal_keys"])
    assert tuple(key for key, spec in CATALOG.crystals.items() if spec.E_grid is not None) == tuple(
        golden["configured_crystal_keys"]
    )
    assert CATALOG.material_keys == tuple(golden["material_keys"])

    for key, expected in golden["crystals"].items():
        actual = CATALOG.crystal(key)
        actual_lattice = dict(actual.lattice)
        expected_lattice = expected["lattice"]
        assert tuple(actual_lattice) == tuple(expected_lattice)
        for field, expected_value in expected_lattice.items():
            actual_value = actual_lattice[field]
            if isinstance(expected_value, float):
                assert actual_value == pytest.approx(expected_value, rel=2e-15, abs=0.0)
            else:
                assert actual_value == expected_value
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
        assert actual.validation.crystal_database_match == expected.get("validation", {}).get(
            "crystal_database_match"
        )
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
    from pyrite.materials import CATALOG

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
    from pyrite.materials import load_material_catalog

    text = _minimal_catalog(
        material_rows="""
[materials.sample]
label = "sample"
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
    from pyrite.materials import CATALOG
    from pyrite.materials import crystal as crystal_module

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
