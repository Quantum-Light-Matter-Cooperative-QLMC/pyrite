"""Focused tests for the immutable TOML material catalog."""

import hashlib
import json
import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest


def test_public_schema_types_keep_catalog_pickle_identity():
    from pyrite.materials import MaterialValidationSpec

    assert MaterialValidationSpec.__module__ == "pyrite.materials.catalog"
    assert pickle.loads(pickle.dumps(MaterialValidationSpec())) == MaterialValidationSpec()


def test_bundled_catalog_uses_automatic_line_grids_for_every_material():
    """Bundled profiles must not silently bypass the converged automatic policy."""
    from pyrite.materials import CATALOG

    assert all(
        not CATALOG.material(material).scan.E_grid_line_by_energy
        for material in CATALOG.material_keys
    )
    assert all(not refs for refs in CATALOG.profile_energy_grid_refs.values())


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
formula = "MoS2"
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
display_name = "sample"
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
    # A material's own [energy_grids.<material>] entry is authoritative;
    # its E_grid_brem
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


def test_per_beam_line_grid_rejects_duplicate_beam_energies(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    replacement = "{ energy_keV = 25.0, grid = 50.0 },\n  { energy_keV = 25.0, grid = 60.0 }"
    text = _catalog_with_per_beam_line_grids().replace(PER_BEAM_ENTRIES, replacement)
    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))
    assert "energy_grids.sample.line_by_energy[1].energy_keV" in str(caught.value)


def test_partial_store_coverage_loads_and_keeps_only_covered_rows(tmp_path):
    """Issue #101: a beam energy with no stored row is no longer a catalog
    error. The covered energies stay in the mapping; the uncovered one falls
    through to automatic case-local resolution in campaign.sweep."""
    from pyrite.materials import load_material_catalog

    text = _catalog_with_per_beam_line_grids().replace(
        PER_BEAM_ENTRIES, '{ energy_keV = 25.0, grid = 50.0, source = "derived" }'
    )
    scan = load_material_catalog(_write_catalog(tmp_path, text)).material("sample").scan
    assert scan.E_grid_line is None
    assert tuple(scan.E_grid_line_by_energy) == (25.0,)


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
    from pyrite import _energy_grid_artifacts as artifacts
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

    # The profile owns the swept axis and here sweeps a subset of the derived
    # energies; only the derived line/brem grids come out of the artifact.
    assert scan.energy_keV.tolist() == [25.0, 30.0]
    assert scan.E_grid_line_by_energy is not None
    assert list(scan.E_grid_line_by_energy) == [25.0, 30.0]
    assert scan.E_grid_line_by_energy[25.0].tolist() == [11.0, 23.0, 35.0, 47.0, 59.0]
    assert scan.E_grid_brem.tolist() == [0.0, 20.0, 40.0, 60.0, 80.0]
    assert catalog.profile_energy_grid_ref("standard", "sample") == stored.digest
    assert catalog.resolved_energy_grid_refs == {"sample": stored.digest}


def test_profile_artifact_ref_rejects_beam_energy_the_artifact_never_covered(tmp_path):
    """A hand-copied ref must fail at load time, not silently swap the axis.

    ``beam_energies_keV`` is a record of what `energy-grid add` derived, written
    *from* the profile; a profile sweeping an energy outside it means the ref
    does not belong to that profile.
    """
    from pyrite import _energy_grid_artifacts as artifacts
    from pyrite.materials import MaterialConfigError, load_material_catalog

    identity = artifacts.artifact_identity(
        "sample",
        # A line row for 40 keV, so the missing-line-grid check cannot fire and
        # mask the beam-energy check under test.
        [
            {"energy_keV": 25, "start_eV": 11, "stop_eV": 59, "num": 5},
            {"energy_keV": 30, "start_eV": 12, "stop_eV": 72, "num": 4},
            {"energy_keV": 40, "start_eV": 20, "stop_eV": 80, "num": 3},
        ],
        {"start_eV": 0, "stop_eV": 100, "step_eV": 20},
        [25, 30],
    )
    stored = artifacts.write_artifact(tmp_path / "energy-grid-artifacts", identity)
    text = _catalog_with_artifact_ref(stored.digest).replace(
        "energy_keV = { values = [25.0, 30.0] }",
        "energy_keV = { values = [25.0, 30.0, 40.0] }",
    )

    with pytest.raises(MaterialConfigError, match="do not include .40.0."):
        load_material_catalog(_write_catalog(tmp_path, text))


def test_profile_artifact_ref_matches_equivalent_legacy_resolution(tmp_path):
    from pyrite import _energy_grid_artifacts as artifacts
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
    from pyrite import _energy_grid_artifacts as artifacts
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
    """Legacy profile-named store which must not supply material "sample"."""
    base = _minimal_catalog(
        material_rows="""
[materials.sample]
display_name = "sample"
crystal = "mos2"
"""
    ).replace(_NO_FLAT_LINE_GRID, "")
    return base + f"\n[energy_grids.standard]\nline_by_energy = [\n  {entries},\n]\n"


@pytest.mark.parametrize("profile", ["standard", "narrowed"])
def test_material_without_own_store_entry_ignores_shared_default(tmp_path, profile):
    import tomlkit

    from pyrite.materials import load_material_catalog

    document = tomlkit.parse(_catalog_with_default_store_only())
    if profile != "standard":
        document["profiles"][profile] = document["profiles"]["standard"].copy()
        document["energy_grids"][profile] = document["energy_grids"]["standard"].copy()
    scan = (
        load_material_catalog(_write_catalog(tmp_path, tomlkit.dumps(document)), profile=profile)
        .material("sample")
        .scan
    )
    assert scan.E_grid_line_by_energy == {}


def test_missing_beam_energy_loads_without_default_store_coverage(tmp_path):
    """Even partial profile-named rows are ignored by material resolution."""
    from pyrite.materials import load_material_catalog

    only_25 = '{ energy_keV = 25.0, grid = { values = [1.0, 2.0] }, source = "derived" }'
    text = _catalog_with_default_store_only(entries=only_25)

    scan = load_material_catalog(_write_catalog(tmp_path, text)).material("sample").scan

    assert scan.E_grid_line_by_energy == {}


def test_material_with_no_line_grid_at_all_is_valid(tmp_path):
    """Neither E_grid_line nor any store row: still a loadable material."""
    from pyrite.materials import load_material_catalog

    text = _minimal_catalog(
        material_rows="""
[materials.sample]
display_name = "sample"
crystal = "mos2"
"""
    ).replace(_NO_FLAT_LINE_GRID, "")

    scan = load_material_catalog(_write_catalog(tmp_path, text)).material("sample").scan

    assert scan.E_grid_line is None
    assert scan.E_grid_line_by_energy == {}


@pytest.mark.parametrize("own_entries", [None, PER_BEAM_ENTRIES.split(",\n")[0]])
def test_missing_material_rows_reach_automatic_case_policy(tmp_path, monkeypatch, own_entries):
    from pyrite._energy_grid_encoding import decode_energy_grid
    from pyrite.campaign import config
    from pyrite.campaign.sweep import build_cases
    from pyrite.materials import load_material_catalog

    text = _catalog_with_default_store_only()
    if own_entries is not None:
        text += f"\n[energy_grids.sample]\nline_by_energy = [{own_entries}]\n"
    catalog = load_material_catalog(_write_catalog(tmp_path, text))
    monkeypatch.setattr(config, "CATALOG", catalog)
    cases = build_cases(config.material_sweep("sample"))
    for case in cases:
        if own_entries is not None and case["E0_keV"] == 25.0:
            assert "line_grid_policy" not in case
            np.testing.assert_array_equal(
                decode_energy_grid(case["E_grid"]), np.linspace(10.0, 58.0, 17)
            )
        else:
            assert case["line_grid_policy"]["resolution"]["policy"] == "sinc-nyquist"
            assert case["line_grid_policy"]["bandwidth"]["policy"] == "kinematic-ceiling"


def test_material_config_error_groups_identical_messages_across_materials(tmp_path):
    # A profile-wide setting invalid for every material must not repeat one
    # near-duplicate line per material.
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = _minimal_catalog(
        material_rows="""
[materials.sample_a]
display_name = "sample_a"
crystal = "mos2"

[materials.sample_b]
display_name = "sample_b"
crystal = "mos2"
"""
    ).replace("tilt_azim_deg = 0.0", "tilt_azim_deg = 400.0")

    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))

    assert len(caught.value.errors) == 3
    message = str(caught.value)
    assert message.count("0 <= tilt_azim_deg <= 360") == 1
    assert (
        "3 paths (profiles.standard.tilt_azim_deg, materials.sample_a.scan.tilt_azim_deg, "
        "materials.sample_b.scan.tilt_azim_deg)"
    ) in message


def test_named_profile_scan_defaults_apply_only_to_members(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = _minimal_catalog(
        material_rows="""
[materials.sample_a]
display_name = "sample_a"
crystal = "mos2"

[materials.sample_b]
display_name = "sample_b"
crystal = "mos2"

[profiles.narrowed]
thickness_ang = 1000.0
energy_keV = 35.0
tilt_deg = 5.0
tilt_azim_deg = 400.0
E_grid_brem = 0.0
materials = ["sample_a"]
"""
    )

    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text), profile="narrowed")

    message = str(caught.value)
    assert "materials.sample_a.scan.tilt_azim_deg" in message
    assert "materials.sample_b.scan" not in message


def _catalog_with_two_profiles(tmp_path: Path) -> Path:
    """``narrowed`` restricts membership to "mos2"; ``standard`` (no
    ``materials`` row) allows every catalog material -- Phase 3 scan/submit
    ``--profile`` intersection semantics read this membership."""
    text = _minimal_catalog(
        material_rows="""
[materials.mos2]
display_name = "mos2"
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


def test_profile_transport_numerics_are_validated_and_exposed(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    path = _catalog_with_two_profiles(tmp_path)
    text = path.read_text().replace(
        "[profiles.narrowed]\n",
        '[profiles.narrowed]\nstraggling = true\nenergy_model = "midpoint"\nmax_dE_frac = 0.02\n',
    )
    path.write_text(text)
    catalog = load_material_catalog(path)
    assert dict(catalog.profile_numerics("narrowed")) == {
        "straggling": True,
        "energy_model": "midpoint",
        "max_dE_frac": 0.02,
    }

    path.write_text(text.replace('energy_model = "midpoint"', 'energy_model = "frozen"'))
    with pytest.raises(MaterialConfigError, match="requires energy_model"):
        load_material_catalog(path)


def test_profile_convergence_numerics_are_validated_and_exposed(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    path = _catalog_with_two_profiles(tmp_path)
    text = path.read_text().replace(
        "[profiles.narrowed]\n",
        "[profiles.narrowed]\nn_families = 3\nmax_reflections = 6\n"
        'mosaic_nodes = 7\nmosaic_route = "mc"\n',
    )
    path.write_text(text)
    catalog = load_material_catalog(path)
    assert dict(catalog.profile_numerics("narrowed")) == {
        "n_families": 3,
        "max_reflections": 6,
        "mosaic_nodes": 7,
        "mosaic_route": "mc",
    }

    path.write_text(text.replace("n_families = 3", "n_families = 0"))
    with pytest.raises(MaterialConfigError, match="profiles.narrowed.n_families"):
        load_material_catalog(path)


@pytest.mark.parametrize("field", ["n_families", "max_reflections", "mosaic_nodes"])
@pytest.mark.parametrize("invalid", ["0", "-1", "true", "1.5"])
def test_profile_convergence_rejects_invalid_counts_after_rewrite(tmp_path, field, invalid):
    import os

    from pyrite.materials import MaterialConfigError, load_material_catalog

    path = _catalog_with_two_profiles(tmp_path)
    text = path.read_text().replace("[profiles.narrowed]\n", f"[profiles.narrowed]\n{field} = 3\n")
    path.write_text(text)
    assert load_material_catalog(path).profile_numerics("narrowed")[field] == 3

    stat = path.stat()
    path.write_text(text.replace(f"{field} = 3", f"{field} = {invalid}"))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(MaterialConfigError, match=rf"profiles\.narrowed\.{field}"):
        load_material_catalog(path)


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
    assert CATALOG.profile_materials("high_energy") == ("hbn", "mose2", "mos2")


def test_material_validation_metadata_is_typed_and_validated(tmp_path):
    from pyrite.materials import MaterialConfigError, MaterialValidationSpec, load_material_catalog

    text = _minimal_catalog(
        material_rows="""
[materials.sample]
display_name = "sample"
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
display_name = "mos2"
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
display_name = "mos2"
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


def test_named_detector_reference_resolves_to_same_value_as_inline_block(tmp_path, monkeypatch):
    from pyrite.campaign import config
    from pyrite.campaign.config import default_settings
    from pyrite.campaign.profiles import dataset_identity
    from pyrite.detectors import Detector
    from pyrite.detectors.spec import Timepix3
    from pyrite.materials import load_material_catalog

    material_rows = """
[materials.mos2]
display_name = "mos2"
crystal = "mos2"
"""
    ref_text = _minimal_catalog(
        material_rows=material_rows,
        profile_extra='detector = "eds"',
    ) + (
        '\n[detectors.eds]\nlabel = "SEM EDS"\n'
        "observation_angle_deg = 119.0\n"
        "polar_acceptance_deg = 16.6\n"
        "solid_angle_sr = 0.066\n"
    )
    inline_text = _minimal_catalog(material_rows=material_rows) + (
        "\n[profiles.standard.detector]\n"
        "observation_angle_deg = 119.0\n"
        "polar_acceptance_deg = 16.6\n"
        "solid_angle_sr = 0.066\n"
    )
    (tmp_path / "ref").mkdir()
    (tmp_path / "inline").mkdir()
    ref_catalog = load_material_catalog(_write_catalog(tmp_path / "ref", ref_text))
    inline_catalog = load_material_catalog(_write_catalog(tmp_path / "inline", inline_text))

    assert ref_catalog.profile_detector("standard") == inline_catalog.profile_detector("standard")
    # The catalog resolves acceptance fields; config builds the detector.
    expected_spec = {
        "observation_angle_deg": 119.0,
        "polar_acceptance_deg": 16.6,
        "solid_angle_sr": 0.066,
    }
    assert dict(ref_catalog.profile_detector("standard")) == expected_spec
    assert ref_catalog.detector_keys == ("eds",)
    assert dict(ref_catalog.detectors["eds"]) == expected_spec
    assert ref_catalog.detector_labels["eds"] == "SEM EDS"
    monkeypatch.setattr(config, "_catalog", lambda catalog_profile="standard": ref_catalog)
    assert config.catalog_detector("standard") == Detector(119.0, 16.6, 0.066, response=Timepix3())
    monkeypatch.setattr(config, "_catalog", lambda catalog_profile="standard": ref_catalog)
    ref_sweep = config.material_sweep("mos2")
    monkeypatch.setattr(config, "_catalog", lambda catalog_profile="standard": inline_catalog)
    inline_sweep = config.material_sweep("mos2")
    assert (
        dataset_identity("mos2", "full", default_settings(), ref_sweep)["parameter_sha256"]
        == dataset_identity("mos2", "full", default_settings(), inline_sweep)["parameter_sha256"]
    )


def test_named_detector_unknown_reference_errors_with_profile_path(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = _minimal_catalog(
        material_rows="""
[materials.mos2]
display_name = "mos2"
crystal = "mos2"
""",
        profile_extra='detector = "missing"',
    )

    with pytest.raises(MaterialConfigError) as caught:
        load_material_catalog(_write_catalog(tmp_path, text))

    assert "profiles.standard.detector: unknown detector 'missing'" in str(caught.value)


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
    from pyrite.detectors.spec import Timepix3
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

    monkeypatch.setattr(config, "_catalog", lambda catalog_profile="standard": catalog)
    assert config.catalog_detector("standard") == Detector(91.0, 12.0, 0.05, response=Timepix3())
    selected = Detector(119.0, 16.6, 0.066, response=Timepix3())
    assert config.catalog_detector("narrowed") == selected

    profiled = config.material_sweep("mos2", catalog_profile="narrowed")
    explicit = config.material_sweep(
        "mos2",
        catalog_profile="narrowed",
        detector=Detector(100.0, 8.0, 0.01),
    )
    assert replace(profiled.detector, energy_bins=EnergyBins()) == selected
    assert replace(explicit.detector, energy_bins=EnergyBins()) == Detector(100.0, 8.0, 0.01)


def test_profile_detector_omission_inherits_standard_then_legacy_fallback(tmp_path, monkeypatch):
    from pyrite.campaign import config
    from pyrite.detectors import Detector
    from pyrite.detectors.spec import Timepix3
    from pyrite.materials import load_material_catalog

    fallback = load_material_catalog(
        _write_catalog(
            tmp_path,
            _minimal_catalog(
                material_rows="""
[materials.mos2]
display_name = "mos2"
crystal = "mos2"
"""
            ),
        )
    )
    assert dict(fallback.profile_detector("standard")) == {}
    monkeypatch.setattr(config, "_catalog", lambda catalog_profile="standard": fallback)
    assert config.catalog_detector("standard") == Detector(response=Timepix3())

    text = (
        _catalog_with_two_profiles(tmp_path).read_text()
        + "\n[profiles.standard.detector]\nobservation_angle_deg = 91.0\n"
    )
    inherited = load_material_catalog(_write_catalog(tmp_path, text), profile="narrowed")
    assert dict(inherited.profile_detector("narrowed")) == {"observation_angle_deg": 91.0}
    monkeypatch.setattr(config, "_catalog", lambda catalog_profile="standard": inherited)
    assert config.catalog_detector("narrowed") == Detector(91.0, response=Timepix3())


def test_profile_detector_rejects_bad_fields_with_catalog_path(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = (
        _minimal_catalog(
            material_rows="""
[materials.mos2]
display_name = "mos2"
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


@pytest.mark.parametrize(
    "field, value",
    [
        ("observation_angle_deg", 181.0),
        ("observation_angle_deg", -1.0),
        ("polar_acceptance_deg", 0.0),
        ("polar_acceptance_deg", 181.0),
        ("polar_acceptance_deg", -2.0),
        ("solid_angle_sr", 0.0),
        ("solid_angle_sr", 13.0),
        ("solid_angle_sr", -0.5),
    ],
)
def test_catalog_detector_bounds_match_the_detector_dataclass(tmp_path, field, value):
    """The catalog no longer builds a Detector, so its ranges are a copy.

    Hold the copy to the original: every value the dataclass rejects must be
    rejected at catalog-load time with the same message, on the block's path.
    """
    from pyrite.detectors import Detector
    from pyrite.materials import MaterialConfigError, load_material_catalog

    with pytest.raises((TypeError, ValueError)) as dataclass_error:
        Detector(**{field: value})

    text = (
        _minimal_catalog(
            material_rows="""
[materials.mos2]
display_name = "mos2"
crystal = "mos2"
"""
        )
        + f"\n[profiles.standard.detector]\n{field} = {value!r}\n"
    )
    with pytest.raises(MaterialConfigError) as catalog_error:
        load_material_catalog(_write_catalog(tmp_path, text))

    message = str(catalog_error.value)
    assert "profiles.standard.detector" in message
    assert str(dataclass_error.value) in message


def test_profile_filter_and_physical_detector_blocks_decode_without_run_wiring(tmp_path):
    from pyrite.materials import load_material_catalog

    text = (
        _minimal_catalog(
            material_rows="""
[materials.mos2]
display_name = "mos2"
crystal = "mos2"
"""
        )
        + """
[profiles.standard.physical_detector]
distance_mm = 400.0
polar_deg = 90.0
shape = [2, 3]
pitch_mm = [0.1, 0.2]

[[profiles.standard.filters]]
name = "half"
material = "sio2"
thickness_mm = 0.1
size_mm = [2.0, 3.0]
distance_mm = 200.0
offset_mm = [0.5, 0.0]
"""
    )
    catalog = load_material_catalog(_write_catalog(tmp_path, text))

    assert catalog.profile_physical_detectors["standard"]["shape"] == (2, 3)
    assert catalog.profile_filters["standard"][0]["name"] == "half"
    assert catalog.profile_filters["standard"][0]["offset_mm"] == (0.5, 0.0)


def test_profile_observation_schema_decodes_and_resolves_without_cli(tmp_path):
    from pyrite.campaign.observation import resolve_profile_observation
    from pyrite.detectors import Timepix3
    from pyrite.materials import load_material_catalog

    text = (
        _minimal_catalog(
            material_rows="""
[materials.mos2]
display_name = "mos2"
crystal = "mos2"
"""
        )
        + """
[profiles.standard.physical_detector]
distance_mm = 400.0
polar_deg = 60.0
shape = [8, 10]
pitch_mm = [0.055, 0.055]

[profiles.standard.physical_detector.scorer]
reconstruction = "nearest_tile"
angular_shape = [3, 5]

[profiles.standard.physical_detector.response]
kind = "timepix3"
thickness_um = 500.0
bias_v = 100.0
n_mc = 20000
seed = 7

[profiles.standard.physical_detector.acquisition]
exposure_s = 2.0
measured_min_eV = 0.0
measured_max_eV = 2000.0
measured_bin_width_eV = 400.0
hit_threshold_eV = 500.0
mode = "poisson"
seed = 19
"""
    )
    catalog = load_material_catalog(_write_catalog(tmp_path, text))
    observation = resolve_profile_observation(catalog, "standard")

    assert observation is not None
    assert observation.detector.pixels.shape == (8, 10)
    assert observation.scorer.angular_shape == (3, 5)
    assert observation.scorer.reconstruction == "nearest_tile"
    assert observation.acquisition.measured_edges_eV == (
        0.0,
        400.0,
        800.0,
        1200.0,
        1600.0,
        2000.0,
    )
    assert observation.acquisition.mode == "poisson"
    assert observation.acquisition.seed == 19
    assert isinstance(observation.detector.response, Timepix3)
    assert observation.detector.response.seed == 7


def test_profile_without_acquisition_does_not_resolve_a_counting_observation(tmp_path):
    from pyrite.campaign.observation import (
        physical_detector_from_config,
        resolve_profile_observation,
    )
    from pyrite.materials import load_material_catalog

    text = (
        _minimal_catalog(
            material_rows="""
[materials.mos2]
display_name = "mos2"
crystal = "mos2"
"""
        )
        + """
[profiles.standard.physical_detector]
distance_mm = 400.0
"""
    )
    catalog = load_material_catalog(_write_catalog(tmp_path, text))

    row = catalog.profile_physical_detectors["standard"]
    assert physical_detector_from_config(row).response is None
    assert resolve_profile_observation(catalog, "standard") is None


@pytest.mark.parametrize(
    ("nested", "message"),
    [
        (
            """
[profiles.standard.physical_detector.scorer]
reconstruction = "bilinear"
""",
            "reconstruction: must be 'nearest_tile'",
        ),
        (
            """
[profiles.standard.physical_detector.response]
kind = "ideal"
n_mc = 10
""",
            "ideal response accepts only kind",
        ),
        (
            """
[profiles.standard.physical_detector.acquisition]
exposure_s = 1.0
measured_edges_eV = [0.0, 1000.0]
measured_min_eV = 0.0
measured_max_eV = 1000.0
measured_bin_width_eV = 100.0
""",
            "use measured_edges_eV or min/max/bin-width, not both",
        ),
    ],
)
def test_profile_observation_schema_rejects_ambiguous_fields(tmp_path, nested, message):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    text = (
        _minimal_catalog(
            material_rows="""
[materials.mos2]
display_name = "mos2"
crystal = "mos2"
"""
        )
        + """
[profiles.standard.physical_detector]
distance_mm = 400.0
"""
        + nested
    )

    with pytest.raises(MaterialConfigError, match=message):
        load_material_catalog(_write_catalog(tmp_path, text))


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
    from pyrite._catalog_layout import bundled_catalog, read_raw

    raw = read_raw(bundled_catalog())
    validation_ids = {
        row["validation_id"].strip()
        for row in raw["crystals"].values()
        if isinstance(row.get("validation_id"), str) and row["validation_id"].strip()
    }
    from pyrite.devtools.validation_ledger import part_paths

    index = Path(__file__).parents[2] / "docs" / "validation" / "physics-validation-ledger.md"
    ledger = "\n".join(path.read_text() for path in part_paths(index))

    missing = sorted(
        validation_id for validation_id in validation_ids if f"`{validation_id}`" not in ledger
    )
    assert not missing, f"catalog validation IDs missing from ledger: {missing}"


_ALLOWED_PHASES = frozenset({"1T", "1T'", "2H", "3R", "4H", "6H", "Td", "beta"})


def test_reduce_indices_collapses_unreduced_slab_normals():
    from pyrite.materials._identity import reduce_indices

    assert reduce_indices((0, 0, 2)) == (0, 0, 1)
    assert reduce_indices((4, 4, 0)) == (1, 1, 0)
    assert reduce_indices((4, 0, 0)) == (1, 0, 0)
    assert reduce_indices((0, 2, 0)) == (0, 1, 0)
    assert reduce_indices((1, 0, -1)) == (1, 0, -1)
    assert reduce_indices((2, 0, -2)) == (1, 0, -1)
    # an all-zero normal has no primitive representative to reduce to
    assert reduce_indices((0, 0, 0)) == (0, 0, 0)


def test_format_indices_uses_the_bracket_convention_for_each_frame():
    from pyrite.materials._identity import format_indices

    assert format_indices((0, 0, 1), "plane") == "(001)"
    assert format_indices((0, 0, 1), "direction") == "[001]"
    # multi-digit or negative components would be ambiguous concatenated
    assert format_indices((1, 0, -1), "plane") == "(1 0 -1)"
    assert format_indices((12, 0, 1), "direction") == "[12 0 1]"
    # a hexagonal setting renders a plane in four Miller--Bravais indices
    assert format_indices((0, 0, 2), "plane", hexagonal=True) == "(0001)"
    assert format_indices((1, 1, 0), "plane", hexagonal=True) == "(1 1 -2 0)"
    # the direction transform is rational, so a direction stays in three
    assert format_indices((0, 0, 1), "direction", hexagonal=True) == "[001]"


def test_hexagonal_setting_reads_the_cell_parameters_not_the_system_field():
    from pyrite.materials import CATALOG
    from pyrite.materials._identity import hexagonal_setting

    hexagonal = {"a": 3.16, "b": 3.16, "c": 12.294, "alpha": 90.0, "beta": 90.0, "gamma": 120.0}
    assert hexagonal_setting(hexagonal)
    # CIF-derived gamma arrives a few ulps off 120
    assert hexagonal_setting({**hexagonal, "gamma": 119.99999999999999})
    assert not hexagonal_setting({**hexagonal, "gamma": 90.0})
    assert not hexagonal_setting({**hexagonal, "b": 3.30})
    assert not hexagonal_setting({**hexagonal, "alpha": 77.6})
    assert not hexagonal_setting({})
    # the stored "system" field is always "general" and must not be consulted
    assert CATALOG.crystal("mos2").lattice["system"] == "general"
    assert CATALOG.crystal("mos2").hexagonal
    assert not CATALOG.crystal("silicon").hexagonal


def test_every_bundled_material_label_is_derived_from_its_crystal():
    """No hand-authored label can drift from the record it describes."""
    from pyrite.materials import CATALOG
    from pyrite.materials._identity import MaterialIdentity

    for key in CATALOG.material_keys:
        spec = CATALOG.material(key)
        crystal = CATALOG.crystal(spec.crystal_key)
        expected = MaterialIdentity(
            formula=crystal.formula,
            phase=crystal.phase,
            full_name=crystal.full_name,
            cut=crystal.cut,
            cut_frame=crystal.cut_frame,
            display_name=spec.identity.display_name,
            hexagonal=crystal.hexagonal,
        )
        assert spec.label == expected.label, key


def test_every_bundled_material_label_agrees_with_its_crystal_phase_and_cut():
    from pyrite.materials import CATALOG
    from pyrite.materials._identity import format_indices, reduce_indices

    for key in CATALOG.material_keys:
        spec = CATALOG.material(key)
        crystal = CATALOG.crystal(spec.crystal_key)

        # the cut is the declared slab normal, reduced -- never a pinned reflection
        declared = crystal.surface_hkl if crystal.surface_hkl is not None else crystal.beam_uvw
        assert spec.cut is not None and declared is not None, key
        assert spec.cut_frame == ("plane" if crystal.surface_hkl is not None else "direction"), key
        assert format_indices(declared, spec.cut_frame, hexagonal=spec.hexagonal) in spec.label, key

        # a declared phase is always shown, unless a display_name replaces the name
        if crystal.phase is not None and spec.identity.display_name is None:
            assert spec.label.startswith(f"{crystal.phase}-"), key

        # no label may quote a pinned reflection family pointing along a
        # different axis than the cut (orders along the cut axis reduce to it)
        for family in crystal.hkl_families:
            if reduce_indices(family) != tuple(spec.cut):
                assert (
                    format_indices(family, spec.cut_frame, hexagonal=spec.hexagonal)
                    not in spec.label
                ), key


def test_every_bundled_crystal_declares_its_cut_as_a_plane():
    """The packaged catalog spells every slab normal ``surface_hkl``.

    ``g_hkl`` is the normal of the cut face by construction in any lattice; the
    direct axis of the same indices coincides with it only under symmetry the
    catalog does not assert. A crystal added with ``beam_uvw`` would still be
    valid config -- the parser accepts either, and Sweep/Layer overrides use the
    direct spelling -- but it would render ``[uvw]`` beside its neighbours'
    ``(hkl)``, which is exactly the label drift ``_identity`` exists to prevent.
    """
    from pyrite.materials import CATALOG

    direct = sorted(key for key, spec in CATALOG.crystals.items() if spec.beam_uvw is not None)
    assert direct == [], (
        f"crystals declaring beam_uvw: {direct}; "
        "re-declare the cut as surface_hkl with the same indices"
    )


def test_hexagonal_cuts_render_four_index_miller_bravais():
    """``(0001)``, not ``(001)``, wherever the cell is on hexagonal axes."""
    from pyrite.materials import CATALOG
    from pyrite.materials._identity import bravais_indices

    hexagonal = [key for key in CATALOG.material_keys if CATALOG.material(key).hexagonal]
    assert {"mos2", "hopg", "hbn", "sapphire", "4h_sic"} <= set(hexagonal)
    for key in hexagonal:
        spec = CATALOG.material(key)
        assert spec.cut_frame == "plane", key
        assert spec.label.endswith("(0001)"), key
    # non-hexagonal settings keep three indices, including the low-symmetry
    # cells where the plane normal genuinely differs from the direct axis
    for key in ("silicon", "res2", "wte2", "gep"):
        assert "(0001)" not in CATALOG.material(key).label, key
    assert bravais_indices((1, 1, 0)) == (1, 1, -2, 0)
    assert bravais_indices((0, 0, 1)) == (0, 0, 0, 1)


def test_bundled_material_labels_are_unique_and_ascii():
    from pyrite.materials import CATALOG

    labels = [CATALOG.material(key).label for key in CATALOG.material_keys]
    assert len(set(labels)) == len(labels), "material labels must be distinguishable"
    assert all(label.isascii() for label in labels)


def test_case_names_do_not_track_the_display_label():
    """Case names key persisted checkpoint records, so a relabel must not rename them."""
    from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases

    sweep = Sweep(
        material="mose2",
        thickness_ang=1.0e4,
        beam=BeamSpec(energy_keV=30.0),
        tilt_deg=45.0,
        tilt_azim_deg=180.0,
        crystal_width_mm=None,
        crystal_height_mm=None,
    )
    name = build_cases(sweep)[0]["name"]
    assert name.startswith("mose2 ")
    from pyrite.materials import CATALOG

    assert CATALOG.material("mose2").label not in name


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
    assert len(CATALOG.crystals) == 49
    assert len(CATALOG.materials) == 50
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


def test_standard_profile_preserves_scan_axes_without_shared_line_rows():
    from pyrite.materials import CATALOG

    expected_energies = [30.0, 40.0, 50.0, 60.0, 100.0, 150.0, 200.0, 250.0, 300.0]
    # Shared scan axes remain valid; line rows must belong to this material.
    scan = CATALOG.material("silicon").scan
    np.testing.assert_array_equal(scan.energy_keV, expected_energies)
    np.testing.assert_array_equal(scan.tilt_deg, [5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0])
    np.testing.assert_array_equal(
        scan.tilt_azim_deg, [95.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0]
    )
    assert scan.E_grid_line is None
    assert scan.E_grid_line_by_energy == {}


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
display_name = "MoS2"

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
display_name = "sample"
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
display_name = "sample"
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
display_name = "MoS2"
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
formula = "MoS2"
B_ang2 = -1.0
beam_uvw = [0, 0, 0]
hkl_families = [[0, 0, 0]]
[media.bad]
composition = { Xe = -1.0 }
[materials.bad]
display_name = ""
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
        "materials.bad.display_name",
        "materials.bad.crystal",
        "materials.bad.stack[0].material",
    ):
        assert expected in message


def test_crystal_requires_exactly_one_orientation(tmp_path):
    from pyrite.materials import MaterialConfigError, load_material_catalog

    material = """
[materials.mos2]
display_name = "MoS2"
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
display_name = "MoS2"
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
display_name = "LiF"
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
display_name = "WS2"
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
        assert actual.formula == config["formula"]
        assert actual.full_name == config["full_name"]
        assert actual.phase == config["phase"]
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
        identity_golden = expected["identity"]
        assert actual.formula == identity_golden["formula"]
        assert actual.phase == identity_golden["phase"]
        assert actual.full_name == identity_golden["full_name"]
        assert (list(actual.cut) if actual.cut is not None else None) == identity_golden["cut"]
        assert actual.cut_frame == identity_golden["cut_frame"]
        assert actual.identity.display_name == identity_golden["display_name"]
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
display_name = "sample"
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


def _observation_catalog(tmp_path, acquisition=True):
    from pyrite.materials import load_material_catalog

    text = (
        _minimal_catalog(
            material_rows="""
[materials.mos2]
display_name = "mos2"
crystal = "mos2"
"""
        )
        + """
[profiles.standard.physical_detector]
distance_mm = 400.0
polar_deg = 60.0
shape = [8, 10]
pitch_mm = [0.055, 0.055]
"""
        + (
            """
[profiles.standard.physical_detector.acquisition]
exposure_s = 2.0
measured_edges_eV = [0.0, 1000.0, 2000.0]
"""
            if acquisition
            else ""
        )
    )
    return load_material_catalog(_write_catalog(tmp_path, text))


def test_counting_observation_profile_sweeps_its_physical_projection(tmp_path, monkeypatch):
    from pyrite.campaign import config
    from pyrite.campaign.observation import resolve_profile_observation

    catalog = _observation_catalog(tmp_path)
    monkeypatch.setattr(config, "_catalog", lambda _profile="standard": catalog)
    projection = resolve_profile_observation(catalog, "standard").scalar_detector()

    detector = config.material_sweep("mos2").detector

    assert detector.observation_angle_deg == pytest.approx(60.0)
    assert detector.observation_angle_deg == projection.observation_angle_deg
    assert detector.polar_acceptance_deg == projection.polar_acceptance_deg
    assert detector.solid_angle_sr == projection.solid_angle_sr


@pytest.mark.parametrize(
    "override",
    [{"theta_obs_deg": 90.0}, {"dtheta_obs_deg": 1.0}, {"domega_sr": 0.01}],
)
def test_counting_observation_profile_rejects_scalar_detector_overrides(
    tmp_path, monkeypatch, override
):
    from pyrite.campaign import config

    catalog = _observation_catalog(tmp_path)
    monkeypatch.setattr(config, "_catalog", lambda _profile="standard": catalog)

    with pytest.raises(ValueError, match="counting physical detector"):
        config.material_sweep("mos2", **override)


def test_geometry_only_physical_detector_keeps_the_scalar_detector(tmp_path, monkeypatch):
    from pyrite.campaign import config

    catalog = _observation_catalog(tmp_path, acquisition=False)
    monkeypatch.setattr(config, "_catalog", lambda _profile="standard": catalog)

    detector = config.material_sweep("mos2").detector

    assert detector.observation_angle_deg == config.catalog_detector().observation_angle_deg
