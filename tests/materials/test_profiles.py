"""Named profile, provenance, and variant-storage regression contracts."""

import json
import tomllib
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest
import tomlkit

from pyrite import DATA_DIR
from pyrite._continuum_grid import (
    CONTINUUM_GRID_MODEL,
    CONTINUUM_LOG_STEP,
    STACK_CONTINUUM_GRID_MODEL,
    automatic_continuum_grid,
)
from pyrite._photon_continuum_floor import continuum_medium_key
from pyrite.campaign import profile_lifecycle, profile_summary, profiles
from pyrite.campaign._continuum_case_grid import automatic_continuum_stack_media
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.profiles import (
    IDENTITY_MIGRATIONS,
    LEGACY_IDENTITY_TIER,
    STOPPING_MODEL,
    case_content_key,
    dataset_identity,
    identity_from_stem,
    named_profile_identity,
    named_profile_stem,
    variant_stem,
)
from pyrite.campaign.sweep import (
    BeamSpec,
    Stack,
    build_cases,
    target_replace,
)
from pyrite.checkpoints import _checkpoint_store
from pyrite.detectors import AngularAcceptance, Detector, Timepix3
from pyrite.materials import CATALOG
from pyrite.montecarlo.spectrum import (
    BREMSSTRAHLUNG_MODEL,
    CHARACTERISTIC_MODEL,
    CHARACTERISTIC_XRAYDB_VERSION,
)
from pyrite.montecarlo.spectrum.brem_bremslib import BREMSSTRAHLUNG_BREMSLIB_MODEL
from tests.helpers.user_catalog import USER_CATALOG_FIXTURE


def _resolve_auto_as_installed(model, _elements):
    return "bremslib" if model == "auto" else model


@pytest.fixture(autouse=True)
def _pin_default_continuum_to_bremslib(monkeypatch):
    """The default ``"auto"`` continuum is BremsLib only where its tables are
    installed. Pin that outcome so the digests below do not depend on the
    machine; the EEDL-fallback identity is covered by an explicit test."""
    monkeypatch.setattr(profiles, "resolve_bremsstrahlung_model", _resolve_auto_as_installed)
    monkeypatch.setattr(
        "pyrite.xsgen.bremslib.tables.resolve_bremsstrahlung_model", _resolve_auto_as_installed
    )


BUNDLED_PROFILES = (
    "coherent_demo",
    "detector_demo",
    "high_energy_demo",
    "hopg_emittance_demo",
    "quickstart",
    "standard",
    "sweep_demo",
    "trajectory_demo",
)


def _cases_by_key(material, catalog_profile):
    sweep = material_sweep(material, catalog_profile=catalog_profile)
    cases = build_cases(sweep, n_electrons=300, n_electrons_brem=150)
    return {(c["name"], c["E0_keV"]): c for c in cases}


@pytest.mark.slow
def test_case_content_key_matches_across_profiles_for_shared_cases():
    """The whole feature: sub_100keV's energy grid is a prefix of standard's with
    identical thickness/tilt/azim grids, so every case they share (energies <=100
    keV) resolves to a bit-identical case dict -- including the grid-derived seed --
    and hashes to the SAME content key even though the two profiles are named
    differently. The profile name never reaches the key."""
    standard = _cases_by_key("hopg", "standard")
    sub = _cases_by_key("hopg", "sub_100keV")
    shared = set(standard) & set(sub)
    assert shared, "standard and sub_100keV must share hopg cases"
    for key in shared:
        assert case_content_key(standard[key]) == case_content_key(sub[key])
    # And a >100 keV case, present only in standard, is absent from sub_100keV's
    # keyspace (nothing wrong shared).
    assert any(E0 > 100.0 for _, E0 in standard) and not any(E0 > 100.0 for _, E0 in sub)


def test_typed_case_content_key_matches_pre_case_golden():
    case = build_cases(material_sweep("hopg"), n_electrons=300, n_electrons_brem=150)[0]

    assert case_content_key(case) == case_content_key(case.to_dict())
    # Moved 2026-09-13 when hopg's B_ang2 went 0.8 -> 1.30 (Trucano & Chen
    # U33; see docs/validation/materials/hopg-debye-waller-00l.md). B_ang2 is
    # hashed into the content key, so a catalog value change orphans records
    # minted under the old value rather than serving them for a case that now
    # diffracts differently -- the same rule the stopping-model marker follows.
    # Re-minted again for issue #88's physical Lorentzian-window marker, and
    # again for issue #100's derived photon-continuum floor: a case's
    # E_grid_brem now starts at the medium's own floor instead of 0 eV, so the
    # band the record was computed over genuinely changed and pre-floor records
    # must be orphaned rather than served for a different band. Re-minted again
    # for issue #91's `l-shell-ck-lorentzian-v5` marker: L-shell Coster--Kronig
    # redistribution changes the L line yields themselves, so v4 records are not
    # the same spectrum. Re-minted again for issue #89: default cases now select
    # the ELSEPA elastic model, which changes every trajectory. Re-minted again
    # for issue #181's `line_escape_model` marker: the incoherent line route
    # now scores the segment-mean escape, and the coherent route the per-piece
    # formation integral under absorption. Re-minted again for issue #91's
    # `eadl-cascade` marker: the EADL relaxation cascade replaces direct-vacancy
    # xraydb yields, so v6 records are not the same spectrum. Re-minted again
    # for issue #274's `attenuation_model` marker: every escape factor now
    # reads EPDL2025 mu instead of Chantler f2 + Elam scattering. Re-minted
    # again for issue #281: hopg has shell data, so the default
    # inelastic_model="auto" adds the shell soft/hard keys and every trajectory
    # changes. Re-minted again for issue #317: the default
    # atomic_electron_deflection="kawrakow" adds its key and rescales every
    # elastic rate. Re-minted for #187’s first-order Snell line marker.
    # #340: literal CIF parameters from Gemmi change rounding and the content key.
    # #513: static CBS potential U_g bumps the line-escape marker to v4.
    # #439: cancellation-free line kinematics bump the line-escape marker to v5.
    # #788: the default radiative-moment policy now participates in identity.
    # #278: an undeclared continuum grid resolves the refined geometric grid.
    # #278: angle-f64-v1 preserves forward emission angles on FP32 backends.
    # #1019: fp32-cell-mb-v1 forms FP32 BremsLib cells in millibarn.
    assert case_content_key(case) == (
        "28cb08fd9fd955372c8573f54ef64c39ce453d3c0e30d1b248782921f7e795b2"
    )


def test_dataset_identity_dispatches_through_recorded_v1():
    sweep = material_sweep("hopg")
    identity = dataset_identity("hopg", default_settings(), sweep)

    assert set(IDENTITY_MIGRATIONS) == {1}
    assert identity["identity_version"] == 1
    assert identity["parameter_sha256"] == (
        "fabe61e93e8396967e7a8d54875591e69af2c459d505980e46e48993ca03f03f"
    )
    with pytest.raises(ValueError, match="unsupported dataset identity version"):
        dataset_identity("hopg", default_settings(), sweep, identity_version=2)


def test_explicit_uniform_brem_grid_does_not_share_catalog_band_identity():
    """Same nodes, different case grid (exact vs stop at E0): digests must differ (#815)."""
    automatic = material_sweep("hopg")
    uniform = replace(automatic.energy_bins, brem=np.arange(0.0, 30000.0, 25.0))
    band = replace(automatic, energy_bins=uniform, brem_band=True)
    explicit = replace(band, brem_band=False)
    band_identity = dataset_identity("hopg", default_settings(), band)
    explicit_identity = dataset_identity("hopg", default_settings(), explicit)

    assert "brem_grid_exact" not in band_identity["resolved_parameters"]["sweep"]
    assert explicit_identity["resolved_parameters"]["sweep"]["brem_grid_exact"] is True
    assert explicit_identity["parameter_sha256"] != band_identity["parameter_sha256"]
    # The automatic continuum grid (#278) forks from both declared spellings.
    automatic_identity = dataset_identity("hopg", default_settings(), automatic)
    assert automatic_identity["resolved_parameters"]["sweep"]["continuum_grid_model"] == (
        CONTINUUM_GRID_MODEL
    )
    assert "continuum_grid_model" not in band_identity["resolved_parameters"]["sweep"]
    assert automatic_identity["parameter_sha256"] not in {
        band_identity["parameter_sha256"],
        explicit_identity["parameter_sha256"],
    }


def test_stack_continuum_grid_forks_only_stack_identities():
    """#1164: a stack's grid spans its layers' media, so only stacks fork the model."""
    slab = material_sweep("mos2")
    stack = replace(
        slab,
        target=Stack.on_substrate("mos2", 2000.0, "silicon", 2.0e5, footprint=None, tilt_deg=30.0),
    )
    assert automatic_continuum_stack_media(slab) == ()
    assert automatic_continuum_stack_media(stack) == ("silicon",)
    slab_sweep = dataset_identity("mos2", default_settings(), slab)["resolved_parameters"]["sweep"]
    stack_sweep = dataset_identity("mos2", default_settings(), stack)["resolved_parameters"][
        "sweep"
    ]
    assert slab_sweep["continuum_grid_model"] == CONTINUUM_GRID_MODEL
    assert stack_sweep["continuum_grid_model"] == STACK_CONTINUUM_GRID_MODEL


def test_high_energy_profile_range_is_part_of_dataset_identity():
    sweep = material_sweep("hbn", catalog_profile="high_energy")
    identity = named_profile_identity("hbn", catalog_profile="high_energy")

    energies = [100.0, 500.0, 1000.0, 5000.0]
    np.testing.assert_array_equal(sweep.beam.energy_keV, energies)
    assert identity["catalog_profile"] == "high_energy"
    assert identity["resolved_parameters"]["sweep"]["energy_keV"] == energies
    # 5 MeV is shared by every member since e7713682; #358 removed h-BN's
    # now-equal override row, which left its resolved range unchanged.
    np.testing.assert_array_equal(
        material_sweep("mose2", catalog_profile="high_energy").beam.energy_keV, energies
    )
    stored = tomllib.loads((USER_CATALOG_FIXTURE / "profiles/high_energy.toml").read_text())
    assert "overrides" not in stored


def test_high_energy_profile_selects_the_measured_line_grid():
    """#192: the profile's named policies reach every case and its identity."""
    policy = {
        "bandwidth": "resonance-population",
        "resolution": "resonance-local",
        "quadrature": "bin-mean",
    }
    sweep = material_sweep("hbn", catalog_profile="high_energy")
    assert sweep.cxr_grid_policy == policy
    identity = named_profile_identity("hbn", catalog_profile="high_energy")
    assert identity["resolved_parameters"]["sweep"]["line_grid_policy"] == policy
    case = build_cases(sweep, n_electrons=10, n_electrons_brem=10)[-1]
    assert case["E0_keV"] == 5000.0
    assert case["line_grid_policy"]["bandwidth"]["policy"] == "resonance-population"
    assert case["line_grid_policy"]["resolution"]["policy"] == "resonance-local"
    assert case["line_grid_policy"]["quadrature"] == "bin-mean"
    assert material_sweep("hbn").cxr_grid_policy is None


def test_hopg_short_keeps_finite_footprint_and_attosecond_bunch():
    sweep = material_sweep("hopg", catalog_profile="hopg_short")
    case = build_cases(sweep, n_electrons=300, n_electrons_brem=150)[0]

    assert sweep.target is not None
    assert sweep.target.footprint is not None
    assert case["crystal_width_mm"] == 5.0
    assert case["crystal_height_mm"] == 5.0
    longitudinal = case["longitudinal_distribution"]
    assert longitudinal["kind"] == "gaussian"
    assert longitudinal["rms_duration_fs"] == 0.001


def test_bundled_profiles_name_packaged_beam_and_detector():
    """Every bundled profile names a packaged beam and detector, so it runs even
    when a user catalog layer removes the bundled-catalog instrument exemption."""
    catalog = DATA_DIR / "catalog"
    for path in sorted((catalog / "profiles").glob("*.toml")):
        row = tomllib.loads(path.read_text())
        assert isinstance(row.get("beam"), str), path.stem
        assert (catalog / "beams" / f"{row['beam']}.toml").is_file(), path.stem
        references = row.get("detectors", {"default": row.get("detector")})
        for reference in references.values():
            assert isinstance(reference, str), path.stem
            assert (catalog / "detectors" / f"{reference}.toml").is_file(), path.stem


def test_example_beam_is_the_implicit_beam_spec():
    """Naming ``example`` must leave the identities of formerly beamless profiles alone."""
    beam = material_sweep("hopg", catalog_profile="quickstart").beam
    assert beam == BeamSpec(energy_keV=beam.energy_keV)


def test_trajectory_demo_is_one_fixed_count_case_with_micron_beam():
    assert CATALOG.profile_materials("trajectory_demo") == ("hopg",)
    sweep = material_sweep("hopg", catalog_profile="trajectory_demo")
    settings = profiles.profile_run_settings(default_settings(), "trajectory_demo", sweep=sweep)
    cases = build_cases(sweep)

    assert len(cases) == 1
    case = cases[0]
    assert case["crystal"] == "hopg"
    assert case["E0_keV"] == 30.0
    assert case["thickness_ang"] == 20_000.0
    assert case["tilt_deg"] == 45.0 and case["tilt_azim_deg"] == 180.0
    assert case["beam_fwhm_mm"] == 0.001  # 1 micrometre, not the default 1 mm spot
    assert sweep.beam.transverse_fwhm_y_mm == 0.001
    assert case["Ne"] == case["Ne_brem"] == 100
    assert settings.precision is None  # trajectory capture requires fixed counts


def _channeling_demo():
    sweep = material_sweep("silicon", catalog_profile="channeling_demo")
    settings = profiles.profile_run_settings(default_settings(), "channeling_demo", sweep=sweep)
    return sweep, settings


def test_channeling_demo_selects_one_fixed_count_planar_case_on_axis():
    from pyrite.detectors import SensorAcceptance

    assert CATALOG.profile_materials("channeling_demo") == ("silicon",)
    sweep, settings = _channeling_demo()
    assert settings.channeling_model == "planar"
    assert settings.channeling_plane == (1, -1, 0)
    assert settings.radiative_model == "uncoupled"
    assert settings.energy_model == "midpoint"
    assert settings.emission == "incoherent"
    assert settings.precision is None  # adaptive precision refuses channeling
    # The packaged forward detector is the documented API sensor resolved on axis.
    polar, _, omega = SensorAcceptance(distance_mm=1000.0, width_mm=3.0, height_mm=3.0).resolve()
    assert sweep.detector.observation_angle_deg == 0.0
    assert sweep.detector.polar_acceptance_deg == polar
    assert sweep.detector.solid_angle_sr == omega
    cxr = np.asarray(sweep.energy_bins.cxr)
    assert cxr[0] == 1000.0 and cxr[-1] == 23950.0  # Si K and the on-axis CR lines


def test_channeling_demo_lowers_to_one_planar_case():
    from pyrite import api

    sweep, settings = _channeling_demo()
    cases = api.build_configured_cases(sweep, settings)
    assert len(cases) == 1
    case = cases[0]
    assert case["crystal"] == "silicon"
    assert case["E0_keV"] == 17_000.0 and case["thickness_ang"] == 20_000.0
    assert case["Ne"] == case["Ne_brem"] == 200
    assert case["theta_obs_rad"] == 0.0
    assert case["channeling"]["plane_hkl"] == (1, -1, 0)
    assert case["channeling"]["basis_size"] == 16
    # The demo is the admitted planar entry and carries its measured budget (#1105).
    assert case["channeling"]["admission"]["case_key"] == "si110-17MeV-2um"
    assert case["channeling"]["admission"]["epsilon"] < 0.05
    assert settings.channeling_admission == "admitted"


def test_case_content_key_excludes_label_perf_and_flux_scale_fields():
    case = next(iter(_cases_by_key("hopg", "standard").values()))
    base = case_content_key(case)
    # Denylisted fields: changing any leaves the key unchanged.
    for field, value in (
        ("domega_sr", (case.get("domega_sr") or 0.0) + 1.0),
        ("beam_current_na", 99.0),
        ("rep_rate_hz", 12345.0),
        ("bunch_charge_pc", 7.0),
        ("mosaic_fwhm_rad", 0.01),
        ("spec_chunk", 4096),
        ("brem_chunk", 2048),
    ):
        assert case_content_key({**case, field: value}) == base, field
    # A profile-name-like label is not even a case field, so injecting one is
    # ignored too (proves label independence directly).
    assert case_content_key({**case, "catalog_profile": "whatever"}) == base
    assert case_content_key({**case, "name": "purely cosmetic label"}) == base
    # In-key physics fields DO change the key.
    assert case_content_key({**case, "E0_keV": case["E0_keV"] + 1.0}) != base
    assert case_content_key({**case, "seed": case["seed"] + 1}) != base
    assert case_content_key({**case, "thickness_ang": case["thickness_ang"] * 2}) != base


def test_full_profile_preserves_production_defaults_exactly():
    implicit_settings = default_settings()
    explicit_settings = default_settings()
    implicit_sweep = material_sweep("mose2")
    explicit_sweep = material_sweep("mose2")

    assert implicit_settings == explicit_settings
    assert implicit_settings.n_electrons == 300
    assert implicit_settings.n_electrons_brem == 150
    implicit_identity = dataset_identity("mose2", implicit_settings, implicit_sweep)
    explicit_identity = dataset_identity("mose2", explicit_settings, explicit_sweep)
    assert implicit_identity == explicit_identity


def test_transport_numerics_fork_dataset_identity_only_when_nondefault():
    settings = default_settings()
    sweep = material_sweep("hopg")
    base = dataset_identity("hopg", settings, sweep)
    explicit_defaults = dataset_identity(
        "hopg",
        replace(settings, straggling=False, energy_model="midpoint", max_dE_frac=0.0),
        sweep,
    )
    active = dataset_identity(
        "hopg",
        replace(settings, straggling=True, energy_model="midpoint", max_dE_frac=0.02),
        sweep,
    )

    assert explicit_defaults["parameter_sha256"] == base["parameter_sha256"]
    assert base["resolved_parameters"]["transport_numerics"] == {
        "energy_model": "midpoint",
        "inelastic_model": "shell-soft-hard",
        "inelastic_cutoff_eV": 50.0,
        "elastic_model": "elsepa",
        "atomic_electron_deflection": "kawrakow",
        "radiative_model": "bremslib-soft-hard",
        "radiative_moment_model": "radiative-moments-v1/positive-linear-log-energy-v1/rtol2e-4",
        "radiative_cutoff_eV": 1000.0,
    }
    assert active["parameter_sha256"] != base["parameter_sha256"]
    assert active["resolved_parameters"]["transport_numerics"] == {
        "straggling": True,
        "energy_model": "midpoint",
        "max_dE_frac": 0.02,
        "inelastic_model": "shell-soft-hard",
        "inelastic_cutoff_eV": 50.0,
        "elastic_model": "elsepa",
        "atomic_electron_deflection": "kawrakow",
        "radiative_model": "bremslib-soft-hard",
        "radiative_moment_model": "radiative-moments-v1/positive-linear-log-energy-v1/rtol2e-4",
        "radiative_cutoff_eV": 1000.0,
    }


def test_profile_convergence_overrides_defaults_and_forks_identity(monkeypatch):
    from types import MappingProxyType

    from pyrite.campaign import config
    from pyrite.materials import CATALOG

    baseline = material_sweep("hopg")
    custom = replace(
        CATALOG,
        profile_transport_numerics=MappingProxyType(
            {
                **CATALOG.profile_transport_numerics,
                "standard": MappingProxyType(
                    {
                        "n_families": 3,
                        "max_reflections": 6,
                        "mosaic_nodes": 7,
                        "mosaic_route": "mc",
                    }
                ),
            }
        ),
    )
    monkeypatch.setattr(config, "CATALOG", custom)

    resolved = config.material_sweep("hopg")
    assert resolved.n_families == 3
    assert resolved.max_reflections == 6
    assert resolved.mosaic_nodes == 7
    assert resolved.mosaic_route == "mc"
    overridden = config.material_sweep("hopg", n_families=8)
    assert overridden.n_families == 8
    assert (
        dataset_identity("hopg", default_settings(), resolved)["parameter_sha256"]
        != dataset_identity("hopg", default_settings(), baseline)["parameter_sha256"]
    )


def test_dataset_identity_is_deterministic_and_resolved_parameter_sensitive():
    settings = default_settings()
    sweep = material_sweep("mose2")
    first = dataset_identity("mose2", settings, sweep)
    same = dataset_identity("mose2", settings, sweep)
    changed = dataset_identity(
        "mose2",
        settings,
        replace(sweep, n_families=sweep.n_families + 1),
    )

    assert first == same
    assert first["parameter_sha256"] != changed["parameter_sha256"]
    assert first["resolved_parameters"]["settings"]["n_electrons"] == 300
    # Frozen legacy identity vocabulary (#1079): hashed and recorded verbatim.
    assert first["fidelity"] == first["resolved_parameters"]["fidelity"] == LEGACY_IDENTITY_TIER
    assert LEGACY_IDENTITY_TIER == "full"


def test_variant_stem_preserves_canonical_full_and_isolates_variants():
    full_settings = default_settings()
    full_sweep = material_sweep("hopg")
    full = dataset_identity("hopg", full_settings, full_sweep)
    quick = {**full, "variant": "quick"}
    # A retired survey-tier identity read back from its sidecar keeps its label.
    survey = {**full, "fidelity": "survey"}

    assert variant_stem(full, canonical_full=True) == "hopg"
    assert variant_stem(full).startswith("hopg@full-")
    assert variant_stem(quick, canonical_full=True) == "hopg_quick"
    assert variant_stem(survey).startswith("hopg@survey-")
    assert variant_stem(full) != variant_stem(survey)


def test_catalog_profile_leaves_standard_hash_bit_for_bit():
    """Phase 3 decision 2: catalog_profile only joins the hashed payload when
    it diverges from "standard", so every pre-existing standard-profile
    identity keeps its historical parameter_sha256 (and checkpoint stem)."""
    settings = default_settings()
    sweep = material_sweep("hopg")
    implicit = dataset_identity("hopg", settings, sweep)
    explicit = dataset_identity("hopg", settings, sweep, catalog_profile="standard")

    assert implicit["parameter_sha256"] == explicit["parameter_sha256"]
    assert implicit["catalog_profile"] == "standard"
    assert "catalog_profile" not in implicit["resolved_parameters"]


def test_line_kinematics_marker_is_a_constant_that_orphans_vacuum_era_digests():
    """The in-medium dispersion is unconditional physics, so it is not a
    divergence key any more -- but every digest minted before it became
    mandatory described the retired vacuum kinematics. The payload therefore
    carries a CONSTANT generation marker, hashed on every run, so those
    checkpoints cannot be resumed into (rev-and-re-run)."""
    identity = dataset_identity("hopg", default_settings(), material_sweep("hopg"))

    assert identity["resolved_parameters"]["line_kinematics"] == "in-medium"
    assert "xray_dispersion" not in identity["resolved_parameters"]
    assert "xray_dispersion" not in identity["resolved_parameters"]["settings"]


def test_stopping_model_marker_is_a_constant_that_orphans_splice_era_digests():
    """SBETHE is unconditional production physics and moves every old digest."""
    identity = dataset_identity("hopg", default_settings(), material_sweep("hopg"))

    assert identity["resolved_parameters"]["stopping_model"] == STOPPING_MODEL
    assert STOPPING_MODEL == "sbethe-corrected-v2"


def test_case_content_key_separates_stopping_models():
    """The stopping model is not a case field, so nothing in the case dict keeps
    a CAS blob computed under the retired pure Joy--Luo model from being served
    for a case that now transports differently. The key hashes the marker."""
    case = build_cases(material_sweep("hopg"), n_electrons=300, n_electrons_brem=150)[0]
    current = case_content_key(case)
    with mock.patch.object(profiles, "STOPPING_MODEL", "joy-luo-only"):
        joy_luo_era = case_content_key(case)

    assert joy_luo_era != current


def test_characteristic_model_marker_orphans_previous_line_models():
    """Unconditional characteristic physics, data, and line shape are hashed."""
    identity = dataset_identity("hopg", default_settings(), material_sweep("hopg"))

    assert identity["resolved_parameters"]["characteristic_model"] == CHARACTERISTIC_MODEL
    assert "eedl" in CHARACTERISTIC_MODEL
    assert "eadl" in CHARACTERISTIC_MODEL
    assert f"xraydb-{CHARACTERISTIC_XRAYDB_VERSION}" in CHARACTERISTIC_MODEL
    assert CHARACTERISTIC_MODEL.endswith("eadl-cascade-eadl-yields-lorentzian-segment-escape-v7")


def test_case_content_key_separates_characteristic_models():
    case = build_cases(material_sweep("hopg"), n_electrons=300, n_electrons_brem=150)[0]
    current = case_content_key(case)
    with mock.patch.object(profiles, "CHARACTERISTIC_MODEL", "characteristic-disabled"):
        pre_characteristic = case_content_key(case)

    assert pre_characteristic != current


def test_bremsstrahlung_model_marker_orphans_bethe_heitler_era_digests():
    settings = replace(
        default_settings(),
        bremsstrahlung_model="eedl",
        radiative_model="uncoupled",
        radiative_cutoff_eV=None,
    )
    identity = dataset_identity("hopg", settings, material_sweep("hopg"))

    assert identity["resolved_parameters"]["bremsstrahlung_model"] == BREMSSTRAHLUNG_MODEL
    assert "mf23-527-mf26-527" in BREMSSTRAHLUNG_MODEL


def test_default_continuum_is_bremslib_when_installed_and_eedl_otherwise(monkeypatch):
    installed = dataset_identity("hopg", default_settings(), material_sweep("hopg"))
    monkeypatch.setattr(profiles, "resolve_bremsstrahlung_model", lambda model, _e: "eedl")
    fallback = dataset_identity("hopg", default_settings(), material_sweep("hopg"))

    assert installed["resolved_parameters"]["bremsstrahlung_model"] == BREMSSTRAHLUNG_BREMSLIB_MODEL
    assert fallback["resolved_parameters"]["bremsstrahlung_model"] == BREMSSTRAHLUNG_MODEL
    assert installed["parameter_sha256"] != fallback["parameter_sha256"]


def test_case_content_key_separates_bremsstrahlung_models():
    case = build_cases(
        material_sweep("hopg"),
        n_electrons=300,
        n_electrons_brem=150,
        bremsstrahlung_model="eedl",
        radiative_model="uncoupled",
    )[0]
    current = case_content_key(case)
    with mock.patch.object(profiles, "BREMSSTRAHLUNG_MODEL", "bethe-heitler-only"):
        bethe_heitler_era = case_content_key(case)

    assert bethe_heitler_era != current


def test_standard_detector_keeps_current_payload_and_digest_bit_for_bit():
    identity = dataset_identity("hopg", default_settings(), material_sweep("hopg"))
    sweep_payload = identity["resolved_parameters"]["sweep"]

    assert identity["parameter_sha256"] == (
        "fabe61e93e8396967e7a8d54875591e69af2c459d505980e46e48993ca03f03f"
    )
    assert "detector" not in sweep_payload
    assert sweep_payload["theta_obs_deg"] == 90.0
    assert sweep_payload["dtheta_obs_deg"] is None
    assert sweep_payload["domega_sr"] is None


@pytest.mark.parametrize(
    ("catalog_profile", "material", "digest"),
    [
        (
            "hopg_hbn_gaussian_200fs",
            "hopg",
            "307283f1bbf91a1ba36030088a865790a65353d768ff0decce9938f4bba9bd78",
        ),
        (
            "hopg_hbn_gaussian_200fs",
            "hbn",
            "936e06223fb22ca2a2d605df47e8cbbe7d1951ee7a2443767fa25abac6035e0a",
        ),
        (
            "hopg_hbn_microtrain_200fs",
            "hopg",
            "30e8609523002485dac161ca53473bfecb065442c32cf8c435c3dc46f4d1633b",
        ),
        (
            "hopg_hbn_microtrain_200fs",
            "hbn",
            "fc8a950f22464d2ef300f603655f7f2985f0163a0e0059505fb06c49108a6421",
        ),
        (
            "hopg_hbn_compressed_microbunch",
            "hopg",
            "090d5197d979ffefcc997beae0cb1484fe47edb7540692663e18286aab18beee",
        ),
        (
            "hopg_hbn_compressed_microbunch",
            "hbn",
            "d54444000b19edf136272f179bfedc8c7c414a001f5b2ad8b6d5bfe8f7980bde",
        ),
        (
            "hopg_emittance_demo",
            "hopg",
            "5b8354201944a94e43acadf30fc51adb7bd02baf697c0723b31d42a481277e3c",
        ),
        (
            "promising_low_ne",
            "hopg",
            # #465: coherent operator identity advanced to physical-distinct-pairs-v4.
            "a069259def39e687dca5a4ee4b70a7922f72770b1e08f6abcaa54951aeaa6695",
        ),
    ],
)
def test_named_beam_migration_keeps_shipped_profile_digests_bit_for_bit(
    lab_catalog, catalog_profile, material, digest
):
    """Every shipped profile that carried an inline ``[profiles.NAME.beam]`` block
    now carries ``beam = "NAME"`` instead. The reference resolves to values before
    hashing, so the migration itself moved no digest. These pins were re-minted
    once when the in-medium line kinematics became unconditional, which
    deliberately orphaned every vacuum-era checkpoint stem; they must stay
    bit-for-bit from here. They were re-minted again when unconditional EEDL
    characteristic radiation was introduced, for the EEDL bremsstrahlung
    generation marker, for natural Lorentzian characteristic profiles, for
    issue #88's physical finite-window convention, for issue #91's
    `l-shell-ck-lorentzian-v5` L-shell Coster--Kronig relaxation marker, for
    issue #89's ELSEPA elastic model becoming the default, for issue
    #181's segment-mean line-escape marker, and for issue #317's default
    atomic-electron deflection, and for #187’s Snell resonance marker.
    ``promising_low_ne`` alone moved for #361, when predicted identities began
    applying the profile's own run settings: its ``emission = "both"`` is in
    every identity the runner writes, and the earlier pin omitted it.
    The profiles without fixed electron counts moved again for #361's
    adaptive default: their identities now carry the default precision policy.
    ``promising_low_ne`` alone moved again for #350 (afc4c418): identities with
    any coherent emission now carry the ``coherent_population_model`` marker
    (``physical-distinct-pairs-v1``) for the physical-bunch-charge pair
    weighting. The other pinned profiles use incoherent emission, which omits
    the marker, so their digests held. #370 moved it once more: the marker
    became ``physical-distinct-pairs-v2`` (finite-footprint transverse form
    factor and tilted face-arrival delay), then ``physical-distinct-pairs-v3``
    for #365 (flat omission certified by ``F_z F_perp``).
    Issue #439 re-minted every pin for the ``segment-mean-v5-precise-kinematics``
    line-escape marker and the BremsLib
    ``…/lin-x18-geo4-ln-t-lin-theta-v2/tm-coupled-v2/ee-sb85-v1`` marker.

    Issue #1099 re-minted every pin for the ``segment-mean-v6-exact-recoil``
    line-escape marker.

    Issue #1134 re-minted every pin for the
    ``segment-mean-v7-quantum-elements`` line-escape marker.

    Issue #938 re-minted adaptive BremsLib/brem pins for the conditional
    ``brem_band_reduction_model`` generation; fixed-count identities hold.

    Issue #278 re-minted every pin: a profile that declares no ``E_grid_brem``
    now carries ``continuum_grid_model`` (``refined-geometric-v2/dlnE=0.0025``),
    so results on the earlier uniform 25 eV default never share a dataset.

    Issue #278 re-minted the pins of profiles whose fixtures declared a uniform
    ``E_grid_brem``: they now resolve the automatic refined grid, whose
    ``continuum_grid_model`` (``refined-geometric-v2/dlnE=0.0025``) is in the identity.

    Issue #1171 re-minted every automatic-grid pin for
    ``refined-geometric-v3``: anchors survive overlapping refinement marks.

    Issue #278 re-minted BremsLib pins for ``angle-f64-v1``: direction cosines
    retain binary64 precision through inverse cosine on every backend.

    Issue #1019 re-minted BremsLib pins for ``fp32-cell-mb-v1``: FP32 cells
    are formed in millibarn; binary64 output is unchanged.

    Issue #100's derived photon-continuum floor deliberately did NOT move these:
    it raises a brem grid's ``start`` where the band meets the material, in
    ``build_cases``, so the declared sweep payload these digests hash is
    untouched. The *case* content key does move; see
    :func:`test_typed_case_content_key_matches_pre_case_golden`."""
    identity = named_profile_identity(material, catalog_profile=catalog_profile)

    assert identity["parameter_sha256"] == digest


def test_nondefault_detector_round_trips_cases_identity_and_stem():
    settings = default_settings()
    standard_sweep = material_sweep("hopg")
    detector = Detector(
        119.0,
        response=Timepix3(thickness_um=500.0),
        acceptance=AngularAcceptance(polar_deg=16.6, solid_angle_sr=0.066),
    )
    custom_sweep = material_sweep("hopg", detector=detector)
    standard = dataset_identity("hopg", settings, standard_sweep)
    custom = dataset_identity("hopg", settings, custom_sweep)
    case = build_cases(custom_sweep)[0]
    payload = custom["resolved_parameters"]["sweep"]

    assert custom_sweep.detector == detector
    assert case["theta_obs_rad"] == pytest.approx(np.deg2rad(119.0))
    assert case["dtheta_obs_rad"] == pytest.approx(np.deg2rad(16.6))
    assert case["domega_sr"] == pytest.approx(0.066)
    assert payload["theta_obs_deg"] == 119.0
    assert payload["dtheta_obs_deg"] == 16.6
    assert payload["domega_sr"] == 0.066
    assert "detector" not in payload
    assert custom["parameter_sha256"] != standard["parameter_sha256"]
    assert variant_stem(custom) != variant_stem(standard)


@pytest.mark.parametrize(
    "detector",
    [
        Detector(observation_angle_deg=119.0),
        Detector(_legacy_acceptance=(16.6, None)),
        Detector(_legacy_acceptance=(None, 0.066)),
    ],
)
def test_every_nondefault_detector_acceptance_field_changes_identity(detector):
    settings = default_settings()
    standard_sweep = material_sweep("hopg")
    standard = dataset_identity("hopg", settings, standard_sweep)
    changed = dataset_identity("hopg", settings, replace(standard_sweep, detector=detector))

    assert changed["parameter_sha256"] != standard["parameter_sha256"]


def test_read_time_response_does_not_change_transport_identity():
    settings = default_settings()
    standard_sweep = material_sweep("hopg")
    rescored = replace(
        standard_sweep,
        detector=replace(
            standard_sweep.detector,
            response=Timepix3(thickness_um=500.0),
        ),
    )

    assert (
        dataset_identity("hopg", settings, rescored)["parameter_sha256"]
        == dataset_identity("hopg", settings, standard_sweep)["parameter_sha256"]
    )


def test_catalog_profile_changes_hash_and_stem_when_not_standard():
    settings = default_settings()
    sweep = material_sweep("hopg")
    standard = dataset_identity("hopg", settings, sweep, catalog_profile="standard")
    other = dataset_identity("hopg", settings, sweep, catalog_profile="sub_100keV")

    assert other["catalog_profile"] == "sub_100keV"
    assert other["resolved_parameters"]["catalog_profile"] == "sub_100keV"
    assert other["parameter_sha256"] != standard["parameter_sha256"]
    # variant_stem itself takes canonical_full as given; a non-standard
    # catalog profile's dataset never sets canonical_full=True (callers,
    # e.g. named_profile_stem/scan._resolved_run, withhold it) -- see
    # test_named_profile_stem_default_catalog_profile_stays_canonical and
    # tests/materials/test_material_catalog.py for that end-to-end contract.
    assert variant_stem(other) != "hopg"
    # @-stem now carries the non-standard catalog_profile name as its label.
    assert variant_stem(other).startswith("hopg@sub_100keV-")


def test_named_profile_stem_default_catalog_profile_is_adaptive_and_hashed():
    """standard runs adaptive counts by default (#361): its stem is a variant."""
    assert named_profile_stem("hopg").startswith("hopg@full-")
    assert named_profile_stem("hopg", catalog_profile="standard") == named_profile_stem("hopg")


def test_default_settings_use_incoherent_emission():
    """Back-compat: the built-in run settings stay on the incoherent policy,
    so existing runs keep the bit-for-bit incoherent path."""
    for settings in (default_settings(), default_settings().__class__()):
        assert settings.emission == "incoherent"
        assert settings.coherent_emission is False


@pytest.mark.parametrize(
    ("emission", "coherent"),
    [("incoherent", False), ("coherent", True), ("both", True)],
)
def test_emission_mode_resolves_through_profile_to_settings(emission, coherent):
    """A profile emission rides profile_run_settings onto Settings, and the
    derived coherent_emission property agrees."""
    catalog = mock.Mock()
    catalog.profile_emission.return_value = emission
    catalog.profile_temporal_profile.return_value = False
    catalog.profile_numerics.return_value = None
    catalog.profile_precision.return_value = None
    resolved = profiles.profile_run_settings(default_settings(), "probe", catalog)
    assert resolved.emission == emission
    assert resolved.coherent_emission is coherent


def test_emission_modes_yield_three_distinct_digests_incoherent_unchanged():
    """dataset_identity emits a distinct digest per emission mode, and the
    incoherent path still adds no key of its own to the hashed payload."""
    sweep = material_sweep("hopg")
    base = default_settings()
    incoherent = dataset_identity("hopg", replace(base, emission="incoherent"), sweep)
    coherent = dataset_identity("hopg", replace(base, emission="coherent"), sweep)
    both = dataset_identity("hopg", replace(base, emission="both"), sweep)

    digests = {
        incoherent["parameter_sha256"],
        coherent["parameter_sha256"],
        both["parameter_sha256"],
    }
    assert len(digests) == 3

    # incoherent adds no divergence key; coherent/both each surface their token.
    assert "emission" not in incoherent["resolved_parameters"]
    assert coherent["resolved_parameters"]["emission"] == "coherent"
    assert both["resolved_parameters"]["emission"] == "both"

    # Known incoherent digest (current baseline, re-minted when the in-medium
    # line kinematics became unconditional, again when the Joy--Luo/
    # Berger--Seltzer stopping splice did, and again when every crystal cut
    # moved to the surface_hkl spelling, and again for natural Lorentzian
    # characteristic profiles, and again for issue #88's physical finite-window
    # convention, again for issue #125's automatic bundled line grids, again
    # for issue #91's L-shell Coster--Kronig relaxation marker, and again for
    # issue #89's default ELSEPA elastic model, and again for the BremsLib
    # default continuum of issue #86, and again for issue #181's line-escape
    # marker, and again for issue #91's EADL relaxation cascade, and again for
    # issue #256's dropped uniform E_grid_brem override, which the sweep-level
    # payload hashes although no case grid changed) must stay bit-for-bit.
    assert incoherent["parameter_sha256"] == (
        "fabe61e93e8396967e7a8d54875591e69af2c459d505980e46e48993ca03f03f"
    )


def test_identity_from_stem_resolves_non_standard_catalog_profile():
    """Regression: ``pyrite remote pull --profile`` died here -- a variant stem
    names only material/label/digest, and identity_from_stem used to
    recompute the digest under "standard" only, so a sub_100keV checkpoint
    could never match (remote `pyrite slim --grid` SystemExit, 21/21 failed)."""
    stem = named_profile_stem("hopg", catalog_profile="sub_100keV")
    identity = identity_from_stem(stem)

    assert identity is not None
    assert identity["material"] == "hopg"
    assert identity["fidelity"] == "full"
    assert identity["catalog_profile"] == "sub_100keV"
    # standard-profile variant stems keep resolving as before
    standard = named_profile_stem("hopg")
    assert identity_from_stem(standard)["catalog_profile"] == "standard"


def test_identity_from_stem_dual_reads_legacy_hyphen_stem():
    """Dual-read: checkpoints written under the pre-2026-07-29
    ``<material>--<tier>-<digest>`` scheme still resolve after the @-stem
    migration, so existing on-disk checkpoints are never orphaned. The @-stem
    and its legacy twin share a digest -- only the stem text differs."""
    identity = named_profile_identity("hopg")
    digest = identity["parameter_sha256"][:12]
    legacy_stem = f"hopg--full-{digest}"
    at_stem = named_profile_stem("hopg")

    assert at_stem == f"hopg@full-{digest}"  # new scheme is what gets written
    resolved = identity_from_stem(legacy_stem)  # old scheme still reads back
    assert resolved is not None
    assert (resolved["material"], resolved["fidelity"], resolved["catalog_profile"]) == (
        "hopg",
        "full",
        "standard",
    )


@pytest.mark.parametrize("stem", ["hopg--survey-0123456789ab", "hopg@survey-0123456789ab"])
def test_retired_survey_stems_resolve_only_from_their_sidecar(tmp_path, stem):
    """#1079: the survey tier's reduced grid is gone, so a sidecar-less survey
    stem cannot be recomputed -- but its recorded identity still reads back."""
    assert identity_from_stem(stem) is None
    recorded = {
        **named_profile_identity("hopg"),
        "fidelity": "survey",
        "parameter_sha256": "0123456789ab" + "0" * 52,
    }
    manifest = _checkpoint_store.manifest_path(stem, tmp_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"dataset_identity": recorded}))
    assert identity_from_stem(stem, tmp_path)["fidelity"] == "survey"


# Golden pins (#1079): retiring the fidelity machinery must leave every named
# stem byte-identical. Re-pin together with the digests above on a *_MODEL bump.
@pytest.mark.parametrize(
    ("profile", "material", "stem"),
    [
        ("standard", "hopg", "hopg@full-a56397dab3df"),
        ("standard", "mose2", "mose2@full-ec76b1a02157"),
    ],
)
def test_named_profile_stems_are_pinned(profile, material, stem):
    assert named_profile_stem(material, catalog_profile=profile) == stem


def test_identity_from_stem_reads_sidecar_when_profile_edited_after_run(tmp_path):
    """Regression (2026-07-29): a checkpoint written under a named profile that
    is edited *after* the run no longer resolves via recompute -- the stem's
    digest was fixed at run time, and the edited profile now hashes to a
    different value, so no candidate recompute matches. But the run-time
    ``dataset_identity`` recorded in the stem's ``meta.json`` sidecar is
    authoritative and still resolves. Mirrors the two on-disk
    named-profile stems found that day (two runs a minute apart,
    profile edited in between, neither recoverable by recompute)."""
    # Identity as recorded at run time. A fabricated digest stands in for "the
    # backing profile was later edited": the stem commits to this run-time
    # digest, which no recompute against the current catalog can reproduce.
    run_time_identity = {
        **named_profile_identity("hopg", catalog_profile="sub_100keV"),
        "parameter_sha256": "dead" * 16,  # 64 hex chars, un-recomputable today
    }
    stem = variant_stem(run_time_identity)
    assert stem == "hopg@sub_100keV-deaddeaddead"

    # Recompute-only path fails exactly as it did in the field.
    assert identity_from_stem(stem) is None

    # Write the run-time sidecar the sweep would have left beside the checkpoint.
    manifest = _checkpoint_store.manifest_path(stem, tmp_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"dataset_identity": run_time_identity}))

    # Sidecar read recovers the run-time identity regardless of the edit. The
    # fabricated digest proves the result came from the sidecar, not recompute.
    recovered = identity_from_stem(stem, tmp_path)
    assert recovered is not None
    assert recovered["parameter_sha256"] == "dead" * 16
    assert recovered["material"] == "hopg"
    assert recovered["fidelity"] == "full"
    assert recovered["catalog_profile"] == "sub_100keV"

    # A stem with no sidecar under the given root still falls back to recompute
    # (which returns None for this fabricated digest).
    assert identity_from_stem(stem, tmp_path / "no-such-checkpoint") is None


@pytest.mark.slow
def test_shell_inelastic_mode_forks_identity_and_case_payload_only_when_on():
    default = replace(default_settings(), energy_model="midpoint")
    settings = replace(default, inelastic_model="continuous", inelastic_cutoff_eV=None)
    sweep = material_sweep("silicon")
    base = dataset_identity("silicon", settings, sweep)
    shell = replace(settings, inelastic_model="shell-soft-hard", inelastic_cutoff_eV=50.0)
    active = dataset_identity("silicon", shell, sweep)
    # The "auto" default (#281) resolves silicon to the shell mode at 50 eV.
    explicit_default = dataset_identity("silicon", default, sweep)
    other_cutoff = dataset_identity("silicon", replace(shell, inelastic_cutoff_eV=100.0), sweep)

    assert explicit_default["parameter_sha256"] == active["parameter_sha256"]
    assert "inelastic_model" not in base["resolved_parameters"]["transport_numerics"]
    assert active["resolved_parameters"]["transport_numerics"] == {
        "energy_model": "midpoint",
        "inelastic_model": "shell-soft-hard",
        "inelastic_cutoff_eV": 50.0,
        "elastic_model": "elsepa",
        "atomic_electron_deflection": "kawrakow",
        "radiative_model": "bremslib-soft-hard",
        "radiative_moment_model": "radiative-moments-v1/positive-linear-log-energy-v1/rtol2e-4",
        "radiative_cutoff_eV": 1000.0,
    }
    assert (
        len(
            {base["parameter_sha256"], active["parameter_sha256"], other_cutoff["parameter_sha256"]}
        )
        == 3
    )

    legacy_case = build_cases(sweep, 4, 4, energy_model="midpoint", inelastic_model="continuous")[0]
    shell_case = build_cases(
        sweep,
        4,
        4,
        energy_model="midpoint",
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
    )[0]
    assert "inelastic_model" not in legacy_case
    assert shell_case["inelastic_model"] == "shell-soft-hard"
    assert shell_case["inelastic_cutoff_eV"] == 50.0
    assert profiles.case_content_key(shell_case) != profiles.case_content_key(legacy_case)


def test_secondary_threshold_forks_identity_and_case_payload_only_when_set():
    """Secondary transport (#94) is a divergence-only transport key."""
    settings = replace(
        default_settings(),
        energy_model="midpoint",
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
    )
    sweep = material_sweep("silicon")
    base = dataset_identity("silicon", settings, sweep)
    on = dataset_identity("silicon", replace(settings, secondary_threshold_eV=1000.0), sweep)
    other = dataset_identity("silicon", replace(settings, secondary_threshold_eV=2000.0), sweep)
    assert "secondary_threshold_eV" not in base["resolved_parameters"]["transport_numerics"]
    assert on["resolved_parameters"]["transport_numerics"]["secondary_threshold_eV"] == 1000.0
    assert len({base["parameter_sha256"], on["parameter_sha256"], other["parameter_sha256"]}) == 3

    shell = {"energy_model": "midpoint", "inelastic_model": "shell-soft-hard"}
    plain = build_cases(sweep, 4, 4, inelastic_cutoff_eV=50.0, **shell)[0]
    secondary = build_cases(
        sweep, 4, 4, inelastic_cutoff_eV=50.0, secondary_threshold_eV=1000.0, **shell
    )[0]
    assert "secondary_threshold_eV" not in plain
    assert secondary["secondary_threshold_eV"] == 1000.0
    assert profiles.case_content_key(secondary) != profiles.case_content_key(plain)
    with pytest.raises(ValueError, match="requires inelastic_model='shell-soft-hard'"):
        build_cases(sweep, 4, 4, inelastic_model="continuous", secondary_threshold_eV=1000.0)
    with pytest.raises(ValueError, match="requires inelastic_model='shell-soft-hard'"):
        replace(
            default_settings(),
            inelastic_model="continuous",
            inelastic_cutoff_eV=None,
            secondary_threshold_eV=1000.0,
        )


def test_default_elsepa_model_forks_identity_and_case_payload_from_mott():
    """ELSEPA is the default; the historical Mott model keeps its old payload."""
    settings = default_settings()
    sweep = material_sweep("silicon")
    base = dataset_identity("silicon", settings, sweep)
    explicit_default = dataset_identity("silicon", replace(settings, elastic_model="elsepa"), sweep)
    mott = dataset_identity("silicon", replace(settings, elastic_model="mott"), sweep)

    assert settings.elastic_model == "elsepa"
    assert explicit_default["parameter_sha256"] == base["parameter_sha256"]
    assert base["resolved_parameters"]["transport_numerics"] == {
        "energy_model": "midpoint",
        "inelastic_model": "shell-soft-hard",
        "inelastic_cutoff_eV": 50.0,
        "elastic_model": "elsepa",
        "atomic_electron_deflection": "kawrakow",
        "radiative_model": "bremslib-soft-hard",
        "radiative_moment_model": "radiative-moments-v1/positive-linear-log-energy-v1/rtol2e-4",
        "radiative_cutoff_eV": 1000.0,
    }
    assert "elastic_model" not in mott["resolved_parameters"]["transport_numerics"]
    assert mott["parameter_sha256"] != base["parameter_sha256"]

    default_case = build_cases(sweep, 4, 4)[0]
    mott_case = build_cases(sweep, 4, 4, elastic_model="mott")[0]
    assert default_case["elastic_model"] == "elsepa"
    assert "elastic_model" not in mott_case
    assert profiles.case_content_key(default_case) != profiles.case_content_key(mott_case)
    with pytest.raises(ValueError, match="elastic_model"):
        build_cases(sweep, 4, 4, elastic_model="nope")


def test_mixed_transport_forks_identity_by_parameter():
    """#894: the default adds no identity key; mode and parameters each fork it."""
    settings = default_settings()
    sweep = material_sweep("hopg")
    digests = {
        dataset_identity("hopg", variant, sweep)["parameter_sha256"]
        for variant in (
            settings,
            replace(settings, transport_mode="mixed"),
            replace(settings, transport_mode="mixed", mixed_c1=0.1),
            replace(settings, transport_mode="mixed", mixed_max_step_um=5.0),
        )
    }
    assert len(digests) == 4


def test_atomic_electron_deflection_forks_identity_and_none_restores_it():
    """#317 is on by default; "none" omits its key; pins include #187’s Snell line marker."""
    settings = default_settings()
    sweep = material_sweep("hopg")
    off = replace(settings, atomic_electron_deflection="none")

    assert settings.atomic_electron_deflection == "kawrakow"
    # The pre-#317 pins of test_dataset_identity_dispatches_through_recorded_v1
    # and test_typed_case_content_key_matches_pre_case_golden.
    assert dataset_identity("hopg", off, sweep)["parameter_sha256"] == (
        "bec00b5a89da7972bebd5c801f6ac49e50fa2301dd588ae515b8224fd54b0b44"
    )
    case = build_cases(
        sweep, n_electrons=300, n_electrons_brem=150, atomic_electron_deflection="none"
    )[0]
    assert "atomic_electron_deflection" not in case
    assert case_content_key(case) == (
        "ed9520ba7a3d1b23ee62e00691d6a3a84b60fe0136a70acd925660157e804dc4"
    )
    with pytest.raises(ValueError, match="atomic_electron_deflection"):
        build_cases(sweep, 4, 4, atomic_electron_deflection="z1")


def test_mott_shell_cases_keep_elastic_only_deflection_with_a_warning():
    sweep = material_sweep("silicon")
    with pytest.warns(UserWarning, match="needs elastic_model='elsepa'"):
        shell = build_cases(sweep, 4, 4, elastic_model="mott")[0]
    continuous = build_cases(sweep, 4, 4, elastic_model="mott", inelastic_model="continuous")[0]
    assert shell["inelastic_model"] == "shell-soft-hard"
    assert "atomic_electron_deflection" not in shell
    assert continuous["atomic_electron_deflection"] == "kawrakow"
    mott = dataset_identity("silicon", replace(default_settings(), elastic_model="mott"), sweep)
    assert "atomic_electron_deflection" not in mott["resolved_parameters"]["transport_numerics"]


_PACKAGED_PROFILES = sorted(
    path.stem for path in (DATA_DIR / "catalog" / "profiles").glob("*.toml")
)


@pytest.mark.parametrize("catalog_profile", _PACKAGED_PROFILES)
def test_undeclared_case_brem_grid_is_automatic_refined_geometric(catalog_profile):
    """Issue #278: a profile that declares no ``E_grid_brem`` resolves the
    automatic refined geometric continuum grid per case (medium floor to just
    past E0, ``dlnE = CONTINUUM_LOG_STEP`` plus edge/endpoint anchors) instead
    of the historical uniform 25 eV band."""
    for material in CATALOG.material_keys:
        sweep = material_sweep(
            material,
            catalog_profile=catalog_profile,
            thickness_ang=1e4,
            tilt_deg=45.0,
            tilt_azim_deg=135.0,
        )
        assert sweep.energy_bins.brem is None, (catalog_profile, material)
        medium = continuum_medium_key(sweep.material)
        # A catalog stack's grid also spans its layers' media (#1164).
        stack_media = automatic_continuum_stack_media(sweep)
        for case in build_cases(
            sweep, n_electrons=300, n_electrons_brem=150, inelastic_model="auto"
        ):
            endpoint = case["E0_keV"] * 1e3
            grid = np.asarray(case["E_grid_brem"], dtype=float)
            np.testing.assert_array_equal(
                grid, automatic_continuum_grid(medium, endpoint, stack_media=stack_media)
            )
            assert np.all(np.diff(grid) > 0.0)
            assert grid[-1] > endpoint
            # Anchor merging drops samples near an anchor, widening a neighbour
            # interval by a fraction of a step; it never doubles one.
            assert np.max(np.diff(np.log(grid))) < 2.0 * CONTINUUM_LOG_STEP


@pytest.mark.parametrize("catalog_profile", _PACKAGED_PROFILES)
def test_packaged_profiles_store_no_uniform_per_material_brem_override(catalog_profile):
    raw = tomllib.loads((DATA_DIR / "catalog" / "profiles" / f"{catalog_profile}.toml").read_text())
    assert "E_grid_brem" not in raw
    for material, row in raw.get("overrides", {}).items():
        assert "arange" not in row.get("E_grid_brem", {}), material


def test_member_overrides_copies_only_member_rows_without_uniform_brem():
    document = {
        "materials": {"sapphire": {}, "mote2_product": {}, "hopg": {}},
    }
    packaged = (
        "[overrides.sapphire]\nthickness_ang = 5000000.0\n"
        "[overrides.mote2_product]\nthickness_layers = 3\n"
        "E_grid_brem = { arange = { start = 75.0, stop = 43400.0, step = 25.0 } }\n"
        "[overrides.hopg]\n"
        "E_grid_brem = { arange = { start = 50.0, stop = 40000.0, step = 25.0 } }\n"
    )
    with mock.patch.object(
        profile_lifecycle, "_packaged_standard", return_value=tomlkit.parse(packaged)
    ):
        members = profile_lifecycle.member_overrides(document, {"mote2_product", "hopg"})
        implicit = profile_lifecycle.member_overrides(document, None)

    assert members.unwrap() == {"mote2_product": {"thickness_layers": 3}}
    assert implicit.unwrap() == {
        "sapphire": {"thickness_ang": 5000000.0},
        "mote2_product": {"thickness_layers": 3},
    }


@pytest.mark.parametrize("name", BUNDLED_PROFILES)
def test_reference_detector_migration_preserves_dataset_and_case_identity(name):
    from pyrite.campaign.config import catalog_detector
    from pyrite.campaign.sweep import TIMEPIX3_DOMEGA_SR, TIMEPIX3_DTHETA_OBS_DEG

    detector = catalog_detector(name, "default")
    assert detector.observation_angle_deg == 90.0
    assert detector.polar_acceptance_deg == TIMEPIX3_DTHETA_OBS_DEG
    assert detector.solid_angle_sr == TIMEPIX3_DOMEGA_SR
    assert detector.response is None
    spectrum = np.array([2.0, 5.0, 3.0])
    np.testing.assert_array_equal(
        detector.score(np.array([100.0, 200.0, 300.0]), spectrum), spectrum
    )

    sweep = material_sweep("hopg", catalog_profile=name, detector_id="default")
    # Before #423 bundled scalar objects contained only observation_angle_deg.
    legacy = replace(
        sweep,
        detector=Detector(
            observation_angle_deg=sweep.detector.observation_angle_deg,
            response=sweep.detector.response,
        ),
        energy_bins=sweep.energy_bins,
    )
    settings = profiles.profile_run_settings(default_settings(), name)
    assert dataset_identity("hopg", settings, sweep, catalog_profile=name) == dataset_identity(
        "hopg", settings, legacy, catalog_profile=name
    )
    # Reduce the product without transport; cases must retain the same CAS keys.
    changes = {
        "beam": replace(sweep.beam, energy_keV=np.array([30.0])),
        "target": target_replace(
            sweep.target,
            thickness_ang=np.array([2000.0]),
            tilt_deg=np.array([45.0]),
            tilt_azim_deg=np.array([180.0]),
        ),
    }
    new_case = build_cases(replace(sweep, **changes), 4, 4)[0]
    old_case = build_cases(replace(legacy, **changes), 4, 4)[0]
    assert case_content_key(new_case) == case_content_key(old_case)


def test_bundled_profile_display_has_explicit_detector_acceptance():
    from pyrite._catalog_layout import bundled_catalog, read_text

    document = tomlkit.parse(read_text(bundled_catalog()))
    assert "default" not in document["detectors"]
    for name in BUNDLED_PROFILES:
        payload = profile_summary.profile_payload(document, name)
        assert payload["detector_ref"] == "reference"
        for detector in payload["detectors"].values():
            assert all(value is not None for value in detector["acceptance"].values())


@pytest.mark.parametrize("material", ["hopg", "mos2", "mote2_product"])
def test_explicit_brem_grid_keeps_its_nodes_and_forks_from_automatic(material):
    sweep = material_sweep(material, thickness_ang=1e4, tilt_deg=45.0, tilt_azim_deg=135.0)
    explicit = replace(
        sweep, energy_bins=replace(sweep.energy_bins, brem=np.arange(0.0, 30000.0, 25.0))
    )
    settings = default_settings()
    implicit_identity = dataset_identity(material, settings, sweep)
    explicit_identity = dataset_identity(material, settings, explicit)
    assert implicit_identity["parameter_sha256"] != explicit_identity["parameter_sha256"]
    assert "continuum_grid_model" not in explicit_identity["resolved_parameters"]["sweep"]
    implicit_cases = build_cases(sweep, n_electrons=1, n_electrons_brem=1, inelastic_model="auto")
    explicit_cases = build_cases(
        explicit, n_electrons=1, n_electrons_brem=1, inelastic_model="auto"
    )
    for implicit_case, explicit_case in zip(implicit_cases, explicit_cases, strict=True):
        # A declared uniform grid keeps its tuple spelling; the automatic one is exact nodes.
        assert isinstance(explicit_case["E_grid_brem"], tuple)
        assert isinstance(implicit_case["E_grid_brem"], np.ndarray)
        assert case_content_key(implicit_case) != case_content_key(explicit_case)


def test_declared_brem_keeps_declared_grid_and_drops_builtin_default():
    from pyrite.campaign.config import _material_spec, declared_brem

    scan = _material_spec("hopg").scan
    assert not scan.E_grid_brem_declared
    assert declared_brem(scan) is None
    declared = replace(scan, E_grid_brem_declared=True)
    np.testing.assert_array_equal(declared_brem(declared), scan.E_grid_brem)


def test_brem_band_reduction_generation_forks_only_adaptive_bremslib(monkeypatch):
    from pyrite import _precision
    from pyrite._precision import Precision
    from pyrite.campaign.physics_view import resolved_physics_view

    sweep = material_sweep("hopg")
    fixed = default_settings()
    adaptive = replace(fixed, precision=Precision(0.05, 100, 200, 100, observables=("brem",)))
    cxr_only = replace(adaptive, precision=replace(adaptive.precision, observables=("cxr",)))
    configurations = [
        (fixed, "bremslib"),
        (adaptive, "bremslib"),
        (cxr_only, "bremslib"),
        (adaptive, "eedl"),
    ]
    identities = [
        dataset_identity("hopg", settings, sweep, bremsstrahlung_model=model)
        for settings, model in configurations
    ]
    case = {
        **build_cases(sweep)[0],
        "adaptive_precision": adaptive.precision.to_dict(),
        "bremsstrahlung_model": "bremslib",
    }
    old_key = case_content_key(case)
    view = resolved_physics_view(case, table_markers={})
    assert (
        dict(view.model_markers)["brem_band_reduction_model"]
        == _precision._BREM_BAND_REDUCTION_MODEL
    )
    monkeypatch.setattr(_precision, "_BREM_BAND_REDUCTION_MODEL", "test-next-generation")
    updated = [
        dataset_identity("hopg", settings, sweep, bremsstrahlung_model=model)
        for settings, model in configurations
    ]
    assert [
        a["parameter_sha256"] != b["parameter_sha256"]
        for a, b in zip(identities, updated, strict=True)
    ] == [False, True, False, False]
    assert case_content_key(case) != old_key
    assert (
        dict(resolved_physics_view(case, table_markers={}).model_markers)[
            "brem_band_reduction_model"
        ]
        == "test-next-generation"
    )


def test_bin_mean_reduction_generation_moves_dataset_and_case_keys_only_for_bins():
    sweep = replace(material_sweep("hopg"), cxr_grid_policy={"quadrature": "bin-mean"})
    settings = default_settings()
    current = dataset_identity("hopg", settings, sweep)
    case = {"line_quadrature": "bin-mean", "seed": 1}
    current_case = case_content_key(case)
    node_case = {"seed": 1}
    node_before = case_content_key(node_case)
    with mock.patch.object(profiles, "_bin_reduction_markers", return_value={}):
        previous = dataset_identity("hopg", settings, sweep)
        previous_case = case_content_key(case)
        assert case_content_key(node_case) == node_before
    assert current["parameter_sha256"] != previous["parameter_sha256"]
    assert current_case != previous_case
