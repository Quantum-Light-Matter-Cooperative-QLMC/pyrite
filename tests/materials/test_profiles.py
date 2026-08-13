"""Named profile, provenance, and variant-storage regression contracts."""

import json
from dataclasses import replace

import numpy as np
import pytest

from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.profiles import (
    FIDELITY_NAMES,
    FidelityPreset,
    case_content_key,
    dataset_identity,
    get_fidelity_preset,
    identity_from_stem,
    named_profile_identity,
    named_profile_stem,
    variant_stem,
)
from pyrite.campaign.sweep import build_cases
from pyrite.checkpoints import _checkpoint_store
from pyrite.detectors import DetectorSpec


def _cases_by_key(material, catalog_profile):
    sweep = material_sweep(material, catalog_profile=catalog_profile)
    cases = build_cases(sweep, n_electrons=300, n_electrons_brem=150)
    return {(c["name"], c["E0_keV"]): c for c in cases}


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
    assert case_content_key(case) == (
        "469035680a013567be35f189059e3b76eeb22acc2d49653592613c784fc9079b"
    )


def test_high_energy_profile_range_is_part_of_dataset_identity():
    sweep = material_sweep("tise2", catalog_profile="high_energy")
    identity = named_profile_identity("tise2", catalog_profile="high_energy")

    np.testing.assert_array_equal(sweep.beam.energy_keV, [100.0, 150.0, 200.0, 250.0, 300.0])
    assert identity["catalog_profile"] == "high_energy"
    assert identity["resolved_parameters"]["sweep"]["energy_keV"] == [
        100.0,
        150.0,
        200.0,
        250.0,
        300.0,
    ]


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
    explicit_settings = default_settings("full")
    implicit_sweep = material_sweep("mose2")
    explicit_sweep = material_sweep("mose2", fidelity="full")

    assert FIDELITY_NAMES == ("full", "survey")
    assert implicit_settings == explicit_settings
    assert implicit_settings.n_electrons == 300
    assert implicit_settings.n_electrons_brem == 150
    implicit_identity = dataset_identity("mose2", "full", implicit_settings, implicit_sweep)
    explicit_identity = dataset_identity("mose2", "full", explicit_settings, explicit_sweep)
    assert implicit_identity == explicit_identity
    assert get_fidelity_preset("full").provisional is False


def test_survey_profile_reduces_every_expensive_sweep_dimension():
    full = material_sweep("mose2")
    survey = material_sweep("mose2", fidelity="survey")
    settings = default_settings("survey")

    assert settings.n_electrons == 60
    assert settings.n_electrons_brem == 30
    assert len(np.atleast_1d(survey.beam.energy_keV)) <= 2
    assert len(np.atleast_1d(survey.thickness_ang)) <= 3
    assert len(np.atleast_1d(survey.tilt_deg)) <= 5
    assert len(np.atleast_1d(survey.tilt_azim_deg)) <= 2
    assert survey.n_families == 2
    assert len(survey.E_grid_brem) < len(full.E_grid_brem)
    assert max(len(grid) for grid in survey.E_grid_line_by_energy.values()) < max(
        len(grid) for grid in full.E_grid_line_by_energy.values()
    )
    cases = build_cases(survey, settings.n_electrons, settings.n_electrons_brem)
    assert cases
    assert all(len(case["hkl_list"]) <= 4 for case in cases)


def test_dataset_identity_is_deterministic_and_resolved_parameter_sensitive():
    settings = default_settings("survey")
    sweep = material_sweep("mose2", fidelity="survey")
    first = dataset_identity("mose2", "survey", settings, sweep)
    same = dataset_identity("mose2", "survey", settings, sweep)
    changed = dataset_identity(
        "mose2",
        "survey",
        settings,
        replace(sweep, n_families=sweep.n_families + 1),
    )

    assert first == same
    assert first["parameter_sha256"] != changed["parameter_sha256"]
    assert first["resolved_parameters"]["settings"]["n_electrons"] == 60
    assert first["resolved_parameters"]["sweep"]["n_families"] == 2


def test_variant_stem_preserves_canonical_full_and_isolates_variants():
    full_settings = default_settings()
    full_sweep = material_sweep("hopg")
    full = dataset_identity("hopg", "full", full_settings, full_sweep)
    survey = dataset_identity(
        "hopg",
        "survey",
        default_settings("survey"),
        material_sweep("hopg", fidelity="survey"),
    )

    assert variant_stem(full, canonical_full=True) == "hopg"
    assert variant_stem(survey).startswith("hopg@survey-")
    assert variant_stem(full) != variant_stem(survey)


def test_catalog_profile_leaves_standard_hash_bit_for_bit():
    """Phase 3 decision 2: catalog_profile only joins the hashed payload when
    it diverges from "standard", so every pre-existing standard-profile
    identity keeps its historical parameter_sha256 (and checkpoint stem)."""
    settings = default_settings()
    sweep = material_sweep("hopg")
    implicit = dataset_identity("hopg", "full", settings, sweep)
    explicit = dataset_identity("hopg", "full", settings, sweep, catalog_profile="standard")

    assert implicit["parameter_sha256"] == explicit["parameter_sha256"]
    assert implicit["catalog_profile"] == "standard"
    assert "catalog_profile" not in implicit["resolved_parameters"]


def test_vacuum_dispersion_leaves_the_hash_bit_for_bit_and_refractive_diverges():
    """``Settings.xray_dispersion`` follows the emission divergence-only rule:
    the "vacuum" default adds no key (so historical digests survive the new
    field), while "refractive" gets its OWN digest -- otherwise a refractive run
    would resume into its vacuum twin's checkpoint."""
    settings = default_settings()
    sweep = material_sweep("hopg")
    vacuum = dataset_identity("hopg", "full", settings, sweep)
    refractive = dataset_identity(
        "hopg", "full", replace(settings, xray_dispersion="refractive"), sweep
    )

    assert "xray_dispersion" not in vacuum["resolved_parameters"]
    assert "xray_dispersion" not in vacuum["resolved_parameters"]["settings"]
    assert refractive["resolved_parameters"]["xray_dispersion"] == "refractive"
    assert refractive["parameter_sha256"] != vacuum["parameter_sha256"]


def test_xray_dispersion_value_mirrors_agree():
    """``_XRAY_DISPERSION_VALUES`` is copied into the catalog and the profile CLI
    because ``crystal`` cannot be imported from ``catalog`` (import cycle). Pin
    the copies to the kernels' own list so they cannot drift apart."""
    from pyrite.cli.commands.profile import _XRAY_DISPERSION_VALUES as cli_values
    from pyrite.materials.catalog import _XRAY_DISPERSION_VALUES as catalog_values
    from pyrite.materials.crystal import XRAY_DISPERSION_MODELS

    assert catalog_values == XRAY_DISPERSION_MODELS
    assert cli_values == XRAY_DISPERSION_MODELS


def test_standard_detector_keeps_historical_payload_and_digest_bit_for_bit():
    identity = dataset_identity("hopg", "full", default_settings(), material_sweep("hopg"))
    sweep_payload = identity["resolved_parameters"]["sweep"]

    assert identity["parameter_sha256"] == (
        "d0bb205f2268b8cd30801b1146de8daf7e745ca70399919718519542a7c9b45c"
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
            "5822690aeba1edf5bcea74ca53cd0504788bb6694dd9145f8af61016fd835865",
        ),
        (
            "hopg_hbn_gaussian_200fs",
            "hbn",
            "c4c14c13bb588154984dc41f8563e9334a73686471cb82651ff853988f64364a",
        ),
        (
            "hopg_hbn_microtrain_200fs",
            "hopg",
            "5ac000d15c7ac4315508e45d74851ec359a917f6986c6934c5ca5c8dff9cdb70",
        ),
        (
            "hopg_hbn_microtrain_200fs",
            "hbn",
            "6a7c899190fc56618bb9e32bfb851a1567b552e0fa4990d252719055f178eaf4",
        ),
        (
            "hopg_hbn_compressed_microbunch",
            "hopg",
            "230e7c1e58aaa127ff55bad3eea403ea7523f6d2f16660731af24f15af2235fc",
        ),
        (
            "hopg_hbn_compressed_microbunch",
            "hbn",
            "08c8328a65eb3decbf53f84fca257bab9225a6d14b2b65944c5393fe020b7deb",
        ),
        (
            "hopg_emittance_demo",
            "hopg",
            "6822cad824a6e6f0147e9a5aa675e77dfd7e12c010f844d67d73a68f6ef75180",
        ),
        (
            "promising_low_ne",
            "hopg",
            "2232f2f65ad43ec50aee27cf89021b557e97bf17edc22d84a077a916551e4722",
        ),
    ],
)
def test_named_beam_migration_keeps_shipped_profile_digests_bit_for_bit(
    catalog_profile, material, digest
):
    """Every shipped profile that carried an inline ``[profiles.NAME.beam]`` block
    now carries ``beam = "NAME"`` instead. The reference resolves to values before
    hashing, so these digests -- and therefore every existing checkpoint stem --
    are the pre-migration ones."""
    identity = named_profile_identity(material, catalog_profile=catalog_profile)

    assert identity["parameter_sha256"] == digest


def test_nondefault_detector_round_trips_cases_identity_and_stem():
    settings = default_settings()
    standard_sweep = material_sweep("hopg")
    detector = DetectorSpec(
        119.0,
        16.6,
        0.066,
        response_model="registry/zhai",
        qe_curve="package-data/qe/zhai.csv",
    )
    custom_sweep = material_sweep("hopg", detector=detector)
    standard = dataset_identity("hopg", "full", settings, standard_sweep)
    custom = dataset_identity("hopg", "full", settings, custom_sweep)
    case = build_cases(custom_sweep)[0]
    payload = custom["resolved_parameters"]["sweep"]

    assert custom_sweep.detector == detector
    assert case["theta_obs_rad"] == pytest.approx(np.deg2rad(119.0))
    assert case["dtheta_obs_rad"] == pytest.approx(np.deg2rad(16.6))
    assert case["domega_sr"] == pytest.approx(0.066)
    assert payload["theta_obs_deg"] == 119.0
    assert payload["dtheta_obs_deg"] == 16.6
    assert payload["domega_sr"] == 0.066
    assert payload["detector"] == {
        "qe_curve": "package-data/qe/zhai.csv",
        "response_model": "registry/zhai",
    }
    assert custom["parameter_sha256"] != standard["parameter_sha256"]
    assert variant_stem(custom) != variant_stem(standard)


@pytest.mark.parametrize(
    "detector",
    [
        DetectorSpec(observation_angle_deg=119.0),
        DetectorSpec(polar_acceptance_deg=16.6),
        DetectorSpec(solid_angle_sr=0.066),
        DetectorSpec(response_model="registry/test"),
        DetectorSpec(qe_curve="package-data/qe/test.csv"),
        DetectorSpec(pixel_pitch_um=55.0),
        DetectorSpec(sensor_thickness_um=500.0),
        DetectorSpec(distance_mm=400.0),
        DetectorSpec(threshold_eV=100.0),
    ],
)
def test_every_nondefault_detector_field_changes_identity(detector):
    settings = default_settings()
    standard_sweep = material_sweep("hopg")
    standard = dataset_identity("hopg", "full", settings, standard_sweep)
    changed = dataset_identity("hopg", "full", settings, replace(standard_sweep, detector=detector))

    assert changed["parameter_sha256"] != standard["parameter_sha256"]


def test_catalog_profile_changes_hash_and_stem_when_not_standard():
    settings = default_settings()
    sweep = material_sweep("hopg")
    standard = dataset_identity("hopg", "full", settings, sweep, catalog_profile="standard")
    other = dataset_identity("hopg", "full", settings, sweep, catalog_profile="sub_100keV")

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


def test_named_profile_stem_default_catalog_profile_stays_canonical():
    assert named_profile_stem("hopg", "full") == "hopg"
    assert named_profile_stem("hopg", "full", catalog_profile="standard") == "hopg"


def test_named_profiles_default_to_incoherent_emission():
    """Back-compat: the tri-state emission rename leaves both presets on the
    incoherent (default) policy, with the derived coherent_emission property
    off, so existing runs keep the bit-for-bit incoherent path."""
    for name in FIDELITY_NAMES:
        preset = get_fidelity_preset(name)
        assert preset.emission == "incoherent"
        assert preset.coherent_emission is False
    # Settings resolved for those profiles carry the same incoherent default.
    settings = default_settings("full").__class__()
    assert settings.emission == "incoherent"
    assert settings.coherent_emission is False


@pytest.mark.parametrize(
    ("emission", "coherent"),
    [("incoherent", False), ("coherent", True), ("both", True)],
)
def test_emission_mode_resolves_through_profile_to_settings(emission, coherent):
    """Each emission mode rides FidelityPreset.apply_settings onto Settings, and
    the derived coherent_emission property agrees on both sides."""
    preset = FidelityPreset("probe", n_electrons=10, n_electrons_brem=5, emission=emission)
    assert preset.coherent_emission is coherent

    resolved = preset.apply_settings(default_settings())
    assert resolved.emission == emission
    assert resolved.coherent_emission is coherent


def test_emission_modes_yield_three_distinct_digests_incoherent_unchanged():
    """dataset_identity emits a distinct digest per emission mode, and the
    incoherent digest is bit-for-bit identical to the pre-rename default (the
    incoherent path adds no key to the hashed payload)."""
    sweep = material_sweep("hopg")
    base = default_settings()
    incoherent = dataset_identity("hopg", "full", replace(base, emission="incoherent"), sweep)
    coherent = dataset_identity("hopg", "full", replace(base, emission="coherent"), sweep)
    both = dataset_identity("hopg", "full", replace(base, emission="both"), sweep)

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

    # Known pre-change incoherent digests (captured before the tri-state rename)
    # must stay bit-for-bit -- orphaning old coherent stems but never incoherent.
    assert incoherent["parameter_sha256"] == (
        "d0bb205f2268b8cd30801b1146de8daf7e745ca70399919718519542a7c9b45c"
    )
    survey_incoherent = dataset_identity(
        "mose2", "survey", default_settings("survey"), material_sweep("mose2", fidelity="survey")
    )
    assert survey_incoherent["parameter_sha256"] == (
        "a6d8116bf5f0aef9b61f0b43cc4e4e4522e6e8d51167e51450d0fc3232417a7e"
    )


def test_identity_from_stem_resolves_non_standard_catalog_profile():
    """Regression: ``cxr remote pull --profile`` died here -- a variant stem
    names only material/fidelity/digest, and identity_from_stem used to
    recompute the digest under "standard" only, so a sub_100keV checkpoint
    could never match (remote `cxr slim --grid` SystemExit, 21/21 failed)."""
    stem = named_profile_stem("hopg", "full", catalog_profile="sub_100keV")
    identity = identity_from_stem(stem)

    assert identity is not None
    assert identity["material"] == "hopg"
    assert identity["fidelity"] == "full"
    assert identity["catalog_profile"] == "sub_100keV"
    # standard-profile variant stems keep resolving as before
    survey = named_profile_stem("hopg", "survey")
    assert identity_from_stem(survey)["catalog_profile"] == "standard"


def test_identity_from_stem_dual_reads_legacy_hyphen_stem():
    """Dual-read: checkpoints written under the pre-2026-07-29
    ``<material>--<fidelity>-<digest>`` scheme still resolve after the @-stem
    migration, so existing on-disk checkpoints are never orphaned. The @-stem
    and its legacy twin share a digest -- only the stem text differs."""
    identity = named_profile_identity("hopg", "survey")
    digest = identity["parameter_sha256"][:12]
    legacy_stem = f"hopg--survey-{digest}"
    at_stem = named_profile_stem("hopg", "survey")

    assert at_stem == f"hopg@survey-{digest}"  # new scheme is what gets written
    resolved = identity_from_stem(legacy_stem)  # old scheme still reads back
    assert resolved is not None
    assert (resolved["material"], resolved["fidelity"], resolved["catalog_profile"]) == (
        "hopg",
        "survey",
        "standard",
    )


def test_identity_from_stem_reads_sidecar_when_profile_edited_after_run(tmp_path):
    """Regression (2026-07-29): a checkpoint written under a named profile that
    is edited *after* the run no longer resolves via recompute -- the stem's
    digest was fixed at run time, and the edited profile now hashes to a
    different value, so no candidate recompute matches. But the run-time
    ``dataset_identity`` recorded in the stem's ``meta.json`` sidecar is
    authoritative and still resolves. Mirrors the two on-disk
    ``hopg_hbn_microtrain_200fs`` stems found that day (two runs a minute apart,
    profile edited in between, neither recoverable by recompute)."""
    # Identity as recorded at run time. A fabricated digest stands in for "the
    # backing profile was later edited": the stem commits to this run-time
    # digest, which no recompute against the current catalog can reproduce.
    run_time_identity = {
        **named_profile_identity("hopg", "full", catalog_profile="sub_100keV"),
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
