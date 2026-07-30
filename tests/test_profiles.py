"""Named profile, provenance, and variant-storage regression contracts."""

import json
from dataclasses import replace

import numpy as np
import pytest

from cxr_mc import _checkpoint_store
from cxr_mc.config import default_settings, material_sweep
from cxr_mc.detectors import DetectorSpec
from cxr_mc.profiles import (
    FIDELITY_NAMES,
    SweepProfile,
    dataset_identity,
    get_profile,
    identity_from_stem,
    named_profile_identity,
    named_profile_stem,
    variant_stem,
)
from cxr_mc.sweep import build_cases


def test_full_profile_preserves_production_defaults_exactly():
    implicit_settings = default_settings()
    explicit_settings = default_settings("full")
    implicit_sweep = material_sweep("mose2")
    explicit_sweep = material_sweep("mose2", profile="full")

    assert FIDELITY_NAMES == ("full", "survey")
    assert implicit_settings == explicit_settings
    assert implicit_settings.n_electrons == 300
    assert implicit_settings.n_electrons_brem == 150
    implicit_identity = dataset_identity("mose2", "full", implicit_settings, implicit_sweep)
    explicit_identity = dataset_identity("mose2", "full", explicit_settings, explicit_sweep)
    assert implicit_identity == explicit_identity
    assert get_profile("full").provisional is False


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
    assert variant_stem(survey).startswith("hopg--survey-")
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
    changed = dataset_identity(
        "hopg", "full", settings, replace(standard_sweep, detector=detector)
    )

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
    # tests/test_material_catalog.py for that end-to-end contract.
    assert variant_stem(other) != "hopg"
    assert variant_stem(other).startswith("hopg--full-")


def test_named_profile_stem_default_catalog_profile_stays_canonical():
    assert named_profile_stem("hopg", "full") == "hopg"
    assert named_profile_stem("hopg", "full", catalog_profile="standard") == "hopg"


def test_named_profiles_default_to_incoherent_emission():
    """Back-compat: the tri-state emission rename leaves both presets on the
    incoherent (default) policy, with the derived coherent_emission property
    off, so existing runs keep the bit-for-bit incoherent path."""
    for name in FIDELITY_NAMES:
        profile = get_profile(name)
        assert profile.emission == "incoherent"
        assert profile.coherent_emission is False
    # Settings resolved for those profiles carry the same incoherent default.
    settings = default_settings("full").__class__()
    assert settings.emission == "incoherent"
    assert settings.coherent_emission is False


@pytest.mark.parametrize(
    ("emission", "coherent"),
    [("incoherent", False), ("coherent", True), ("both", True)],
)
def test_emission_mode_resolves_through_profile_to_settings(emission, coherent):
    """Each emission mode rides SweepProfile.apply_settings onto Settings, and
    the derived coherent_emission property agrees on both sides."""
    profile = SweepProfile("probe", n_electrons=10, n_electrons_brem=5, emission=emission)
    assert profile.coherent_emission is coherent

    resolved = profile.apply_settings(default_settings())
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
    assert stem == "hopg--full-deaddeaddead"

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
