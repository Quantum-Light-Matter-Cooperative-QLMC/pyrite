"""Named profile, provenance, and variant-storage regression contracts."""

import json
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

from pyrite.campaign import profiles
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.profiles import (
    FIDELITY_NAMES,
    IDENTITY_MIGRATIONS,
    STOPPING_MODEL,
    FidelityPreset,
    case_content_key,
    dataset_identity,
    get_fidelity_preset,
    identity_from_stem,
    named_profile_identity,
    named_profile_stem,
    variant_stem,
)
from pyrite.campaign.sweep import build_cases, target_flat_fields
from pyrite.checkpoints import _checkpoint_store
from pyrite.detectors import Detector, EnergyBins, Timepix3
from pyrite.montecarlo.spectrum import (
    BREMSSTRAHLUNG_MODEL,
    CHARACTERISTIC_MODEL,
    CHARACTERISTIC_XRAYDB_VERSION,
)


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
    # the ELSEPA elastic model, which changes every trajectory.
    assert case_content_key(case) == (
        "7edf048cd4a329377db46f72d97adee2ce07c427aaf38e653d8b74b7a7d3985d"
    )


def test_dataset_identity_dispatches_through_recorded_v1():
    sweep = material_sweep("hopg")
    identity = dataset_identity("hopg", "full", default_settings(), sweep)

    assert set(IDENTITY_MIGRATIONS) == {1}
    assert identity["identity_version"] == 1
    assert identity["parameter_sha256"] == (
        "9186f29caf429c136e7a2e17c7157a39d323df1bc3016bad0981ee5f9b6b8849"
    )
    with pytest.raises(ValueError, match="unsupported dataset identity version"):
        dataset_identity("hopg", "full", default_settings(), sweep, identity_version=2)


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


def test_transport_numerics_fork_dataset_identity_only_when_nondefault():
    settings = default_settings()
    sweep = material_sweep("hopg")
    base = dataset_identity("hopg", "full", settings, sweep)
    explicit_defaults = dataset_identity(
        "hopg",
        "full",
        replace(settings, straggling=False, energy_model="frozen", max_dE_frac=0.0),
        sweep,
    )
    active = dataset_identity(
        "hopg",
        "full",
        replace(settings, straggling=True, energy_model="midpoint", max_dE_frac=0.02),
        sweep,
    )

    assert explicit_defaults["parameter_sha256"] == base["parameter_sha256"]
    # The default ELSEPA elastic model is the one transport numeric that diverges
    # from the historical payload, so it is the only one the default records.
    assert base["resolved_parameters"]["transport_numerics"] == {"elastic_model": "elsepa"}
    assert active["parameter_sha256"] != base["parameter_sha256"]
    assert active["resolved_parameters"]["transport_numerics"] == {
        "straggling": True,
        "energy_model": "midpoint",
        "max_dE_frac": 0.02,
        "elastic_model": "elsepa",
    }


def test_survey_profile_reduces_configured_expensive_sweep_dimensions():
    full = material_sweep("mose2")
    survey = material_sweep("mose2", fidelity="survey")
    settings = default_settings("survey")

    assert settings.n_electrons == 60
    assert settings.n_electrons_brem == 30
    assert len(np.atleast_1d(survey.beam.energy_keV)) <= 2
    survey_geometry = target_flat_fields(survey.target)
    assert len(np.atleast_1d(survey_geometry["thickness_ang"])) <= 3
    assert len(np.atleast_1d(survey_geometry["tilt_deg"])) <= 5
    assert len(np.atleast_1d(survey_geometry["tilt_azim_deg"])) <= 2
    assert survey.n_families == 2
    assert len(survey.detector.energy_bins.brem) < len(full.detector.energy_bins.brem)
    assert not survey.detector.energy_bins.line_by_energy
    assert not full.detector.energy_bins.line_by_energy
    cases = build_cases(survey, settings.n_electrons, settings.n_electrons_brem)
    assert cases
    assert all(len(case["hkl_list"]) <= 4 for case in cases)
    assert all(case.get("line_grid_policy") is not None for case in cases)


def test_profile_convergence_overrides_fidelity_and_forks_identity(monkeypatch):
    from types import MappingProxyType

    from pyrite.campaign import config
    from pyrite.materials import CATALOG

    baseline = material_sweep("hopg", fidelity="survey")
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

    resolved = config.material_sweep("hopg", fidelity="survey")
    assert resolved.n_families == 3
    assert resolved.max_reflections == 6
    assert resolved.mosaic_nodes == 7
    assert resolved.mosaic_route == "mc"
    overridden = config.material_sweep("hopg", fidelity="survey", n_families=8)
    assert overridden.n_families == 8
    assert (
        dataset_identity("hopg", "survey", default_settings("survey"), resolved)["parameter_sha256"]
        != dataset_identity("hopg", "survey", default_settings("survey"), baseline)[
            "parameter_sha256"
        ]
    )


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


def test_line_kinematics_marker_is_a_constant_that_orphans_vacuum_era_digests():
    """The in-medium dispersion is unconditional physics, so it is not a
    divergence key any more -- but every digest minted before it became
    mandatory described the retired vacuum kinematics. The payload therefore
    carries a CONSTANT generation marker, hashed on every run, so those
    checkpoints cannot be resumed into (rev-and-re-run)."""
    identity = dataset_identity("hopg", "full", default_settings(), material_sweep("hopg"))

    assert identity["resolved_parameters"]["line_kinematics"] == "in-medium"
    assert "xray_dispersion" not in identity["resolved_parameters"]
    assert "xray_dispersion" not in identity["resolved_parameters"]["settings"]


def test_stopping_model_marker_is_a_constant_that_orphans_splice_era_digests():
    """SBETHE is unconditional production physics and moves every old digest."""
    identity = dataset_identity("hopg", "full", default_settings(), material_sweep("hopg"))

    assert identity["resolved_parameters"]["stopping_model"] == STOPPING_MODEL
    assert STOPPING_MODEL == "sbethe-corrected-v1"


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
    identity = dataset_identity("hopg", "full", default_settings(), material_sweep("hopg"))

    assert identity["resolved_parameters"]["characteristic_model"] == CHARACTERISTIC_MODEL
    assert "eedl" in CHARACTERISTIC_MODEL
    assert f"xraydb-{CHARACTERISTIC_XRAYDB_VERSION}" in CHARACTERISTIC_MODEL
    assert CHARACTERISTIC_MODEL.endswith("l-shell-ck-lorentzian-v5")


def test_case_content_key_separates_characteristic_models():
    case = build_cases(material_sweep("hopg"), n_electrons=300, n_electrons_brem=150)[0]
    current = case_content_key(case)
    with mock.patch.object(profiles, "CHARACTERISTIC_MODEL", "characteristic-disabled"):
        pre_characteristic = case_content_key(case)

    assert pre_characteristic != current


def test_bremsstrahlung_model_marker_orphans_bethe_heitler_era_digests():
    identity = dataset_identity("hopg", "full", default_settings(), material_sweep("hopg"))

    assert identity["resolved_parameters"]["bremsstrahlung_model"] == BREMSSTRAHLUNG_MODEL
    assert "mf23-527-mf26-527" in BREMSSTRAHLUNG_MODEL


def test_case_content_key_separates_bremsstrahlung_models():
    case = build_cases(material_sweep("hopg"), n_electrons=300, n_electrons_brem=150)[0]
    current = case_content_key(case)
    with mock.patch.object(profiles, "BREMSSTRAHLUNG_MODEL", "bethe-heitler-only"):
        bethe_heitler_era = case_content_key(case)

    assert bethe_heitler_era != current


def test_standard_detector_keeps_current_payload_and_digest_bit_for_bit():
    identity = dataset_identity("hopg", "full", default_settings(), material_sweep("hopg"))
    sweep_payload = identity["resolved_parameters"]["sweep"]

    assert identity["parameter_sha256"] == (
        "9186f29caf429c136e7a2e17c7157a39d323df1bc3016bad0981ee5f9b6b8849"
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
            "e5cc87acb0beb487f3b2ef6945b655d8590fb162a906979b0062add28854eb55",
        ),
        (
            "hopg_hbn_gaussian_200fs",
            "hbn",
            "0b31256eda8cbb9b6d797b5c5393d77908718d2788587bbf8f6afa1fe044e10d",
        ),
        (
            "hopg_hbn_microtrain_200fs",
            "hopg",
            "c1334593380d955d69be20f01d558699b38000fd53688f9123529e389f7b395e",
        ),
        (
            "hopg_hbn_microtrain_200fs",
            "hbn",
            "a75ef33494842c6002332d85a3da3e43e61590a731350fa798723577d49e04e8",
        ),
        (
            "hopg_hbn_compressed_microbunch",
            "hopg",
            "b33158964d798439132190b890613e1a059e4d1502b4f69366cecb332f47032a",
        ),
        (
            "hopg_hbn_compressed_microbunch",
            "hbn",
            "960d236f37f46a4372119583b5ebfae858d4fbf280cc388a4e1bee2cdfee0e78",
        ),
        (
            "hopg_emittance_demo",
            "hopg",
            "393fc64b0eadf5bcf2a6fe22d998b3d79b47382d5721bf164d4cc6f9c9b07713",
        ),
        (
            "promising_low_ne",
            "hopg",
            "82bc585dae67c424b863bff958e30344a4b29a88ea66efcb48c882112ae3c66b",
        ),
    ],
)
def test_named_beam_migration_keeps_shipped_profile_digests_bit_for_bit(
    catalog_profile, material, digest
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
    `l-shell-ck-lorentzian-v5` L-shell Coster--Kronig relaxation marker, and
    for issue #89's ELSEPA elastic model becoming the default.

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
    detector = Detector(119.0, 16.6, 0.066, response=Timepix3(thickness_um=500.0))
    custom_sweep = material_sweep("hopg", detector=detector)
    standard = dataset_identity("hopg", "full", settings, standard_sweep)
    custom = dataset_identity("hopg", "full", settings, custom_sweep)
    case = build_cases(custom_sweep)[0]
    payload = custom["resolved_parameters"]["sweep"]

    assert replace(custom_sweep.detector, energy_bins=EnergyBins()) == detector
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
        Detector(polar_acceptance_deg=16.6),
        Detector(solid_angle_sr=0.066),
    ],
)
def test_every_nondefault_detector_acceptance_field_changes_identity(detector):
    settings = default_settings()
    standard_sweep = material_sweep("hopg")
    standard = dataset_identity("hopg", "full", settings, standard_sweep)
    changed = dataset_identity("hopg", "full", settings, replace(standard_sweep, detector=detector))

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
        dataset_identity("hopg", "full", settings, rescored)["parameter_sha256"]
        == dataset_identity("hopg", "full", settings, standard_sweep)["parameter_sha256"]
    )


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
    incoherent path still adds no key of its own to the hashed payload."""
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

    # Known incoherent digest (current baseline, re-minted when the in-medium
    # line kinematics became unconditional, again when the Joy--Luo/
    # Berger--Seltzer stopping splice did, and again when every crystal cut
    # moved to the surface_hkl spelling, and again for natural Lorentzian
    # characteristic profiles, and again for issue #88's physical finite-window
    # convention, again for issue #125's automatic bundled line grids, again
    # for issue #91's L-shell Coster--Kronig relaxation marker, and again for
    # issue #89's default ELSEPA elastic model) must stay bit-for-bit.
    assert incoherent["parameter_sha256"] == (
        "9186f29caf429c136e7a2e17c7157a39d323df1bc3016bad0981ee5f9b6b8849"
    )
    survey_incoherent = dataset_identity(
        "mose2", "survey", default_settings("survey"), material_sweep("mose2", fidelity="survey")
    )
    # Re-minted for issue #100: mose2 carries its own E_grid_brem override, and
    # that row's `start` now records the medium's derived photon-continuum floor
    # instead of 0 eV, so the declared sweep payload this digest hashes moved.
    # Artifact-backed materials (hopg, hbn) did not: their stored 0.0 is a
    # bandwidth request the resolver raises, so it was left alone. Re-minted
    # again for issue #89's default ELSEPA elastic model.
    assert survey_incoherent["parameter_sha256"] == (
        "de65c05cffb5ea139ee4c921c6856c61275376a8258262ddc070c929dd68e644"
    )


def test_identity_from_stem_resolves_non_standard_catalog_profile():
    """Regression: ``pyrite remote pull --profile`` died here -- a variant stem
    names only material/fidelity/digest, and identity_from_stem used to
    recompute the digest under "standard" only, so a sub_100keV checkpoint
    could never match (remote `pyrite slim --grid` SystemExit, 21/21 failed)."""
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


def test_shell_inelastic_mode_forks_identity_and_case_payload_only_when_on():
    settings = replace(default_settings(), energy_model="midpoint")
    sweep = material_sweep("silicon")
    base = dataset_identity("silicon", "full", settings, sweep)
    explicit_default = dataset_identity(
        "silicon",
        "full",
        replace(settings, inelastic_model="continuous", inelastic_cutoff_eV=None),
        sweep,
    )
    shell = replace(settings, inelastic_model="shell-soft-hard", inelastic_cutoff_eV=50.0)
    active = dataset_identity("silicon", "full", shell, sweep)
    other_cutoff = dataset_identity(
        "silicon", "full", replace(shell, inelastic_cutoff_eV=100.0), sweep
    )

    assert explicit_default["parameter_sha256"] == base["parameter_sha256"]
    assert "inelastic_model" not in base["resolved_parameters"]["transport_numerics"]
    assert active["resolved_parameters"]["transport_numerics"] == {
        "energy_model": "midpoint",
        "inelastic_model": "shell-soft-hard",
        "inelastic_cutoff_eV": 50.0,
        "elastic_model": "elsepa",
    }
    assert (
        len(
            {base["parameter_sha256"], active["parameter_sha256"], other_cutoff["parameter_sha256"]}
        )
        == 3
    )

    legacy_case = build_cases(sweep, 4, 4, energy_model="midpoint")[0]
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


def test_default_elsepa_model_forks_identity_and_case_payload_from_mott():
    """ELSEPA is the default; the historical Mott model keeps its old payload."""
    settings = default_settings()
    sweep = material_sweep("silicon")
    base = dataset_identity("silicon", "full", settings, sweep)
    explicit_default = dataset_identity(
        "silicon", "full", replace(settings, elastic_model="elsepa"), sweep
    )
    mott = dataset_identity("silicon", "full", replace(settings, elastic_model="mott"), sweep)

    assert settings.elastic_model == "elsepa"
    assert explicit_default["parameter_sha256"] == base["parameter_sha256"]
    assert base["resolved_parameters"]["transport_numerics"] == {"elastic_model": "elsepa"}
    assert "transport_numerics" not in mott["resolved_parameters"]
    assert mott["parameter_sha256"] != base["parameter_sha256"]

    default_case = build_cases(sweep, 4, 4)[0]
    mott_case = build_cases(sweep, 4, 4, elastic_model="mott")[0]
    assert default_case["elastic_model"] == "elsepa"
    assert "elastic_model" not in mott_case
    assert profiles.case_content_key(default_case) != profiles.case_content_key(mott_case)
    with pytest.raises(ValueError, match="elastic_model"):
        build_cases(sweep, 4, 4, elastic_model="nope")
