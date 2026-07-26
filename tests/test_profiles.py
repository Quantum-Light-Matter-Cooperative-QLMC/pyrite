"""Named profile, provenance, and variant-storage regression contracts."""

from dataclasses import replace

import numpy as np

from cxr_mc.config import default_settings, material_sweep
from cxr_mc.profiles import PROFILE_NAMES, dataset_identity, get_profile, variant_stem
from cxr_mc.sweep import build_cases


def test_full_profile_preserves_production_defaults_exactly():
    implicit_settings = default_settings()
    explicit_settings = default_settings("full")
    implicit_sweep = material_sweep("mose2")
    explicit_sweep = material_sweep("mose2", profile="full")

    assert PROFILE_NAMES == ("full", "survey")
    assert implicit_settings == explicit_settings
    assert implicit_settings.n_electrons == 300
    assert implicit_settings.n_electrons_brem == 150
    implicit_identity = dataset_identity("mose2", "full", implicit_settings, implicit_sweep)
    explicit_identity = dataset_identity("mose2", "full", explicit_settings, explicit_sweep)
    assert implicit_identity == explicit_identity
    assert get_profile("full").provisional is False


def test_survey_profile_reduces_every_expensive_sweep_dimension():
    full = material_sweep("mose2")
    survey = material_sweep("mose2", profile="survey")
    settings = default_settings("survey")

    assert settings.n_electrons == 60
    assert settings.n_electrons_brem == 30
    assert len(np.atleast_1d(survey.energy_keV)) <= 2
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
    sweep = material_sweep("mose2", profile="survey")
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
        material_sweep("hopg", profile="survey"),
    )

    assert variant_stem(full, canonical_full=True) == "hopg"
    assert variant_stem(survey).startswith("hopg--survey-")
    assert variant_stem(full) != variant_stem(survey)
