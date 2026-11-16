"""The corrected basal h-BN coefficient invalidates both checkpoint caches."""

from dataclasses import replace

import pytest

from pyrite.campaign import profiles
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.sweep import Layer, Stack, build_cases


@pytest.mark.parametrize("material,layered", [("hbn", False), ("hopg", True), ("hopg", False)])
def test_hbn_basal_generation_invalidates_only_affected_identities(monkeypatch, material, layered):
    sweep = material_sweep(material)
    if layered:
        sweep = replace(
            sweep,
            target=Stack((Layer("hopg", 2000.0), Layer("hbn", 3000.0)), 45.0, footprint=None),
        )
    settings = default_settings()
    case = build_cases(sweep, n_electrons=1, n_electrons_brem=1)[0]
    current_dataset = profiles.dataset_identity(material, settings, sweep)["parameter_sha256"]
    current_case = profiles.case_content_key(case)
    # Simulate the previous generation without changing geometry, seeds or tables.
    monkeypatch.setattr(profiles, "_HBN_DEBYE_WALLER_MODEL", "retired-b33-3.45", raising=False)
    changed_dataset = profiles.dataset_identity(material, settings, sweep)["parameter_sha256"]
    changed_case = profiles.case_content_key(case)
    affected = material == "hbn" or layered
    assert (current_dataset != changed_dataset) == affected
    assert (current_case != changed_case) == affected
