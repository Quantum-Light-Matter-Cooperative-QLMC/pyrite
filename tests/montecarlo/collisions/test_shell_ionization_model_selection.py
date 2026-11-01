"""Run-level selection of the characteristic shell-ionization model (#92).

Validation: bote-salvat-ionization-backend
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite._numerics import Physics
from pyrite._physics_contract import BUILTIN_DESCRIPTORS
from pyrite.campaign import profiles
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.physics_view import resolved_physics_view
from pyrite.campaign.profiles import case_content_key, dataset_identity
from pyrite.campaign.sweep import build_cases
from pyrite.campaign.transition_cases import resolve_shell_ionization
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import emission
from pyrite.montecarlo.runner.emission_adapters import select_emission
from pyrite.montecarlo.spectrum import CHARACTERISTIC_MODEL, characteristic
from pyrite.results import Settings

BOTE = characteristic.characteristic_model_marker(ionization_model="bote-salvat")


@pytest.fixture(scope="module")
def case():
    return build_cases(material_sweep("hopg"), n_electrons=300, n_electrons_brem=150)[0]


def test_settings_validate_and_default_to_eedl():
    from pyrite._numerics import SHELL_IONIZATION_MODELS

    assert Physics().shell_ionization_model == "eedl"
    assert Settings().shell_ionization_model == "eedl"
    assert SHELL_IONIZATION_MODELS == characteristic._IONIZATION_MODELS
    with pytest.raises(ValueError, match="shell_ionization_model"):
        Physics(shell_ionization_model="penelope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shell_ionization_model"):
        Settings(shell_ionization_model="penelope")  # type: ignore[arg-type]


def test_case_records_only_the_opt_in_model(case):
    assert "shell_ionization_model" not in case
    assert resolve_shell_ionization([case], Physics()) == [case]
    (bote,) = resolve_shell_ionization([case], Physics(shell_ionization_model="bote-salvat"))

    assert bote["shell_ionization_model"] == "bote-salvat"
    with pytest.raises(ValueError, match="absent or 'bote-salvat'"):
        replace(case, shell_ionization_model="eedl")  # type: ignore[arg-type]


def test_identity_is_unchanged_for_eedl_and_forks_for_bote_salvat(case):
    sweep = material_sweep("hopg")
    default = dataset_identity("hopg", "full", default_settings(), sweep)
    explicit = dataset_identity(
        "hopg", "full", replace(default_settings(), shell_ionization_model="eedl"), sweep
    )
    bote = dataset_identity(
        "hopg", "full", replace(default_settings(), shell_ionization_model="bote-salvat"), sweep
    )

    assert explicit["parameter_sha256"] == default["parameter_sha256"]
    assert "shell_ionization_model" not in default["resolved_parameters"]
    assert default["resolved_parameters"]["characteristic_model"] == CHARACTERISTIC_MODEL
    assert bote["parameter_sha256"] != default["parameter_sha256"]
    assert bote["resolved_parameters"]["shell_ionization_model"] == "bote-salvat"
    assert bote["resolved_parameters"]["characteristic_model"] == BOTE

    bote_case = replace(case, shell_ionization_model="bote-salvat")
    assert profiles.case_characteristic_marker(case) == CHARACTERISTIC_MODEL
    assert profiles.case_characteristic_marker(bote_case) == BOTE
    assert case_content_key(bote_case) != case_content_key(case)


def test_provenance_and_emission_select_the_bote_salvat_descriptor(case):
    bote_case = replace(case, shell_ionization_model="bote-salvat")
    default_view = resolved_physics_view(case, table_markers={})
    bote_view = resolved_physics_view(bote_case, table_markers={})
    descriptor = BUILTIN_DESCRIPTORS.get("emission.characteristic", "bote-salvat")

    assert descriptor.required_datasets == ("bote-salvat",)
    assert select_emission(case).characteristic.descriptor.model == "characteristic"
    assert select_emission(bote_case).characteristic.descriptor == descriptor
    assert dict(default_view.model_markers)["characteristic_model"] == CHARACTERISTIC_MODEL
    assert dict(bote_view.model_markers)["characteristic_model"] == BOTE
    assert "emission.characteristic/bote-salvat" in bote_view.descriptor_ids
    assert "emission.characteristic/characteristic" in default_view.descriptor_ids


@pytest.mark.parametrize("layers", [1, 2])
def test_runner_passes_the_case_model_to_every_layer(monkeypatch, case, layers):
    seen = []

    def score(segments, energy, **kwargs):
        seen.append(kwargs["ionization_model"])
        return np.zeros_like(energy)

    monkeypatch.setattr(runner, "mc_characteristic_spectrum", score)
    monkeypatch.setattr(runner, "_admit_chunk", lambda chunk, _size: 1)
    monkeypatch.setattr(runner, "_segments_in_layer", lambda segs, _i: {"L_ang": np.ones(1)})
    segments = {"n_layers": layers}
    layer_stack = [(0.0, 1.0, case["composition"])] * layers
    energy = np.linspace(100.0, 200.0, 3)
    for model, expected in ((None, "eedl"), ("bote-salvat", "bote-salvat")):
        selected = case if model is None else replace(case, shell_ionization_model=model)
        seen.clear()
        emission._characteristic_from_segments(segments, energy, selected, None, layer_stack)
        assert seen == [expected] * layers
