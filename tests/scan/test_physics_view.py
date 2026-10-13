"""Private extension metadata preserves resolved-case semantics (issue #485)."""

import json
import pickle
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from pyrite._physics_contract import (
    BUILTIN_DESCRIPTORS,
    BackendAvailability,
    DescriptorCatalog,
    ModelDescriptor,
    ResolvedPhysicsView,
)
from pyrite.campaign.physics_view import resolved_physics_view
from pyrite.montecarlo.case import Case


@pytest.fixture
def baseline_cases():
    return json.loads(
        (Path(__file__).parents[1] / "data/physics-baseline/resolved.json").read_text()
    )


def test_metadata_import_has_no_scientific_or_device_dependencies():
    script = """
import sys
from pyrite._physics_contract import BUILTIN_DESCRIPTORS
assert BUILTIN_DESCRIPTORS.get("transport.elastic", "elsepa")
for prefix in ("numpy", "scipy", "cupy", "numba", "pyrite.campaign", "pyrite.montecarlo", "pyrite.materials", "pyrite._backend"):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
from pyrite.campaign.physics_view import resolved_physics_view
for prefix in ("numpy", "scipy", "cupy", "numba", "pyrite.montecarlo", "pyrite.materials", "pyrite._backend"):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=10)


def test_catalog_rejects_duplicate_ids_and_copies_mutable_inputs():
    requirements = ["tables"]
    descriptor = ModelDescriptor("test", "model", "test.owner", requirements=requirements)
    entries = [descriptor]
    catalog = DescriptorCatalog(entries)
    entries.clear()
    requirements.append("changed")
    assert catalog.descriptors == (descriptor,)
    assert descriptor.requirements == ("tables",)
    with pytest.raises(FrozenInstanceError):
        descriptor.model = "changed"
    with pytest.raises(FrozenInstanceError):
        catalog.descriptors = ()
    with pytest.raises(ValueError, match="duplicate"):
        DescriptorCatalog((descriptor, descriptor))
    # The same local ID in a different family is distinct.
    assert (
        len(DescriptorCatalog((descriptor, replace(descriptor, family="other"))).descriptors) == 2
    )
    with pytest.raises(ValueError, match="unknown"):
        catalog.get("test", "missing")
    assert pickle.loads(pickle.dumps(catalog)) == catalog


def test_native_capabilities_are_not_backend_availability_or_physical_domain(baseline_cases):
    descriptor = BUILTIN_DESCRIPTORS.get("transport.inelastic", "shell-soft-hard")
    assert "cuda.exact" in descriptor.implementations
    assert "cuda.lut" not in descriptor.implementations
    assert descriptor.validation_domain == "not-assessed"
    case = baseline_cases["cascade"]["case"]
    view = resolved_physics_view(case, table_markers={})
    assert view.backend_availability == ()
    assert "ledger-validated physical domain" in view.applicability.pending
    unavailable = resolved_physics_view(
        case,
        table_markers={},
        backend_availability=(BackendAvailability("cuda", False, "test device unavailable"),),
    )
    assert unavailable.descriptor_ids == view.descriptor_ids
    assert unavailable.model_markers == view.model_markers
    assert unavailable.parameters == view.parameters
    assert not unavailable.backend_availability[0].available


@pytest.mark.parametrize(
    "kind", ["legacy", "elsepa", "eedl-fallback", "coupled", "multilayer", "coherent", "cascade"]
)
def test_resolved_view_retains_case_and_selected_models(baseline_cases, kind):
    payload = baseline_cases[kind]["case"]
    before = pickle.dumps(payload)
    view = resolved_physics_view(payload, table_markers={"selected-table": "a" * 64})
    assert pickle.dumps(payload) == before
    assert view.parameters == payload
    assert "transport.elastic/" + payload.get("elastic_model", "mott") in view.descriptor_ids
    assert (
        "emission.continuum/" + payload.get("bremsstrahlung_model", "eedl") in view.descriptor_ids
    )
    assert (
        "transport.radiative/" + payload.get("radiative_model", "uncoupled") in view.descriptor_ids
    )
    assert ("transport.pair/penelope-2024" in view.descriptor_ids) == (kind == "cascade")
    assert ("transport.positron/positron" in view.descriptor_ids) == (kind == "cascade")
    assert ("coherent_population_model" in dict(view.model_markers)) == (kind == "coherent")
    assert ResolvedPhysicsView.from_json(view.to_json()) == view
    assert pickle.loads(pickle.dumps(view)) == view
    assert (
        resolved_physics_view(Case(**payload), table_markers={"selected-table": "a" * 64}) == view
    )
    if kind in ("legacy", "eedl-fallback"):
        assert "elastic_model" not in view.parameters
        assert "bremsstrahlung_model" not in view.parameters
        assert "radiative_model" not in view.parameters


@pytest.mark.parametrize(
    "key", ["elastic_model", "inelastic_model", "bremsstrahlung_model", "radiative_model"]
)
def test_unresolved_auto_is_refused_without_running_another_resolver(
    baseline_cases, monkeypatch, key
):
    def forbidden(*args, **kwargs):
        pytest.fail("view construction must not resolve auto or load tables")

    monkeypatch.setattr("pyrite.xsgen.bremslib.tables.resolve_bremsstrahlung_model", forbidden)
    monkeypatch.setattr("pyrite.xsgen.bremslib.tables.load_bremsstrahlung_tables", forbidden)
    with pytest.raises(ValueError, match=key):
        resolved_physics_view({**baseline_cases["legacy"]["case"], key: "auto"}, table_markers={})


def test_table_inventory_is_supplied_by_existing_owner_and_copied(baseline_cases, monkeypatch):
    from pyrite.montecarlo.runner import case_tables

    supplied = {"sbethe-selected": "b" * 64, "elsepa-selected": "a" * 64}
    monkeypatch.setattr(case_tables, "case_table_markers", lambda case: supplied)
    case = Case(**baseline_cases["elsepa"]["case"])
    view = resolved_physics_view(case, table_markers=case_tables.case_table_markers(case))
    supplied["unrelated-dataset-table"] = "c" * 64
    assert dict(view.table_markers) == {"sbethe-selected": "b" * 64, "elsepa-selected": "a" * 64}
    changed = resolved_physics_view(case, table_markers={"elsepa-selected": "c" * 64})
    assert changed.table_markers != view.table_markers
    assert changed.descriptor_ids == view.descriptor_ids
    assert "unrelated-dataset-table" not in view.to_json()


@pytest.mark.parametrize("resolved_model", ["eedl", "bremslib"])
def test_campaign_auto_resolution_precedes_view_construction(monkeypatch, resolved_model):
    from tests.helpers.physics_baseline import resolved_baseline

    calls = []

    def resolve(model, elements):
        calls.append((model, tuple(elements)))
        return resolved_model

    monkeypatch.setattr("pyrite.xsgen.bremslib.tables.resolve_bremsstrahlung_model", resolve)
    payload = resolved_baseline("eedl-fallback")["case"]
    resolution_calls = list(calls)
    view = resolved_physics_view(payload, table_markers={})
    assert resolution_calls and calls == resolution_calls
    assert f"emission.continuum/{resolved_model}" in view.descriptor_ids
    assert view.parameters == json.loads(json.dumps(payload))


def test_view_is_deeply_immutable_and_serialization_is_deterministic(baseline_cases):
    case = baseline_cases["multilayer"]["case"]
    markers = {"z": "b", "a": "a"}
    view = resolved_physics_view(case, table_markers=markers)
    reordered = resolved_physics_view(
        dict(reversed(list(case.items()))), table_markers=dict(reversed(list(markers.items())))
    )
    assert view.to_json() == reordered.to_json()
    detached = view.parameters
    detached["composition"][0][1] = 123.0
    case["composition"][0][1] = 456.0
    assert view.parameters["composition"][0][1] not in (123.0, 456.0)
    with pytest.raises(FrozenInstanceError):
        view.schema = "changed"
    assert hash(view)
    with pytest.raises(ValueError, match="unknown resolved"):
        ResolvedPhysicsView.from_json(
            view.to_json().replace("pyrite.resolved-physics.v1", "unknown")
        )
    with pytest.raises(ValueError, match="duplicate marker"):
        replace(view, table_markers=(("a", "x"), ("a", "y")))
    with pytest.raises(ValueError, match="key/value"):
        replace(view, table_markers=("ab",))


def test_array_and_path_parameters_are_data_only(baseline_cases, tmp_path):
    import numpy as np

    payload = baseline_cases["legacy"]["case"]
    energies = np.asarray(payload["E_grid_line"])
    view = resolved_physics_view(
        {**payload, "E_grid_line": energies, "brem_file": tmp_path / "continuum.dat"},
        table_markers={},
    )
    assert view.parameters["brem_file"] == str(tmp_path / "continuum.dat")
    assert view.parameters["E_grid_line"] == energies.tolist()
    energies[0] = 99.0
    assert view.parameters["E_grid_line"][0] == payload["E_grid_line"][0]
    assert ResolvedPhysicsView.from_json(view.to_json()) == view
    with pytest.raises(TypeError, match="data-only"):
        resolved_physics_view({**payload, "brem_file": object()}, table_markers={})


def test_unused_descriptor_does_not_enter_view_or_current_identities(baseline_cases):
    from pyrite.campaign.profiles import case_content_key

    case = baseline_cases["legacy"]["case"]
    before = resolved_physics_view(case, table_markers={})
    expanded = DescriptorCatalog(
        (*BUILTIN_DESCRIPTORS.descriptors, ModelDescriptor("test", "unused", "test.owner"))
    )
    assert expanded.get("test", "unused")
    assert resolved_physics_view(case, table_markers={}) == before
    assert case_content_key(case) == baseline_cases["legacy"]["case_content_key"]
    assert "test/unused" not in before.to_json()


def test_descriptors_inventory_existing_native_restrictions():
    elsepa = BUILTIN_DESCRIPTORS.get("transport.elastic", "elsepa")
    assert "cuda.exact" in elsepa.implementations
    assert "cuda.lut" not in elsepa.implementations
    assert "exact cores even when LUT configuration is enabled" in elsepa.requirements
    coupled = BUILTIN_DESCRIPTORS.get("transport.radiative", "bremslib-soft-hard")
    assert "no CUDA LUT" in coupled.requirements
    assert "photon rows completed on host" in coupled.requirements
    pair = BUILTIN_DESCRIPTORS.get("transport.pair", "penelope-2024")
    assert {"coupled hard photons", "shell-soft-hard", "secondary threshold"} <= set(
        pair.requirements
    )
    positron = BUILTIN_DESCRIPTORS.get("transport.positron", "positron")
    assert "no Mott positron model" in positron.requirements
