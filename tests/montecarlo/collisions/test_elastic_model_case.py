"""Case-level ELSEPA elastic-model selection and table resolution (issue #89)."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite._numerics import Physics
from pyrite.materials import CATALOG
from pyrite.montecarlo.runner.case_tables import (
    _case_elastic_kwargs,
    _case_elastic_table_records,
)


def test_physics_accepts_only_known_elastic_models():
    assert Physics().elastic_model == "elsepa"
    assert Physics(elastic_model="mott").elastic_model == "mott"
    with pytest.raises(ValueError, match="elastic_model"):
        Physics(elastic_model="sr")  # type: ignore[arg-type]


def test_mott_cases_name_mott_and_resolve_no_elastic_tables():
    """Absent key is Mott; it is spelled out since transport defaults to ELSEPA (#293)."""
    case = {"composition": [("Si", 0.05)]}

    assert _case_elastic_kwargs(case) == {"elastic_model": "mott"}
    assert _case_elastic_table_records(case) == []


def test_elsepa_cases_resolve_one_entry_per_layer_element(monkeypatch):
    calls = []

    def fake(composition):
        calls.append(list(composition))
        return tuple(
            SimpleNamespace(arrays={"element": element}, tables=(f"table-{element}",))
            for element, _ in composition
        )

    monkeypatch.setattr("pyrite.xsgen.elsepa.catalog.resolve_layer_tables", fake)
    si = list(CATALOG.crystal("silicon").info.composition)
    case = {
        "elastic_model": "elsepa",
        "composition": [("Mo", 0.02), ("S", 0.04)],
        "abs_layers": [
            (0.0, 1.0, [("Mo", 0.02), ("S", 0.04)]),
            (1.0, 2.0, si),
        ],
    }

    kwargs = _case_elastic_kwargs(case)

    assert kwargs["elastic_model"] == "elsepa"
    assert kwargs["elastic_tables"] == [
        [{"element": "Mo"}, {"element": "S"}],
        [{"element": "Si"}],
    ]
    assert _case_elastic_table_records(case) == ["table-Mo", "table-S", "table-Si"]
    assert calls[:2] == [case["abs_layers"][0][2], si]


def test_case_carries_only_the_opt_in_elastic_model():
    """Absent is the default, so a case never spells ``"mott"`` explicitly."""
    case = _any_case()

    assert replace(case, elastic_model="elsepa")["elastic_model"] == "elsepa"
    with pytest.raises(ValueError, match="elastic_model"):
        replace(case, elastic_model="mott")


def _any_case():
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.sweep import build_cases

    return build_cases(material_sweep("silicon"), 4, 4)[0]


def test_elastic_kwargs_are_numpy_ready(monkeypatch):
    table = {"energy_eV": np.array([1e2, 1e8])}
    monkeypatch.setattr(
        "pyrite.xsgen.elsepa.catalog.resolve_layer_tables",
        lambda composition: (SimpleNamespace(arrays=table, tables=()),),
    )

    kwargs = _case_elastic_kwargs({"elastic_model": "elsepa", "composition": [("Si", 0.05)]})

    assert kwargs["elastic_tables"][0][0] is table


@pytest.mark.parametrize("model", ["mott", "elsepa"])
def test_every_transport_call_site_names_the_case_elastic_model(monkeypatch, model):
    """Live transport, brem repair and the penetration gate all pass the case model (#293)."""
    from pyrite.campaign import config
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.sweep import build_cases
    from pyrite.montecarlo import runner

    monkeypatch.setattr(
        "pyrite.xsgen.elsepa.catalog.resolve_layer_tables",
        lambda composition: (SimpleNamespace(arrays={}, tables=()),),
    )
    case = build_cases(material_sweep("silicon"), 4, 4, elastic_model=model)[0]
    seen = []

    def stop(*_args, **kwargs):
        seen.append(kwargs.get("elastic_model"))
        raise RuntimeError("stop")

    def transmit(_E0_keV, Ne, _thickness_ang, **kwargs):
        seen.append(kwargs.get("elastic_model"))
        return {"n_transmitted": Ne}

    # _transport_case reads case_driver's binding; _brem_for_case reads the package's.
    monkeypatch.setattr(runner.case_driver, "simulate_trajectories", stop)
    monkeypatch.setattr(runner, "simulate_trajectories", stop)
    monkeypatch.setattr(config, "simulate_trajectories", transmit)
    with pytest.raises(RuntimeError, match="stop"):
        runner._transport_case(case)
    with pytest.raises(RuntimeError, match="stop"):
        runner._brem_for_case(case, np.array([5000.0]))
    config.gate_cases_by_penetration([case], Ne=10)

    assert seen == [model, model, model]


@pytest.mark.parametrize("model", ["mott", "elsepa"])
def test_live_transport_uses_the_selected_preparation_adapter(monkeypatch, model):
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.sweep import build_cases
    from pyrite.montecarlo import runner
    from pyrite.montecarlo.runner import case_tables

    case = build_cases(material_sweep("silicon"), 4, 4, elastic_model=model)[0]
    monkeypatch.setattr(
        "pyrite.xsgen.elsepa.catalog.resolve_layer_tables",
        lambda composition: (SimpleNamespace(arrays={}, tables=()),),
    )
    selected = []
    original = case_tables._ELASTIC_PREPARERS

    def prepare(entries):
        selected.append(model)
        return original[f"transport.elastic/{model}"](entries)

    monkeypatch.setattr(case_tables, "_ELASTIC_PREPARERS", {f"transport.elastic/{model}": prepare})

    def stop(*args, **kwargs):
        assert kwargs["elastic_model"] == model
        raise RuntimeError("native dispatch reached")

    monkeypatch.setattr(runner.case_driver, "simulate_trajectories", stop)
    with pytest.raises(RuntimeError, match="native dispatch reached"):
        runner._transport_case(case)
    assert selected == [model]


@pytest.mark.parametrize("model", ["auto", "sr", "unknown", None])
def test_preparation_rejects_unresolved_or_unsupported_case_models(monkeypatch, model):
    def forbidden(composition):
        pytest.fail("invalid selection must fail before ELSEPA table resolution")

    monkeypatch.setattr("pyrite.xsgen.elsepa.catalog.resolve_layer_tables", forbidden)
    with pytest.raises(ValueError, match="unknown physics descriptor"):
        _case_elastic_kwargs({"elastic_model": model, "composition": [("Si", 0.05)]})


def test_preparation_preserves_atomic_deflection_and_entry_resolution_seam(monkeypatch):
    from pyrite.montecarlo.runner import case_tables

    arrays = {"energy_eV": np.array([1e2, 1e8])}
    seen = []

    def entries(case):
        seen.append(case)
        return [(SimpleNamespace(arrays=arrays, tables=()),)]

    monkeypatch.setattr(case_tables, "_case_elastic_entries", entries)
    case = {"elastic_model": "elsepa", "atomic_electron_deflection": "kawrakow"}
    kwargs = _case_elastic_kwargs(case)
    assert seen == [case]
    assert list(kwargs) == ["elastic_model", "elastic_tables", "atomic_electron_deflection"]
    assert kwargs["elastic_tables"][0][0] is arrays
    assert kwargs["atomic_electron_deflection"] == "kawrakow"


def test_missing_elsepa_tables_keep_the_existing_error(monkeypatch):
    from pyrite.xsgen._errors import TableNotFoundError

    failure = TableNotFoundError("existing missing-table instruction")

    def missing(composition):
        raise failure

    monkeypatch.setattr("pyrite.xsgen.elsepa.catalog.resolve_layer_tables", missing)
    with pytest.raises(TableNotFoundError) as caught:
        _case_elastic_kwargs({"elastic_model": "elsepa", "composition": [("Si", 0.05)]})
    assert caught.value is failure


def test_runtime_adapter_catalog_is_immutable_and_matches_descriptor_ids():
    from pyrite._physics_contract import BUILTIN_DESCRIPTORS
    from pyrite.montecarlo.runner import case_tables

    ids = {
        descriptor.id
        for descriptor in BUILTIN_DESCRIPTORS.descriptors
        if descriptor.family == "transport.elastic"
    }
    assert set(case_tables._ELASTIC_PREPARERS) == ids
    with pytest.raises(TypeError):
        case_tables._ELASTIC_PREPARERS["transport.elastic/new"] = lambda entries: {}
