"""Case-level ELSEPA elastic-model selection and table resolution (issue #89)."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pyrite._numerics import Numerics
from pyrite.materials import CATALOG
from pyrite.montecarlo.runner.case_tables import (
    _case_elastic_kwargs,
    _case_elastic_table_records,
)


def test_numerics_accepts_only_known_elastic_models():
    assert Numerics().elastic_model == "mott"
    assert Numerics(elastic_model="elsepa").elastic_model == "elsepa"
    with pytest.raises(ValueError, match="elastic_model"):
        Numerics(elastic_model="sr")  # type: ignore[arg-type]


def test_default_cases_resolve_no_elastic_tables():
    case = {"composition": [("Si", 0.05)]}

    assert _case_elastic_kwargs(case) == {}
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
