"""Case-level BremsLib continuum selection, table resolution and identity (issue #86)."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyrite._numerics import Numerics
from pyrite.campaign.profiles import case_bremsstrahlung_marker, case_content_key
from pyrite.montecarlo.runner.case_tables import (
    _case_bremslib_table_records,
    _case_bremslib_tables,
)
from pyrite.montecarlo.spectrum import BREMSSTRAHLUNG_MODEL
from pyrite.montecarlo.spectrum.brem_bremslib import BREMSSTRAHLUNG_BREMSLIB_MODEL


def _any_case():
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.sweep import build_cases

    return build_cases(material_sweep("silicon"), 4, 4)[0]


def test_numerics_accepts_only_known_bremsstrahlung_models():
    assert Numerics().bremsstrahlung_model == "eedl"
    assert Numerics(bremsstrahlung_model="bremslib").bremsstrahlung_model == "bremslib"
    with pytest.raises(ValueError, match="bremsstrahlung_model"):
        Numerics(bremsstrahlung_model="bethe-heitler")  # type: ignore[arg-type]


def test_default_cases_resolve_no_bremslib_tables():
    case = {"composition": [("Si", 0.05)]}

    assert _case_bremslib_tables(case) is None
    assert _case_bremslib_table_records(case) == []


def test_bremslib_cases_load_tables_for_every_layer_element(monkeypatch):
    seen = []

    def fake(elements):
        seen.append(list(elements))
        return {
            e: SimpleNamespace(key=f"key-{e}", digest=f"digest-{e}")
            for e in dict.fromkeys(elements)
        }

    monkeypatch.setattr("pyrite.xsgen.bremslib.tables.load_bremsstrahlung_tables", fake)
    case = {
        "bremsstrahlung_model": "bremslib",
        "composition": [("Mo", 0.02), ("S", 0.04)],
        "abs_layers": [
            (0.0, 1.0, [("Mo", 0.02), ("S", 0.04)]),
            (1.0, 2.0, [("Si", 0.05)]),
        ],
    }

    assert set(_case_bremslib_tables(case)) == {"Mo", "S", "Si"}
    assert seen[0] == ["Mo", "S", "Si"]
    assert [t.key for t in _case_bremslib_table_records(case)] == ["key-Mo", "key-S", "key-Si"]


def test_case_carries_only_the_opt_in_bremsstrahlung_model():
    """Absent is EEDL, so a case never spells ``"eedl"`` explicitly."""
    case = _any_case()

    assert replace(case, bremsstrahlung_model="bremslib")["bremsstrahlung_model"] == "bremslib"
    with pytest.raises(ValueError, match="bremsstrahlung_model"):
        replace(case, bremsstrahlung_model="eedl")


def test_bremslib_forks_the_case_content_key_only_when_selected():
    case = _any_case()
    forked = replace(case, bremsstrahlung_model="bremslib")

    assert case_bremsstrahlung_marker(case) == BREMSSTRAHLUNG_MODEL
    assert case_bremsstrahlung_marker(forked) == BREMSSTRAHLUNG_BREMSLIB_MODEL
    assert case_content_key(case) != case_content_key(forked)
