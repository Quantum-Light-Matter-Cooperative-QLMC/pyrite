"""Case-level BremsLib continuum selection, table resolution and identity (issue #86)."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyrite._numerics import Physics
from pyrite.campaign.profiles import case_bremsstrahlung_marker, case_content_key
from pyrite.montecarlo.runner.case_tables import (
    _case_bremslib_table_records,
    _case_bremslib_tables,
)
from pyrite.montecarlo.spectrum import BREMSSTRAHLUNG_MODEL
from pyrite.montecarlo.spectrum.brem_bremslib import BREMSSTRAHLUNG_BREMSLIB_MODEL


def _any_case(model="eedl"):
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.sweep import build_cases

    return build_cases(material_sweep("silicon"), 4, 4, bremsstrahlung_model=model)[0]


def test_physics_accepts_only_known_bremsstrahlung_models():
    assert Physics().bremsstrahlung_model == "auto"
    assert Physics(bremsstrahlung_model="eedl").bremsstrahlung_model == "eedl"
    assert Physics(bremsstrahlung_model="bremslib").bremsstrahlung_model == "bremslib"
    with pytest.raises(ValueError, match="bremsstrahlung_model"):
        Physics(bremsstrahlung_model="bethe-heitler")  # type: ignore[arg-type]


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
    case = _any_case("eedl")

    assert replace(case, bremsstrahlung_model="bremslib")["bremsstrahlung_model"] == "bremslib"
    with pytest.raises(ValueError, match="bremsstrahlung_model"):
        replace(case, bremsstrahlung_model="eedl")


def test_bremslib_forks_the_case_content_key_only_when_selected():
    case = _any_case("eedl")
    forked = replace(case, bremsstrahlung_model="bremslib")

    assert case_bremsstrahlung_marker(case) == BREMSSTRAHLUNG_MODEL
    assert case_bremsstrahlung_marker(forked) == BREMSSTRAHLUNG_BREMSLIB_MODEL
    assert case_content_key(case) != case_content_key(forked)


def test_auto_resolves_to_bremslib_only_when_every_table_is_installed(monkeypatch):
    from pyrite.xsgen import _errors
    from pyrite.xsgen.bremslib import tables

    monkeypatch.setattr(
        tables, "load_bremsstrahlung_tables", lambda elements: {e: object() for e in elements}
    )
    assert tables.resolve_bremsstrahlung_model("auto", ["C", "Si"]) == "bremslib"

    monkeypatch.setattr(tables, "load_bremsstrahlung_tables", lambda elements: {"C": object()})
    with pytest.warns(RuntimeWarning, match="pyrite tables fetch bremslib"):
        assert tables.resolve_bremsstrahlung_model("auto", ["C", "Si"]) == "eedl"

    def missing(elements):
        raise _errors.TableNotFoundError("not fetched")

    monkeypatch.setattr(tables, "load_bremsstrahlung_tables", missing)
    with pytest.warns(RuntimeWarning):
        assert tables.resolve_bremsstrahlung_model("auto", ["C"]) == "eedl"
    # Explicit choices pass through untouched, so "bremslib" stays strict.
    assert tables.resolve_bremsstrahlung_model("bremslib", ["C"]) == "bremslib"
    assert tables.resolve_bremsstrahlung_model("eedl", ["C"]) == "eedl"


def test_build_cases_records_the_resolved_continuum(monkeypatch):
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.sweep import build_cases

    monkeypatch.setattr(
        "pyrite.xsgen.bremslib.tables.resolve_bremsstrahlung_model",
        lambda model, elements: "bremslib",
    )
    resolved = build_cases(material_sweep("silicon"), 4, 4)[0]
    monkeypatch.setattr(
        "pyrite.xsgen.bremslib.tables.resolve_bremsstrahlung_model",
        lambda model, elements: "eedl",
    )
    fallback = build_cases(material_sweep("silicon"), 4, 4)[0]

    assert resolved["bremsstrahlung_model"] == "bremslib"
    assert "bremsstrahlung_model" not in fallback


def _mev_cases(monkeypatch, energy_keV, resolved, **kwargs):
    from pyrite.campaign.config import material_sweep
    from pyrite.campaign.sweep import BeamSpec, build_cases

    monkeypatch.setattr(
        "pyrite.xsgen.bremslib.tables.resolve_bremsstrahlung_model",
        lambda model, elements: resolved,
    )
    sweep = replace(material_sweep("silicon"), beam=BeamSpec(energy_keV=energy_keV))
    return build_cases(sweep, 4, 4, **kwargs)


def test_mev_cases_refuse_to_drop_radiative_loss_silently(monkeypatch):
    """#439: above 30 MeV an EEDL fallback may not run uncoupled unless asked."""
    with pytest.raises(ValueError, match="radiative_model='uncoupled'"):
        _mev_cases(monkeypatch, 5.0e4, "eedl")
    with pytest.raises(ValueError, match="drops radiative energy loss"):
        _mev_cases(monkeypatch, 1.0e5, "bremslib", bremsstrahlung_model="eedl")
    # Coupled cases, and the historical uncoupled fallback up to 30 MeV, pass.
    assert _mev_cases(monkeypatch, 5.0e4, "bremslib")[0]["radiative_model"] == (
        "bremslib-soft-hard"
    )
    assert "radiative_model" not in _mev_cases(monkeypatch, 3.0e4, "eedl")[0]
    # Above the tables' 100 MeV no coupled mode exists (channeling is out of scope).
    assert "radiative_model" not in _mev_cases(monkeypatch, 8.55e5, "eedl")[0]


def test_explicit_uncoupled_mev_case_records_the_opt_in(monkeypatch):
    opted = _mev_cases(monkeypatch, 5.0e4, "bremslib", radiative_model="uncoupled")[0]
    coupled = _mev_cases(monkeypatch, 5.0e4, "bremslib")[0]

    assert "radiative_model" not in opted
    assert opted["bremsstrahlung_model"] == "bremslib"
    assert case_content_key(opted) != case_content_key(coupled)
