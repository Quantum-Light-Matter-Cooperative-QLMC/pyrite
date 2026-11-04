"""Run preflight refuses implicit changes to radiative transport."""

from argparse import Namespace

import pytest

from pyrite._physics_fallback import PhysicsFallbackWarning
from pyrite.campaign.bremsstrahlung_cases import resolve_run_radiative_model


@pytest.mark.parametrize("source,grooved", [("eedl", False), ("bremslib", True)])
def test_auto_requires_opt_in_for_uncoupled_transport(source, grooved):
    with pytest.raises(ValueError, match="--allow-physics-fallback"):
        resolve_run_radiative_model("auto", source, grooved=grooved)
    with pytest.warns(PhysicsFallbackWarning, match="uncoupled"):
        model, cutoff, events = resolve_run_radiative_model(
            "auto", source, grooved=grooved, allow_fallback=True, cutoff_eV=1000.0
        )
    assert model == "uncoupled"
    assert cutoff is None
    assert events[0]["model_used"] == "uncoupled"
    assert events[0]["fraction_segments_affected"] is None


def test_explicit_coupled_request_cannot_be_degraded_by_flag():
    with pytest.raises(ValueError, match="grooved targets"):
        resolve_run_radiative_model(
            "bremslib-soft-hard", "bremslib", grooved=True, allow_fallback=True
        )


def test_supported_coupling_and_explicit_uncoupled_do_not_report_fallbacks():
    assert resolve_run_radiative_model("auto", "bremslib", grooved=False) == (
        "bremslib-soft-hard",
        None,
        [],
    )
    assert resolve_run_radiative_model("uncoupled", "eedl", grooved=True) == ("uncoupled", None, [])


def test_scan_records_both_source_and_radiative_fallbacks(monkeypatch):
    from pyrite.runs import scan
    from pyrite.xsgen.bremslib import tables

    monkeypatch.setattr(tables, "resolve_bremsstrahlung_model", lambda *_a, **_k: "eedl")
    args = Namespace(fidelity="full", catalog_profile="standard", allow_physics_fallback=True)
    with pytest.warns(PhysicsFallbackWarning):
        settings, _, identity, _ = scan._resolved_run(args, "hopg")
    assert settings.radiative_model == "uncoupled"
    events = identity["physics_fallbacks"]
    assert [event["model_used"] for event in events] == ["eedl", "uncoupled"]
    assert events[1]["elements"] == events[0]["elements"]
    assert "radiative_model" not in identity["resolved_parameters"].get("transport_numerics", {})


def test_scan_refuses_before_minting_an_uncoupled_identity(monkeypatch):
    from pyrite.runs import scan
    from pyrite.xsgen.bremslib import tables

    monkeypatch.setattr(tables, "resolve_bremsstrahlung_model", lambda *_a, **_k: "eedl")
    with pytest.raises(ValueError, match="--allow-physics-fallback"):
        scan._resolved_run(Namespace(fidelity="full", catalog_profile="standard"), "hopg")
