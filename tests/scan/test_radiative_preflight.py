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


def test_explicit_coupled_default_degrades_only_under_the_flag():
    """The coupled default (#642) is strict unless the run opts into fallback."""
    with pytest.raises(ValueError, match="grooved targets.*--allow-physics-fallback"):
        resolve_run_radiative_model("bremslib-soft-hard", "bremslib", grooved=True)
    with pytest.warns(PhysicsFallbackWarning, match="uncoupled"):
        model, cutoff, events = resolve_run_radiative_model(
            "bremslib-soft-hard", "bremslib", grooved=True, allow_fallback=True, cutoff_eV=1000.0
        )
    assert (model, cutoff) == ("uncoupled", None)
    assert events[0]["model_requested"] == "bremslib-soft-hard"
    assert events[0]["model_used"] == "uncoupled"


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
    args = Namespace(catalog_profile="standard", allow_physics_fallback=True)
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
        scan._resolved_run(Namespace(catalog_profile="standard"), "hopg")


def test_run_settings_opt_in_overrides_ambient_warning_errors(monkeypatch):
    import warnings
    from dataclasses import replace

    from pyrite.runs import scan
    from pyrite.xsgen.bremslib import tables

    scan._load_runtime()
    original = scan.default_settings
    assert original is not None
    monkeypatch.setattr(
        scan, "default_settings", lambda *_a, **_k: replace(original(), allow_fallback=True)
    )
    monkeypatch.setattr(tables, "resolve_bremsstrahlung_model", lambda *_a, **_k: "eedl")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PhysicsFallbackWarning)
        with pytest.warns(PhysicsFallbackWarning):
            settings, _, identity, _ = scan._resolved_run(
                Namespace(catalog_profile="standard"), "hopg"
            )
    assert settings.allow_fallback
    assert identity["resolved_parameters"]["transport_numerics"]["allow_physics_fallback"] is True


def test_opt_in_and_strict_have_distinct_dataset_cache_admission():
    from dataclasses import replace

    from pyrite.campaign.config import default_settings, material_sweep
    from pyrite.campaign.profiles import dataset_identity

    sweep = material_sweep("hopg")
    settings = default_settings()
    strict = dataset_identity("hopg", settings, sweep)
    allowed = dataset_identity("hopg", replace(settings, allow_fallback=True), sweep)
    assert strict["parameter_sha256"] != allowed["parameter_sha256"]
    assert "allow_physics_fallback" not in strict["resolved_parameters"]["transport_numerics"]
