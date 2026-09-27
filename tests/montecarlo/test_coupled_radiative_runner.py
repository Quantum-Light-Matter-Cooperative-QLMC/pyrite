"""Coupled BremsLib radiative transport wired through numerics, cases and the runner.

Issue #172: the opt-in ``radiative_model="bremslib-soft-hard"`` reaches
production transport, continuum scoring and the brem repair path, as
divergence-only identity keys.

Validation: bremslib-radiative-partition, bremslib-radiative-event-spectrum
"""

from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite import api
from pyrite.campaign import profiles
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.profiles import dataset_identity
from pyrite.campaign.sweep import build_cases
from pyrite.detectors import EnergyBins
from pyrite.montecarlo import runner
from pyrite.montecarlo.spectrum.brem import mc_brem_spectrum
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.spectrum.brem_events import mc_soft_brem_spectrum
from pyrite.montecarlo.transport.events import EVENT_CUTOFF, EVENT_HARD_RADIATIVE
from pyrite.xsgen.bremslib import tables as bremslib_tables
from tests.helpers.bremslib import synthetic_bremslib_arrays

COUPLED = {"radiative_model": "bremslib-soft-hard", "radiative_cutoff_eV": 1000.0}


def test_numerics_validate_coupled_radiative_requirements():
    numerics = pr.Numerics(energy_model="midpoint", **COUPLED)
    assert numerics.radiative_model == "bremslib-soft-hard"
    with pytest.raises(ValueError, match="radiative_cutoff_eV requires"):
        pr.Numerics(radiative_cutoff_eV=1000.0)
    with pytest.raises(ValueError, match="finite positive radiative_cutoff_eV"):
        pr.Numerics(energy_model="midpoint", radiative_model="bremslib-soft-hard")
    with pytest.raises(ValueError, match="energy_model='midpoint'"):
        pr.Numerics(**COUPLED)
    assert pr.Numerics(energy_model="midpoint", straggling=True, **COUPLED).straggling
    with pytest.raises(ValueError, match="requires BremsLib"):
        pr.Numerics(energy_model="midpoint", bremsstrahlung_model="eedl", **COUPLED)
    with pytest.raises(ValueError, match="radiative_model must be one of"):
        pr.Numerics(radiative_model="coupled")


def test_case_keys_are_divergence_only_and_need_bremslib():
    sweep = material_sweep("silicon")
    plain = build_cases(sweep, 4, 4, energy_model="midpoint", bremsstrahlung_model="bremslib")[0]
    coupled = build_cases(
        sweep, 4, 4, energy_model="midpoint", bremsstrahlung_model="bremslib", **COUPLED
    )[0]
    assert "radiative_model" not in plain and "radiative_cutoff_eV" not in plain
    assert coupled["radiative_model"] == "bremslib-soft-hard"
    assert coupled["radiative_cutoff_eV"] == 1000.0
    assert profiles.case_content_key(coupled) != profiles.case_content_key(plain)
    eedl = build_cases(sweep, 4, 4, energy_model="midpoint", bremsstrahlung_model="eedl")[0]
    with pytest.raises(ValueError, match="requires bremsstrahlung_model='bremslib'"):
        replace(eedl, **COUPLED)
    with pytest.raises(ValueError, match="must not exceed the continuum electron cutoff"):
        replace(coupled, radiative_cutoff_eV=1500.0)
    with pytest.raises(ValueError, match="E_cut_brem_keV <= E_cut_lines_keV"):
        replace(coupled, E_cut_brem_keV=6.0)
    with pytest.raises(ValueError, match="set together"):
        replace(plain, radiative_model="bremslib-soft-hard")
    assert replace(coupled, straggling=True)["straggling"] is True


@pytest.mark.parametrize("resolved", ["bremslib", "eedl"])
def test_auto_bremsstrahlung_couples_only_cases_that_resolve_to_bremslib(monkeypatch, resolved):
    monkeypatch.setattr(
        bremslib_tables, "resolve_bremsstrahlung_model", lambda _model, _elements: resolved
    )
    case = build_cases(material_sweep("silicon"), 4, 4, energy_model="midpoint", **COUPLED)[0]
    assert ("radiative_model" in case) is (resolved == "bremslib")
    assert (case.get("bremsstrahlung_model") == "bremslib") is (resolved == "bremslib")


@pytest.mark.parametrize("resolved", ["bremslib", "eedl"])
def test_identity_records_coupling_only_when_bremslib_runs(monkeypatch, resolved):
    monkeypatch.setattr(
        profiles, "resolve_bremsstrahlung_model", lambda _model, _elements: resolved
    )
    sweep = material_sweep("silicon")
    settings = replace(default_settings(), energy_model="midpoint")
    base = dataset_identity("silicon", "full", settings, sweep)
    coupled = dataset_identity("silicon", "full", replace(settings, **COUPLED), sweep)
    other = dataset_identity(
        "silicon", "full", replace(settings, **{**COUPLED, "radiative_cutoff_eV": 500.0}), sweep
    )
    numerics = coupled["resolved_parameters"]["transport_numerics"]
    assert "radiative_model" not in base["resolved_parameters"]["transport_numerics"]
    if resolved == "bremslib":
        assert numerics["radiative_model"] == "bremslib-soft-hard"
        assert numerics["radiative_cutoff_eV"] == 1000.0
        digests = {base["parameter_sha256"], coupled["parameter_sha256"], other["parameter_sha256"]}
        assert len(digests) == 3
    else:
        assert "radiative_model" not in numerics
        assert coupled["parameter_sha256"] == base["parameter_sha256"]


def _synthetic_table(element, atomic_number):
    arrays = synthetic_bremslib_arrays(t1_MeV=np.array([5.0e-3, 1.0e-2, 5.0e-2, 1.0e-1, 2.0e-1]))
    table = prepare_bremslib_table(
        arrays, atomic_number=atomic_number, key=f"bremslib/{element}", digest="0" * 64
    )
    # Inflated so a dozen electrons emit hard photons (as test_hard_radiative_transport).
    return replace(
        table,
        scaled_sdcs_mb=table.scaled_sdcs_mb * 1e4,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * 1e4,
    )


@pytest.fixture
def synthetic_si_tables(monkeypatch):
    """Inflated 5-200 keV synthetic Si and O tables in place of the released ones."""
    tables = {"Si": _synthetic_table("Si", 14), "O": _synthetic_table("O", 8)}
    monkeypatch.setattr(
        bremslib_tables,
        "load_bremsstrahlung_tables",
        lambda elements, **_kwargs: {e: tables[e] for e in dict.fromkeys(elements) if e in tables},
    )
    return {"Si": tables["Si"]}


def _coupled_case(target=None, **numerics_overrides):
    detector = pr.Detector(
        energy_bins=EnergyBins(
            line=np.linspace(1500.0, 2000.0, 20), brem=np.arange(5000.0, 40_001.0, 500.0)
        )
    )
    numerics = pr.Numerics(
        n_electrons=12,
        n_electrons_brem=12,
        energy_model="midpoint",
        elastic_model="mott",
        bremsstrahlung_model="bremslib",
        **{**COUPLED, **numerics_overrides},
    )
    if target is None:
        target = pr.Slab("silicon", thickness_ang=2.0e4, tilt_deg=30.0)
    scene = pr.Scene(pr.Beam(energy_keV=40.0), target, detector)
    # The synthetic table starts at 5 keV, so the electrons stop there.
    return replace(api.build_case(scene, numerics), E_cut_lines_keV=5.0, E_cut_brem_keV=5.0)


@pytest.mark.parametrize("straggling", [False, True])
def test_runner_transport_and_scoring_use_the_coupled_mode(synthetic_si_tables, straggling):
    case = _coupled_case(straggling=straggling)
    assert case["radiative_model"] == "bremslib-soft-hard"
    tp = runner._transport_case(case)
    segments = tp["segs"]
    assert segments["radiative"]["model"] == "bremslib-soft-hard"
    assert segments["radiative"]["cutoff_eV"] == 1000.0
    photons = segments["hard_radiative_k_eV"]
    kind = segments["event_kind"]
    assert np.any((kind == EVENT_HARD_RADIATIVE) | ((kind == EVENT_CUTOFF) & (photons > 0)))

    E_brem = tp["E_brem"]
    brem = runner._brem_wide_from_segments(
        segments, E_brem, case, tp["n_hat"], case.get("abs_layers"), Ne=case["Ne_brem"]
    )
    assert brem.shape == E_brem.shape and np.all(np.isfinite(brem)) and np.any(brem > 0.0)
    # The 1 keV cutoff lies below the brem grid, so every on-grid photon is hard.
    soft = mc_soft_brem_spectrum(
        segments,
        E_brem,
        cutoff_eV=1000.0,
        bremslib_tables=synthetic_si_tables,
        composition=case["composition"],
        n_hat=tp["n_hat"],
    )
    np.testing.assert_array_equal(soft, 0.0)
    with pytest.raises(ValueError, match="coupled radiative tracks"):
        mc_brem_spectrum(segments, E_brem, composition=case["composition"], n_hat=tp["n_hat"])

    # The brem repair replays the coupled transport on the same streams.
    np.testing.assert_allclose(runner._brem_for_case(case, E_brem), brem, rtol=1e-10, atol=0.0)


def test_uncoupled_case_transport_passes_no_radiative_arguments(monkeypatch):
    case = build_cases(material_sweep("silicon"), 4, 4, bremsstrahlung_model="eedl")[0]
    assert runner._case_radiative_kwargs(case) == {}
    seen = {}

    def fake(*_args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(runner, "simulate_trajectories", fake)
    with pytest.raises(RuntimeError, match="stop"):
        runner._transport_case(case)
    assert "radiative_model" not in seen and "bremslib_tables" not in seen


def test_layered_stack_scores_soft_per_layer_and_hard_through_the_stack(synthetic_si_tables):
    from pyrite.campaign.geometry import Layer, Stack

    case = _coupled_case(
        Stack(layers=(Layer("silicon", 5.0e3), Layer("sio2", 2.0e4)), tilt_deg=30.0)
    )
    assert case["abs_layers"] is not None and len(case["abs_layers"]) == 2
    tp = runner._transport_case(case)
    segments = tp["segs"]
    assert set(np.unique(segments["hard_radiative_Z"])) >= {14}
    brem = runner._brem_wide_from_segments(
        segments, tp["E_brem"], case, tp["n_hat"], case["abs_layers"], Ne=case["Ne_brem"]
    )
    assert np.all(np.isfinite(brem)) and np.any(brem > 0.0)
    np.testing.assert_allclose(runner._brem_for_case(case, tp["E_brem"]), brem, rtol=1e-10)
