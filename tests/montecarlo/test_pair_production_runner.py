"""Pair conversion (#275) wired through numerics, cases, identity and the runner.

``pair_production_model`` is a divergence-only key that rides on coupled
radiative transport and the secondary cascade.

Validation: photon-pair-first-interaction
"""

from dataclasses import replace

import pytest

import pyrite as pr
from pyrite import _numerics
from pyrite.campaign import profiles
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.profiles import dataset_identity
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo.case import _ABSENT
from pyrite.montecarlo.runner.case_tables import _case_radiative_kwargs
from pyrite.montecarlo.transport import pair_production
from pyrite.xsgen.bremslib import tables as bremslib_tables

COUPLED = {"radiative_model": "bremslib-soft-hard", "radiative_cutoff_eV": 1000.0}
SHELL = {
    "energy_model": "midpoint",
    "inelastic_model": "shell-soft-hard",
    "inelastic_cutoff_eV": 50.0,
    "secondary_threshold_eV": 100_000.0,
}
PAIR = {"pair_production_model": "penelope-2024"}


def test_model_names_stay_in_step_with_transport():
    assert _numerics.PAIR_PRODUCTION_MODELS == pair_production.PAIR_PRODUCTION_MODELS


def test_numerics_and_settings_validate_pair_requirements():
    assert pr.Numerics().pair_production_model is None
    numerics = pr.Numerics(**SHELL, **COUPLED, **PAIR)
    assert numerics.pair_production_model == "penelope-2024"
    with pytest.raises(ValueError, match="requires secondary_threshold_eV"):
        pr.Numerics(energy_model="midpoint", **COUPLED, **PAIR)
    with pytest.raises(ValueError, match="requires coupled BremsLib"):
        pr.Numerics(**SHELL, radiative_model="uncoupled", **PAIR)
    with pytest.raises(ValueError, match="requires coupled BremsLib"):
        pr.Numerics(**SHELL, bremsstrahlung_model="eedl", **PAIR)
    with pytest.raises(ValueError, match="pair_production_model must be one of"):
        pr.Numerics(**SHELL, pair_production_model="bethe-heitler")
    with pytest.raises(ValueError, match="requires secondary_threshold_eV"):
        replace(default_settings(), **PAIR)


@pytest.mark.slow
@pytest.mark.parametrize("resolved", ["bremslib", "eedl"])
def test_case_key_joins_only_coupled_cases(monkeypatch, resolved):
    monkeypatch.setattr(
        bremslib_tables, "resolve_bremsstrahlung_model", lambda _model, _elements: resolved
    )
    sweep = material_sweep("silicon")
    plain = build_cases(sweep, 4, 4, **SHELL, **COUPLED)[0]
    paired = build_cases(sweep, 4, 4, **SHELL, **COUPLED, **PAIR)[0]
    assert "pair_production_model" not in plain
    if resolved == "bremslib":
        assert paired["pair_production_model"] == "penelope-2024"
        assert profiles.case_content_key(paired) != profiles.case_content_key(plain)
        assert _case_radiative_kwargs(paired)["pair_production_model"] == "penelope-2024"
        assert "pair_production_model" not in _case_radiative_kwargs(plain)
        with pytest.raises(ValueError, match="requires secondary_threshold_eV"):
            replace(paired, secondary_threshold_eV=_ABSENT)
    else:
        assert "pair_production_model" not in paired and "radiative_model" not in paired
    with pytest.raises(ValueError, match="must be absent or 'penelope-2024'"):
        replace(plain, pair_production_model="other")


@pytest.mark.parametrize("resolved", ["bremslib", "eedl"])
def test_identity_forks_only_when_pairs_can_run(monkeypatch, resolved):
    monkeypatch.setattr(
        profiles, "resolve_bremsstrahlung_model", lambda _model, _elements: resolved
    )
    sweep = material_sweep("silicon")
    settings = replace(default_settings(), **SHELL, **COUPLED)
    base = dataset_identity("silicon", "full", settings, sweep)
    paired = dataset_identity("silicon", "full", replace(settings, **PAIR), sweep)
    numerics = paired["resolved_parameters"]["transport_numerics"]
    assert "pair_production_model" not in base["resolved_parameters"]["transport_numerics"]
    if resolved == "bremslib":
        assert numerics["pair_production_model"] == "penelope-2024"
        assert paired["parameter_sha256"] != base["parameter_sha256"]
    else:
        assert "pair_production_model" not in numerics
        assert paired["parameter_sha256"] == base["parameter_sha256"]


def test_case_pair_model_reaches_runner_transport(monkeypatch):
    """A Numerics-built case runs the cascade with pair conversion enabled."""
    import numpy as np

    from pyrite import api
    from pyrite.detectors import EnergyBins
    from pyrite.montecarlo import shell_configuration as config
    from pyrite.montecarlo.runner import _transport_case, case_tables
    from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
    from tests.helpers.bremslib import synthetic_bremslib_arrays

    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    monkeypatch.setattr(bremslib_tables, "resolve_bremsstrahlung_model", lambda *_: "bremslib")
    arrays = synthetic_bremslib_arrays(t1_MeV=np.array([1e-3, 1e-2, 5e-2, 1e-1]))
    table = prepare_bremslib_table(arrays, atomic_number=14)
    monkeypatch.setattr(case_tables, "_case_bremslib_tables", lambda _case: {"Si": table})
    numerics = pr.Numerics(
        n_electrons=4,
        n_electrons_brem=4,
        elastic_model="mott",
        **{**SHELL, "secondary_threshold_eV": 5000.0},
        **COUPLED,
        **PAIR,
    )
    detector = pr.Detector(
        energy_bins=EnergyBins(line=np.linspace(1500, 2000, 20), brem=np.linspace(1500, 9000, 20))
    )
    scene = pr.Scene(pr.Beam(energy_keV=20.0), pr.Slab("silicon", thickness_ang=5.0e4), detector)
    case = api.build_case(scene, numerics)
    assert case["pair_production_model"] == "penelope-2024"
    segments = _transport_case(case, transport_core="per-electron")["segs"]
    # 20 keV photons cannot convert; the mode still runs and records nothing.
    assert segments["pair_production"]["model"] == "penelope-2024"
    assert segments["pair_production"]["photon_counts"]["photons"] == 0
    assert segments["secondary_tracks"]["launch_kind"].max() <= 1
