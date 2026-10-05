"""Positron transport (#276) wired through numerics, cases, identity and the runner.

``positron_transport`` is a divergence-only key that rides on pair conversion.

Validation: bhabha-close
"""

from dataclasses import replace

import pytest

import pyrite as pr
from pyrite.campaign import profiles
from pyrite.campaign.config import default_settings, material_sweep
from pyrite.campaign.profiles import dataset_identity
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo.case import _ABSENT
from pyrite.montecarlo.runner.case_tables import _case_radiative_kwargs
from pyrite.xsgen.bremslib import tables as bremslib_tables

from .test_pair_production_runner import COUPLED, PAIR, SHELL

POSITRON = {"positron_transport": True}


def test_numerics_and_settings_require_pair_conversion():
    assert pr.Numerics().positron_transport is False
    assert pr.Numerics(**SHELL, **COUPLED, **PAIR, **POSITRON).positron_transport is True
    with pytest.raises(ValueError, match="positron_transport requires pair_production_model"):
        pr.Numerics(**SHELL, **COUPLED, **POSITRON)
    with pytest.raises(TypeError, match="must be a bool"):
        pr.Numerics(**SHELL, **COUPLED, **PAIR, positron_transport="yes")
    with pytest.raises(ValueError, match="positron_transport requires pair_production_model"):
        replace(default_settings(), **SHELL, **COUPLED, **POSITRON)


@pytest.mark.parametrize("resolved", ["bremslib", "eedl"])
def test_case_key_joins_only_coupled_pair_cases(monkeypatch, resolved):
    monkeypatch.setattr(
        bremslib_tables, "resolve_bremsstrahlung_model", lambda _model, _elements: resolved
    )
    sweep = material_sweep("silicon")
    paired = build_cases(sweep, 4, 4, **SHELL, **COUPLED, **PAIR)[0]
    positron = build_cases(sweep, 4, 4, **SHELL, **COUPLED, **PAIR, **POSITRON)[0]
    assert "positron_transport" not in paired
    if resolved == "bremslib":
        assert positron["positron_transport"] is True
        assert profiles.case_content_key(positron) != profiles.case_content_key(paired)
        assert _case_radiative_kwargs(positron)["positron_transport"] is True
        assert "positron_transport" not in _case_radiative_kwargs(paired)
        with pytest.raises(ValueError, match="requires pair_production_model"):
            replace(positron, pair_production_model=_ABSENT)
    else:
        assert "positron_transport" not in positron
    with pytest.raises(ValueError, match="must be absent or True"):
        replace(paired, positron_transport=False)


@pytest.mark.parametrize("resolved", ["bremslib", "eedl"])
def test_identity_forks_only_when_positrons_can_run(monkeypatch, resolved):
    monkeypatch.setattr(
        profiles, "resolve_bremsstrahlung_model", lambda _model, _elements: resolved
    )
    sweep = material_sweep("silicon")
    settings = replace(default_settings(), **SHELL, **COUPLED, **PAIR)
    base = dataset_identity("silicon", "full", settings, sweep)
    positron = dataset_identity("silicon", "full", replace(settings, **POSITRON), sweep)
    numerics = positron["resolved_parameters"]["transport_numerics"]
    assert "positron_transport" not in base["resolved_parameters"]["transport_numerics"]
    if resolved == "bremslib":
        assert numerics["positron_transport"] is True
        assert positron["parameter_sha256"] != base["parameter_sha256"]
    else:
        assert "positron_transport" not in numerics
        assert positron["parameter_sha256"] == base["parameter_sha256"]
