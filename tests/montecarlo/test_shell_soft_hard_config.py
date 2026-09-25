"""Configuration threading of the opt-in shell soft/hard inelastic mode.

Validation: shell-soft-hard-transport
"""

from dataclasses import replace

import pytest

from pyrite import _numerics
from pyrite._numerics import Numerics
from pyrite.campaign.config import material_sweep
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo import runner
from pyrite.montecarlo.case import Case
from pyrite.montecarlo.transport.hard_inelastic import INELASTIC_MODELS
from pyrite.results.store import Settings


def test_numerics_inelastic_models_track_the_transport_constant():
    assert _numerics.INELASTIC_MODELS == INELASTIC_MODELS


@pytest.mark.parametrize("owner", (Numerics, Settings))
def test_inelastic_settings_validate_coupling(owner):
    assert owner().inelastic_model == "continuous"
    owner(energy_model="midpoint", inelastic_model="shell-soft-hard", inelastic_cutoff_eV=50.0)
    with pytest.raises(ValueError, match="requires energy_model='midpoint'"):
        owner(inelastic_model="shell-soft-hard", inelastic_cutoff_eV=50.0)
    with pytest.raises(ValueError, match="finite positive inelastic_cutoff_eV"):
        owner(energy_model="midpoint", inelastic_model="shell-soft-hard")
    with pytest.raises(ValueError, match="requires inelastic_model"):
        owner(inelastic_cutoff_eV=50.0)
    with pytest.raises(ValueError, match="inelastic_model must be one of"):
        owner(inelastic_model="dielectric")


def _case(**kw):
    case = build_cases(material_sweep("silicon"), 4, 4, energy_model="midpoint", **kw)[0]
    return case


def test_case_schema_and_runner_kwargs():
    legacy = _case()
    assert runner._case_inelastic_kwargs(legacy) == {}
    shell = _case(inelastic_model="shell-soft-hard", inelastic_cutoff_eV=50.0)
    assert isinstance(shell, Case)
    assert runner._case_inelastic_kwargs(shell) == {
        "inelastic_model": "shell-soft-hard",
        "inelastic_cutoff_eV": 50.0,
        "inelastic_materials": ["silicon"],
    }
    with pytest.raises(ValueError, match="set together"):
        replace(shell, inelastic_cutoff_eV=legacy_absent(legacy))
    stacked = dict(shell)
    stacked["abs_layers"] = [(0.0, 10.0, shell["composition"]), (10.0, 20.0, shell["composition"])]
    with pytest.raises(ValueError, match="absorber layer 1 has none"):
        runner._case_inelastic_kwargs(stacked)
    stacked["layer_radiators"] = [None, {"crystal": "mos2"}]
    assert runner._case_inelastic_kwargs(stacked)["inelastic_materials"] == ["silicon", "mos2"]


def test_build_cases_rejects_shell_mode_without_cutoff():
    with pytest.raises(ValueError, match="finite positive inelastic_cutoff_eV"):
        _case(inelastic_model="shell-soft-hard")


def test_shell_tables_accept_one_material_key():
    from pyrite.montecarlo.transport.shell_transport import build_shell_inelastic_tables
    from pyrite.montecarlo.transport.stopping import prepare_sbethe_stopping_table
    from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

    prepared = prepare_sbethe_stopping_table(resolve_catalog_table("silicon").arrays())
    assert build_shell_inelastic_tables("silicon", 50.0, [prepared], 5.0, 30.0).materials == (
        "silicon",
    )


def legacy_absent(case):
    return type(case).__dataclass_fields__["inelastic_cutoff_eV"].default


def test_shell_case_runs_on_the_exact_cuda_core(monkeypatch):
    from pyrite.montecarlo.transport import batching

    monkeypatch.setattr(batching, "_cuda_transport_available", lambda: True)
    monkeypatch.delenv("PYRITE_MC_TRANSPORT_CORE", raising=False)
    shell = dict(_case(inelastic_model="shell-soft-hard", inelastic_cutoff_eV=50.0))
    shell["Ne"] = 10**5
    assert runner._case_transport_core(shell) == "cuda"

    class Launched(Exception):
        pass

    seen = {}

    def launch(*args, **kwargs):
        seen.update(kwargs)
        raise Launched

    monkeypatch.setattr(runner, "simulate_trajectories", launch)
    with pytest.raises(Launched):
        runner._transport_case(shell)
    # The CUDA LUT kernel has no shell mode, so the runner takes the exact one.
    assert seen["transport_core"] == "cuda"
    assert seen["transport_lut_config"].enabled is False
    assert seen["inelastic_model"] == "shell-soft-hard"
