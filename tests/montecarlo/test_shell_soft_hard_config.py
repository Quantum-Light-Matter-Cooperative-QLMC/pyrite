"""Configuration threading of the shell soft/hard inelastic mode and its ``auto`` default.

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
    assert _numerics.INELASTIC_MODELS == ("auto", *INELASTIC_MODELS)


@pytest.mark.parametrize("owner", (Numerics, Settings))
def test_inelastic_settings_validate_coupling(owner):
    assert owner().inelastic_model == "auto"
    owner(inelastic_cutoff_eV=60.0)
    owner(energy_model="midpoint", inelastic_model="shell-soft-hard", inelastic_cutoff_eV=50.0)
    with pytest.raises(ValueError, match="requires energy_model='midpoint'"):
        owner(
            energy_model="frozen",
            radiative_model="uncoupled",
            inelastic_model="shell-soft-hard",
            inelastic_cutoff_eV=50.0,
        )
    with pytest.raises(ValueError, match="finite positive inelastic_cutoff_eV"):
        owner(energy_model="midpoint", inelastic_model="shell-soft-hard")
    with pytest.raises(ValueError, match="requires inelastic_model"):
        owner(inelastic_model="continuous", inelastic_cutoff_eV=50.0)
    with pytest.raises(ValueError, match="requires inelastic_model"):
        owner(secondary_threshold_eV=100.0)
    with pytest.raises(ValueError, match="inelastic_model must be one of"):
        owner(inelastic_model="dielectric")


def _case(**kw):
    case = build_cases(material_sweep("silicon"), 4, 4, energy_model="midpoint", **kw)[0]
    return case


def test_case_schema_and_runner_kwargs():
    legacy = _case(inelastic_model="continuous")
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


def test_auto_resolves_covered_material_to_shell_mode():
    case = _case()
    assert case["inelastic_model"] == "shell-soft-hard"
    assert case["inelastic_cutoff_eV"] == _numerics.DEFAULT_INELASTIC_CUTOFF_EV
    assert _case(inelastic_cutoff_eV=80.0)["inelastic_cutoff_eV"] == 80.0
    with pytest.raises(ValueError, match="conduction-band resonance"):
        _case(inelastic_cutoff_eV=10.0)


@pytest.mark.parametrize(
    ("material", "energy_model", "reason"),
    (
        ("hfs2", "midpoint", "no conduction-band shell data for 'hfs2'"),
        ("silicon", "frozen", "energy_model='frozen'"),
    ),
)
def test_auto_falls_back_to_continuous_with_a_warning(material, energy_model, reason):
    with pytest.warns(UserWarning, match=reason):
        cases = build_cases(
            material_sweep(material),
            4,
            4,
            energy_model=energy_model,
            radiative_model="uncoupled",
        )
    assert all(case.get("inelastic_model") is None for case in cases)
    assert runner._case_inelastic_kwargs(cases[0]) == {}


@pytest.mark.parametrize(("material", "shell"), (("silicon", True), ("hfs2", False)))
def test_auto_identity_records_the_resolved_model(material, shell):
    from pyrite.campaign.config import default_settings
    from pyrite.campaign.profiles import dataset_identity

    settings = default_settings()
    identity = dataset_identity(material, "full", settings, material_sweep(material))
    numerics = identity["resolved_parameters"]["transport_numerics"]
    if shell:
        assert numerics["inelastic_model"] == "shell-soft-hard"
        assert numerics["inelastic_cutoff_eV"] == 50.0
    else:
        assert "inelastic_model" not in numerics
        continuous = dataset_identity(
            material,
            "full",
            replace(settings, inelastic_model="continuous"),
            material_sweep(material),
        )
        assert continuous["parameter_sha256"] == identity["parameter_sha256"]


def test_missing_shell_reference_data_names_the_fetch_command(monkeypatch, tmp_path):
    from pyrite.montecarlo import shell_configuration
    from pyrite.xsgen._errors import DataFetchError

    monkeypatch.setattr(shell_configuration, "_default_path", lambda: tmp_path / "pdatconf.p14")
    with pytest.raises(DataFetchError, match="pyrite tables fetch sbethe"):
        shell_configuration.load_atomic_shells()


def test_uncovered_material_names_the_covered_set():
    from pyrite.montecarlo.transport.shell_rates import catalog_shell_oscillators

    with pytest.raises(ValueError, match="'hfs2' has no conduction-band shell data.*silicon"):
        catalog_shell_oscillators("hfs2")
