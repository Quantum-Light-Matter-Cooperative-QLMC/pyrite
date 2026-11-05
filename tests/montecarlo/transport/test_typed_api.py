"""Typed public transport preserves the resolved engine's seeded trajectories."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite import Scene
from pyrite.beams import Beam, PointBunch, PointSpot, Spot
from pyrite.campaign.lowering import build_case
from pyrite.config import (
    AdaptiveSampling,
    ExecutionConfig,
    FixedSampling,
    Numerics,
    Physics,
    TransportKernelTuning,
    TransportNumerics,
)
from pyrite.geometry import Slab
from pyrite.montecarlo.transport import _simulate_trajectories
from pyrite.montecarlo.transport import api as transport_api
from pyrite.transport import simulate_trajectories


@pytest.fixture
def inputs(monkeypatch):
    # Isolate boundary parity from external SBETHE files. Both calls use the
    # existing analytic reference stopping law and the synthetic Mott fixture.
    monkeypatch.setattr(
        "pyrite.montecarlo.runner.case_tables._case_stopping_tables", lambda case: None
    )
    return (
        Beam(30.0, transverse=PointSpot(), longitudinal=PointBunch()),
        Slab("silicon", 100.0, tilt_deg=30.0, footprint=None),
        Physics(
            elastic_model="mott",
            inelastic_model="continuous",
            radiative_model="uncoupled",
            bremsstrahlung_model="eedl",
            atomic_electron_deflection="none",
        ),
        Numerics(
            sampling=FixedSampling(3, 2),
            seed=19,
            transport=TransportNumerics(energy_model="midpoint", max_dE_frac=0.02),
            execution=ExecutionConfig(transport_core="lockstep"),
        ),
    )


@pytest.mark.parametrize("spot", [False, True])
def test_typed_seeded_parity(inputs, spot):
    beam, target, physics, numerics = inputs
    if spot:
        beam = replace(beam, transverse=Spot(0.01, 0.02), energy_spread_frac=0.01)
    case = build_case(Scene(beam, target, detector=None), physics=physics, numerics=numerics)
    from pyrite.montecarlo.geometry import tilted_geometry

    polar = np.deg2rad(case["tilt_deg"])
    azimuth = np.deg2rad(case["tilt_azim_deg"])
    direction, _ = tilted_geometry(case["theta_obs_rad"], polar, azimuth)
    reference = _simulate_trajectories(
        case["E0_keV"],
        3,
        case["thickness_ang"],
        composition=case["composition"],
        seed=19,
        beam_dir=direction,
        tilt_polar_rad=polar,
        tilt_azim_rad=azimuth,
        beam_fwhm_mm=0.01 if spot else None,
        beam_fwhm_y_mm=0.02 if spot else None,
        energy_spread_frac=0.01 if spot else None,
        elastic_model="mott",
        radiative_model="uncoupled",
        atomic_electron_deflection=None,
        transport_core="lockstep",
        energy_model="midpoint",
        max_dE_frac=0.02,
    )
    actual = simulate_trajectories(beam, target, physics=physics, numerics=numerics)
    assert actual.keys() == reference.keys()
    for key, expected in reference.items():
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual[key], expected, err_msg=key)
        else:
            assert actual[key] == expected, key
    assert actual["Ne"] == 3


def test_removed_flat_arguments_fail():
    with pytest.raises(TypeError):
        simulate_trajectories(30.0, 3, 100.0, elastic_model="mott")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {
                "numerics": Numerics(
                    sampling=AdaptiveSampling(target_rse=0.1, max_electrons=100, block_electrons=10)
                )
            },
            "fixed sampling",
        ),
        (
            {"numerics": Numerics(execution=ExecutionConfig(stream_block_electrons=2))},
            "stream_block",
        ),
        ({"physics": "continuous"}, "Physics"),
        ({"numerics": 3}, "Numerics"),
    ],
)
def test_refuses_unsupported_inputs_before_engine(inputs, monkeypatch, kwargs, message):
    beam, target, _, _ = inputs
    monkeypatch.setattr(
        transport_api, "_simulate_trajectories", lambda **kw: pytest.fail("engine ran")
    )
    with pytest.raises((TypeError, ValueError), match=message):
        simulate_trajectories(beam, target, **kwargs)


def test_execution_and_output_controls(inputs, monkeypatch):
    beam, target, physics, numerics = inputs
    tuning = TransportKernelTuning(
        electron_batch=7,
        scratch_mib=8,
        kernel_threads=64,
        schedule="full-history",
        precision="fp64",
    )
    numerics = replace(
        numerics,
        execution=ExecutionConfig(transport_core="per-electron", transport=tuning),
        transport_core="auto",
    )
    monkeypatch.setattr(transport_api, "_simulate_trajectories", lambda **kw: kw)

    def callback(done, total):
        return None

    kw = simulate_trajectories(
        beam,
        target,
        physics=physics,
        numerics=numerics,
        E_cut_keV=2.0,
        max_steps=20,
        collect_diagnostics=True,
        transport_progress=callback,
    )
    assert kw["transport_core"] == "per-electron"
    assert kw["per_electron_config"].max_batch == 7
    assert kw["per_electron_config"].scratch_budget_bytes == 8 * 1024**2
    assert kw["per_electron_config"].kernel_threads == 64
    assert kw["transport_precision"] == "fp64"
    assert kw["E_cut_keV"] == 2.0
    assert kw["max_steps"] == 20
    assert kw["collect_diagnostics"] is True
    assert kw["transport_progress"] is callback


@pytest.mark.parametrize("core, disabled", [("cuda", True), ("lockstep", False)])
def test_explicit_inelastic_lut_rule_matches_case_driver(inputs, monkeypatch, core, disabled):
    beam, target, physics, numerics = inputs
    physics = replace(physics, inelastic_model="shell-soft-hard", inelastic_cutoff_eV=50.0)
    numerics = replace(
        numerics, execution=ExecutionConfig(transport_core=core), transport_core="auto"
    )
    monkeypatch.setattr(transport_api, "_simulate_trajectories", lambda **kw: kw)
    kw = simulate_trajectories(beam, target, physics=physics, numerics=numerics)
    assert kw["transport_core"] == core
    assert kw["inelastic_model"] == "shell-soft-hard"
    assert ("transport_lut_config" in kw) is disabled
    if disabled:
        assert kw["transport_lut_config"].enabled is False
