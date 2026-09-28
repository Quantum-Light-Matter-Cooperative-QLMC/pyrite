"""Production-surface contracts for energy-loss straggling controls."""

from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

from pyrite.campaign.config import material_sweep
from pyrite.campaign.model import Numerics
from pyrite.campaign.profiles import case_content_key
from pyrite.campaign.sweep import build_cases
from pyrite.montecarlo.runner import _transport_case


def test_numerics_validates_coupled_energy_controls():
    with pytest.raises(ValueError, match="requires energy_model='midpoint'"):
        Numerics(energy_model="frozen", max_dE_frac=0.02)
    with pytest.raises(ValueError, match="finite and non-negative"):
        Numerics(energy_model="midpoint", max_dE_frac=float("nan"))


def test_transport_controls_lower_into_case_and_content_identity():
    sweep = material_sweep("hopg", energy_keV=[30.0], thickness_ang=[1000.0])
    base = build_cases(sweep, n_electrons=1, n_electrons_brem=1)[0]
    active = build_cases(
        sweep,
        n_electrons=1,
        n_electrons_brem=1,
        straggling=True,
        energy_model="midpoint",
        max_dE_frac=0.02,
    )[0]

    assert "straggling" not in base
    assert base["energy_model"] == "midpoint"
    assert "max_dE_frac" not in base
    assert active["straggling"] is True
    assert active["energy_model"] == "midpoint"
    assert active["max_dE_frac"] == 0.02
    assert case_content_key(active) != case_content_key(base)


def test_runner_falls_back_from_cuda_lut_to_exact_for_straggling():
    # Fixed line grid keeps this fallback-plumbing test independent of
    # automatic line-grid resolution, which needs real transport segments.
    case = build_cases(
        material_sweep(
            "hopg",
            energy_keV=[30.0],
            thickness_ang=[1000.0],
            E_grid_line=np.array([10.0, 2600.0]),
        ),
        n_electrons=1,
        n_electrons_brem=1,
        straggling=True,
        energy_model="midpoint",
        max_dE_frac=0.02,
    )[0]
    assert case.get("line_grid_policy") is None
    segments = {"sentinel": object()}

    with (
        mock.patch("pyrite.montecarlo.runner._case_transport_core", return_value="cuda"),
        mock.patch(
            "pyrite.montecarlo.runner.simulate_trajectories", return_value=segments
        ) as simulate,
    ):
        result = _transport_case(case, transport_core="cuda")

    kwargs = simulate.call_args.kwargs
    assert kwargs["straggling"] is True
    assert kwargs["energy_model"] == "midpoint"
    assert kwargs["max_dE_frac"] == 0.02
    assert kwargs["transport_lut_config"].enabled is False
    assert result["segs"] is segments


def test_case_rejects_inconsistent_manual_transport_controls():
    case = build_cases(
        material_sweep("hopg", energy_keV=[30.0], thickness_ang=[1000.0]),
        n_electrons=1,
        n_electrons_brem=1,
        energy_model="frozen",
        radiative_model="uncoupled",
    )[0]
    with pytest.raises(ValueError, match="requires energy_model='midpoint'"):
        replace(case, max_dE_frac=0.02)
