"""Default lookup, exact replay and persisted approximation identity (#788)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite._radiative_moment_policy import RADIATIVE_MOMENT_MODEL
from pyrite.montecarlo.runner import case_tables
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.transport import simulate_trajectories
from tests.helpers.bremslib import synthetic_bremslib_arrays


def _table():
    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6)
    energy = table.incident_energy_keV.copy()
    energy[0], energy[-1] = 10.0, 200.0
    return replace(
        table,
        incident_energy_keV=energy,
        scaled_sdcs_mb=table.scaled_sdcs_mb * 1e5,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * 1e5,
    )


@pytest.mark.slow
@pytest.mark.parametrize("core", ["lockstep", "per-electron"])
def test_default_table_and_explicit_exact_replay(core):
    kwargs = dict(
        E0_keV=60.0,
        Ne=8,
        thickness_ang=4000.0,
        composition=[("C", 0.1)],
        E_cut_keV=10.0,
        seed=42,
        transport_core=core,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1000.0,
        bremslib_tables={"C": _table()},
    )
    default = simulate_trajectories(**kwargs)
    table = simulate_trajectories(**kwargs, radiative_moments="table")
    exact = simulate_trajectories(**kwargs, radiative_moments="exact")
    replay = simulate_trajectories(**kwargs, radiative_moments="exact")
    for name in ["E_end_keV", "L_ang", "hard_radiative_k_eV", "hard_radiative_direction"]:
        np.testing.assert_array_equal(default[name], table[name])
        np.testing.assert_array_equal(exact[name], replay[name])
        assert np.isfinite(default[name]).all()
    assert default["radiative"]["moment_model"] == RADIATIVE_MOMENT_MODEL
    assert len(default["radiative"]["moment_table_sha256"]) == 64
    assert exact["radiative"]["moment_model"] == "exact"
    assert "moment_table_sha256" not in exact["radiative"]
    assert len(default["radiative"]["plasma_energy_eV_by_layer"]) == 1


def test_legacy_case_replays_exact_while_new_cases_select_table(monkeypatch):
    monkeypatch.setattr(case_tables, "_case_bremslib_tables", lambda case: {})
    legacy = dict(radiative_model="bremslib-soft-hard", radiative_cutoff_eV=1000.0)
    assert case_tables._case_radiative_kwargs(legacy)["radiative_moments"] == "exact"
    current = legacy | {"radiative_moment_model": RADIATIVE_MOMENT_MODEL}
    assert case_tables._case_radiative_kwargs(current)["radiative_moments"] == "table"


def test_unknown_saved_moment_policy_is_refused(monkeypatch):
    monkeypatch.setattr(case_tables, "_case_bremslib_tables", lambda case: {})
    with pytest.raises(ValueError, match="unsupported radiative_moment_model"):
        case_tables._case_radiative_kwargs(
            {
                "radiative_model": "bremslib-soft-hard",
                "radiative_cutoff_eV": 1000.0,
                "radiative_moment_model": "unknown-policy",
            }
        )
