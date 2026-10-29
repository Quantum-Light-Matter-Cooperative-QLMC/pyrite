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


def test_production_endpoint_admission_matches_exact_without_relaxing_query_bounds():
    from pyrite.montecarlo.transport._jit_moment_lookup import transport_layer_moments
    from pyrite.montecarlo.transport._jit_radiative import radiative_layer_moments_scalar
    from pyrite.montecarlo.transport.hard_radiative import radiative_core_args
    from pyrite.montecarlo.transport.radiative_moments import prepare_radiative_moments

    table = _table()
    grid = table.incident_energy_keV.copy()
    grid[0] = np.nextafter(10.0, np.inf)
    table = replace(table, incident_energy_keV=grid)
    compositions, tables = [[("C", 0.1)]], {"C": table}
    args = (compositions, tables, 1000.0, np.array([10.0]), np.array([60.0]), 42, 1)
    exact = radiative_core_args(*args)
    candidate = radiative_core_args(*args, moment_mode="table")
    # Source-refinement roundoff remains admissible exactly where it was before.
    expected = radiative_layer_moments_scalar(
        *exact[2:-1], 0, 10000.0, 1000.0, np.empty(1), exact[-1]
    )
    actual = transport_layer_moments(
        *candidate[2:-1], 0, 10000.0, 1000.0, np.empty(1), candidate[-1]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
    # The standalone numerical API remains strict about source coverage.
    with pytest.raises(ValueError, match="coverage outside"):
        prepare_radiative_moments(compositions, tables, 10000.0, 60000.0, 1000.0)
    # Production interpolation must still refuse even adjacent outside queries.
    with pytest.raises(ValueError, match="outside prepared"):
        transport_layer_moments(
            *candidate[2:-1], 0, np.nextafter(10000.0, -np.inf), 1000.0, np.empty(1), candidate[-1]
        )
    # Nor may a physically unsupported source range be admitted by preparation.
    with pytest.raises(ValueError, match="outside the BremsLib table"):
        radiative_core_args(*(args[:3] + (np.array([9.0]),) + args[4:]), moment_mode="table")
