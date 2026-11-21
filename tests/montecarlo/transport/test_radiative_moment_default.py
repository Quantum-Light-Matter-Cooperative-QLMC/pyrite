"""Default lookup, exact replay and persisted approximation identity (#788)."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite._radiative_moment_policy import RADIATIVE_MOMENT_MODEL
from pyrite.montecarlo.runner import case_tables
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.transport import _simulate_trajectories as simulate_trajectories
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
    from pyrite.montecarlo.transport.radiative_moments import MomentPolicy

    policy = MomentPolicy(rtol=2.5e-5, atol=(1.25e-11, 1.25e-15, 1.25e-11))
    diagnostic = simulate_trajectories(**kwargs, _radiative_moment_policy=policy)
    assert diagnostic["radiative"]["moment_policy"]["rtol"] == policy.rtol
    assert diagnostic["radiative"]["moment_model"] == "diagnostic-radiative-moments-v1"
    assert diagnostic["radiative"]["moment_energy_eV_by_layer"][0][0] == 10000.0
    assert np.isfinite(diagnostic["E_end_keV"]).all()
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


def test_uncoupled_case_explicitly_disables_automatic_radiative_fallback():
    assert case_tables._case_radiative_kwargs({}) == {"radiative_model": "uncoupled"}


def test_identical_passes_reuse_moment_buffer_and_inputs_change_the_key(monkeypatch):
    from pyrite.montecarlo.transport import radiative_moment_transport as transport
    from pyrite.montecarlo.transport.hard_radiative import radiative_core_args

    monkeypatch.setattr(transport, "_cache", type(transport._cache)())
    tables = {"C": _table()}
    args = ([[("C", 0.1)]], tables, 1000.0, np.array([10.0]), np.array([60.0]), 42, 1)
    first = radiative_core_args(*args, moment_mode="table")[-1]
    built = first.copy()

    def unexpected(*args, **kwargs):
        raise AssertionError("identical inputs must not rebuild moments")

    monkeypatch.setattr(transport, "prepare_radiative_moments", unexpected)
    # A different seed/block start leaves the numerical tables unchanged.
    again = radiative_core_args(*args[:5], 7, 4, start=4, moment_mode="table")[-1]
    assert again is first
    np.testing.assert_array_equal(again, built)
    assert not again.flags.writeable
    # Density, cutoff, range and SDCS bytes each select a distinct table.
    edited = replace(tables["C"], scaled_sdcs_mb=tables["C"].scaled_sdcs_mb * 1.01)
    variants = [
        ([[("C", 0.2)]],) + args[1:],
        args[:2] + (2000.0,) + args[3:],
        args[:4] + (np.array([50.0]),) + args[5:],
        (args[0], {"C": edited}) + args[2:],
    ]
    for variant in variants:
        with pytest.raises(AssertionError, match="must not rebuild"):
            radiative_core_args(*variant, moment_mode="table")


def test_diagnostic_policy_changes_preparation_cache_and_retained_identity(monkeypatch):
    from dataclasses import asdict

    from pyrite.montecarlo.transport import radiative_moment_transport as transport
    from pyrite.montecarlo.transport.hard_radiative import radiative_core_args
    from pyrite.montecarlo.transport.radiative_moments import MomentPolicy

    monkeypatch.setattr(transport, "_cache", type(transport._cache)())
    args = ([[("C", 0.1)]], {"C": _table()}, 1000.0, np.array([10.0]), np.array([60.0]), 42, 1)
    nominal = radiative_core_args(*args, moment_mode="table")
    policy = MomentPolicy(rtol=2.5e-5, atol=(1.25e-11, 1.25e-15, 1.25e-11))
    candidate = radiative_core_args(*args, moment_mode="table", moment_policy=policy)
    result = {"radiative": {}}
    transport.add_moment_identity(result, candidate, policy)
    audit = result["radiative"]
    assert audit["moment_model"] != RADIATIVE_MOMENT_MODEL
    assert audit["moment_policy"] == asdict(policy)
    grid = np.asarray(audit["moment_energy_eV_by_layer"][0])
    assert grid[0] == 10000.0 and grid[-1] == 60000.0
    assert np.all(np.diff(grid) > 0) and grid.size <= policy.max_nodes
    assert len(audit["moment_table_sha256"]) == 64
    assert candidate[-1] is not nominal[-1]
    assert not candidate[-1].flags.writeable

    def unexpected(*args, **kwargs):
        raise AssertionError("new policy must rebuild")

    monkeypatch.setattr(transport, "prepare_radiative_moments", unexpected)
    # Equal policy values reuse; any budget/validation change must rebuild.
    assert (
        radiative_core_args(
            *args, moment_mode="table", moment_policy=MomentPolicy(**asdict(policy))
        )[-1]
        is candidate[-1]
    )
    for changed in (
        replace(policy, rtol=1e-5),
        replace(policy, atol=(1e-12, 1e-16, 1e-12)),
        replace(policy, max_nodes=16384),
        replace(policy, held_out_count=4095),
    ):
        with pytest.raises(AssertionError, match="must rebuild"):
            radiative_core_args(*args, moment_mode="table", moment_policy=changed)


def test_diagnostic_policy_cap_failure_never_falls_back_to_cached_nominal(monkeypatch):
    from pyrite.montecarlo.transport import radiative_moment_transport as transport
    from pyrite.montecarlo.transport.hard_radiative import radiative_core_args
    from pyrite.montecarlo.transport.radiative_moments import MomentPolicy

    monkeypatch.setattr(transport, "_cache", type(transport._cache)())
    args = ([[("C", 0.1)]], {"C": _table()}, 1000.0, np.array([10.0]), np.array([60.0]), 42, 1)
    nominal = radiative_core_args(*args, moment_mode="table")
    with pytest.raises(ValueError, match="exceeded max_nodes"):
        radiative_core_args(*args, moment_mode="table", moment_policy=MomentPolicy(max_nodes=2))
    assert radiative_core_args(*args, moment_mode="table")[-1] is nominal[-1]


@pytest.mark.parametrize("mode,coupled", [("exact", True), ("table", False)])
def test_diagnostic_policy_cannot_be_silently_ignored(mode, coupled):
    from pyrite.montecarlo.transport.radiative_moment_transport import validate_moment_policy
    from pyrite.montecarlo.transport.radiative_moments import MomentPolicy

    with pytest.raises(ValueError, match="coupled table"):
        validate_moment_policy(MomentPolicy(), mode, coupled)
    with pytest.raises(TypeError, match="MomentPolicy"):
        validate_moment_policy({}, "table", True)


def test_private_transport_entry_forwards_diagnostic_policy(monkeypatch):
    from pyrite.montecarlo.transport import api
    from pyrite.montecarlo.transport.radiative_moments import MomentPolicy

    policy = MomentPolicy(rtol=2.5e-5)
    seen = []
    monkeypatch.setattr(api, "run_transport_pass", lambda inputs: seen.append(inputs))
    api._simulate_trajectories(60.0, 1, 4000.0, _radiative_moment_policy=policy)
    assert seen[0]._radiative_moment_policy is policy
    assert seen[0].as_kwargs()["_radiative_moment_policy"] is policy
