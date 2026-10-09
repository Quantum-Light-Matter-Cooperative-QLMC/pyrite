"""Opt-in CPU hard-radiative flights and photon row payloads.

Both exact CPU cores run the mode: lockstep, and the per-electron reference
of the CUDA kernel (whose device half is ``test_hard_radiative_cuda.py``).

Validation: bremslib-radiative-partition, bremslib-radiative-event-spectrum,
bremslib-coupled-expected-spectrum
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.montecarlo import shell_configuration
from pyrite.montecarlo.spectrum.brem import mc_brem_spectrum
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.spectrum.brem_events import (
    mc_coupled_brem_spectrum,
    mc_hard_brem_event_spectrum,
    mc_soft_brem_spectrum,
)
from pyrite.montecarlo.transport import (
    TransportLUTConfig,
    check_segment_event_contract,
    simulate_trajectories,
)
from pyrite.montecarlo.transport.events import EVENT_CUTOFF, EVENT_HARD_RADIATIVE
from pyrite.montecarlo.transport.hard_radiative import build_radiative_partition
from pyrite.montecarlo.transport.stopping import spliced_stopping_keV_per_ang
from tests.helpers.bremslib import synthetic_bremslib_arrays


def _table(scale):
    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6)
    return replace(
        table,
        scaled_sdcs_mb=table.scaled_sdcs_mb * scale,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * scale,
    )


CPU_CORES = ["lockstep", "per-electron"]


def _run(table, core="lockstep", **kwargs):
    return simulate_trajectories(
        E0_keV=60.0,
        Ne=80,
        thickness_ang=4_000.0,
        composition=[("C", 0.1)],
        E_cut_keV=10.0,
        seed=42,
        energy_model="midpoint",
        transport_core=core,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"C": table},
        **kwargs,
    )


@pytest.mark.slow
@pytest.mark.parametrize("core", CPU_CORES)
@pytest.mark.parametrize("straggling", [False, True])
def test_hard_radiative_events_debit_energy_and_close_cpu_flights(core, straggling):
    table = _table(1e5)
    result = _run(table, core, straggling=straggling)
    check_segment_event_contract(result)
    photons = result["hard_radiative_k_eV"]
    events = result["event_kind"] == EVENT_HARD_RADIATIVE
    terminal = (result["event_kind"] == EVENT_CUTOFF) & (photons > 0)
    assert np.count_nonzero(events | terminal) > 0
    assert np.all(photons[events | terminal] >= 1_000.0)
    assert np.all(result["hard_radiative_Z"][events | terminal] == 6)
    assert np.all(photons[~(events | terminal)] == 0.0)
    assert result["radiative"]["model"] == "bremslib-soft-hard"
    direction = result["hard_radiative_direction"]
    target = result["hard_radiative_target_momentum_eV_c"]
    assert np.all(direction[~(events | terminal)] == 0.0)
    assert np.all(target[~(events | terminal)] == 0.0)
    np.testing.assert_allclose(np.linalg.norm(direction[events | terminal], axis=1), 1.0)
    incident_eV = result["E_end_keV"][events | terminal] * 1e3
    photon_eV = photons[events | terminal]
    electron = result["v_hat"][events | terminal]
    rest_eV = 510_998.95
    p_in = np.sqrt(incident_eV * (incident_eV + 2.0 * rest_eV))
    remaining = incident_eV - photon_eV
    p_out = np.sqrt(remaining * (remaining + 2.0 * rest_eV))
    np.testing.assert_allclose(
        p_in[:, None] * electron,
        photon_eV[:, None] * direction[events | terminal]
        + p_out[:, None] * electron
        + target[events | terminal],
        rtol=1e-12,
        atol=1e-9,
    )

    grid = np.arange(500.0, 60_500.0, 1_000.0)
    scoring = dict(
        cutoff_eV=1_000.0,
        bremslib_tables={"C": table},
        composition=[("C", 0.1)],
    )
    soft = mc_soft_brem_spectrum(result, grid, **scoring)
    hard = mc_hard_brem_event_spectrum(result, grid, **scoring)
    assert soft[0] > 0.0 and np.all(soft[1:] == 0.0)
    assert hard[0] == 0.0 and hard[1:].sum() > 0.0
    with pytest.raises(ValueError, match="coupled radiative tracks"):
        mc_brem_spectrum(
            result,
            grid,
            composition=[("C", 0.1)],
            cross_section_model="bremslib",
            bremslib_tables={"C": table},
        )
    with pytest.raises(ValueError, match="spectrum cutoff must match"):
        mc_soft_brem_spectrum(result, grid, **(scoring | {"cutoff_eV": 2_000.0}))
    other_table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=14)
    with pytest.raises(ValueError, match="tables must match"):
        mc_soft_brem_spectrum(
            result,
            grid,
            **(scoring | {"bremslib_tables": {"C": other_table}}),
        )


@pytest.mark.parametrize("core", CPU_CORES)
def test_expected_value_continuum_matches_the_sampled_photons_on_the_same_tracks(core):
    """Track length over the full DDCS is the expectation of the analog sum.

    On one set of coupled tracks, the soft bins are the soft scorer's exactly
    and the hard photons' yield agrees with the expected-value estimate within
    four standard errors of the ~370 sampled events (seeded, so deterministic).
    """
    table = _table(1e6)
    result = _run(table, core)
    grid = np.arange(500.0, 60_500.0, 1_000.0)
    scoring = dict(cutoff_eV=1_000.0, bremslib_tables={"C": table}, composition=[("C", 0.1)])
    expected = mc_coupled_brem_spectrum(result, grid, **scoring)
    soft = mc_soft_brem_spectrum(result, grid, **scoring)
    hard = mc_hard_brem_event_spectrum(result, grid, **scoring)

    assert expected[0] == soft[0] > 0.0
    events = np.count_nonzero(result["hard_radiative_k_eV"] > 0.0)
    assert events > 300
    relative = hard[1:].sum() / expected[1:].sum() - 1.0
    assert abs(relative) < 4.0 / np.sqrt(events)
    with pytest.raises(ValueError, match="spectrum cutoff must match"):
        mc_coupled_brem_spectrum(result, grid, **(scoring | {"cutoff_eV": 2_000.0}))


@pytest.mark.parametrize("core", CPU_CORES)
@pytest.mark.parametrize("straggling", [False, True])
def test_radiative_mode_replays_and_zero_cross_section_preserves_legacy_tracks(core, straggling):
    table = _table(1e5)
    first = _run(table, core, straggling=straggling)
    replay = _run(table, core, straggling=straggling)
    for field in (
        "event_kind",
        "L_ang",
        "E_start_keV",
        "E_end_keV",
        "v_hat",
        "hard_radiative_k_eV",
        "hard_radiative_Z",
        "hard_radiative_direction",
        "hard_radiative_target_momentum_eV_c",
    ):
        np.testing.assert_array_equal(first[field], replay[field])

    zero = _table(0.0)
    coupled = _run(zero, core, straggling=straggling)
    legacy = simulate_trajectories(
        E0_keV=60.0,
        Ne=80,
        thickness_ang=4_000.0,
        composition=[("C", 0.1)],
        E_cut_keV=10.0,
        seed=42,
        energy_model="midpoint",
        transport_core=core,
        transport_lut_config=TransportLUTConfig(enabled=False),
        straggling=straggling,
    )
    for field in ("event_kind", "L_ang", "E_start_keV", "E_end_keV", "v_hat"):
        np.testing.assert_array_equal(coupled[field], legacy[field])


@pytest.mark.slow
@pytest.mark.parametrize("core", CPU_CORES)
@pytest.mark.parametrize("straggling", [False, True])
def test_radiative_and_shell_collision_modes_share_the_cpu_event_contract(core, straggling):
    if not shell_configuration._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")
    from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=14)
    table = replace(
        table,
        scaled_sdcs_mb=table.scaled_sdcs_mb * 1e5,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * 1e5,
    )
    result = simulate_trajectories(
        E0_keV=60.0,
        Ne=20,
        thickness_ang=20_000.0,
        composition=CATALOG.crystal("silicon").composition,
        E_cut_keV=10.0,
        seed=17,
        energy_model="midpoint",
        transport_core=core,
        straggling=straggling,
        stopping_tables=[resolve_catalog_table("silicon").arrays()],
        inelastic_model="shell-soft-hard",
        inelastic_cutoff_eV=50.0,
        inelastic_materials=["silicon"],
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"Si": table},
    )
    check_segment_event_contract(result)
    assert result["inelastic"]["model"] == "shell-soft-hard"
    assert result["radiative"]["model"] == "bremslib-soft-hard"


@pytest.mark.slow
@pytest.mark.parametrize("core", CPU_CORES)
@pytest.mark.parametrize("cutoff_eV", [100.0, 10_000.0])
def test_soft_and_hard_radiative_loss_reproduce_the_full_moment(cutoff_eV, core):
    # Radiative loss dominates the synthetic table. Continuous row loss must be
    # collision plus the soft first moment only; hard events must supply the
    # rest at the partition's rate, so the sum does not depend on kc.
    table = _table(3e4)
    composition = [("C", 0.1)]
    result = simulate_trajectories(
        E0_keV=60.0,
        Ne=2_000,
        thickness_ang=3_000.0,
        composition=composition,
        E_cut_keV=10.0,
        seed=7,
        energy_model="midpoint",
        transport_core=core,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=cutoff_eV,
        bremslib_tables={"C": table},
    )
    E_start, E_end, L = result["E_start_keV"], result["E_end_keV"], result["L_ang"]
    k_keV = result["hard_radiative_k_eV"] * 1e-3
    per_ang = 0.1e24 * 1e-8  # atoms/cm^3 x cm/Angstrom

    midpoint = 0.5 * (E_start + E_end)
    collision = -np.asarray(spliced_stopping_keV_per_ang(composition, midpoint))
    mid_parts = [build_radiative_partition(table, e * 1e3, cutoff_eV) for e in midpoint]
    soft = np.array([p.soft_stopping_cs_eV_cm2 for p in mid_parts]) * per_ang * 1e-3
    total = np.array([p.total_stopping_cs_eV_cm2 for p in mid_parts]) * per_ang * 1e-3
    np.testing.assert_allclose((E_start - E_end).sum(), (L * (collision + soft)).sum(), rtol=1e-4)

    # The flight hazard is frozen at row-start energy.
    start_parts = [build_radiative_partition(table, e * 1e3, cutoff_eV) for e in E_start]
    rate = np.array([p.hard_rate_cs_cm2 for p in start_parts]) * per_ang
    expected_count = (L * rate).sum()
    count = np.count_nonzero(k_keV)
    assert abs(count - expected_count) < 4.0 * np.sqrt(expected_count)

    radiative = (L * total).sum()
    observed = (L * soft).sum() + k_keV.sum()
    assert (L * total).sum() > 4.0 * (L * collision).sum()
    assert abs(observed - radiative) < 4.0 * np.sqrt((k_keV**2).sum())
