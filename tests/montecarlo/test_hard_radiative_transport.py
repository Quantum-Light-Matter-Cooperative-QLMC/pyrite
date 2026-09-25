"""Opt-in CPU hard-radiative flights and photon row payloads.

Validation: bremslib-radiative-partition, bremslib-radiative-event-spectrum
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials import CATALOG
from pyrite.montecarlo import shell_configuration
from pyrite.montecarlo.spectrum.brem import mc_brem_spectrum
from pyrite.montecarlo.spectrum.brem_bremslib import prepare_bremslib_table
from pyrite.montecarlo.spectrum.brem_events import (
    mc_hard_brem_event_spectrum,
    mc_soft_brem_spectrum,
)
from pyrite.montecarlo.transport import (
    TransportLUTConfig,
    check_segment_event_contract,
    simulate_trajectories,
)
from pyrite.montecarlo.transport.events import EVENT_CUTOFF, EVENT_HARD_RADIATIVE
from tests.helpers.bremslib import synthetic_bremslib_arrays


def _table(scale):
    table = prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6)
    return replace(
        table,
        scaled_sdcs_mb=table.scaled_sdcs_mb * scale,
        scaled_ddcs_mb_sr=table.scaled_ddcs_mb_sr * scale,
    )


def _run(table, **kwargs):
    return simulate_trajectories(
        E0_keV=60.0,
        Ne=80,
        thickness_ang=4_000.0,
        composition=[("C", 0.1)],
        E_cut_keV=10.0,
        seed=42,
        energy_model="midpoint",
        transport_core="lockstep",
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=1_000.0,
        bremslib_tables={"C": table},
        **kwargs,
    )


def test_hard_radiative_events_debit_energy_and_close_cpu_flights():
    table = _table(1e5)
    result = _run(table)
    check_segment_event_contract(result)
    photons = result["hard_radiative_k_eV"]
    events = result["event_kind"] == EVENT_HARD_RADIATIVE
    terminal = (result["event_kind"] == EVENT_CUTOFF) & (photons > 0)
    assert np.count_nonzero(events | terminal) > 0
    assert np.all(photons[events | terminal] >= 1_000.0)
    assert np.all(result["hard_radiative_Z"][events | terminal] == 6)
    assert np.all(photons[~(events | terminal)] == 0.0)
    assert result["radiative"]["model"] == "bremslib-soft-hard"

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


def test_radiative_mode_replays_and_zero_cross_section_preserves_legacy_tracks():
    table = _table(1e5)
    first = _run(table)
    replay = _run(table)
    for field in (
        "event_kind",
        "L_ang",
        "E_start_keV",
        "E_end_keV",
        "v_hat",
        "hard_radiative_k_eV",
        "hard_radiative_Z",
    ):
        np.testing.assert_array_equal(first[field], replay[field])

    zero = _table(0.0)
    coupled = _run(zero)
    legacy = simulate_trajectories(
        E0_keV=60.0,
        Ne=80,
        thickness_ang=4_000.0,
        composition=[("C", 0.1)],
        E_cut_keV=10.0,
        seed=42,
        energy_model="midpoint",
        transport_core="lockstep",
        transport_lut_config=TransportLUTConfig(enabled=False),
    )
    for field in ("event_kind", "L_ang", "E_start_keV", "E_end_keV", "v_hat"):
        np.testing.assert_array_equal(coupled[field], legacy[field])


def test_radiative_and_shell_collision_modes_share_the_cpu_event_contract():
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
        transport_core="lockstep",
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
