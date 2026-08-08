"""Profile-aware quick beam-energy resolution.

Covers ``scan._select_quick_energies``: ``--quick`` must pick a bounded,
deterministic beam-energy subset whose line grids the effective profile/material
configures, so ``build_cases`` never requests an absent per-energy line grid
(the fixed ``[30, 50]`` failure). ``sweep._line_grid_for_energy`` stays strict.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cxr_mc.config import material_sweep
from cxr_mc.scan import _select_quick_energies
from cxr_mc.sweep import beam_replace, build_cases


def _grid(energy):
    return np.array([float(energy) * 1e3 - 100.0, float(energy) * 1e3], dtype=float)


def test_both_nominal_valid_preserves_int_literals():
    # Standard-like: 30 and 50 both configured -> historical [30, 50] ints,
    # keeping the standard profile's checkpoint identity bit-for-bit.
    by_energy = {e: _grid(e) for e in (30.0, 40.0, 50.0, 60.0, 100.0)}
    chosen = _select_quick_energies(by_energy, None)
    assert chosen == [30, 50]
    assert all(isinstance(e, int) for e in chosen)


def test_missing_50_fills_nearest_target():
    by_energy = {e: _grid(e) for e in (30.0, 60.0, 100.0, 200.0, 300.0)}
    # 30 valid, 50 absent -> nearest valid to 50 is 60.
    assert _select_quick_energies(by_energy, None) == [30, 60.0]


def test_missing_both_nominal_fills_nearest_each():
    by_energy = {e: _grid(e) for e in (35.0, 55.0, 300.0)}
    # nearest to 30 -> 35, nearest to 50 -> 55.
    assert _select_quick_energies(by_energy, None) == [35.0, 55.0]


def test_single_valid_energy_runs_it():
    by_energy = {300.0: _grid(300.0)}
    assert _select_quick_energies(by_energy, None) == [300.0]


def test_absent_mapping_keeps_nominal():
    assert _select_quick_energies(None, None) == [30, 50]


def test_fixed_line_grid_keeps_nominal():
    # A fixed E_grid_line resolves for every beam energy regardless of the map.
    by_energy = {300.0: _grid(300.0)}
    assert _select_quick_energies(by_energy, np.array([1.0, 2.0])) == [30, 50]


def test_empty_mapping_falls_through_to_nominal():
    # Falls through so the strict missing-grid error still fires downstream.
    assert _select_quick_energies({}, None) == [30, 50]


def test_determinism_and_bounded_count():
    by_energy = {e: _grid(e) for e in (30.0, 40.0, 60.0, 100.0)}
    first = _select_quick_energies(by_energy, None)
    assert first == _select_quick_energies(by_energy, None)
    assert len(first) <= 2


def test_standard_hopg_quick_resolves_30_50():
    sweep = material_sweep("hopg")
    chosen = _select_quick_energies(sweep.E_grid_line_by_energy, sweep.E_grid_line)
    assert chosen == [30, 50]
    quick = replace(sweep, beam=beam_replace(sweep.beam, energy_keV=chosen))
    cases = build_cases(quick)
    assert sorted({float(c["E0_keV"]) for c in cases}) == [30.0, 50.0]


def test_profile_lacking_50_reaches_valid_bounded_cases():
    # Simulate a profile whose per-energy line grids omit 50 keV: dropping 50
    # from a real standard sweep reproduces the pre-fix ValueError condition.
    sweep = material_sweep("hopg")
    by_energy = {float(e): g for e, g in sweep.E_grid_line_by_energy.items() if float(e) != 50.0}
    lacking = replace(sweep, E_grid_line_by_energy=by_energy)
    chosen = _select_quick_energies(lacking.E_grid_line_by_energy, lacking.E_grid_line)
    assert 50.0 not in chosen
    assert chosen == [30, 40.0]  # tie 40/60 vs 50 breaks low
    quick = replace(lacking, beam=beam_replace(lacking.beam, energy_keV=chosen))
    cases = build_cases(quick)  # must not raise the missing-grid error
    assert sorted({float(c["E0_keV"]) for c in cases}) == [30.0, 40.0]


def test_line_grid_for_energy_still_strict_on_missing():
    # The downstream invariant remains: an absent per-energy grid fails closed.
    sweep = material_sweep("hopg")
    by_energy = {float(e): g for e, g in sweep.E_grid_line_by_energy.items() if float(e) != 50.0}
    lacking = replace(sweep, E_grid_line_by_energy=by_energy)
    forced = replace(lacking, beam=beam_replace(lacking.beam, energy_keV=[50]))
    with pytest.raises(ValueError, match="no E_grid_line configured for beam energy 50"):
        build_cases(forced)
