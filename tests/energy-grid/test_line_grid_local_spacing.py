"""Energy-dependent ``resonance-local`` line spacing (issue #192).

Pins the halo rule of ``local_spacing_seeds`` -- every measured line narrower
than the backbone is resolved inside a halo whose extent bounds the line's
out-of-window mass -- and the runner's measured local grid built from
collected production lines.
"""

import numpy as np
import pytest

from pyrite._line_grid_policy import (
    DEFAULT_LOCAL_HALO_LIMIT,
    LOCAL_RESOLUTION_POLICY,
    RESONANCE_BANDWIDTH_POLICY,
    RESONANCE_LINE_GRID_POLICY_SCHEMA,
    LineGridToleranceError,
    resolve_line_grid_policy,
)
from pyrite.montecarlo.runner.line_grid import _measured_line_grid
from pyrite.montecarlo.spectrum.line_seeds import (
    LOCAL_SPACING_SOURCE,
    ResonancePopulation,
    local_spacing_seeds,
)

_HALO = 1e-4


def _population(energy, width, weight=None):
    energy = np.atleast_1d(np.asarray(energy, dtype=float))
    width = np.broadcast_to(np.asarray(width, dtype=float), energy.shape).copy()
    weight = np.ones_like(energy) if weight is None else np.asarray(weight, dtype=float)
    return ResonancePopulation("(0 0 2)", energy, weight, width)


def _seeds(energies, widths, *, start=50.0, stop=20_000.0, floor=0.5, backbone=3.0):
    return local_spacing_seeds(
        [_population(energies, widths)],
        start_eV=start,
        stop_eV=stop,
        floor_spacing_eV=floor,
        max_spacing_eV=backbone,
        halo_limit=_HALO,
        bin_eV=100.0,
    )


def _halo_extent(width):
    return width / (np.pi**2 * _HALO)


def test_isolated_narrow_line_gets_one_halo_window():
    seeds, summary = _seeds([7000.0], [2.0])
    assert len(seeds) == 1 == summary["windows"]
    assert summary["narrow_lines"] == 1
    seed = seeds[0]
    assert seed.source == LOCAL_SPACING_SOURCE
    # 100 eV bins round the halo E_res +- w / (pi**2 halo_limit) outward.
    lo = 50.0 + 100.0 * np.floor((7000.0 - _halo_extent(2.0) - 50.0) / 100.0)
    hi = 50.0 + 100.0 * np.ceil((7000.0 + _halo_extent(2.0) - 50.0) / 100.0)
    assert seed.centre_eV == lo
    assert seed.below_eV == 0.0
    assert seed.above_eV == pytest.approx(hi - lo)
    # No coarser than the line's first-zero width.
    assert seed.spacing_eV == 2.0


def test_lines_as_wide_as_the_backbone_seed_nothing():
    seeds, summary = _seeds([7000.0, 9000.0], [3.0, 10.0])
    assert seeds == []
    assert summary["narrow_lines"] == 0
    assert summary["windows"] == 0


def test_lines_narrower_than_the_floor_hold_the_floor():
    seeds, _ = _seeds([7000.0], [0.1])
    assert [seed.spacing_eV for seed in seeds] == [0.5]


def test_halos_outside_the_band_seed_nothing():
    seeds, summary = _seeds([30.0, 100_000.0], [0.01, 1.0])
    assert seeds == []
    assert summary["windows"] == 0


def test_empty_population_seeds_nothing():
    seeds, summary = local_spacing_seeds(
        [_population([], 1.0)],
        start_eV=50.0,
        stop_eV=1000.0,
        floor_spacing_eV=0.5,
        max_spacing_eV=3.0,
        halo_limit=_HALO,
    )
    assert seeds == []
    assert summary["windows"] == 0


def test_overlapping_halos_take_the_finest_level():
    seeds, _ = _seeds([7000.0, 7050.0], [2.0, 0.6])
    assert [seed.spacing_eV for seed in seeds] == [2.0, 0.5, 2.0]
    fine = seeds[1]
    assert fine.centre_eV <= 7050.0 - _halo_extent(0.6)
    assert fine.centre_eV + fine.above_eV >= 7050.0 + _halo_extent(0.6)


def test_invalid_spacing_and_halo_arguments_raise():
    with pytest.raises(ValueError, match="floor_spacing_eV"):
        _seeds([7000.0], [1.0], floor=0.0)
    with pytest.raises(ValueError, match="halo_limit"):
        local_spacing_seeds(
            [_population(7000.0, 1.0)],
            start_eV=50.0,
            stop_eV=1000.0,
            floor_spacing_eV=0.5,
            max_spacing_eV=3.0,
            halo_limit=1.0,
        )


def _local_policy(**per_call):
    return resolve_line_grid_policy(
        start_eV=50.0,
        stop_eV=100_000.0,
        per_call={
            "bandwidth": RESONANCE_BANDWIDTH_POLICY,
            "resolution": LOCAL_RESOLUTION_POLICY,
            **per_call,
        },
    ).payload()


def test_local_policy_payload_carries_its_halo_share():
    payload = _local_policy()
    assert payload["schema"] == RESONANCE_LINE_GRID_POLICY_SCHEMA
    assert payload["resolution"]["policy"] == LOCAL_RESOLUTION_POLICY
    assert payload["resolution"]["halo_limit"] == DEFAULT_LOCAL_HALO_LIMIT
    assert payload["sources"]["resolution"] == "per-call"


def test_local_resolution_needs_the_measured_bandwidth():
    with pytest.raises(ValueError, match="needs the resonance-population bandwidth"):
        resolve_line_grid_policy(
            start_eV=50.0, stop_eV=1000.0, per_call={"resolution": LOCAL_RESOLUTION_POLICY}
        )


def _collecting_lines(monkeypatch, energy, width, weight):
    from pyrite.montecarlo import runner

    collected = (
        np.asarray(energy, dtype=float),
        np.asarray(width, dtype=float),
        np.asarray(weight, dtype=float),
    )

    def collect(_segments, axis, _case, _direction, _layers, _groove, **kwargs):
        kwargs["truncation_audit"]["collect"].append(collected)
        return np.zeros(axis.size)

    monkeypatch.setattr(runner, "_lines_for_segments", collect)


def _measured_local_grid(monkeypatch, collected, **per_call):
    _collecting_lines(monkeypatch, *collected)
    case = {"name": "hbn test", "E0_keV": 5000.0, "composition": [("B", 0.5), ("N", 0.5)]}
    return _measured_line_grid(
        _local_policy(**per_call), case, {}, np.array([0.0, 0.0, 1.0]), 2, 0.25, None, None
    )


def test_measured_local_grid_refines_only_around_narrow_lines(monkeypatch):
    grid, record, bandwidth = _measured_local_grid(monkeypatch, ([5000.0], [1.0], [1.0]))
    stop = bandwidth["stop_eV"]
    # One straight-flight line: E_res + w / (pi**2 * limit/proxy_safety), rounded up.
    assert bandwidth["kinematic"]["stop_eV"] == 7100.0
    assert grid[0] == 50.0
    assert grid[-1] == stop >= 7100.0
    steps = np.diff(grid)
    assert steps.min() == pytest.approx(1.0)  # the line's first-zero width
    assert steps.max() == pytest.approx(3.0)  # the backbone
    fine_nodes = grid[:-1][steps < 3.0 - 1e-9]
    assert fine_nodes.size
    assert np.all(np.abs(fine_nodes - 5000.0) <= _halo_extent(1.0) + 100.0)
    assert record["resolution_policy"] == LOCAL_RESOLUTION_POLICY
    assert record["local_spacing"]["narrow_lines"] == 1
    assert record["local_spacing"]["windows"] >= 1


def test_measured_local_grid_refuses_beyond_the_point_budget(monkeypatch):
    with pytest.raises(LineGridToleranceError, match="above the 100-point budget"):
        _measured_local_grid(monkeypatch, ([5000.0], [1.0], [1.0]), max_points=100)


def test_block_streaming_matches_one_block(monkeypatch):
    from pyrite.montecarlo.spectrum import line_seeds

    rng = np.random.default_rng(0)
    energy = rng.uniform(1000.0, 20_000.0, 200)
    width = rng.uniform(0.2, 5.0, 200)
    weight = rng.uniform(0.0, 1.0, 200)
    weight[:5] = [0.0, np.nan, -1.0, np.inf, 0.0]
    whole = [_population(energy, width, weight)]
    split = [
        _population(energy[:77], width[:77], weight[:77]),
        _population(energy[77:], width[77:], weight[77:]),
    ]
    kwargs = dict(
        start_eV=50.0,
        stop_eV=40_000.0,
        floor_spacing_eV=0.5,
        max_spacing_eV=3.0,
        halo_limit=_HALO,
        bin_eV=100.0,
    )
    expected = local_spacing_seeds(whole, **kwargs)
    expected_stop = line_seeds.resonance_population_stop_eV(
        whole, ceiling_eV=1e6, truncation_limit=1e-4
    )
    monkeypatch.setattr(line_seeds, "_POPULATION_BLOCK_SIZE", 7)
    assert local_spacing_seeds(split, **kwargs) == expected
    stop, summary = line_seeds.resonance_population_stop_eV(
        split, ceiling_eV=1e6, truncation_limit=1e-4
    )
    assert stop == expected_stop[0]
    assert summary == pytest.approx(expected_stop[1])
