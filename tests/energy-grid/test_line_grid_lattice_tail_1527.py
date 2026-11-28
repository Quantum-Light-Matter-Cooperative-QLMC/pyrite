"""Fixed-size stop-lattice evidence for full-fiber measured grids (#1527).

Validation: line-grid-resonance-bandwidth
"""

import numpy as np
import pytest

from pyrite.montecarlo.runner import cxr_grid, streamed_line_grid
from pyrite.montecarlo.spectrum import _lattice_tail as lt
from pyrite.montecarlo.spectrum.cxr_seeds import (
    ResonancePopulation,
    _lattice_stop,
    resonance_population_stop_eV,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("limit", [0.001, 0.05, 0.5])
def test_accumulated_lattice_matches_retained_stop(dtype, limit, monkeypatch):
    monkeypatch.setattr(lt, "REAL", dtype)
    rng = np.random.default_rng(1527)
    ceiling = 5432.5  # includes a non-lattice terminal candidate
    stops = lt.stop_lattice(ceiling)
    summary = None
    populations = []
    for n, roll in ((101, 1.0), (227, 1 / 3), (511, 1 / 3)):
        energy = rng.uniform(300, 2700, n).astype(dtype)
        width = rng.uniform(0.1, 15, n).astype(dtype)
        weight = rng.lognormal(0, 1, n).astype(dtype)
        weight[:3] = [0, -1, np.nan]
        populations.append(ResonancePopulation("lines", energy, weight * roll, width))
        part = lt.summarize_lines(energy, width, weight, stops, roll)
        summary = lt.merge_tail(summary, part)
    tail = lt.LatticeTail(stops, summary)
    expected, record = resonance_population_stop_eV(
        populations, ceiling_eV=ceiling, truncation_limit=limit
    )
    actual, resolved = _lattice_stop(tail, ceiling, limit, 100.0)
    assert actual == expected
    for key in ("n_lines", "max_resonance_eV", "capped"):
        assert resolved[key] == record[key]
    # The old host denominator sums float32 blocks in float32. The new
    # accumulator sums the same per-line products in float64 across chunks.
    assert resolved["proxy_truncated_fraction"] == pytest.approx(
        record["proxy_truncated_fraction"], rel=3e-7
    )
    assert tail.raw_lines == 839
    assert tail.sums.nbytes == stops.size * 8


def test_empty_lattice_has_no_resonance_stop():
    tail = lt.LatticeTail(lt.stop_lattice(1234.5), None)
    stop, record = _lattice_stop(tail, 1234.5, 0.01, 100.0)
    assert stop is None and record["n_lines"] == 0
    assert tail.population_bytes == 0


def test_streamed_fiber_summary_does_not_keep_line_populations(monkeypatch):
    case = {"cxr_fiber": {}, "Ne": 10, "Ne_brem": 10}
    payload = {
        "bandwidth": {"policy": "resonance-population", "start_eV": 100, "stop_eV": 950},
        "resolution": {"policy": "sinc-oversampled"},
    }
    direction = np.array([0.0, 0.0, 1.0])
    evidence = streamed_line_grid.BlockGridEvidence(case, payload, direction, 10)
    monkeypatch.setattr(
        streamed_line_grid, "_block_sinc_samples", lambda *args: (np.ones(1), np.ones(1))
    )
    part = lt.summarize_lines(
        np.array([250.0, 300.0]), np.ones(2), np.ones(2), lt.stop_lattice(950)
    )
    monkeypatch.setattr(cxr_grid, "_direction_tail", lambda *args: part)

    def refuse_collection(*args, **kwargs):
        pytest.fail("uniform measured fiber grid collected production lines")

    monkeypatch.setattr(cxr_grid, "_direction_populations", refuse_collection)
    for _ in range(10):
        evidence.add({"L_ang": np.ones(1), "Ne": 1})
    assert not evidence.keeps_lines
    assert not evidence._populations and evidence._seen[2] == 0
    tail = evidence.tail(direction, 100, 950, None)
    assert tail is not None and tail.n_lines == 20
    assert tail.sums.nbytes == lt.stop_lattice(950).size * 8
    with pytest.raises(streamed_line_grid._NeedsLinePopulation) as exc:
        evidence.populations(direction, 100, 950, None)
    assert exc.value.required_bytes == tail.population_bytes
    evidence.close()


def test_device_stop_tiles_and_remainder_match_host(monkeypatch):
    # Exercise the tiled broadcast/reduction geometry without a GPU. The
    # production GPU uses the existing fused DeviceTail reduction kernel.
    def reduction(energy, weight, width, stop, *, axis):
        mass = weight * width
        distance = (stop - energy).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            bound = np.where(distance > 0, width / (np.pi**2 * distance), 1.0)
        return np.sum(mass * np.minimum(1.0, bound), axis=axis)

    monkeypatch.setattr(lt, "_kernels", lambda _: (None, reduction))
    monkeypatch.setattr(lt, "_ROW_LINES", 7)
    monkeypatch.setattr(lt, "_STOP_TILE", 3)
    energy = np.linspace(10, 500, 32)
    width = np.linspace(1, 5, 32)
    weight = np.linspace(0.2, 1, 32)
    stops = lt.stop_lattice(999.5)
    actual = lt._device_stop_sums(np, energy, weight, width, stops)
    expected = np.array([reduction(energy, weight, width, s, axis=0) for s in stops])
    np.testing.assert_allclose(actual, expected, rtol=1e-14)
