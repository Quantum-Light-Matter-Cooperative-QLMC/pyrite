"""Stacked #1527 regression tests: block transactions and deferred collection.

Validation: line-grid-resonance-bandwidth
"""

import numpy as np
import pytest

from pyrite.montecarlo.runner import cxr_grid, streamed_line_grid
from pyrite.montecarlo.spectrum import _lattice_tail as lt
from pyrite.montecarlo.spectrum.cxr_seeds import ResonancePopulation


def _evidence(monkeypatch, *, fiber=True, measured=True, keep_lines=False):
    case = {"Ne": 10, "Ne_brem": 10, "cxr_fiber": {} if fiber else None}
    payload = {
        "bandwidth": {
            "policy": "resonance-population" if measured else "kinematic-ceiling",
            "start_eV": 100,
            "stop_eV": 950,
        },
        "resolution": {"policy": "sinc-oversampled"},
    }
    direction = np.array([0.0, 0.0, 1.0])
    monkeypatch.setattr(
        streamed_line_grid, "_block_sinc_samples", lambda *args: (np.ones(1), np.ones(1))
    )
    return streamed_line_grid.BlockGridEvidence(case, payload, direction, 10, keep_lines=keep_lines)


def _block():
    return {"L_ang": np.ones(1), "Ne": 1}


@pytest.mark.parametrize("fiber, keep_lines", [(False, False), (True, True)])
def test_line_collection_is_reachable_and_atomic(monkeypatch, fiber, keep_lines):
    evidence = _evidence(monkeypatch, fiber=fiber, keep_lines=keep_lines)
    population = ResonancePopulation("lines", np.ones(1), np.ones(1), np.ones(1))

    def broken(_):
        yield [population], 1
        raise MemoryError("injected after first slice")

    monkeypatch.setattr(evidence, "_line_slices", broken)
    with pytest.raises(MemoryError, match="injected"):
        evidence.add(_block())
    assert evidence._seen == [0, 0, 0, 0]
    assert not evidence._populations and not evidence._width and not evidence._weight
    monkeypatch.setattr(evidence, "_line_slices", lambda _: iter([([population], 1)]))
    evidence.add(_block())
    assert evidence._seen == [1, 16, 24, 1]
    assert len(evidence.populations(evidence.n_hat, *evidence.band, None)) == 1
    evidence.close()


def test_tail_failure_rolls_back_samples_before_replay(monkeypatch):
    evidence = _evidence(monkeypatch)
    part = lt.summarize_lines(
        np.array([250.0, 300.0]), np.ones(2), np.ones(2), lt.stop_lattice(950)
    )
    monkeypatch.setattr(cxr_grid, "_direction_tail", lambda *args: part)
    evidence.add(_block())
    before = evidence._tail_summary

    def broken(*args):
        raise MemoryError("injected tail failure")

    monkeypatch.setattr(cxr_grid, "_direction_tail", broken)
    with pytest.raises(MemoryError, match="tail failure"):
        evidence.add(_block())
    assert evidence._seen == [1, 16, 0, 0]
    assert evidence._tail_summary is before
    monkeypatch.setattr(cxr_grid, "_direction_tail", lambda *args: part)
    evidence.add(_block())
    assert evidence._seen == [2, 32, 0, 0]
    tail = evidence.tail(evidence.n_hat, *evidence.band, None)
    assert tail is not None and tail.n_lines == 4
    assert evidence.line_evidence_record()["line_population_bytes"] == 0
    evidence.close()


def test_kinematic_fiber_grid_defers_line_collection(monkeypatch):
    evidence = _evidence(monkeypatch, measured=False)

    def refuse(*args):
        pytest.fail("uniform kinematic grid must not generate production lines")

    monkeypatch.setattr(evidence, "_line_slices", refuse)
    monkeypatch.setattr(cxr_grid, "_direction_tail", refuse)
    evidence.add(_block())
    assert evidence.line_evidence_mode == "spacing-only"
    assert evidence._seen == [1, 16, 0, 0]
    assert evidence.tail(evidence.n_hat, *evidence.band, None) is None
    with pytest.raises(streamed_line_grid._NeedsLinePopulation) as exc:
        evidence.populations(evidence.n_hat, *evidence.band, None)
    assert exc.value.required_bytes is None
    evidence.close()


@pytest.mark.hardware
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_actual_device_tail_tiles_match_host(dtype, monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        devices = cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA/ROCm device")
    if not devices:
        pytest.skip("no CUDA/ROCm device")
    monkeypatch.setattr(lt, "REAL", dtype)
    monkeypatch.setattr(
        lt, "_to_cpu", lambda a: cp.asnumpy(a) if isinstance(a, cp.ndarray) else np.asarray(a)
    )
    rng = np.random.default_rng(1527)
    n = 8195  # two full reduction rows and a remainder
    energy = rng.uniform(100, 2700, n).astype(dtype)
    width = rng.uniform(0.1, 15, n).astype(dtype)
    weight = rng.lognormal(0, 1, n).astype(dtype)
    weight[:3] = [0, -1, np.nan]
    stops = lt.stop_lattice(5432.5)  # several tiles and a non-lattice ceiling
    host = lt.LatticeTail(stops, lt.summarize_lines(energy, width, weight, stops, 1 / 3))
    device_part = lt.summarize_lines(
        cp.asarray(energy), cp.asarray(width), cp.asarray(weight), stops, 1 / 3
    )
    device = lt.LatticeTail(stops, device_part)
    assert device.n_lines == host.n_lines
    assert device.raw_lines == host.raw_lines
    assert device.total == pytest.approx(host.total, rel=1e-12)
    np.testing.assert_allclose(device.sums, host.sums, rtol=1e-12, atol=0)
