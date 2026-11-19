"""Device route of the measured line-axis sizing (#1275).

The route runs on NumPy arrays without a device, so its arithmetic is pinned
against the exact host route here; the ``hardware`` tests repeat the
comparison on CuPy arrays, where only the documented reductions differ.
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum import _device_sizing
from pyrite.montecarlo.spectrum.cxr_seeds import (
    ResonancePopulation,
    _HostTail,
    _lattice_stop,
    resonance_population_stop_eV,
)
from pyrite.montecarlo.spectrum.diagnostics import (
    sinc_feature_samples,
    sinc_spacing_from_samples,
)


def _segments(rng, n, *, flights=False, dtype=np.float64):
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    segments = {
        "E_keV": rng.uniform(20.0, 1000.0, n).astype(dtype),
        "L_ang": rng.lognormal(3.0, 1.5, n).astype(dtype),
        "v_hat": direction.astype(dtype),
        "elec_id": np.sort(rng.integers(0, max(1, n // 4), n)).astype(np.int32),
    }
    segments["L_ang"][:3] = (0.0, -1.0, np.nan)
    if flights:
        segments["flight_id"] = np.zeros(n, dtype=np.int32)
    else:
        segments["flight_id"] = np.arange(n, dtype=np.int32)
    return segments


def _population(rng, n, dtype):
    energy = rng.uniform(1.0e3, 4.0e4, n) * (1.0 + rng.exponential(0.3, n) * (rng.random(n) < 0.01))
    weight = rng.lognormal(0.0, 2.0, n)
    width = rng.lognormal(np.log(rng.uniform(1.0, 300.0)), 1.0, n)
    weight[:2] = (0.0, -1.0)
    width[2] = np.inf
    return ResonancePopulation(
        "lines", energy.astype(dtype), weight.astype(dtype), width.astype(dtype)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("electron_limit", [None, 300])
def test_device_samples_reproduce_the_host_samples(dtype, electron_limit):
    rng = np.random.default_rng(1)
    segments = _segments(rng, 20_000, dtype=dtype)
    host = sinc_feature_samples(segments, (0.3, -0.2, 0.93), electron_limit=electron_limit)
    device = _device_sizing.device_sinc_samples(
        segments, (0.3, -0.2, 0.93), electron_limit=electron_limit
    )
    assert device is not None
    _assert_samples_agree(device, host)


def _assert_samples_agree(device, host):
    """Weights bitwise; widths up to the dot product's rounding (module doc).

    ``width = 2 pi hbar c / ((1 - beta v.n) t_L)``: a dot product off by a few
    ulp moves ``1 - beta v.n`` by about ``eps beta |v.n|``, relative
    ``eps / (1 - beta v.n)`` near forward emission.
    """
    width, weight = device
    host_width, host_weight = host
    np.testing.assert_array_equal(weight, host_weight)
    assert width.shape == host_width.shape
    np.testing.assert_allclose(width, host_width, rtol=1e-13, atol=0.0)


def test_grouped_flights_keep_the_host_route():
    rng = np.random.default_rng(2)
    segments = _segments(rng, 400, flights=True)
    assert _device_sizing.device_sinc_samples(segments, (0.0, 0.0, 1.0)) is None


def test_device_spacing_matches_the_host_quantile():
    rng = np.random.default_rng(3)
    width = np.where(rng.random(200_000) < 0.3, 17.25, rng.lognormal(3.0, 1.0, 200_000))
    weight = rng.lognormal(0.0, 2.0, width.size)
    host = sinc_spacing_from_samples([width.copy(), weight.copy()], aliased_weight_limit=0.01)
    device = _device_sizing.device_sinc_spacing(width, weight, aliased_weight_limit=0.01)
    assert device[0] == host[0] and device[2] == host[2]
    assert device[1] == pytest.approx(host[1], rel=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("limit", [1.0e-3, 1.0e-2, 0.3])
def test_device_tail_reproduces_the_host_stop(dtype, limit):
    rng = np.random.default_rng(7)
    for _ in range(4):
        populations = [
            _population(rng, int(rng.integers(5, 3000)), dtype)
            for _ in range(int(rng.integers(1, 4)))
        ]
        for ceiling in (5.0e4, 1.2345e5):
            args = (ceiling, limit, 100.0)
            host = _lattice_stop(_HostTail(populations), *args)
            device = _lattice_stop(_device_sizing.DeviceTail(populations), *args)
            assert device[0] == host[0]
            for key in ("n_lines", "max_resonance_eV", "capped", "stop_eV"):
                assert device[1][key] == host[1][key]
            # float64 total mass against the host's float32 pairwise total.
            rtol = 1e-6 if dtype == np.float32 else 1e-12
            assert device[1]["proxy_truncated_fraction"] == pytest.approx(
                host[1]["proxy_truncated_fraction"], rel=rtol
            )


def test_host_populations_keep_the_host_route(monkeypatch):
    rng = np.random.default_rng(8)
    populations = [_population(rng, 500, np.float32)]

    def refuse(*_):
        raise AssertionError("host populations must not take the device route")

    monkeypatch.setattr(_device_sizing.DeviceTail, "__init__", refuse)
    resonance_population_stop_eV(populations, ceiling_eV=5.0e4, truncation_limit=1e-2)


def test_collector_moves_an_over_budget_population_to_the_host(monkeypatch):
    monkeypatch.setattr(_device_sizing, "is_device_array", lambda array: True)
    chunk = tuple(np.arange(4, dtype=float) + k for k in range(3))
    size = 3 * 4 * np.dtype(_device_sizing.REAL).itemsize
    audit = {"collect": [], "device_bytes": 2 * size}
    for _ in range(2):
        _device_sizing.collect_lines(audit, chunk)
    assert audit["device_bytes"] == 0
    _device_sizing.collect_lines(audit, chunk)
    assert audit["device_bytes"] is None and len(audit["collect"]) == 3
    for collected in audit["collect"]:
        for got, want in zip(collected, chunk, strict=True):
            assert isinstance(got, np.ndarray) and got.dtype == _device_sizing.REAL
            np.testing.assert_array_equal(got, want.astype(_device_sizing.REAL))


def _cupy():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:
        pass
    pytest.skip("no CUDA device")


@pytest.mark.hardware
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cuda_sizing_matches_the_host_route(dtype):
    cupy = _cupy()
    rng = np.random.default_rng(11)
    segments = _segments(rng, 200_000)
    device_segments = {key: cupy.asarray(value) for key, value in segments.items()}
    n_hat = (0.3, -0.2, 0.93)
    host_samples = sinc_feature_samples(segments, n_hat, electron_limit=None)
    device_samples = _device_sizing.device_sinc_samples(device_segments, n_hat)
    _assert_samples_agree([cupy.asnumpy(a) for a in device_samples], host_samples)
    host = sinc_spacing_from_samples([a.copy() for a in host_samples], aliased_weight_limit=0.01)
    device = _device_sizing.device_sinc_spacing(*device_samples, aliased_weight_limit=0.01)
    assert device[0] == host[0] and device[2] == host[2]
    assert device[1] == pytest.approx(host[1], rel=1e-12)

    populations = [_population(rng, 300_000, dtype) for _ in range(3)]
    resident = [
        ResonancePopulation(
            p.label, *(cupy.asarray(a) for a in (p.energy_eV, p.weight, p.width_eV))
        )
        for p in populations
    ]
    tail = _device_sizing.DeviceTail(resident)
    for stop in (2.0e3, 2.04e4, 4.4e4, 1.0e6):
        numerator = _HostTail(populations).numerator(stop)
        assert tail.numerator(stop) == pytest.approx(numerator, rel=1e-12)
    host_stop = resonance_population_stop_eV(populations, ceiling_eV=1.0e5, truncation_limit=1e-2)
    device_stop = resonance_population_stop_eV(resident, ceiling_eV=1.0e5, truncation_limit=1e-2)
    assert device_stop[0] == host_stop[0]
    assert device_stop[1]["n_lines"] == host_stop[1]["n_lines"]


def test_host_chunks_are_not_joined():
    chunks = [tuple(np.arange(3.0) for _ in range(3)) for _ in range(2)]
    assert _device_sizing.join_resident_chunks(chunks) is chunks


@pytest.mark.hardware
def test_cuda_chunks_join_in_collection_order():
    cupy = _cupy()
    rng = np.random.default_rng(12)
    host = [tuple(rng.random(int(n)) for _ in range(3)) for n in rng.integers(1, 50, 7)]
    joined = _device_sizing.join_resident_chunks(
        [tuple(cupy.asarray(a) for a in chunk) for chunk in host]
    )
    assert len(joined) == 1
    for got, want in zip(joined[0], zip(*host, strict=True), strict=True):
        np.testing.assert_array_equal(cupy.asnumpy(got), np.concatenate(want))


@pytest.mark.hardware
def test_cuda_staged_upload_is_exact(monkeypatch):
    cupy = _cupy()
    monkeypatch.setattr(_device_sizing, "_STAGE_BYTES", 4096)
    rng = np.random.default_rng(13)
    for array in (rng.random((10_001, 3)), rng.integers(0, 9, 7_777).astype(np.int32)):
        got = _device_sizing.upload(cupy, array)
        assert got.shape == array.shape and got.dtype == array.dtype
        np.testing.assert_array_equal(cupy.asnumpy(got), array)
    strided = rng.random((5000, 4))[:, 1:]
    np.testing.assert_array_equal(cupy.asnumpy(_device_sizing.upload(cupy, strided)), strided)


@pytest.mark.hardware
def test_cuda_spacing_repeats_bitwise():
    cupy = _cupy()
    rng = np.random.default_rng(14)
    width = cupy.asarray(rng.lognormal(3.0, 1.0, 5_000_003))
    weight = rng.lognormal(0.0, 2.0, width.size)
    cumulative = _device_sizing._fixed_order_cumsum(cupy, cupy.asarray(weight))
    np.testing.assert_allclose(cupy.asnumpy(cumulative), np.cumsum(weight), rtol=1e-12)
    results = {
        _device_sizing.device_sinc_spacing(width, cupy.asarray(weight), aliased_weight_limit=0.01)
        for _ in range(4)
    }
    assert len(results) == 1
