"""Device Heitler sampling against source DCS, invariants, and host replay.

Validation: heitler-annihilation, positron-annihilation-at-rest
"""

import numpy as np
import pytest
from scipy import stats

from pyrite.montecarlo.transport import annihilation as ann
from pyrite.montecarlo.transport.kinematics import _stream_uniform_scalar

try:
    import cupy as cp

    _HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

pytestmark = [pytest.mark.hardware, pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device")]


class _CounterRNG:
    """Replay the device's keyed uniform sequence through the anchored host sampler."""

    def __init__(self, key):
        self.key, self.counter = np.uint64(key), 0

    def random(self, size=None):
        count = 1 if size is None else int(np.prod(size))
        with np.errstate(over="ignore"):
            values = np.array(
                [
                    _stream_uniform_scalar(self.key, np.uint64(self.counter + j))
                    for j in range(count)
                ]
            )
        self.counter += count
        return float(values[0]) if size is None else values.reshape(size)


def _sample(kinds, kinetic, direction, keys, **kwargs):
    from pyrite.montecarlo.transport._jit_annihilation import sample_annihilation_device

    return sample_annihilation_device(
        cp.asarray(kinds), cp.asarray(kinetic), cp.asarray(direction), cp.asarray(keys), **kwargs
    )


def test_device_cross_sections_match_host_and_limits():
    from pyrite.montecarlo.transport._jit_annihilation import heitler_cross_section_device

    # Non-contiguous device input; covers slow, relativistic, and Dirac limits.
    kinetic = np.geomspace(1e-3, 1e9, 100).reshape(10, 10)[:, ::2]
    result = heitler_cross_section_device(cp.asarray(kinetic)[:, ::-1])
    assert isinstance(result, cp.ndarray) and result.dtype == cp.float64
    np.testing.assert_allclose(
        result.get(), ann.heitler_cross_section_ang2(kinetic[:, ::-1]), rtol=2e-10
    )
    scalar = heitler_cross_section_device(cp.asarray(1000.0))
    assert scalar.shape == ()
    assert scalar.get() == pytest.approx(ann.heitler_cross_section_ang2(1000.0), rel=1e-13)
    assert heitler_cross_section_device(cp.empty((0, 2))).shape == (0, 2)


def test_device_photons_and_scoring_draws_replay_on_host():
    rng = np.random.default_rng(295)
    kinetic = np.geomspace(0.1, 1e5, 64)
    kinds = np.ones(64, dtype=np.int8)
    kinds[::4] = ann.ANNIHILATION_AT_REST
    direction = rng.normal(size=(64, 3))  # sampler normalizes the in-flight direction
    keys = ann.annihilation_stream_keys(295, np.arange(64), np.zeros(64), in_flight=False)
    result = _sample(kinds, kinetic, direction, keys)
    assert all(isinstance(v, cp.ndarray) for v in result.values())
    assert result["valid"].get().all()
    host_energy, host_direction, host_uniform = [], [], []
    for kind, energy, axis, key in zip(kinds, kinetic, direction, keys, strict=True):
        uniform = _CounterRNG(key)
        if kind == ann.ANNIHILATION_IN_FLIGHT:
            energies, dirs = ann.sample_heitler(energy, axis, uniform)
        else:
            energies, dirs = (
                np.full(2, ann.ELECTRON_REST_KEV),
                ann.sample_at_rest_directions(uniform),
            )
        host_energy.append(energies)
        host_direction.append(dirs)
        host_uniform.append(uniform.random((2, 2)))
    np.testing.assert_allclose(result["k_keV"].get(), host_energy, rtol=2e-13)
    np.testing.assert_allclose(result["direction"].get(), host_direction, atol=2e-11, rtol=2e-11)
    np.testing.assert_array_equal(result["uniforms"].get(), host_uniform)
    energy, photons = result["k_keV"].get(), result["direction"].get()
    flight = kinds == ann.ANNIHILATION_IN_FLIGHT
    total = kinetic[flight] + 2 * ann.ELECTRON_REST_KEV
    np.testing.assert_allclose(energy[flight].sum(axis=1), total, rtol=3e-16)
    momentum = np.sqrt(kinetic[flight] * total)[:, None] * (
        direction[flight] / np.linalg.norm(direction[flight], axis=1)[:, None]
    )
    np.testing.assert_allclose(
        (energy[flight, :, None] * photons[flight]).sum(axis=1), momentum, atol=1e-7, rtol=1e-9
    )
    np.testing.assert_allclose(np.linalg.norm(photons, axis=2), 1.0, atol=5e-16)


@pytest.mark.parametrize("kinetic_keV", (10.0, 1_000.0, 10_000.0))
def test_device_zeta_matches_independent_heitler_cdf(kinetic_keV):
    n = 20_000
    keys = ann.annihilation_stream_keys(295, np.arange(n), np.zeros(n), in_flight=False)
    result = _sample(
        np.ones(n, dtype=np.int8), np.full(n, kinetic_keV), np.tile([0.0, 0.0, 1.0], (n, 1)), keys
    )
    assert result["valid"].get().all()
    zeta = result["k_keV"][:, 0].get() / (kinetic_keV + 2 * ann.ELECTRON_REST_KEV)
    # Source Eqs. 3.187--3.188 integrated analytically, independent of sampler/pdf helpers.
    gamma = 1.0 + kinetic_keV / 510.99895
    lower = 1.0 / (gamma + 1.0 + np.sqrt(gamma**2 - 1.0))

    def primitive(z):
        return -((gamma + 1.0) ** 2) * z + (gamma**2 + 4 * gamma + 1) * np.log(z) + 1 / z

    def integral(z):
        return primitive(z) - primitive(lower) - primitive(1 - z) + primitive(1 - lower)

    assert zeta.min() >= lower and zeta.max() <= 0.5
    fit = stats.kstest(zeta, lambda z: integral(z) / integral(0.5))
    assert fit.pvalue > 1e-3, fit


def test_device_at_rest_is_isotropic_and_exactly_antiparallel():
    n = 20_000
    keys = ann.annihilation_stream_keys(296, np.arange(n), np.zeros(n), in_flight=False)
    # No incoming axis or energy is needed for an at-rest event.
    result = _sample(np.zeros(n, dtype=np.int8), np.full(n, np.nan), np.full((n, 3), np.nan), keys)
    energy, direction = result["k_keV"].get(), result["direction"].get()
    assert result["valid"].get().all()
    np.testing.assert_array_equal(energy, ann.ELECTRON_REST_KEV)
    np.testing.assert_array_equal(direction[:, 0], -direction[:, 1])
    np.testing.assert_allclose(np.linalg.norm(direction, axis=2), 1.0, atol=5e-16)
    assert stats.kstest(direction[:, 0, 2], stats.uniform(-1, 2).cdf).pvalue > 1e-3
    phi = np.arctan2(direction[:, 0, 1], direction[:, 0, 0])
    assert stats.kstest(phi, stats.uniform(-np.pi, 2 * np.pi).cdf).pvalue > 1e-3


def test_device_sampler_is_independent_of_batch_order_and_chunking():
    n = 129  # straddles a block boundary
    kinds = np.arange(n) % 2
    kinetic = np.geomspace(1.0, 1e5, n)
    directions = np.tile([0.3, -0.4, 0.866], (n, 1))
    keys = ann.annihilation_stream_keys(297, np.arange(n), np.zeros(n), in_flight=False)
    whole = _sample(kinds, kinetic, directions, keys)
    reverse = _sample(kinds[::-1], kinetic[::-1], directions[::-1], keys[::-1])
    chunks = [
        _sample(kinds[s], kinetic[s], directions[s], keys[s]) for s in (slice(0, 57), slice(57, n))
    ]
    for name, value in whole.items():
        np.testing.assert_array_equal(value.get(), reverse[name][::-1].get(), err_msg=name)
        np.testing.assert_array_equal(value.get(), cp.concatenate([c[name] for c in chunks]).get())


def test_device_failure_flags_and_empty_batch():
    result = _sample(
        np.array([1, 0]),
        np.array([1000.0, 0.0]),
        np.tile([0.0, 0.0, 1.0], (2, 1)),
        np.array([1, 2], dtype=np.uint64),
        max_rejections=0,
    )
    np.testing.assert_array_equal(result["valid"].get(), [False, True])
    for name in ("k_keV", "direction", "uniforms"):
        assert np.isnan(result[name][0].get()).all()
    empty = _sample(
        np.empty(0, dtype=np.int8), np.empty(0), np.empty((0, 3)), np.empty(0, dtype=np.uint64)
    )
    assert empty["k_keV"].shape == (0, 2) and empty["direction"].shape == (0, 2, 3)
    assert empty["uniforms"].shape == (0, 2, 2) and empty["valid"].shape == (0,)


def test_device_invalid_events_are_flagged_without_kind_wraparound():
    result = _sample(
        np.array([1, 1, 1, 256]),
        np.array([0.0, -1.0, 100.0, 0.0]),
        np.zeros((4, 3)),
        np.arange(4, dtype=np.uint64),
    )
    assert not result["valid"].get().any()
    for name in ("k_keV", "direction", "uniforms"):
        assert np.isnan(result[name].get()).all()


def test_device_degenerate_energies_are_flagged():
    from pyrite.montecarlo.transport._jit_annihilation import heitler_cross_section_device

    kinetic = np.array([1e-14, 1e308, np.nan, np.inf])
    result = _sample(
        np.ones(4, dtype=np.int8),
        kinetic,
        np.tile([0.0, 0.0, 1.0], (4, 1)),
        np.arange(4, dtype=np.uint64),
    )
    assert not result["valid"].get().any()
    assert np.isnan(heitler_cross_section_device(cp.asarray(kinetic)).get()).all()


def test_device_sampler_rejects_host_inputs_and_invalid_shapes():
    from pyrite.montecarlo.transport._jit_annihilation import sample_annihilation_device

    kinds, kinetic, directions, keys = (
        cp.zeros(1, dtype=cp.int8),
        cp.ones(1),
        cp.ones((1, 3)),
        cp.ones(1, dtype=cp.uint64),
    )
    with pytest.raises(TypeError, match="CuPy"):
        sample_annihilation_device(np.zeros(1), kinetic, directions, keys)
    with pytest.raises(ValueError, match="expected"):
        sample_annihilation_device(kinds, kinetic, directions[:, :2], keys)
    with pytest.raises(TypeError, match="integer"):
        sample_annihilation_device(kinds, kinetic, directions, keys.astype(cp.float64))
    with pytest.raises(ValueError, match="max_rejections"):
        sample_annihilation_device(kinds, kinetic, directions, keys, max_rejections=-1)
