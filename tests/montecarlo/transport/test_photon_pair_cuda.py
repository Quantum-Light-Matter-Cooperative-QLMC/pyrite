"""Explicit-draw CPU/CUDA pair parity, including the threshold convention."""

import numpy as np
import pytest

from pyrite.montecarlo.transport.pair_production import PAIR_THRESHOLD_EV, sample_pair
from pyrite.montecarlo.transport.photon_pair_cuda import sample_pairs_cuda


@pytest.fixture
def cp():
    from pyrite._backend import BackendUnavailableError, select_cupy_backend

    try:
        return select_cupy_backend().xp
    except BackendUnavailableError as error:
        pytest.skip(str(error))


class _Draws:
    def __init__(self, values):
        self.values = iter(values)
        self.count = 0

    def random(self):
        self.count += 1
        return next(self.values)


@pytest.mark.parametrize("z", [6, 74, 90])
def test_same_draw_pairs_and_threshold_limit_match_cpu(cp, z):
    rng = np.random.default_rng(341_903)
    energy = np.geomspace(PAIR_THRESHOLD_EV * (1 + 1e-6), 1e8, 48)
    rays = rng.normal(size=(len(energy), 3))
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    uniforms = rng.random((len(energy), 512))
    streams = [_Draws(row) for row in uniforms]
    expected = [
        sample_pair(e, z, ray, draws) for e, ray, draws in zip(energy, rays, streams, strict=True)
    ]
    actual = sample_pairs_cuda(energy, rays, uniforms, atomic_number=z)
    np.testing.assert_array_equal(cp.asnumpy(actual["status"]), 0)
    np.testing.assert_array_equal(
        cp.asnumpy(actual["draws_used"]), [stream.count for stream in streams]
    )
    for output, field in (
        ("electron_energy_eV", "electron_eV"),
        ("positron_energy_eV", "positron_eV"),
    ):
        np.testing.assert_allclose(
            cp.asnumpy(actual[output]), [getattr(e, field) for e in expected], rtol=2e-12, atol=2e-8
        )
    for field in ("electron_direction", "positron_direction"):
        np.testing.assert_allclose(
            cp.asnumpy(actual[field]), [getattr(e, field) for e in expected], rtol=2e-10, atol=2e-12
        )
    np.testing.assert_allclose(
        cp.asnumpy(actual["electron_energy_eV"] + actual["positron_energy_eV"]) + PAIR_THRESHOLD_EV,
        energy,
        rtol=3e-16,
    )


def test_pair_prefix_refill_and_zero_kinetic_threshold(cp):
    energy, rays = [PAIR_THRESHOLD_EV + 100], [[0, 0, 1]]
    uniforms = np.zeros((1, 16))
    short = sample_pairs_cuda(
        energy, rays, uniforms[:, :2], atomic_number=90, require_complete=False
    )
    np.testing.assert_array_equal(cp.asnumpy(short["status"]), [1])
    with pytest.raises(ValueError, match="sequence exhausted"):
        sample_pairs_cuda(energy, rays, uniforms[:, :2], atomic_number=90)
    complete = sample_pairs_cuda(energy, rays, uniforms, atomic_number=90)
    stream = _Draws(uniforms[0])
    expected = sample_pair(energy[0], 90, np.array(rays[0]), stream)
    np.testing.assert_allclose(
        cp.asnumpy(complete["electron_energy_eV"]), [expected.electron_eV], atol=1e-9
    )
    np.testing.assert_array_equal(cp.asnumpy(complete["draws_used"]), [stream.count])
    np.testing.assert_allclose(
        cp.asnumpy(complete["electron_direction"]), [expected.electron_direction], atol=1e-14
    )


def test_pair_domain_fails_before_device_probe(monkeypatch):
    from pyrite import _backend

    monkeypatch.setattr(
        _backend, "select_cupy_backend", lambda *args: pytest.fail("unexpected probe")
    )
    with pytest.raises(ValueError, match="above 2"):
        sample_pairs_cuda([PAIR_THRESHOLD_EV], [[0, 0, 1]], [[0.1]], atomic_number=74)
    with pytest.raises(ValueError, match="Z=1..99"):
        sample_pairs_cuda([3e6], [[0, 0, 1]], [[0.1]], atomic_number=100)
