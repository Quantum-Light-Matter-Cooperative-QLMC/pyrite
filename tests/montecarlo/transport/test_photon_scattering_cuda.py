"""Same-draw conditional scattering parity and explicit exhaustion on CUDA."""

import numpy as np
import pytest

from pyrite.montecarlo.transport import photon_scattering as host
from pyrite.montecarlo.transport import photon_scattering_cuda as device


@pytest.fixture
def cp():
    from pyrite._backend import BackendUnavailableError, select_cupy_backend

    try:
        return select_cupy_backend().xp
    except BackendUnavailableError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("channel", ["coherent", "incoherent"])
@pytest.mark.parametrize("element", ["C", "Si", "W"])
def test_same_draw_epdl_scattering_matches_cpu(cp, channel, element):
    rng = np.random.default_rng(341_901)
    n = 64
    energy = np.geomspace(1, 1e8, n)
    ray = rng.normal(size=(n, 3))
    ray /= np.linalg.norm(ray, axis=1)[:, None]
    uniforms = rng.random((n, 512))
    sampler = host.sample_rayleigh if channel == "coherent" else host.sample_compton
    expected = [sampler(element, e, d, u) for e, d, u in zip(energy, ray, uniforms, strict=True)]
    actual = device.sample_photon_scattering_cuda(element, energy, ray, uniforms, channel=channel)
    assert all(isinstance(value, cp.ndarray) for value in actual.values())
    np.testing.assert_array_equal(cp.asnumpy(actual["status"]), 0)
    for key in ("photon_energy_eV", "photon_direction", "electron_energy_eV", "one_minus_cosine"):
        np.testing.assert_allclose(
            cp.asnumpy(actual[key]), [getattr(e, key) for e in expected], rtol=2e-12, atol=2e-14
        )
    np.testing.assert_array_equal(
        cp.asnumpy(actual["draws_used"]), [e.draws_used for e in expected]
    )
    if channel == "coherent":
        assert bool(cp.isnan(actual["electron_direction"]).all())
        np.testing.assert_array_equal(cp.asnumpy(actual["photon_energy_eV"]), energy)
    else:
        np.testing.assert_allclose(
            cp.asnumpy(actual["electron_direction"]),
            [e.electron_direction for e in expected],
            rtol=2e-12,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            cp.asnumpy(actual["photon_energy_eV"] + actual["electron_energy_eV"]),
            energy,
            rtol=3e-16,
        )


@pytest.mark.parametrize("channel", ["coherent", "incoherent"])
def test_exhausted_prefix_is_explicit_and_refill_preserves_event(cp, channel):
    full = np.random.default_rng(341_902).random((1, 256))
    args = ("Si", [3e5], [[0, 0, 1]])
    short = device.sample_photon_scattering_cuda(
        *args, full[:, :2], channel=channel, require_complete=False
    )
    np.testing.assert_array_equal(cp.asnumpy(short["status"]), [1])
    np.testing.assert_array_equal(cp.asnumpy(short["draws_used"]), [2])
    with pytest.raises(ValueError, match="sequence exhausted"):
        device.sample_photon_scattering_cuda(*args, full[:, :2], channel=channel)
    event = device.sample_photon_scattering_cuda(*args, full, channel=channel)
    used = int(event["draws_used"][0])
    exact = device.sample_photon_scattering_cuda(*args, full[:, :used], channel=channel)
    for key in event:
        np.testing.assert_array_equal(cp.asnumpy(event[key]), cp.asnumpy(exact[key]))


def test_invalid_scatter_inputs_fail_before_device_probe(monkeypatch):
    from pyrite import _backend

    monkeypatch.setattr(
        _backend, "select_cupy_backend", lambda *args: pytest.fail("unexpected GPU probe")
    )
    with pytest.raises(ValueError, match="unit vectors"):
        device.sample_photon_scattering_cuda("Si", [1e5], [[0, 0, 0]], [[0.1]], channel="coherent")
    with pytest.raises(ValueError, match="uniforms"):
        device.sample_photon_scattering_cuda("Si", [1e5], [[0, 0, 1]], [[1.0]], channel="coherent")
    with pytest.raises(ValueError, match="channel"):
        device.sample_photon_scattering_cuda("Si", [1e5], [[0, 0, 1]], [[0.1]], channel="auto")
