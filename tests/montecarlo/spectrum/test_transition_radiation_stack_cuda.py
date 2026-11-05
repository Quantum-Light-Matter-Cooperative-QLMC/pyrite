"""CUDA parity of the O(N) planar stack kernels (#962).

Validation: optical-stack-fast-solver
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum import transition_radiation_stack_kernel
from pyrite.montecarlo.spectrum.transition_radiation import (
    interface_response,
    planar_transition_radiation,
)

try:  # pragma: no cover - depends on the machine, not the branch
    import cupy

    _HAS_CUDA = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not _HAS_CUDA, reason="no CUDA device"),
]

_RNG = np.random.default_rng(9620)
_ENERGY = _RNG.uniform(0.3, 4.0, 300)
_THETA = _RNG.uniform(0.0, 1.45, 300)
_THETA[0] = 0.0
_PHI = _RNG.uniform(0.0, 2 * np.pi, 300)
_N = np.stack(
    [np.sin(_THETA) * np.cos(_PHI), np.sin(_THETA) * np.sin(_PHI), np.cos(_THETA)], axis=-1
)
_STACKS = {
    "bragg": ([(2.1 + 0.0j, 800.0), (6.0 + 0.01j, 500.0)] * 150, 1.0),
    "metal_dielectric": ([(-20.0 + 1.3j, 300.0), (2.1 + 0.0j, 1200.0)] * 40, 2.25 + 0.0j),
    "thick": ([(-30.0 + 2.0j, 2e6), (3.0 + 0.01j, 1e6), (2.0 + 0.0j, 1e5)] * 8, 1.0),
}


@pytest.mark.parametrize("name", _STACKS)
@pytest.mark.parametrize("velocity", [(0.0, 0.0, 0.7), (0.1, -0.2, -0.9)])
def test_amplitude_parity(name, velocity):
    layers, substrate = _STACKS[name]
    v = np.broadcast_to(np.asarray(velocity), _N.shape)
    host_y, host = planar_transition_radiation(
        _ENERGY, _N, v, layers, substrate, return_amplitude=True
    )
    device_y, device = planar_transition_radiation(
        _ENERGY, _N, v, layers, substrate, xp=cupy, return_amplitude=True
    )
    assert isinstance(device, cupy.ndarray)
    assert np.max(np.abs(device.get() - host)) <= 1e-10 * np.max(np.abs(host))
    assert np.max(np.abs(device_y.get() - host_y)) <= 1e-10 * np.max(host_y)


@pytest.mark.parametrize("name", _STACKS)
def test_response_parity(name):
    layers, substrate = _STACKS[name]
    host = interface_response(_ENERGY, _N, layers, substrate)
    device = interface_response(_ENERGY, _N, layers, substrate, xp=cupy)
    assert isinstance(device.response, cupy.ndarray)
    assert np.max(np.abs(device.response.get() - host.response)) <= 1e-10 * np.max(
        np.abs(host.response)
    )
    np.testing.assert_allclose(device.lam_top.get(), host.lam_top, rtol=1e-13)


def test_launch_budget_does_not_change_the_result(monkeypatch):
    layers, substrate = _STACKS["bragg"]
    whole = interface_response(_ENERGY, _N, layers, substrate, xp=cupy).response
    monkeypatch.setattr(transition_radiation_stack_kernel, "_MAX_LAUNCH_STEPS", 64 * 301)
    split = interface_response(_ENERGY, _N, layers, substrate, xp=cupy).response
    assert cupy.array_equal(whole, split)
