"""CUDA parity of the planar-stack transition-radiation solve (tr-detected-omission-bound)."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.transition_radiation import (
    medium_dielectric,
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


def test_cupy_solve_matches_numpy_on_an_absorbing_oblique_multilayer():
    rng = np.random.default_rng(102)
    photon = np.geomspace(40.0, 5e3, 24)
    theta = rng.uniform(0.05, 1.4, 16)
    phi = rng.uniform(0.0, 2 * np.pi, 16)
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
    velocities = np.array([[0.2, 0.1, 0.3], [-0.3, 0.0, -0.8], [0.0, 0.05, 0.995]])
    eps_si = medium_dielectric("silicon", photon)[:, None, None]
    eps_mo = medium_dielectric("mos2", photon)[:, None, None]
    layers = [(eps_si, 2e3), (1.0, 5e3), (eps_mo, 800.0)]
    sapphire = medium_dielectric("sapphire", photon)[:, None, None]
    args = (photon[:, None, None], n[None, :, None, :], velocities[None, None, :, :], layers)
    host, host_amplitude = planar_transition_radiation(*args, sapphire, return_amplitude=True)
    device, device_amplitude = planar_transition_radiation(
        *args, cupy.asarray(sapphire), xp=cupy, return_amplitude=True
    )
    assert isinstance(device, cupy.ndarray) and device.dtype == cupy.float64
    # cuSOLVER and LAPACK round differently in near-cancelling, ~1e-14-of-peak
    # entries; the absolute floor keeps 1e-10 relative on everything above it.
    np.testing.assert_allclose(cupy.asnumpy(device), host, rtol=1e-10, atol=1e-12 * host.max())
    assert isinstance(device_amplitude, cupy.ndarray)
    assert device_amplitude.dtype == cupy.complex128
    np.testing.assert_allclose(
        cupy.asnumpy(device_amplitude),
        host_amplitude,
        rtol=1e-10,
        atol=1e-12 * np.max(abs(host_amplitude)),
    )
