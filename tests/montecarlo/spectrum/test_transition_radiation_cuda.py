"""CUDA parity of the planar-stack transition-radiation solve and its run-case component.

Validation: tr-detected-omission-bound, tr-crossing-superposition, tr-coherent-bunch
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import elementary_charge

from pyrite import _backend
from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.runner import transition_emission
from pyrite.montecarlo.runner.transition_emission import case_transition_spectrum
from pyrite.montecarlo.spectrum import transition_radiation_crossings
from pyrite.montecarlo.spectrum.coherent_population import CoherentSamplingWarning
from pyrite.montecarlo.spectrum.transition_radiation import (
    composition_dielectric,
    interface_response,
    medium_dielectric,
    planar_transition_radiation,
)
from pyrite.montecarlo.spectrum.transition_radiation_crossings import (
    _emission_frame,
    _take,
    boundary_crossings,
    crossing_transition_radiation,
)
from pyrite.montecarlo.spectrum.transition_radiation_jit_kernel import history_sums
from tests.montecarlo.spectrum.test_transition_radiation_crossings import _BILAYER, _run
from tests.montecarlo.spectrum.test_transition_radiation_runner import _case

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


def _assert_parity(device, host):
    assert isinstance(device, cupy.ndarray) and device.dtype == cupy.float64
    peak = np.max(np.abs(host))
    np.testing.assert_allclose(cupy.asnumpy(device), host, rtol=1e-10, atol=1e-12 * peak)


@pytest.fixture(scope="module")
def bilayer_transport():
    return _run(_BILAYER)


@pytest.fixture(scope="module")
def bilayer_crossings(bilayer_transport):
    return boundary_crossings(bilayer_transport, _BILAYER)


@pytest.mark.parametrize("fused", [True, False], ids=["kernel", "array"])
@pytest.mark.parametrize("coherence", ["history", "crossing", "bunch"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_cupy_crossing_yield_matches_numpy(bilayer_crossings, coherence, sign, fused, monkeypatch):
    monkeypatch.setattr(transition_radiation_crossings, "_USE_JIT_TRANSITION", fused)
    energy = np.geomspace(100.0, 3000.0, 40)
    theta = np.linspace(0.05, 1.2, 5)
    n = np.stack([np.sin(theta), np.zeros_like(theta), sign * np.cos(theta)], -1)
    eps = [composition_dielectric(comp, energy)[:, None] for _, _, comp in _BILAYER]
    # A small element budget splits the 60 histories over many array-route chunks.
    kw = dict(n_electrons=60, coherence=coherence, max_elements=4096)
    if coherence == "bunch":
        kw.update(
            physical_electrons=1e5, form_factor=lambda e: np.exp(-((e * 0.2 / HBARC_EV_ANG) ** 2))
        )
    args = (bilayer_crossings, energy[:, None], n[None], eps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoherentSamplingWarning)  # signed bunch estimate
        device = crossing_transition_radiation(*args, xp=cupy, **kw)
        host = crossing_transition_radiation(*args, **kw)
    _assert_parity(device, host)


def test_kernel_sums_are_reproducible_and_launch_geometry_only_reorders(bilayer_crossings):
    crossings = _take(bilayer_crossings, np.argsort(bilayer_crossings.electron_id, kind="stable"))
    energy = np.geomspace(100.0, 3000.0, 7)[:, None]
    n = np.array([[[np.sin(0.4), 0.0, np.cos(0.4)]]])
    eps = [composition_dielectric(comp, energy) for _, _, comp in _BILAYER]
    frame = _emission_frame(crossings, n, eps)
    stack = interface_response(energy, frame.direction, frame.layers, 1.0, xp=cupy)
    history = crossings.electron_id
    starts = np.r_[np.flatnonzero(np.r_[True, history[1:] != history[:-1]]), history.size]
    args = (stack, frame.velocity, crossings.time_ang, frame.position, frame.face,
            crossings.charge.astype(float), starts)  # fmt: skip
    first, again = history_sums(*args), history_sums(*args)
    for a, b in zip(first, again, strict=True):  # no atomics: bit-identical reruns
        np.testing.assert_array_equal(cupy.asnumpy(a), cupy.asnumpy(b))
    for threads, blocks in ((32, 1), (256, 1 << 14)):
        regrouped = history_sums(*args, threads=threads, target_blocks=blocks)
        for a, b in zip(first, regrouped, strict=True):
            a, b = cupy.asnumpy(a), cupy.asnumpy(b)
            np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-13 * np.max(np.abs(a)))


@pytest.mark.parametrize("coherence", ["history", "bunch"])
def test_run_case_component_scores_on_the_device(coherence, bilayer_transport, monkeypatch):
    # The component consumes transported rows; SBETHE/ELSEPA installation is
    # independent of its device dispatch. Reuse the same seeded bilayer as
    # the crossing-parity tests instead of requiring external table downloads.
    case = _case(abs_layers=_BILAYER, Ne=60, Ne_brem=60, thickness_ang=3000.0)
    if coherence == "bunch":  # N = M: the plain coherent sum, no undersampled excess
        case = replace(
            case,
            transition_radiation={"model": "planar-stack", "coherence": "bunch"},
            bunch_charge_pc=case.Ne_brem * elementary_charge / 1e-12,
        )
    transport = dict(
        segs=bilayer_transport,
        E_grid=np.geomspace(100.0, 3000.0, 40),
        n_hat=np.array([np.sin(0.3), 0.0, np.cos(0.3)]),
        Ne_brem=case.Ne_brem,
    )
    host = case_transition_spectrum(case, transport, xp=np)
    used = []
    score = transition_emission.crossing_transition_radiation
    monkeypatch.setattr(
        transition_emission,
        "crossing_transition_radiation",
        lambda *args, **kw: used.append(kw["xp"]) or score(*args, **kw),
    )
    default = case_transition_spectrum(case, transport)
    assert used == [np if _backend.cp is None else cupy]  # CuPy backend: device scoring
    assert type(default) is np.ndarray  # returned to the host
    peak = np.max(np.abs(host))
    np.testing.assert_allclose(default, host, rtol=1e-10, atol=1e-12 * peak)
    np.testing.assert_allclose(
        case_transition_spectrum(case, transport, xp=cupy), host, rtol=1e-10, atol=1e-12 * peak
    )
