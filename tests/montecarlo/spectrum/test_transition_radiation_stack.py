"""O(N) planar stack solve against the dense boundary system (#962).

Validation: optical-stack-fast-solver
"""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.transition_radiation import (
    FAST_STACK_MIN_LAYERS,
    _interface_heights,
    _particular_field,
    _stack_media,
    ginzburg_frank_interface,
    interface_response,
    interface_transition_radiation_amplitude,
    planar_transition_radiation,
)
from pyrite.montecarlo.spectrum.transition_radiation_stack import (
    stack_amplitude,
    stack_response,
    straight_charge_amplitude,
)
from pyrite.montecarlo.spectrum.transition_radiation_stack_kernel import _COUPLING, SOURCE

_CXX = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")

_RNG = np.random.default_rng(962)
_ENERGY = _RNG.uniform(0.3, 4.0, 40)
_THETA = _RNG.uniform(0.0, 1.45, 40)
_THETA[0] = 0.0
_PHI = _RNG.uniform(0.0, 2 * np.pi, 40)
_N = np.stack(
    [np.sin(_THETA) * np.cos(_PHI), np.sin(_THETA) * np.sin(_PHI), np.cos(_THETA)], axis=-1
)
_STACKS = {
    "single_film": ([(2.25 + 0.0j, 3000.0)], 1.0),
    "absorbing_on_substrate": ([(-20.0 + 1.3j, 500.0), (2.1 + 0.0j, 4000.0)], 2.25 + 0.0j),
    "bragg": ([(2.1 + 0.0j, 800.0), (6.0 + 0.01j, 500.0)] * 12, 1.0),
    "thick_opaque": ([(-30.0 + 2.0j, 2e6), (3.0 + 0.01j, 1e6), (-5.0 + 1.0j, 3e5)], 1.0),
    "evanescent_gap": ([(2.25 + 0.0j, 2000.0), (1.0 + 0.0j, 8e5), (2.25 + 0.0j, 2000.0)], 1.0),
}


def _inputs(layers, substrate):
    omega, n = _ENERGY, _N
    media = _stack_media(substrate, layers, omega.shape, np)
    return omega, omega[:, None] * n[:, :2], media, _interface_heights(layers)


@pytest.mark.parametrize("name", _STACKS)
def test_response_matches_the_dense_transposed_solve(name):
    layers, substrate = _STACKS[name]
    dense = interface_response(_ENERGY, _N, layers, substrate)
    omega, k_perp, media, z_if = _inputs(layers, substrate)
    response, lam = stack_response(omega, k_perp, np.stack(media), z_if)
    scale = np.max(np.abs(dense.response))
    assert np.max(np.abs(response - dense.response)) <= 1e-9 * scale
    np.testing.assert_allclose(lam, dense.lam_top, rtol=1e-14)


@pytest.mark.parametrize("name", _STACKS)
@pytest.mark.parametrize("velocity", [(0.0, 0.0, 0.7), (0.1, -0.2, -0.9), (0.3, 0.0, 0.05)])
def test_amplitude_matches_the_dense_solve(name, velocity):
    layers, substrate = _STACKS[name]
    v = np.broadcast_to(np.asarray(velocity), _N.shape)
    _, dense = planar_transition_radiation(_ENERGY, _N, v, layers, substrate, return_amplitude=True)
    omega, k_perp, media, z_if = _inputs(layers, substrate)
    q = (omega - np.sum(k_perp * v[:, :2], axis=-1)) / v[:, 2]
    particular = [_particular_field(eps, omega, k_perp, q, v, np) for eps in media]
    sources = []
    for i, z0 in enumerate(z_if):
        e_jump = particular[i][0][:, :2] - particular[i + 1][0][:, :2]
        h_jump = particular[i][1][:, :2] - particular[i + 1][1][:, :2]
        sources.append(np.concatenate([e_jump, h_jump], axis=-1) * np.exp(1j * q * z0)[:, None])
    e_t, _ = stack_amplitude(omega, k_perp, np.stack(media), z_if, sources)
    assert np.max(np.abs(e_t - dense[:, :2])) <= 1e-9 * np.max(np.abs(dense))


def test_a_ten_micron_absorber_decouples_what_lies_beneath_it():
    gold = -30.0 + 2.0j
    energy, n = np.array([1.5]), np.array([[0.0, 0.0, 1.0]])
    k_perp = energy[:, None] * n[:, :2]
    below = [(2.25 + 0.0j, 900.0)] * 40
    layers = [*below, (gold, 1e5)]
    media = np.stack(_stack_media(1.0, layers, energy.shape, np))
    response, _ = stack_response(energy, k_perp, media, _interface_heights(layers))
    assert np.all(np.isfinite(response))
    assert np.max(np.abs(response[:, :-2])) < 1e-150


@pytest.fixture(scope="module")
def host_kernel(tmp_path_factory):
    if _CXX is None:
        pytest.skip("no host C++ compiler")
    build = tmp_path_factory.mktemp("stack_kernel")
    (build / "stack.cpp").write_text(SOURCE)
    library = build / "stack.so"
    subprocess.run(
        [_CXX, "-std=c++17", "-O1", "-shared", "-fPIC", "-DPYRITE_STACK_HOST",
         str(build / "stack.cpp"), "-o", str(library)],
        check=True, capture_output=True,
    )  # fmt: skip
    return ctypes.CDLL(str(library))


def _ptr(array):
    return array.ctypes.data_as(ctypes.c_void_p)


@pytest.mark.parametrize("name", _STACKS)
def test_host_kernel_response_matches_numpy(host_kernel, name):
    layers, substrate = _STACKS[name]
    omega, k_perp, media, z_if = _inputs(layers, substrate)
    media = np.ascontiguousarray(np.stack(media))
    expected, expected_lam = stack_response(omega, k_perp, media, z_if)
    response = np.zeros_like(expected)
    lam = np.zeros_like(expected_lam)
    trans = np.zeros((omega.size, len(z_if), 2), dtype=complex)
    k_perp = np.ascontiguousarray(k_perp)
    host_kernel.pyrite_stack_host_response(
        ctypes.c_int(omega.size), ctypes.c_int(media.shape[0]), _ptr(omega), _ptr(k_perp),
        _ptr(media), _ptr(z_if), _ptr(response), _ptr(lam), _ptr(trans),
    )  # fmt: skip
    assert np.max(np.abs(response - expected)) <= 1e-11 * np.max(np.abs(expected))
    np.testing.assert_allclose(lam, expected_lam, rtol=1e-14)


@pytest.mark.parametrize("name", _STACKS)
def test_host_kernel_amplitude_matches_numpy(host_kernel, name):
    layers, substrate = _STACKS[name]
    omega, k_perp, media, z_if = _inputs(layers, substrate)
    media = np.ascontiguousarray(np.stack(media))
    k_perp = np.ascontiguousarray(k_perp)
    v = np.ascontiguousarray(np.broadcast_to([0.1, -0.2, -0.9], _N.shape), dtype=float)
    expected, _ = straight_charge_amplitude(omega, k_perp, v, media, z_if)
    e_t = np.zeros_like(expected)
    lam = np.zeros(omega.size, dtype=complex)
    host_kernel.pyrite_stack_host_amplitude(
        ctypes.c_int(omega.size), ctypes.c_int(media.shape[0]), _ptr(omega), _ptr(k_perp), _ptr(v),
        _ptr(media), _ptr(z_if), ctypes.c_double(_COUPLING), _ptr(e_t), _ptr(lam),
    )  # fmt: skip
    assert np.max(np.abs(e_t - expected)) <= 1e-11 * np.max(np.abs(expected))


_MANY = FAST_STACK_MIN_LAYERS + 4


def test_matched_and_vanishing_many_layer_stacks_radiate_nothing():
    n, v = np.array([0.6, 0.0, 0.8]), np.array([0.0, 0.0, 0.9])
    assert planar_transition_radiation(2.0, n, v, [(1.0, 500.0)] * _MANY) == 0.0
    film = planar_transition_radiation(2.0, n, v, [(2.25, 1e-6)] * _MANY)
    assert film < 1e-12 * planar_transition_radiation(2.0, n, v, [(2.25, 500.0)])


def test_uniform_many_layer_stack_is_the_single_interface_closed_form():
    theta = np.radians([5.0, 30.0, 70.0])
    eps = -30.0 + 2.0j
    n = np.stack([np.sin(theta), np.zeros(3), np.cos(theta)], axis=-1)
    v = np.array([0.0, 0.0, 0.8])
    stack = planar_transition_radiation(1.5, n, v, [(eps, 700.0)] * _MANY, eps)
    closed = ginzburg_frank_interface(eps, 0.8, theta, 1.5)
    np.testing.assert_allclose(stack, closed, rtol=1e-9)


def test_opaque_top_layer_reduces_to_its_exit_interface():
    gold = -30.0 + 2.0j
    n, v = np.array([0.5, 0.0, np.sqrt(0.75)]), np.array([0.0, 0.0, 0.9])
    layers = [(2.1 + 0.0j, 800.0), (6.0 + 0.01j, 500.0)] * (_MANY // 2) + [(gold, 1e5)]
    stack = planar_transition_radiation(1.5, n, v, layers)
    exit_only = planar_transition_radiation(1.5, n, v, [], gold)
    np.testing.assert_allclose(stack, exit_only, rtol=1e-9)


def test_many_layer_crossings_sum_to_the_straight_amplitude():
    layers = [(2.1 + 0.0j, 800.0), (6.0 + 0.1j, 500.0)] * 15
    energy, n = np.array([1.2, 2.3]), np.array([0.3, 0.2, np.sqrt(0.87)])
    v = np.array([0.1, -0.2, 0.9])
    _, direct = planar_transition_radiation(energy, n, v, layers, return_amplitude=True)
    z = np.concatenate([[0.0], np.cumsum([d for _, d in layers])])
    time = z / v[2]
    position = np.stack([v[0] * time, v[1] * time, z], axis=-1)
    amplitude = interface_transition_radiation_amplitude(
        energy, n, layers, 1.0, np.broadcast_to(v, (z.size, 3)), time, position, np.arange(z.size)
    )
    np.testing.assert_allclose(amplitude.sum(axis=-2), direct, rtol=1e-9)


@pytest.mark.parametrize("polarization", [0, 1], ids=["te", "tm"])
def test_response_decays_at_the_bloch_rate_inside_a_stop_band(polarization):
    # Quarter-wave stack at the design energy and direction; Bloch dispersion from the period's
    # 2x2 transfer trace, independent of the recursion.
    eps_a, eps_b, energy, sin_t = 2.1, 6.0, 2.0, 0.4
    k = energy * sin_t
    lam_a, lam_b = (np.sqrt(e * energy**2 - k**2) for e in (eps_a, eps_b))
    d_a, d_b = np.pi / (2 * lam_a), np.pi / (2 * lam_b)
    y_a, y_b = (lam_a, lam_b) if polarization == 0 else (eps_a / lam_a, eps_b / lam_b)
    trace = np.cos(lam_a * d_a) * np.cos(lam_b * d_b) - 0.5 * (y_a / y_b + y_b / y_a) * np.sin(
        lam_a * d_a
    ) * np.sin(lam_b * d_b)
    assert abs(trace) > 1.0
    per_period = abs(trace) - np.sqrt(trace**2 - 1.0)
    layers = [(eps_a + 0.0j, d_a * HBARC_EV_ANG), (eps_b + 0.0j, d_b * HBARC_EV_ANG)] * 40
    n = np.array([sin_t, 0.0, np.sqrt(1 - sin_t**2)])
    stack = interface_response(energy, n, layers, 1.0)
    # (E_y -> E_y) is TE and (E_x -> E_x) is TM for k_perp along x.
    component = stack.response[0, :, 1, 1] if polarization == 0 else stack.response[0, :, 0, 0]
    measured = np.abs(component[40] / component[42])
    np.testing.assert_allclose(measured, per_period, rtol=1e-6)


def test_weak_contrast_periodic_stack_peaks_at_the_resonance_condition():
    # Born limit of a periodic stack: crossings add in phase when (q - k_z) * period = 2 pi m.
    beta, period, m = 0.99, 2e4, 3
    eps_a, eps_b = 1.0002, 1.0
    n_z = 0.9
    eps_mean = 0.5 * (eps_a + eps_b)
    expected = 2 * np.pi * m * HBARC_EV_ANG / (period * (1 / beta - np.sqrt(eps_mean - 1 + n_z**2)))
    energy = np.linspace(0.96 * expected, 1.04 * expected, 1601)
    n = np.array([np.sqrt(1 - n_z**2), 0.0, n_z])
    layers = [(eps_a + 0.0j, period / 2), (eps_b + 0.0j, period / 2)] * 200
    # m odd keeps the first-harmonic form factor of a half-filled period nonzero.
    spectrum = planar_transition_radiation(energy, n, np.array([0.0, 0.0, beta]), layers)
    assert abs(energy[np.argmax(spectrum)] / expected - 1.0) < 1e-3
