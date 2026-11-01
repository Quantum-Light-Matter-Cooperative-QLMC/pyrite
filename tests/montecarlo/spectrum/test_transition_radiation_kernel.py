"""Host build of the fused CUDA TR kernel's arithmetic against the NumPy amplitudes (#103).

``transition_radiation_jit_kernel.SOURCE`` compiles as host C++ with a
sequential entry point over the same per-crossing and per-history routines the
device kernel calls, so its physics is checked where no GPU is present. The
device launch, reduction and dispatch are covered in
``test_transition_radiation_cuda.py``.

Validation: tr-crossing-superposition
"""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.transition_radiation import (
    composition_dielectric,
    interface_response,
)
from pyrite.montecarlo.spectrum.transition_radiation_crossings import (
    _emission_frame,
    _take,
    boundary_crossings,
    crossing_amplitudes,
)
from pyrite.montecarlo.spectrum.transition_radiation_jit_kernel import _COUPLING, SOURCE
from tests.montecarlo.spectrum.test_transition_radiation_crossings import _BILAYER, _run

_CXX = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
pytestmark = pytest.mark.skipif(_CXX is None, reason="no host C++ compiler")


@pytest.fixture(scope="module")
def host_sums(tmp_path_factory):
    build = tmp_path_factory.mktemp("tr_kernel")
    (build / "tr.cpp").write_text(SOURCE)
    library = build / "tr.so"
    subprocess.run(
        [_CXX, "-std=c++17", "-O1", "-shared", "-fPIC", "-DPYRITE_TR_HOST",
         str(build / "tr.cpp"), "-o", str(library)],
        check=True, capture_output=True,
    )  # fmt: skip
    entry = ctypes.CDLL(str(library)).pyrite_tr_host_sums
    entry.restype = None

    def run(stack, frame, crossings, starts):
        def ptr(array, dtype):
            array = np.ascontiguousarray(array, dtype=dtype)
            keep.append(array)
            return array.ctypes.data_as(ctypes.c_void_p)

        keep = []
        pairs = stack.omega.size
        out = np.zeros((pairs, 7))
        entry(
            ctypes.c_int(pairs), ctypes.c_int(stack.response.shape[1]),
            ctypes.c_int(starts.size - 1),
            ptr(stack.omega, float), ptr(stack.k_perp, float), ptr(stack.eps, complex),
            ptr(stack.response, complex), ptr(stack.lam_top, complex),
            ptr(frame.velocity, float), ptr(crossings.time_ang, float),
            ptr(frame.position[:, :2], float), ptr(crossings.charge, float),
            ptr(frame.face, np.int32), ptr(starts, np.int64),
            ctypes.c_double(_COUPLING), ctypes.c_double(HBARC_EV_ANG),
            out.ctypes.data_as(ctypes.c_void_p),
        )  # fmt: skip
        return out[:, 0], out[:, 1::2] + 1j * out[:, 2::2]

    return run


@pytest.fixture(scope="module")
def bilayer_crossings():
    crossings = boundary_crossings(_run(_BILAYER), _BILAYER)
    return _take(crossings, np.argsort(crossings.electron_id, kind="stable"))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_host_kernel_sums_match_the_numpy_amplitudes(host_sums, bilayer_crossings, sign):
    crossings = bilayer_crossings
    energy = np.geomspace(100.0, 3000.0, 12)[:, None]
    theta = np.linspace(0.05, 1.2, 3)
    n = np.stack([np.sin(theta), 0.2 * np.ones_like(theta), sign * np.cos(theta)], -1)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    eps = [composition_dielectric(comp, energy) for _, _, comp in _BILAYER]
    frame = _emission_frame(crossings, n[None], eps)
    stack = interface_response(energy, frame.direction, frame.layers, 1.0)
    history = crossings.electron_id
    starts = np.r_[np.flatnonzero(np.r_[True, history[1:] != history[:-1]]), history.size]
    assert starts.size > 30 and np.any(np.diff(starts) > 2)  # multi-crossing histories
    power, field = host_sums(stack, frame, crossings, starts)

    amplitude = crossing_amplitudes(crossings, energy, n[None], eps).reshape(
        stack.omega.size, -1, 3
    )
    per_history = np.add.reduceat(amplitude, starts[:-1], axis=1)
    expected_power = np.sum(np.abs(per_history) ** 2, axis=(1, 2))
    np.testing.assert_allclose(power, expected_power, rtol=1e-11)
    expected_field = np.sum(amplitude, axis=1)
    np.testing.assert_allclose(
        field * frame.flip, expected_field, rtol=1e-10, atol=1e-12 * np.max(np.abs(expected_field))
    )
