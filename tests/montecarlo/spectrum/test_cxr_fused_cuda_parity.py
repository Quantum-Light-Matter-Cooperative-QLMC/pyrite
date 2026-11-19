"""Bitwise parity of the fused CXR line kernels with their CuPy chains (#906).

Each ``_fused`` kernel replaces a chain of float32 CuPy elementwise calls and
claims bit-for-bit equality; NaN, infinity, zero and table-node inputs
exercise the bracket, guard and select branches.
"""

import numpy as np
import pytest

from pyrite._backend import BACKEND, CUPY_BACKENDS
from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum import segment_escape
from pyrite.montecarlo.spectrum.cxr import _fused, _kernels, _line_amplitude

cp = pytest.importorskip("cupy")

try:
    _NDEV = int(cp.cuda.runtime.getDeviceCount())
except Exception:  # pragma: no cover - depends on CUDA runtime presence
    _NDEV = 0

pytestmark = [
    pytest.mark.skipif(_NDEV < 1, reason="CUDA device required"),
    pytest.mark.skipif(
        BACKEND.name not in CUPY_BACKENDS, reason="run with PYRITE_TEST_BACKEND=cuda or rocm"
    ),
]

_N, _NG = 20_000, 6
_RNG = np.random.default_rng(906)
_E_TAB = np.unique(
    np.concatenate([np.linspace(500.0, 2000.0, 1500), [1000.0, np.nextafter(1000.0, 2000.0)]])
).astype(np.float32)


def _f32(values):
    return cp.asarray(np.asarray(values, dtype=np.float32))


def _spiky(shape, lo, hi):
    """Uniform float32 values with NaN, +inf, 0 and 1 sprinkled in."""
    values = _RNG.uniform(lo, hi, shape).astype(np.float32)
    flat = values.reshape(-1)
    k = max(1, flat.size // 50)
    for special in (np.nan, 0.0, np.inf, 1.0):
        flat[_RNG.integers(0, flat.size, k)] = special
    return cp.asarray(values)


def _assert_bitwise(actual, expected):
    actual, expected = cp.asnumpy(actual), cp.asnumpy(expected)
    assert actual.shape == expected.shape
    if actual.dtype.kind == "f":
        nan = np.isnan(expected)
        np.testing.assert_array_equal(np.isnan(actual), nan)
        np.testing.assert_array_equal(actual[~nan].view(np.uint32), expected[~nan].view(np.uint32))
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.fixture
def chain(monkeypatch):
    """Force every ``_fused`` entry point onto its original CuPy chain."""
    return lambda: monkeypatch.setattr(_fused, "_float32_cupy", lambda *a: False)


@pytest.mark.parametrize("with_grad2", [True, False])
def test_in_medium_root_matches_chain(chain, with_grad2):
    E_tab = cp.asarray(_E_TAB)
    delta_tab = _f32(_RNG.uniform(1e-6, 5e-5, _E_TAB.size))
    denom_vac = _f32(_RNG.uniform(1e-3, 0.5, (_N, 1)))
    v_dot_g = _spiky((_N, _NG), -0.15, 0.3)
    v_dot_grad = _f32(_RNG.uniform(-2.0, 2.0, (_N, 1)))
    grad2 = _f32(_RNG.uniform(0.0, 3.0, (_N, 1))) if with_grad2 else None
    args = (denom_vac, v_dot_g, delta_tab, E_tab, v_dot_grad, grad2)
    fused = _kernels._in_medium_kinematics(*args)
    chain()
    for actual, expected in zip(fused, _kernels._in_medium_kinematics(*args), strict=True):
        _assert_bitwise(actual, expected)


def test_line_bracket_matches_chain():
    E_tab = cp.asarray(_E_TAB)
    omega = cp.asnumpy(_spiky((_N, _NG), 0.0, 2200.0 / HBARC_EV_ANG))
    nodes = _RNG.integers(0, omega.size, omega.size // 10)
    omega.reshape(-1)[nodes] = _RNG.choice(_E_TAB, nodes.size) / np.float32(HBARC_EV_ANG)
    omega = cp.asarray(omega)
    delta = _spiky((_N, _NG), -1e-4, 1e-4)
    grad2 = _f32(_RNG.uniform(0.0, 3.0, (_N, 1)))
    grad2[::7] = 0.0
    electron = cp.asarray(_RNG.random((_N, 1)) > 0.1)
    lo_keep, hi_keep = 700.0 + 1e-9, 1050.0 + 3e-9
    fused = _fused.line_bracket(
        omega, delta, grad2, electron, E_tab, hbarc=HBARC_EV_ANG, lo_keep=lo_keep, hi_keep=hi_keep
    )
    k_mag = omega * cp.sqrt(cp.where(grad2 == 0.0, 1.0, 1.0 - 2.0 * delta + delta**2 * grad2))
    E_res = HBARC_EV_ANG * omega
    keep = electron & (E_res > lo_keep) & (E_res > 10.0) & (E_res < hi_keep)
    idx, frac, below, above = _kernels._interp_index(E_res, E_tab)
    log_frac = _kernels._log_interp_fraction(E_res, E_tab, idx)
    expected = (k_mag, E_res, keep, idx, frac, log_frac, below, above)
    for actual, want in zip(fused, expected, strict=True):
        _assert_bitwise(actual, want)


def test_polarization_and_weights_match_chain():
    vx, vy, vz, g0, g1, g2 = (_spiky((_N, 1), -1.0, 1.0) for _ in range(6))
    ex, ey, ez = (_spiky((1, _NG), -1.0, 1.0) for _ in range(3))
    omega, delta = _spiky((_N, _NG), 0.0, 1.0), _spiky((_N, _NG), -1e-4, 1e-4)
    v_dot_e, k_dot_e = _fused.polarization_terms(vx, vy, vz, ex, ey, ez, g0, g1, g2, omega, delta)
    _assert_bitwise(v_dot_e, vx * ex + vy * ey + vz * ez)
    _assert_bitwise(k_dot_e, omega * delta * (g0 * ex + g1 * ey + g2 * ez))

    denom, pref, A2 = (
        _spiky((_N, _NG), -1, 1),
        _spiky((_N, _NG), -1e-3, 1e-2),
        _spiky((_N, _NG), 0, 9),
    )
    t_L, WM = _spiky((_N, 1), 0.0, 1e4), _f32(_RNG.uniform(0.0, 1.0, (1, _NG)))
    keep = cp.asarray(_RNG.random((_N, _NG)) > 0.3)
    a_width, weight, good = _fused.line_weights(
        denom, t_L, pref, A2, WM, keep, two_hbarc=2.0 * HBARC_EV_ANG
    )
    want_weight = pref * A2 * WM
    _assert_bitwise(a_width, denom * t_L / (2.0 * HBARC_EV_ANG) * cp.ones_like(denom))
    _assert_bitwise(weight, want_weight)
    _assert_bitwise(good, keep & cp.isfinite(want_weight) & (want_weight > 0))


@pytest.mark.parametrize("n_pieces", [1, 3])
def test_piece_transmission_matches_chain(n_pieces):
    fraction = _f32(_RNG.uniform(0.0, 1.0, (_N, n_pieces)))
    path_start = _spiky((_N, n_pieces, 1), 0.0, 1e5)
    path_end = _spiky((_N, n_pieces, 1), 0.0, 1e5)
    path_end[::3] = path_start[::3] * np.float32(1.0000001)  # series branch
    mu = _spiky((_N, _NG), 0.0, 1e-3)
    piece_fraction = _f32(_RNG.uniform(0.0, 1.0, (_N, 1)))
    fused = _fused.piece_transmission(fraction, path_start, path_end, mu, piece_fraction)
    expected = segment_escape.piece_mean_transmission(
        fraction, path_start, path_end, mu[..., None], xp=cp
    )
    expected *= piece_fraction
    _assert_bitwise(fused, expected)


def _block_inputs():
    """Row (nb, 1) and reflection (n_g,) operands of one segment block."""
    row = lambda lo, hi: _spiky((_N, 1), lo, hi)  # noqa: E731
    col = lambda lo, hi: _spiky((_NG,), lo, hi)  # noqa: E731
    return {
        "vx": row(-1.0, 1.0),
        "vy": row(-1.0, 1.0),
        "vz": row(-1.0, 1.0),
        "gx": col(-3.0, 3.0),
        "gy": col(-3.0, 3.0),
        "gz": col(-3.0, 3.0),
        "g2": col(0.0, 20.0),
        "n_dot_g": col(-3.0, 3.0),
        "denom_row": row(1e-4, 1.0),
        "inv_E": row(0.0, 1e-3),
    }


def test_recoil_pair_terms_match_chain():
    a = _block_inputs()
    fused = _fused.recoil_pair_terms(*a.values())
    r_num = 0.5 * a["g2"] * a["inv_E"]
    r_den = a["n_dot_g"] * a["inv_E"]
    v_dot_g = (a["vx"] * a["gx"] + a["vy"] * a["gy"] + a["vz"] * a["gz"]) - r_num
    for actual, want in zip(fused, (r_num, r_den, a["denom_row"] + r_den, v_dot_g), strict=True):
        _assert_bitwise(actual, want)


def test_block_n_dot_g_matches_chain():
    a = _block_inputs()
    g0, g1, g2 = (_spiky((_N, 1), -2.0, 2.0) for _ in range(3))
    delta = _spiky((_N, _NG), -1e-4, 1e-4)
    fused = _fused.block_n_dot_g(g0, g1, g2, a["gx"], a["gy"], a["gz"], delta, a["n_dot_g"])
    want = a["n_dot_g"] + delta * (g0 * a["gx"] + g1 * a["gy"] + g2 * a["gz"])
    _assert_bitwise(fused, want)


def test_recoil_slope_keep_matches_chain():
    from pyrite.montecarlo.spectrum.cxr._recoil import _recoil_line_allowed, _recoil_slope_factor

    denom = _spiky((_N, _NG), -0.5, 1.0)
    omega = _spiky((_N, _NG), 0.0, 2.0)
    inv_E = _spiky((_N, 1), 0.0, 1e-3)
    gamma = _spiky((_N, 1), 1.0, 60.0)
    keep = cp.asarray(_RNG.random((_N, _NG)) > 0.3)
    slope, kept = _fused.recoil_slope_keep(denom, omega, inv_E, gamma, keep)
    _assert_bitwise(slope, denom * _recoil_slope_factor(omega, inv_E))
    _assert_bitwise(kept, keep & _recoil_line_allowed(denom, omega, inv_E, gamma))


def test_amplitude_only_kernel_matches_full_kernel():
    """``|A|^2`` alone equals the full kernel's first output bit for bit (#862)."""
    names = "chi_re chi_im u_re u_im g2 g_dot_e om v_dot_e vdg k_dot_g k_dot_v gamma".split()
    args = [_spiky((_N, _NG), 0.1, 3.0) for _ in names]
    args += [_spiky((_N, _NG), -1.0, 1.0), _spiky((_N, _NG), -1.0, 1.0), np.float32(3.86e-3)]
    alone = _line_amplitude._cxr_amp_sq_total_core(*args)
    full = _kernels._cxr_amp_sq_core(*args)
    _assert_bitwise(alone, full[0])
