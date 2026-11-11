"""Deterministic checks of the W <001> finite-beta audit algebra."""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.optimize import brentq

from checks import channeling_submev_w001_kinematics as kin
from pyrite.materials.crystal import M_E_EV


@dataclass(frozen=True)
class _Profile:
    potential_eV: np.ndarray
    max_order: tuple[int, int]
    g_inv_ang: tuple[float, float] = (1.0, 2.0)


def _synth(coefficients, x, y, g):
    m, n = (s // 2 for s in coefficients.shape)
    ex = np.exp(1j * g[0] * np.multiply.outer(x, np.arange(-m, m + 1)))
    ey = np.exp(1j * g[1] * np.multiply.outer(y, np.arange(-n, n + 1)))
    return np.einsum("im,mn,jn->ij", ex, coefficients, ey).real


def test_kg_profile_is_scaled_u_minus_u_squared_over_2e_with_zero_mean():
    rng = np.random.default_rng(5)
    raw = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    u = raw + np.conj(raw[::-1, ::-1])  # Hermitian: real U(x, y).
    u[2, 1] = 0.0
    total, mean = 50.0, -7.0
    out = kin.kg_profile(_Profile(u, (2, 1)), total, mean)
    assert out.max_order == (4, 2)
    x = np.linspace(0.0, 2 * np.pi, 17, endpoint=False)
    y = np.linspace(0.0, np.pi, 13, endpoint=False)
    plain = _synth(u, x, y, (1.0, 2.0))
    expected = plain * (1 - mean / total) - plain**2 / (2 * total)
    expected -= expected.mean()
    assert _synth(out.potential_eV, x, y, (1.0, 2.0)) == pytest.approx(expected, abs=1e-12)


def test_exact_root_matches_independent_solve_for_fixed_levels(monkeypatch):
    lower, upper = -300.0, -70.0
    monkeypatch.setattr(kin, "pair_energies", lambda *args, **kwargs: (lower, upper))
    total = M_E_EV + 30e3

    def cpz(e, eps):
        return np.sqrt(e**2 - M_E_EV**2 - 2 * e * eps)

    reference = brentq(lambda w: cpz(total, upper) - cpz(total - w, lower) - w, 1.0, 1e3)
    got = kin.exact_forward_line(None, total, (1, 1), kg=False)
    assert got == pytest.approx(reference, rel=1e-10)
    beta = np.sqrt(1 - (M_E_EV / total) ** 2)
    # First-order Doppler is the small-gap limit; the correction is O(hw/E).
    doppler = (upper - lower) / (1 - beta)
    assert abs(got / doppler - 1) < 5 * doppler / total
