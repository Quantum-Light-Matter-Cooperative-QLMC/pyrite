"""The fused CPU line reduction matches the dense sinc^2 contraction."""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.cxr._cpu_reduce import accumulate_sincsq
from pyrite.montecarlo.spectrum.cxr._kernels import _sincsq_lineshape


def _dense(E_r, a_width, weights, E_grid):
    S = np.pi * np.where(
        (x := a_width[:, None] * (E_grid[None, :] - E_r[:, None]) / np.pi) == 0.0, 1.0e-20, x
    )
    S = (np.sin(S) / S) ** 2
    return [w @ S for w in weights]


@pytest.mark.parametrize("n_targets", [1, 2, 3])
def test_matches_dense_contraction(n_targets):
    rng = np.random.default_rng(7)
    E_grid = np.linspace(1000.0, 20000.0, 97)
    E_r = rng.uniform(900.0, 20100.0, 513)
    E_r[0] = E_grid[11]  # exact resonance hits the x == 0 guard
    a_width = rng.uniform(1.0e-3, 0.5, E_r.size)
    weights = list(rng.random((n_targets, E_r.size)))
    targets = [np.full(E_grid.size, 2.0) for _ in range(n_targets)]

    accumulate_sincsq(E_r, a_width, weights, E_grid, targets)

    for got, ref in zip(targets, _dense(E_r, a_width, weights, E_grid), strict=True):
        np.testing.assert_allclose(got - 2.0, ref, rtol=1e-12, atol=0.0)


def test_profile_matches_lineshape_helper():
    E_grid = np.linspace(-3.0, 3.0, 13)
    E_r = np.array([0.0])
    a_width = np.array([2.0])
    target = [np.zeros(E_grid.size)]
    accumulate_sincsq(E_r, a_width, [np.ones(1)], E_grid, target)
    ref = _sincsq_lineshape(a_width[:, None], E_grid[None, :], E_r[:, None])[0]
    np.testing.assert_allclose(target[0], ref, rtol=1e-14, atol=0.0)


def test_empty_inputs_leave_targets_unchanged():
    target = [np.ones(4)]
    accumulate_sincsq(np.empty(0), np.empty(0), [np.empty(0)], np.arange(4.0), target)
    np.testing.assert_array_equal(target[0], np.ones(4))
