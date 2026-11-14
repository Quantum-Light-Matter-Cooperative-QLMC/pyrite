"""Single-event weight checks against isotropic spherical-cap integrals.

Implementation-context regression only; not independent estimator validation.
"""

import numpy as np
import pytest
from numba import njit

from pyrite.montecarlo.transport.cone_split import _event_key, cone_split_event


@njit(cache=True)
def _isotropic_trials(direction, theta, target, count):
    mu = np.array([0.0, 1.0])
    cdf = np.array([[0.0, 1.0], [0.0, 1.0]])
    pdf = np.ones((2, 2))
    guide = np.zeros((2, 2), dtype=np.int32)
    energies = np.log(np.array([1000.0, 10000.0]))
    scores = np.zeros((count, 3))
    for i in range(count):
        made, x, y, z, weight = cone_split_event(
            direction[0],
            direction[1],
            direction[2],
            0.0,
            0.0,
            1.0,
            np.cos(theta),
            np.sin(theta),
            target,
            _event_key(np.uint64(918273), i),
            5.0,
            energies,
            cdf,
            pdf,
            mu,
            guide,
            0,
            2,
        )
        if made:
            if weight <= 0.0 or abs(x * x + y * y + z * z - 1.0) >= 1e-12:
                raise ValueError("invalid copy weight or norm")
            if z < np.cos(theta) - 1e-12:
                raise ValueError("copy outside cap")
            scores[i, 0] = weight
            scores[i, 1] = weight * z
            scores[i, 2] = weight * (z >= np.cos(theta / 2.0))
    return scores


@pytest.mark.parametrize("direction", [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)])
@pytest.mark.parametrize("target", [0.001, 0.2])
def test_single_event_weights_preserve_isotropic_cap_integrals(direction, target):
    theta = 0.4
    scores = _isotropic_trials(np.asarray(direction), theta, target, 8192)
    # Integrals of 1, cos(theta_out), and a smaller cap under dOmega/(4*pi).
    expected = np.array(
        [
            (1.0 - np.cos(theta)) / 2.0,
            np.sin(theta) ** 2 / 4.0,
            (1.0 - np.cos(theta / 2.0)) / 2.0,
        ]
    )
    se = scores.std(axis=0, ddof=1) / np.sqrt(scores.shape[0])
    assert np.all(np.abs(scores.mean(axis=0) - expected) <= 6.0 * se + 1e-14)
