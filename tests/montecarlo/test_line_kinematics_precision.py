"""Float32 line kinematics at tens of MeV (issue #439).

``_vacuum_doppler`` and ``_in_medium_kinematics`` must give float32 results
within a few float32 ulps of the float64 ones at 30 and 100 MeV, including a
detector near the beam direction where ``1 - beta v_hat.n_hat`` approaches
``1/(2 gamma^2)``. The naive forms (``gamma`` from ``beta``, ``1 - beta cos``)
are run alongside to show the cancellation the regression guards against.

Validation: line-kinematics-precision
"""

import numpy as np
import pytest

from pyrite.materials.crystal import M_E_EV
from pyrite.montecarlo.spectrum.cxr._kernels import _in_medium_kinematics, _vacuum_doppler

HBARC_EV_ANG = 1973.269804
# Observation angles from the beam: forward (1 mrad), the Cherenkov-like
# 1/gamma scale and a typical backward PXR geometry.
ANGLES_RAD = np.array([1e-3, 5e-3, 0.3, 2.0])


def _beta(E_keV, dtype):
    gamma = 1.0 + np.asarray(E_keV, dtype=dtype) / dtype(M_E_EV / 1e3)
    return np.sqrt(1.0 - 1.0 / (gamma * gamma)).astype(dtype)


def _rows(E_keV, dtype):
    v_hat = np.tile(np.array([0.0, 0.0, 1.0]), (ANGLES_RAD.size, 1))
    n_hat = np.stack([np.sin(ANGLES_RAD), np.zeros_like(ANGLES_RAD), np.cos(ANGLES_RAD)], axis=1)
    energy = np.full(ANGLES_RAD.size, E_keV)
    return energy.astype(dtype), v_hat.astype(dtype), n_hat.astype(dtype)


def _doppler(E_keV, dtype):
    energy, v_hat, n_hat = _rows(E_keV, dtype)
    beta = _beta(energy, dtype)
    gammas, denoms = [], []
    for row in range(ANGLES_RAD.size):
        g, d = _vacuum_doppler(
            energy[row : row + 1], beta[row : row + 1], v_hat[row : row + 1], n_hat[row], xp=np
        )
        gammas.append(g[0])
        denoms.append(d[0])
    return np.array(gammas), np.array(denoms)


def _exact(E_keV):
    gamma = 1.0 + E_keV / (M_E_EV / 1e3)
    one_minus_beta = 1.0 / (gamma * gamma * (1.0 + np.sqrt(1.0 - 1.0 / gamma**2)))
    beta = 1.0 - one_minus_beta
    return gamma, one_minus_beta + beta * 2.0 * np.sin(ANGLES_RAD / 2.0) ** 2


@pytest.mark.parametrize("E_keV", [3.0e4, 1.0e5], ids=["30MeV", "100MeV"])
def test_float32_doppler_factor_matches_float64(E_keV):
    gamma32, denom32 = _doppler(E_keV, np.float32)
    gamma64, denom64 = _doppler(E_keV, np.float64)
    gamma_ref, denom_ref = _exact(E_keV)
    np.testing.assert_allclose(gamma64, gamma_ref, rtol=1e-14)
    np.testing.assert_allclose(denom64, denom_ref, rtol=1e-12)
    eps32 = float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(gamma32, gamma64, rtol=4 * eps32)
    np.testing.assert_allclose(denom32, denom64, rtol=16 * eps32)


@pytest.mark.parametrize("E_keV", [3.0e4, 1.0e5], ids=["30MeV", "100MeV"])
def test_naive_float32_forms_lose_the_digits_the_guard_protects(E_keV):
    """The control: what the cancellation-free forms replace is visibly worse."""
    energy, v_hat, n_hat = _rows(E_keV, np.float32)
    beta = _beta(energy, np.float32)
    naive_gamma = 1.0 / np.sqrt(np.float32(1.0) - beta * beta)
    naive_denom = np.float32(1.0) - beta * np.sum(v_hat * n_hat, axis=1, dtype=np.float32)
    gamma_ref, denom_ref = _exact(E_keV)
    worst = max(
        float(np.max(np.abs(naive_gamma / gamma_ref - 1.0))),
        float(np.abs(naive_denom[0] / denom_ref[0] - 1.0)),
    )
    assert worst > 100 * float(np.finfo(np.float32).eps)


@pytest.mark.parametrize("E_keV", [3.0e4, 1.0e5], ids=["30MeV", "100MeV"])
def test_float32_in_medium_root_matches_float64(E_keV):
    """Forward Snell root with a Si-like decrement ``delta(E) ~ 5e-6 (10 keV/E)^2``.

    The line sits at 50 keV, where ``delta`` (2e-7) is small against the 1 mrad
    Doppler factor (~1e-5 at 100 MeV) and the fixed point contracts. Softer
    forward lines at 100 MeV have ``delta ~ D`` and are rejected as
    non-converged instead, a solver limit outside this precision check.
    """
    E_tab = np.geomspace(1e3, 1e6, 64)
    delta_tab = 5e-6 * (1e4 / E_tab) ** 2  # Thomson-like delta ~ 1/E^2
    roots = {}
    for dtype in (np.float32, np.float64):
        _, denom_vac = _doppler(E_keV, dtype)
        energy, v_hat, n_hat = _rows(E_keV, dtype)
        beta = _beta(energy, dtype)
        # A g along the beam with the vacuum line at 50 keV in the 1 mrad row.
        v_dot_g = (5e4 / HBARC_EV_ANG) * denom_vac[0] * np.ones_like(denom_vac)
        v_dot_grad = -(beta * np.sum(v_hat * n_hat, axis=1)).astype(dtype)  # normal exit
        denom, _ = _in_medium_kinematics(
            denom_vac.astype(dtype),
            v_dot_g.astype(dtype),
            delta_tab.astype(dtype),
            E_tab.astype(dtype),
            v_dot_grad,
            np.ones_like(denom_vac),
        )
        roots[dtype] = (
            HBARC_EV_ANG * v_dot_g.astype(np.float64) / np.asarray(denom, dtype=np.float64)
        )
    finite = np.isfinite(roots[np.float64])
    assert finite[0]
    np.testing.assert_allclose(roots[np.float32][finite], roots[np.float64][finite], rtol=1e-5)
