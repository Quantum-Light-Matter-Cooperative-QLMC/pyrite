"""Published CTR population comparison: Schroeder et al., PRE 69, 016501 (2004), Fig. 3.

A fully coherent bunch (``|F|^2 = 1``) with a 5-MeV-temperature Boltzmann
momentum distribution, ``g(u) = exp(-u/u_t)/u_t`` with ``u = gamma beta`` and
``u_t = 10``, leaves a perfect conductor into vacuum. The paper's Eq. (22) at
``F = 1`` is ``(e^2/pi^2 c) N (N - 1) (|<E_par>|^2 + |<E_perp>|^2)`` with the
single-electron amplitudes of Eqs. (17)/(18). Fig. 3 plots the square root of
that normalized quantity, ``sin(theta)|<...>|``: the dashed curve matches the
square root of Eq. (36) with exponent 0.992 and scale 0.997. Fig. 2 (incoherent)
is reproduced by the same ``g(u)`` as an energy, which fixes ``g``.

Here the population runs through the production crossing scorer: one
conductor-to-vacuum exit crossing per sampled electron, equal-weight
quantile samples of ``u`` and of ``sin^2 psi ~ Exp(<psi^2>)`` with uniform
azimuth, and a point Gaussian source. The normalized distinct-pair yield
``pi^2 E pair / alpha`` is the paper's ``|<E>|^2`` quantity (photons/eV/sr per
electron to ``d^2W/(domega dOmega)`` in Gaussian units: ``x pi^2 hbar E /
(e^2/c)``).

Data: ``tests/data/optical/schroeder-2004-fig3.json`` (digitized by the
repository owner). The divergent (solid) curve departs below 0.08 rad by up to
18 %; an independent quadrature of the paper's own Eqs. (17)/(18) departs from
it identically, so that part is a figure/equation discrepancy of the source and
is excluded.

Validation: optical-ctr-population
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from pyrite.materials.crystal import ALPHA_FS, M_E_EV
from pyrite.montecarlo.spectrum.optical_transition import (
    OpticalBunchSource,
    optical_transition_radiation,
)
from pyrite.montecarlo.spectrum.transition_radiation_crossings import BoundaryCrossings

_DATA = json.loads(
    (Path(__file__).parents[2] / "data" / "optical" / "schroeder-2004-fig3.json").read_text()
)
_UT = _DATA["u_t"]
_ENERGY = 2.0  # eV; the conductor yield per eV is energy independent after pi^2 E / alpha
_CONDUCTOR = [np.array(-1e12 + 1e9j)]


def _exit_crossings(n_u, rms_psi, n_psi=8, n_phi=8):
    """Conductor-to-vacuum exit crossings of quantile-sampled electrons."""
    u = -_UT * np.log1p(-(np.arange(n_u) + 0.5) / n_u)
    if rms_psi == 0.0:
        sin_psi, phi = np.zeros(1), np.zeros(1)
    else:
        x = -np.log1p(-(np.arange(n_psi) + 0.5) / n_psi)
        sin_psi, phi = rms_psi * np.sqrt(x), 2 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    u, s, p = (a.ravel() for a in np.meshgrid(u, sin_psi, phi, indexing="ij"))
    m = u.size
    return m, BoundaryCrossings(
        electron_id=np.arange(m), track_id=np.arange(m), charge=-np.ones(m, np.int8),
        position_ang=np.tile([0.0, 0.0, 1e3], (m, 1)),
        direction=np.stack([s * np.cos(p), s * np.sin(p), np.sqrt(1 - s**2)], -1),
        kinetic_energy_keV=M_E_EV * (np.sqrt(1 + u**2) - 1) / 1e3, time_ang=np.zeros(m),
        interface=np.ones(m, int), normal=np.tile([0.0, 0.0, 1.0], (m, 1)),
        layer_before=np.zeros(m, int), layer_after=-np.ones(m, int),
        z_interfaces_ang=np.array([0.0, 1e3]), n_matched_dropped=0, n_side_exits=0,
    )  # fmt: skip


def _normalized(theta, rms_psi, n_u):
    """``{pi^2 c/[e^2 N(N-1)]} d^2W_CTR/(domega dOmega)`` and the self analogue."""
    m, crossings = _exit_crossings(n_u, rms_psi)
    n = np.stack([np.sin(theta), 0 * theta, np.cos(theta)], -1)
    point = OpticalBunchSource.gaussian(1e3, "point bunch", rms_t_fs=1e-6)
    result = optical_transition_radiation(
        crossings, _ENERGY, n, _CONDUCTOR, source=point, n_histories=m
    )
    scale = np.pi**2 * _ENERGY / ALPHA_FS
    return result.pair_yield * scale, result.self_yield * scale


def _curve(name):
    data = np.array(_DATA[name]["theta_rad_ordinate"])
    keep = data[:, 0] > 0.01  # the first points sit on the axis line
    return data[keep, 0], data[keep, 1], _DATA[name]["rms_divergence_rad"]


def test_collimated_population_reproduces_eq36_and_fig3():
    theta, plotted, rms = _curve("dashed")
    pair, self_ = _normalized(theta, rms, n_u=4000)

    def g(u):
        return np.exp(-u / _UT) / _UT

    eq36 = np.array([
        np.sin(t) ** 2
        * quad(lambda u, t=t: g(u) * u * np.sqrt(1 + u * u) / (1 + (u * np.sin(t)) ** 2),
               0, np.inf, limit=400)[0] ** 2
        for t in theta
    ])  # fmt: skip
    np.testing.assert_allclose(pair, eq36, rtol=1e-2)  # quantile sampling + 1/M pair bias
    relative = np.sqrt(pair) / plotted - 1
    assert np.sqrt(np.mean(relative**2)) < 0.012 and np.max(np.abs(relative)) < 0.05
    # Coherence lowers the per-electron yield: |<S>|^2 <= <|S|^2> (Cauchy-Schwarz).
    assert np.all(pair <= self_ * (1 + 1e-12))


def test_divergent_population_reproduces_fig3_above_the_divergence():
    theta, plotted, rms = _curve("solid")
    pair, _ = _normalized(theta, rms, n_u=200)
    relative = np.sqrt(pair) / plotted - 1
    wide = theta >= 0.08
    assert np.max(np.abs(relative[wide])) < 0.015
    # Below, the figure exceeds the paper's own Eqs. (17)/(18) (module docstring).
    assert np.all(relative[theta < 0.04] < -0.08)
    collimated, _ = _normalized(theta, 0.0, n_u=200)
    assert np.all(pair[theta < 0.05] < collimated[theta < 0.05])


@pytest.mark.parametrize("rms_psi", [0.0, 0.05])
def test_population_is_converged_in_the_sample_count(rms_psi):
    theta = np.array([0.05, 0.1, 0.3])
    coarse, _ = _normalized(theta, rms_psi, n_u=200)
    fine, _ = _normalized(theta, rms_psi, n_u=400)
    np.testing.assert_allclose(coarse, fine, rtol=0.02)
