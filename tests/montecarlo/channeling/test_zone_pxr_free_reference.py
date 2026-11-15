"""Zone PXR source against an independent free-electron Born reference.

A free plane wave entering at true transverse momentum ``p`` (along the plane
normal) is a free electron moving along ``v' = (p/P, 0, sqrt(1 - (p/P)^2))``.
Its first-order Born PXR line in a ``chi_g`` slab is built here by hand: the
exact-recoil root along ``v'``, the vertex ``beta v'.F_s`` and the truncated
line ``phi_L(Delta) / (det + 2 Q_s Delta / s)``. Nothing in the reference uses
``zone_pxr`` beyond the production call under test, so it pins the pre-crystal
vertex, the ``p``--``Q_x`` sign and the per-state entrance weights (#1148
fresh-context verification 2026-11-14), 12 eV from a Bragg pole where the
earlier folded-momentum (2--17%) and entry-weight-only (3.9%) errors show.

The runner then has to hand the kernel an in-plane beam: a tilt off the plane
is carried by ``p`` alone, so a tilted ``SlabGeometry`` would count it twice.

Validation: cr-planar-zone-pxr-modulation
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG, M_E_EV
from pyrite.materials.recoil import recoil_resonance_eV
from pyrite.montecarlo.channeling import zone_pxr as zp

KINETIC_EV = 17e6
GAMMA = 1.0 + KINETIC_EV / M_E_EV
BETA = float(np.sqrt(1.0 - 1.0 / GAMMA**2))
PERIOD = 1.920113109
G_PLANE = 2.0 * np.pi / PERIOD
LENGTH = 20000.0
CHI_G = 4e-6
P_MOMENTUM = GAMMA * BETA * M_E_EV / HBARC_EV_ANG  # 1/Angstrom
N_HAT = np.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
GEO = zp.SlabGeometry(
    beam=np.array([0.0, 0.0, 1.0]),
    x_hat=np.array([1.0, 0.0, 0.0]),
    s_hat=np.array([0.0, 0.0, 1.0]),
    path_ang=LENGTH,
)
ORDERS = np.arange(-8, 9)


def _velocity(p):
    return np.array([p / P_MOMENTUM, 0.0, np.sqrt(1.0 - (p / P_MOMENTUM) ** 2)])


@pytest.fixture(scope="module")
def near_pole_g():
    """``g = (-a, 0, 1.19)`` with the free root closest to its Bragg pole (~12 eV)."""

    def offset(a):
        g = np.array([-a, 0.0, 1.19])
        E0 = float(recoil_resonance_eV(KINETIC_EV, GEO.beam, g, N_HAT))
        return -HBARC_EV_ANG * (g @ g) / (2.0 * N_HAT @ g) - E0

    scan = np.linspace(2.5, 2.95, 451)
    a = scan[int(np.argmin([offset(a) for a in scan]))]
    assert 0.0 < offset(a) < 20.0
    return np.array([-a, 0.0, 1.19])


def _born_band(p, g, E):
    """Hand-built kinematic Born PXR band of a free electron along ``v'(p)``."""
    v = _velocity(p)
    E0 = float(recoil_resonance_eV(KINETIC_EV, v, g, N_HAT))
    k = E / HBARC_EV_ANG
    Q = k[:, None] * N_HAT + g
    det = g @ g + 2.0 * k * (N_HAT @ g)
    delta = (1.0 - BETA * v @ N_HAT) * (E - E0) / (HBARC_EV_ANG * BETA)
    safe = np.where(delta == 0.0, 1.0, delta)
    phi = np.where(delta == 0.0, LENGTH, (np.exp(1j * delta * LENGTH) - 1.0) / (1j * safe))
    e_s = np.cross(N_HAT, g)
    e_s /= np.linalg.norm(e_s)
    total = np.zeros(E.size)
    for e in (e_s, np.cross(e_s, N_HAT)):
        F = Q * (g @ e) - (k**2)[:, None] * e
        amp = CHI_G * BETA * (F @ v) * phi / (det + 2.0 * Q[:, 2] * delta / v[2])
        total += np.abs(amp) ** 2
    prefactor = zp.ALPHA_FS * k / (4.0 * np.pi**2 * HBARC_EV_ANG * BETA**2)
    return np.trapezoid(prefactor * total, E)


def _production_band(states, g, E):
    E0 = float(recoil_resonance_eV(KINETIC_EV, GEO.beam, g, N_HAT))
    spectrum = zp.reflection_spectrum(states, GEO, g, N_HAT, E0, E, CHI_G, dynamical=False)
    return np.trapezoid(spectrum, E)


def _grid(g, *p):
    roots = [float(recoil_resonance_eV(KINETIC_EV, _velocity(v), g, N_HAT)) for v in p]
    return np.arange(min(roots) - 300.0, max(roots) + 300.0, 0.02)


@pytest.mark.parametrize("order", [-3, -1, 1, 2, 4])
def test_free_entry_matches_the_tilted_free_electron_born_band(near_pole_g, order):
    p = 0.3 + order * G_PLANE
    states = zp.free_states(ORDERS, PERIOD, GAMMA, kappa=p)
    states = replace(states, entrance=states.populations)
    E = _grid(near_pole_g, p)
    got = _production_band(states, near_pole_g, E)
    assert got == pytest.approx(_born_band(p, near_pole_g, E), rel=3e-3)


def test_mixed_entry_matches_the_incoherent_tilted_born_sum(near_pole_g):
    p1, p2 = 0.3, 0.3 + 2.0 * G_PLANE
    w = np.where(ORDERS == 0, 0.7, 0.0) + np.where(ORDERS == 2, 0.3, 0.0)
    states = replace(
        zp.free_states(ORDERS, PERIOD, GAMMA, kappa=0.3), populations=w[None], entrance=w[None]
    )
    E = _grid(near_pole_g, p1, p2)
    expected = 0.7 * _born_band(p1, near_pole_g, E) + 0.3 * _born_band(p2, near_pole_g, E)
    assert _production_band(states, near_pole_g, E) == pytest.approx(expected, rel=3e-3)
