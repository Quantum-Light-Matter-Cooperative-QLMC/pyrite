"""Friedel-pair discrimination for the line resonance / coupling pairing.

A line at ``omega = v.g(hkl) / (1 - n_hat.v)`` emits ``k = k0 - g(hkl)``, so
its momentum transfer is ``-g(hkl)`` and, in the repository's ``exp(+i omega
t)`` / ``f' + i f''`` convention, its PXR coupling is ``chi(-hkl)``. Only a
noncentrosymmetric crystal with ``f'' != 0`` separates that from ``chi(+hkl)``.

The test point-reflects one straight segment (``v -> -v``, ``n_hat -> -n_hat``,
``hkl -> -hkl``, segment centred in the slab): every dot product, the line
energy, the escape path and ``U_g`` coupling are unchanged, so the PXR yield
ratio isolates the ``|chi|^2`` pairing.

Validation: line-energy-dispersion
"""

import numpy as np
import pytest

from pyrite.materials.crystal import (
    CRYSTALS,
    HBARC_EV_ANG,
    beta_from_Ee,
    chi_g,
    reciprocal_g_vector,
)
from pyrite.montecarlo import mc_spectrum

CRYSTAL = "4h_sic"  # P6_3mc: noncentrosymmetric, Si K-edge f'' near 3 keV
HKL = np.array([1, 0, 2])
E_KEV = 100.0
THICKNESS = 2000.0


def _geometry(tilt_deg):
    g_vec, g_mag = reciprocal_g_vector(HKL, CRYSTALS[CRYSTAL]["lattice"])
    g_hat = g_vec / g_mag
    perp = np.cross(g_hat, [0.0, 1.0, 0.0])
    perp /= np.linalg.norm(perp)
    tilt = np.radians(tilt_deg)
    v_hat = np.cos(tilt) * g_hat + np.sin(tilt) * perp
    n_hat = perp  # 90 deg from g, in the tilt plane
    return g_mag, g_hat, v_hat, n_hat


def _segment(v_hat):
    return {
        "r_mid": np.array([[0.0, 0.0, 0.5 * THICKNESS]]),
        "v_hat": v_hat[None, :],
        "L_ang": np.array([200.0]),
        "E_keV": np.array([E_KEV]),
        "t_ang": np.zeros(1),
        "t0_ang": np.zeros(1),
        "elec_id": np.zeros(1, dtype=int),
        "layer": np.zeros(1, dtype=int),
        "Ne": 1,
        "thickness_ang": THICKNESS,
        "crystal_width_ang": None,
        "crystal_height_ang": None,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }


@pytest.mark.parametrize("tilt_deg", [-12.0, 0.0, 12.0])
def test_line_couples_to_the_friedel_mate_of_its_resonant_harmonic(tilt_deg, monkeypatch):
    from pyrite.montecarlo.spectrum.lines import _setup

    # Vacuum index so the line centre is the closed-form vacuum root.
    monkeypatch.setattr(
        _setup,
        "refractive_index",
        lambda crystal, energy, *args: np.ones_like(energy, dtype=complex),
    )
    g_mag, g_hat, v_hat, n_hat = _geometry(tilt_deg)
    beta = float(beta_from_Ee(E_KEV * 1e3))
    # Independent closed form for the line centre (vacuum, straight flight).
    E_line = HBARC_EV_ANG * beta * g_mag * (g_hat @ v_hat) / (1.0 - beta * (n_hat @ v_hat))
    grid = np.linspace(0.9 * E_line, 1.1 * E_line, 4001)
    step = grid[1] - grid[0]

    def run(sign):
        hkl = tuple(int(sign * h) for h in HKL)
        return mc_spectrum(
            _segment(sign * v_hat),
            grid,
            crystal=CRYSTAL,
            hkl_list=[hkl],
            n_hat=sign * n_hat,
            B_ang2=0.5,
            use_henke=True,
            components=True,
        )

    total_p, pxr_p, cbs_p = run(+1)
    total_m, pxr_m, cbs_m = run(-1)

    # Line energy: closed form, and opposite tilts are genuinely distinct.
    for total in (total_p, total_m):
        assert abs(grid[np.argmax(total)] - E_line) <= step

    # U_g carries no f'': conj(U(-g)) == U(g), so the CBS yield is invariant.
    np.testing.assert_allclose(cbs_m.sum(), cbs_p.sum(), rtol=1e-9)

    # PXR: the +hkl line couples to chi(-hkl), the -hkl line to chi(+hkl).
    chi_plus = chi_g(CRYSTAL, HKL, E_line, 0.5, True)
    chi_minus = chi_g(CRYSTAL, -HKL, E_line, 0.5, True)
    expected = abs(chi_plus) ** 2 / abs(chi_minus) ** 2
    assert abs(expected - 1.0) > 0.2  # the case actually discriminates
    np.testing.assert_allclose(pxr_m.sum() / pxr_p.sum(), expected, rtol=2e-3)
