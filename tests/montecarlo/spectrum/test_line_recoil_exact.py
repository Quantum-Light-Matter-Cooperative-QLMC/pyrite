"""Production line positions on the exact recoil root (#1099).

Straight single-segment spectra from ``mc_cxr_spectrum`` against
``recoil_resonance_eV`` (Shi et al. 2024, Eq. (6)) and the Huang et al. (2023)
measured graphite/hBN harmonic defect. The ``_RECOIL_HBARC_EV_ANG = 0`` hook is
the ``hbar g -> 0`` classical limit.

Validation: line-recoil-in-medium-root
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, M_E_EV, reciprocal_g_vector
from pyrite.materials.recoil import recoil_resonance_eV, recoil_shift_fraction
from pyrite.montecarlo import mc_cxr_spectrum
from pyrite.montecarlo.spectrum.cxr import _kernels, _setup
from pyrite.montecarlo.transport import beta_from_keV

BEAM = np.array([0.0, 0.0, 1.0])
HUANG = Path(__file__).parents[2] / "data/recoil/huang-2023-table-s4.json"
HUANG_CRYSTAL = {"graphite": "hopg", "hBN": "hbn"}


def _segment(v_hat, T_keV, thickness_ang):
    length = thickness_ang / v_hat[2]
    return {
        "r_mid": (0.5 * length * v_hat)[None, :],
        "v_hat": v_hat[None, :],
        "L_ang": np.array([length]),
        "E_keV": np.array([T_keV]),
        "t_ang": np.zeros(1),
        "elec_id": np.zeros(1, dtype=int),
        "Ne": 1,
        "thickness_ang": thickness_ang,
    }


def _peak(grid, spectrum):
    """Sub-step peak position from a parabola through the five top samples."""
    spectrum = np.asarray(spectrum, dtype=float)
    top = int(np.argmax(spectrum))
    assert 2 <= top < grid.size - 2, "line peak at the grid edge"
    window = slice(top - 2, top + 3)
    curvature, slope, _ = np.polyfit(grid[window] - grid[top], spectrum[window], 2)
    return grid[top] - slope / (2.0 * curvature)


def _line(segments, centre, width, crystal, hkl, n_hat, **kwargs):
    grid = np.linspace(centre - 1.5 * width, centre + 1.5 * width, 601)
    spectrum = mc_cxr_spectrum(segments, grid, crystal, [hkl], n_hat=n_hat, **kwargs)
    return _peak(grid, spectrum), grid[1] - grid[0]


def _width(T_keV, length_ang, v_dot_n):
    """First-zero half-width of the straight segment's sinc^2, in eV."""
    beta = float(beta_from_keV(np.array([T_keV]))[0])
    return 2.0 * np.pi * HBARC_EV_ANG * beta / ((1.0 - beta * v_dot_n) * length_ang)


def _to_sample(axis):
    """Rotation ``R`` with ``R @ axis = +z`` (Rodrigues)."""
    k = np.cross(axis, BEAM)
    s, c = np.linalg.norm(k), float(axis @ BEAM)
    if s < 1e-15:
        return np.eye(3)
    k /= s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + s * K + (1.0 - c) * K @ K


def _huang_defects(classical=False):
    """Measured and ``mc_cxr_spectrum`` ``2 E2 - E4`` per Huang SI Table S4 row."""
    data = json.loads(HUANG.read_text())
    measured, simulated = [], []
    for name, T_keV, e2, e4, theta_til, phi_til in data["rows"]:
        info = data["sets"][name]
        crystal = HUANG_CRYSTAL[info["material"]]
        t, p, o = np.radians([theta_til, phi_til, info["theta_obs_deg"]])
        rotation = _to_sample(np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]))
        v_hat = rotation @ BEAM
        n_hat = rotation @ np.array([np.sin(o), 0.0, np.cos(o)])
        thickness = 10.0 * info["thickness_nm"]
        segments = _segment(v_hat, T_keV, thickness)
        width = _width(T_keV, thickness / v_hat[2], float(n_hat @ v_hat))
        lines = []
        for order in (2, 4):
            g, _ = reciprocal_g_vector((0, 0, order), CRYSTALS[crystal]["lattice"])
            centre = float(recoil_resonance_eV(T_keV * 1e3, v_hat, g, n_hat))
            if classical:
                centre /= 1.0 + float(recoil_shift_fraction(T_keV * 1e3, v_hat, g, n_hat))
            hkl = (0, 0, order)
            lines.append(_line(segments, centre, width, crystal, hkl, n_hat, B_ang2=0.8)[0])
        measured.append(2.0 * e2 - e4)
        simulated.append(2.0 * lines[0] - lines[1])
    sigma_cal = data["detector"]["calibration_uncertainty_eV"]
    return np.array(measured), np.array(simulated), sigma_cal


def test_huang_2023_harmonic_defect_through_production_lines(monkeypatch):
    """Huang 2023 SI Table S4 ``2 E2 - E4`` from production straight-segment lines.

    Includes the production first-order Snell escape refraction; the classical
    (``hbar -> 0``) production lines are excluded.
    """
    measured, simulated, sigma_cal = _huang_defects()
    residual = measured - simulated
    sigma = np.hypot(residual.std(ddof=1) / np.sqrt(residual.size), sigma_cal)
    assert np.all(simulated > 15.0)
    assert abs(residual.mean()) <= 2.0 * sigma

    monkeypatch.setattr(_kernels, "_RECOIL_HBARC_EV_ANG", 0.0)
    _, classical, _ = _huang_defects(classical=True)
    assert np.all(np.abs(classical) < 1.0)  # refraction only
    assert (measured - classical).mean() > 10.0 * sigma


def _vacuum(monkeypatch):
    monkeypatch.setattr(
        _setup,
        "refractive_index",
        lambda crystal, energy, *args: np.ones_like(energy, dtype=complex),
    )


# Si [001] reflections sharing g_z = 2 (2 pi/a): one classical line, split by
# the exact root through Q.n/E (Shi et al. 2024 transverse recoil).
SHI_SHELL = [(2, 0, 2), (-2, 0, 2), (0, 2, 2), (0, -2, 2)]
SHI_FORWARD = [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1)]


def _si_lines(T_keV, length, theta_deg, rows, monkeypatch, *, classical=False):
    _vacuum(monkeypatch)
    if classical:
        monkeypatch.setattr(_kernels, "_RECOIL_HBARC_EV_ANG", 0.0)
    th = np.radians(theta_deg)
    n_hat = np.array([np.sin(th), 0.0, np.cos(th)])
    lattice = CRYSTALS["silicon"]["lattice"]
    g = np.array([reciprocal_g_vector(row, lattice)[0] for row in rows])
    exact = recoil_resonance_eV(T_keV * 1e3, BEAM, g, n_hat)
    reference = (
        exact / (1.0 + recoil_shift_fraction(T_keV * 1e3, BEAM, g, n_hat)) if classical else exact
    )
    width = _width(T_keV, length, float(n_hat[2]))
    segments = _segment(BEAM, T_keV, length)
    peaks, steps = zip(
        *(
            _line(segments, centre, width, "silicon", row, n_hat, beam_uvw=(0, 0, 1), B_ang2=0.4632)
            for centre, row in zip(reference, rows, strict=True)
        ),
        strict=True,
    )
    return np.array(peaks), np.asarray(reference), max(steps)


@pytest.mark.parametrize("T_keV", [30.0, 300.0, 5000.0])
def test_shi_fig2_geometry_lines_sit_on_each_reflections_exact_root(T_keV, monkeypatch):
    """30 nm Si [001], theta = 100 deg (Shi 2024 Fig. 2): per-reflection lines."""
    peaks, exact, step = _si_lines(T_keV, 300.0, 100.0, SHI_SHELL, monkeypatch)
    np.testing.assert_allclose(peaks, exact, rtol=0.0, atol=step)
    split = np.ptp(exact) / exact.mean()
    total = T_keV * 1e3 + M_E_EV
    # The split is the Q.n/E denominator term: ~ hbar c |g_x| / E.
    assert split > 0.2 * HBARC_EV_ANG * 2.0 * np.pi / 5.431 * 2.0 / total


def test_shi_fig2_splits_collapse_toward_mev(monkeypatch):
    splits = {}
    for T_keV in (30.0, 5000.0):
        peaks, _, _ = _si_lines(T_keV, 300.0, 100.0, SHI_SHELL, monkeypatch)
        splits[T_keV] = np.ptp(peaks) / peaks.mean()
    assert splits[30.0] > 1e-2
    assert splits[5000.0] < 0.15 * splits[30.0]


@pytest.mark.parametrize("theta_deg", [0.5, 2.5, 5.0])
def test_shi_si_s10_forward_lines_sit_on_the_exact_root(theta_deg, monkeypatch):
    """SI S10 forward case: 100 keV, 100 nm Si [001], theta = 0-5 deg."""
    peaks, exact, step = _si_lines(100.0, 1000.0, theta_deg, SHI_FORWARD, monkeypatch)
    np.testing.assert_allclose(peaks, exact, rtol=0.0, atol=step)


def test_classical_hook_recovers_the_classical_roots(monkeypatch):
    """``hbar -> 0``: each line on its classical root; the shell is one line."""
    peaks, classical, step = _si_lines(30.0, 300.0, 100.0, SHI_SHELL, monkeypatch, classical=True)
    np.testing.assert_allclose(peaks, classical, rtol=0.0, atol=step)
    assert np.ptp(peaks) <= 2.0 * step
