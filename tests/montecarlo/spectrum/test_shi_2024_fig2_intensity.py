"""Shi et al. (2024) Fig. 2 intensities against production PXR/CBS (#1103).

Straight 30 nm Si [001], theta = 100 deg, |h|,|k| <= 8. Reference curves are
the vector-digitized Fig. 2b/d/f (``tests/data/recoil/shi-2024-fig2.json``).
Shi's "CL" curve is the classical PXR-only rate (Eq. (1) with the trajectory
deflection dropped); "QM" carries PXR, coherent bremsstrahlung and their
interference. QM/CL area ratios within one n_z group do not depend on the
figure's y axes. The ``_RECOIL_HBARC_EV_ANG = 0`` hook is the classical limit.

Validation: shi-2024-fig2-intensity-attribution
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pyrite.materials.crystal import CRYSTALS, reciprocal_g_vector
from pyrite.materials.recoil import recoil_resonance_eV, recoil_shift_fraction
from pyrite.montecarlo import mc_cxr_spectrum
from pyrite.montecarlo.spectrum.cxr import _recoil

FIG2 = json.loads((Path(__file__).parents[2] / "data/recoil/shi-2024-fig2.json").read_text())[
    "curves"
]
BEAM = np.array([0.0, 0.0, 1.0])
N_HAT = np.array([np.sin(np.radians(100.0)), 0.0, np.cos(np.radians(100.0))])
B_SI = 0.4632
LENGTH_ANG = 300.0


def _rows(nz, hmax=8):
    """Diamond-allowed (h, k, nz) rows; PyRITE's sign is the paper's negation."""
    rows = []
    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            if len({h % 2, k % 2, nz % 2}) == 1 and (h % 2 == 1 or (h + k + nz) % 4 == 0):
                rows.append((h, k, nz))
    return rows


def _roots(T_keV, rows):
    lattice = CRYSTALS["silicon"]["lattice"]
    g = np.array([reciprocal_g_vector(row, lattice)[0] for row in rows])
    exact = recoil_resonance_eV(T_keV * 1e3, BEAM, g, N_HAT)
    return exact, exact / (1.0 + recoil_shift_fraction(T_keV * 1e3, BEAM, g, N_HAT))


def _segment(T_keV):
    return {
        "r_mid": np.array([[0.0, 0.0, 0.5 * LENGTH_ANG]]),
        "v_hat": BEAM[None, :].copy(),
        "L_ang": np.array([LENGTH_ANG]),
        "E_keV": np.array([T_keV]),
        "t_ang": np.zeros(1),
        "elec_id": np.zeros(1, dtype=int),
        "Ne": 1,
        "thickness_ang": LENGTH_ANG,
    }


def _curves(T_keV, nz):
    return FIG2[f"{T_keV:g} keV CL nz=-{nz}"], FIG2[f"{T_keV:g} keV QM nz=-{nz}"]


def _spectrum(T_keV, nz, monkeypatch, *, classical, components=True):
    cl, qm = _curves(T_keV, nz)
    grid = np.arange(min(cl["E_eV"][0], qm["E_eV"][0]), max(cl["E_eV"][-1], qm["E_eV"][-1]), 0.5)
    with monkeypatch.context() as m:
        if classical:
            m.setattr(_recoil, "_RECOIL_HBARC_EV_ANG", 0.0)
        out = mc_cxr_spectrum(
            _segment(T_keV),
            grid,
            "silicon",
            _rows(nz),
            n_hat=N_HAT,
            beam_uvw=(0, 0, 1),
            B_ang2=B_SI,
            components=components,
        )
    return grid, [np.asarray(a) for a in out] if components else np.asarray(out)


def _area_ratios(T_keV, nz, monkeypatch):
    """(exact total / classical PXR-only, exact total / classical total) areas."""
    _, (cl_total, cl_pxr, _) = _spectrum(T_keV, nz, monkeypatch, classical=True)
    _, qm_total = _spectrum(T_keV, nz, monkeypatch, classical=False, components=False)
    return qm_total.sum() / cl_pxr.sum(), qm_total.sum() / cl_total.sum()


def _shi_ratio(T_keV, nz):
    cl, qm = _curves(T_keV, nz)
    return qm["area_per_sr"] / cl["area_per_sr"]


@pytest.mark.parametrize("T_keV", [30.0, 300.0, 5000.0])
@pytest.mark.parametrize("nz", [2, 3])
def test_digitized_cl_peaks_sit_on_the_classical_root(T_keV, nz):
    """The pinned CL curves peak on the merged classical line (axis sanity)."""
    _, classical = _roots(T_keV, _rows(nz))
    cl, _ = _curves(T_keV, nz)
    assert cl["peak_eV"] == pytest.approx(np.nanmax(classical), rel=2e-3)


@pytest.mark.parametrize("T_keV", [30.0, 300.0])
def test_digitized_tallest_qm_line_is_the_0_pm2_2_root(T_keV):
    exact, _ = _roots(T_keV, [(0, 2, 2)])
    _, qm = _curves(T_keV, 2)
    assert qm["peak_eV"] == pytest.approx(exact[0], abs=4.0)


@pytest.mark.parametrize(("T_keV", "nz"), [(30.0, 2), (5000.0, 2), (5000.0, 3)])
def test_shi_qm_over_cl_area_is_full_cxr_over_classical_pxr(T_keV, nz, monkeypatch):
    """Shi's QM/CL area ratio is production PXR+CBS over classical PXR alone."""
    with_cbs_in_qm_only, _ = _area_ratios(T_keV, nz, monkeypatch)
    assert with_cbs_in_qm_only == pytest.approx(_shi_ratio(T_keV, nz), rel=0.15)


@pytest.mark.parametrize("T_keV", [30.0, 300.0])
@pytest.mark.parametrize("nz", [2, 3])
def test_sub_mev_area_excess_needs_cbs_on_one_side_only(T_keV, nz, monkeypatch):
    """Shi's sub-MeV QM/CL excess (2.9-5.8) appears only with CBS in QM alone.

    With CBS on both sides the exact-root area is the classical one to within
    the recoil Jacobian and the K-edge couplings at the shifted roots
    (0.89-1.07); the residual factor <= 2.2 at 300 keV and n_z = -3 is
    shared by an independent evaluation of Shi's own inputs
    (``checks/shi_2024_matrix_elements_1103.py``).
    """
    with_cbs_in_qm_only, with_cbs_on_both = _area_ratios(T_keV, nz, monkeypatch)
    shi = _shi_ratio(T_keV, nz)
    assert shi > 2.5
    assert with_cbs_on_both == pytest.approx(1.0, abs=0.12)
    assert 1.0 / 2.2 <= with_cbs_in_qm_only / shi <= 1.2


@pytest.mark.parametrize("nz", [2, 3])
def test_mev_classical_heights_match_shi_cl(nz, monkeypatch):
    """At 5 MeV the CBS share is <= 10 %, so either CL definition matches."""
    _, (total, pxr, _) = _spectrum(5000.0, nz, monkeypatch, classical=True)
    cl, _ = _curves(5000.0, nz)
    assert total.max() == pytest.approx(cl["peak"], rel=0.15)
    assert (total.max() - pxr.max()) / total.max() < 0.12
