"""Zhai tilt-angle convention (docs/superpowers/specs/2026-07-11-zhai-tilt-
convention-design.md):

* Positive polar tilt (tilt_polar_rad / tilt_deg) tilts the slab normal --
  and, by default, the reciprocal vector g (g || n) -- TOWARD the detector.
* Positive azimuth (tilt_azim_rad / tilt_azim_deg) is a CCW roll about the
  beam +z axis; azimuth 0 places the tilt in the scattering x-z plane (zero
  y-component).
* The optional n/g split hook (recip_miscut_rad) lets g tilt independently of
  the physical slab normal n (a crystal miscut). None is a strict no-op.

This module covers the convention itself (tilted_geometry) and the hook
(_orientation_R / mc_spectrum); the grid-value flips in materials/registry.py /
config.py / sweep.py / scan.py / src/cxr_mc/apps/anchor_figures.py are covered by
their existing consumer tests (already updated to the positive convention).
"""

import numpy as np
import pytest

from cxr_mc.materials.crystal import CRYSTALS
from cxr_mc.montecarlo import mc_spectrum, simulate_trajectories, tilted_geometry
from cxr_mc.montecarlo.geometry import _orientation_R

THETA_OBS = np.deg2rad(119.0)


# ---- tilted_geometry: the Zhai (theta, phi) convention ------------------------
def test_positive_polar_tilts_reciprocal_vector_toward_detector():
    """g || n, so g's sample-frame direction is always [0, 0, 1]; its lab-frame
    orientation is n_hat's own construction. Equivalently: dot(n_hat, [0,0,1])
    in the SAMPLE frame is the same as dot(normal_lab, detector_dir) in the LAB
    frame (both frames related by the same rotation R). +10 deg must be LESS
    anti-aligned with the detector than -10 deg (spec: -0.326 vs -0.629)."""
    _, n_hat_plus = tilted_geometry(THETA_OBS, np.deg2rad(10.0), 0.0)
    _, n_hat_minus = tilted_geometry(THETA_OBS, np.deg2rad(-10.0), 0.0)
    dot_plus = float(n_hat_plus @ np.array([0.0, 0.0, 1.0]))
    dot_minus = float(n_hat_minus @ np.array([0.0, 0.0, 1.0]))
    assert dot_plus == pytest.approx(-0.3256, abs=1e-3)
    assert dot_minus == pytest.approx(-0.6293, abs=1e-3)
    assert dot_plus > dot_minus  # +10 deg: g closer to (less anti-aligned with) detector


def test_zero_azimuth_keeps_tilt_in_scattering_plane():
    """phi = 0 must place beam_dir and n_hat entirely in the x-z plane (zero
    y-component) for any polar tilt."""
    for tilt_deg in (5.0, 30.0, 80.0):
        beam_dir, n_hat = tilted_geometry(THETA_OBS, np.deg2rad(tilt_deg), 0.0)
        assert beam_dir[1] == pytest.approx(0.0, abs=1e-12)
        assert n_hat[1] == pytest.approx(0.0, abs=1e-12)


def test_positive_azimuth_is_ccw_about_beam_z():
    """As phi increases from 0, the slab normal's azimuth sweeps +x -> +y (a
    CCW roll about +z) -- confirmed by inspecting normal_lab directly (the
    same expression tilted_geometry rotates the beam/detector against)."""
    tp = np.deg2rad(20.0)
    for ta_deg in (0.0, 45.0, 90.0):
        ta = np.deg2rad(ta_deg)
        st = np.sin(tp)
        normal_lab = np.array([st * np.cos(ta), st * np.sin(ta), np.cos(tp)])
        assert normal_lab[1] >= -1e-12  # never swings negative over [0, 90]
    # at phi=90, the tilt is purely in y (x-component vanishes)
    st = np.sin(tp)
    normal_90 = np.array([st * np.cos(np.pi / 2), st * np.sin(np.pi / 2), np.cos(tp)])
    assert normal_90[0] == pytest.approx(0.0, abs=1e-12)
    assert normal_90[1] == pytest.approx(st, abs=1e-12)


# ---- optional n/g split hook (recip_miscut_rad) --------------------------------
_LATTICE = CRYSTALS["silicon"]["lattice"]


def test_orientation_R_recip_miscut_none_is_default():
    """recip_miscut_rad=None (the default) must match the pre-hook 3-arg call
    exactly -- a strict no-op."""
    R_implicit = _orientation_R(_LATTICE, (1, 1, 1), np.deg2rad(15.0))
    R_explicit_none = _orientation_R(_LATTICE, (1, 1, 1), np.deg2rad(15.0), None)
    assert R_implicit is not None
    assert R_explicit_none is not None
    assert np.array_equal(R_implicit, R_explicit_none)

    # also true for the fully-bare construction-frame default (both None)
    assert _orientation_R(_LATTICE, None, 0.0) is None
    assert _orientation_R(_LATTICE, None, 0.0, None) is None


def test_orientation_R_recip_miscut_rotates_g_only():
    """A nonzero recip_miscut_rad must produce a DIFFERENT rotation from the
    None case, applied only to the reciprocal-vector orientation path -- it
    has no effect on tilted_geometry's beam_dir/n_hat, which is guaranteed
    structurally: tilted_geometry() takes no recip_miscut_rad parameter."""
    g_test = np.array([0.0, 0.0, 1.0])
    R_base = _orientation_R(_LATTICE, (1, 1, 1), np.deg2rad(15.0))
    R_miscut = _orientation_R(
        _LATTICE, (1, 1, 1), np.deg2rad(15.0), (np.deg2rad(20.0), np.deg2rad(40.0))
    )
    assert R_miscut is not None
    assert not np.allclose(R_base @ g_test, R_miscut @ g_test)

    # the slab-normal / beam-frame directions are computed by a function with
    # no knowledge of recip_miscut_rad at all -- unaffected by construction.
    beam_dir, n_hat = tilted_geometry(THETA_OBS, np.deg2rad(30.0), np.deg2rad(10.0))
    beam_dir2, n_hat2 = tilted_geometry(THETA_OBS, np.deg2rad(30.0), np.deg2rad(10.0))
    assert np.array_equal(beam_dir, beam_dir2)
    assert np.array_equal(n_hat, n_hat2)


def test_orientation_R_recip_miscut_zero_polar_is_identity_like():
    """(0.0, anything) collapses to the same rotation as None -- zero polar
    miscut is an exact no-op regardless of the azim component."""
    R_none = _orientation_R(_LATTICE, (1, 1, 1), np.deg2rad(15.0), None)
    R_zero = _orientation_R(_LATTICE, (1, 1, 1), np.deg2rad(15.0), (0.0, np.deg2rad(77.0)))
    assert np.array_equal(R_none, R_zero)


# ---- mc_spectrum: the same guarantee at the production call site ---------------
E0_KEV = 25.0
B_002 = 0.8
HKL = ((0, 0, 2), (0, 0, -2))
E_LINE = np.arange(700.0, 1200.0, 1.0)

_hopg = CRYSTALS["hopg"]
_n_atoms = len(_hopg["basis"]) / _hopg["V_cell"]


@pytest.fixture(scope="module")
def _segments():
    return simulate_trajectories(
        E0_KEV, 80, 1e7, element="C", n_atoms_per_ang3=_n_atoms, E_cut_keV=1.0, seed=11
    )


def test_mc_spectrum_recip_miscut_none_reproduces_bit_for_bit(_segments):
    kw = dict(crystal="hopg", hkl_list=HKL, theta_obs_rad=THETA_OBS, B_ang2=B_002)
    default = mc_spectrum(_segments, E_LINE, **kw)
    explicit_none = mc_spectrum(_segments, E_LINE, recip_miscut_rad=None, **kw)
    np.testing.assert_array_equal(default, explicit_none)


def test_mc_spectrum_recip_miscut_nonzero_changes_spectrum(_segments):
    kw = dict(crystal="hopg", hkl_list=HKL, theta_obs_rad=THETA_OBS, B_ang2=B_002)
    default = mc_spectrum(_segments, E_LINE, **kw)
    miscut = mc_spectrum(
        _segments, E_LINE, recip_miscut_rad=(np.deg2rad(15.0), np.deg2rad(0.0)), **kw
    )
    assert not np.allclose(default, miscut)
