"""Laboratory entry geometry for planar channeling (cut crystals, tilt, refraction)."""

import numpy as np
import pytest

from pyrite.campaign.geometry import CrystalCut, resolve_crystal_cut
from pyrite.materials.channeling import entry_populations, planar_profile
from pyrite.materials.channeling.entry import mean_inner_potential_eV, plane_incidence
from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, M_E_EV
from pyrite.montecarlo.geometry import tilted_geometry

SI_LATTICE = CRYSTALS["silicon"]["lattice"]


def _rotation(cut):
    return resolve_crystal_cut(cut, None, (4, 4, 0)).rotation(SI_LATTICE)


def _p(kinetic_eV):
    return np.sqrt(kinetic_eV * (kinetic_eV + 2 * M_E_EV)) / HBARC_EV_ANG


# ---- mean inner potential (cr-mean-inner-potential) --------------------------


def test_mean_inner_potential_is_attractive_and_plausible():
    u_si = mean_inner_potential_eV("silicon")
    # Independent-atom Si overestimates the measured ~12.6 V by ~10% (bonding).
    assert -15.0 < u_si < -12.5
    assert mean_inner_potential_eV("diamond") < u_si  # denser, more electrons per volume


# ---- incidence and refraction (cr-plane-incidence) ---------------------------


@pytest.mark.parametrize(("tilt", "azim"), [(0.0, 0.0), (0.3, 20.0), (2.0, 135.0), (5.0, 300.0)])
def test_beam_direction_matches_transport_tilt_convention(tilt, azim):
    incidence = plane_incidence(
        "silicon",
        (1, 1, 0),
        40e6,
        tilt_deg=tilt,
        tilt_azim_deg=azim,
        rotation=_rotation(CrystalCut(surface_hkl=(0, 0, 1))),
    )
    beam, _ = tilted_geometry(0.0, np.deg2rad(tilt), np.deg2rad(azim))
    np.testing.assert_allclose(incidence.beam_dir, beam, atol=1e-15)


def test_tilt_across_versus_along_a_perpendicular_plane():
    rotation = _rotation(CrystalCut(surface_hkl=(0, 0, 1)))  # (110) planes stand on the face
    normal_incidence = plane_incidence("silicon", (1, 1, 0), 40e6, rotation=rotation)
    assert normal_incidence.geometric_rad == pytest.approx(0.0, abs=1e-15)
    across = plane_incidence(
        "silicon", (1, 1, 0), 40e6, tilt_deg=0.05, tilt_azim_deg=225.0, rotation=rotation
    )
    along = plane_incidence(
        "silicon", (1, 1, 0), 40e6, tilt_deg=0.05, tilt_azim_deg=135.0, rotation=rotation
    )
    assert across.geometric_rad == pytest.approx(np.deg2rad(0.05), rel=1e-9)
    assert along.geometric_rad == pytest.approx(0.0, abs=1e-15)
    # Planes perpendicular to the face: refraction leaves k_x unchanged.
    assert across.effective_rad == pytest.approx(np.sin(np.deg2rad(0.05)), rel=1e-12)


def test_refraction_conserves_tangential_momentum_and_raises_normal_momentum():
    energy = 30e3
    rotation = _rotation(CrystalCut(surface_hkl=(0, 0, 1)))
    incidence = plane_incidence(
        "silicon", (1, 1, 1), energy, tilt_deg=10.0, tilt_azim_deg=30.0, rotation=rotation
    )
    u0 = incidence.mean_inner_potential_eV
    p_out, p_in = _p(energy), _p(energy - u0)
    beam, normal = incidence.beam_dir, incidence.plane_normal
    p_par = p_out * beam[:2]
    expected = p_par @ normal[:2] + np.sqrt(p_in**2 - p_par @ p_par) * normal[2]
    assert incidence.effective_rad * p_out == pytest.approx(expected, rel=1e-12)
    assert np.sin(incidence.internal_rad) * p_in == pytest.approx(expected, rel=1e-12)
    vacuum = plane_incidence(
        "silicon",
        (1, 1, 1),
        energy,
        tilt_deg=10.0,
        tilt_azim_deg=30.0,
        rotation=rotation,
        refraction=False,
    )
    assert vacuum.mean_inner_potential_eV == 0.0
    assert vacuum.effective_rad == pytest.approx(np.sin(vacuum.geometric_rad), rel=1e-12)
    # Normal momentum grows, so a plane with n_z > 0 sees a larger k_x.
    assert normal[2] > 0 and incidence.effective_rad > vacuum.effective_rad


def test_miscut_tilts_channeling_planes_off_the_face_normal():
    miscut = 1.0
    rotation = _rotation(CrystalCut(surface_hkl=(0, 0, 1), miscut_deg=miscut))
    incidence = plane_incidence("silicon", (1, 0, 0), 40e6, rotation=rotation, refraction=False)
    # (100) planes contain the face normal before the miscut; the cut tilts
    # them by the miscut angle toward the beam.
    assert abs(incidence.geometric_rad) == pytest.approx(np.deg2rad(miscut), rel=1e-9)


def test_catalog_face_orientation_is_default():
    # The catalog Si face is (440): (110) planes lie in the face, 90 deg to the beam.
    incidence = plane_incidence("silicon", (1, 1, 0), 40e6)
    assert abs(incidence.geometric_rad) == pytest.approx(np.pi / 2, rel=1e-12)


def test_beam_must_enter_through_the_face():
    with pytest.raises(ValueError, match="entrance face"):
        plane_incidence("silicon", (1, 1, 0), 40e6, tilt_deg=120.0)


def test_cut_crystal_entry_feeds_populations():
    rotation = _rotation(CrystalCut(surface_hkl=(0, 0, 1)))
    incidence = plane_incidence(
        "silicon", (1, 1, 0), 40e6, tilt_deg=0.0172, tilt_azim_deg=225.0, rotation=rotation
    )
    profile = planar_profile("silicon", (1, 1, 0), 293.15)
    from_lab = entry_populations(profile, 40e6, incidence.effective_rad)
    direct = entry_populations(profile, 40e6, np.sin(np.deg2rad(0.0172)))
    np.testing.assert_allclose(from_lab, direct, atol=1e-14)
    assert from_lab.sum() == pytest.approx(1.0, abs=1e-12)
