"""Per-target crystal cuts: default compatibility, lowering, identity, matching."""

from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite.api import build_case
from pyrite.campaign.flat_geometry import target_flat_fields, target_from_flat, target_replace
from pyrite.campaign.longitudinal import LongitudinalDistribution
from pyrite.campaign.sweep import BeamSpec, Sweep, build_cases
from pyrite.materials import CATALOG
from pyrite.materials.crystal import HBARC_EV_ANG, beta_from_Ee, reciprocal_g_vector
from pyrite.montecarlo.geometry import _orientation_R

HOPG = CATALOG.crystal("hopg")  # hexagonal (gamma = 120 deg): a nonorthogonal cell
CUT = pr.geometry.CrystalCut(
    surface_hkl=(1, 0, 1), roll_deg=17.0, miscut_deg=5.56512, miscut_azim_deg=200.0
)


def _slab(cut=None, tilt=84.5, azimuth=0.0, **kwargs):
    return pr.geometry.Slab(
        "hopg", 500.0, tilt_deg=tilt, tilt_azim_deg=azimuth, footprint=None, cut=cut, **kwargs
    )


def _scene(cut=None, tilt=84.5, azimuth=0.0):
    return pr.Scene(
        beam=pr.Beam(
            energy_keV=30.0,
            transverse=pr.beams.TransverseSpot(0.001, 0.002),
            longitudinal=pr.beams.BunchLength(0.001),
        ),
        target=_slab(cut, tilt, azimuth),
        detector=pr.Detector(observation_angle_deg=145.0),
    )


def _minimal_rotation(src, dst):
    """Independent Rodrigues minimal rotation taking unit ``src`` onto ``dst``."""
    src = src / np.linalg.norm(src)
    dst = dst / np.linalg.norm(dst)
    v = np.cross(src, dst)
    c = float(src @ dst)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _independent_lattice_rotation(cut):
    axis, _ = reciprocal_g_vector(cut.surface_hkl, HOPG.lattice)
    z = np.array([0.0, 0.0, 1.0])
    roll = np.deg2rad(cut.roll_deg)
    Rz = np.array(
        [[np.cos(roll), -np.sin(roll), 0.0], [np.sin(roll), np.cos(roll), 0.0], [0.0, 0.0, 1.0]]
    )
    theta, phi = np.deg2rad([cut.miscut_deg, cut.miscut_azim_deg])
    tilted = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return _minimal_rotation(z, tilted) @ Rz @ _minimal_rotation(axis, z)


@pytest.mark.parametrize(
    "cut", [pr.geometry.CrystalCut(), pr.geometry.CrystalCut(surface_hkl=HOPG.surface_hkl)]
)
def test_noop_cuts_preserve_the_catalog_case_payload(cut):
    assert build_case(_scene(cut)).to_dict() == build_case(_scene()).to_dict()


def test_explicit_cut_reaches_the_case_without_touching_the_catalog():
    before = (HOPG.surface_hkl, HOPG.beam_uvw)
    case = build_case(_scene(CUT))
    assert case["surface_hkl"] == (1, 0, 1)
    assert case["beam_uvw"] is None
    assert case["azimuth_rad"] == pytest.approx(np.deg2rad(17.0), abs=0.0)
    np.testing.assert_allclose(case["recip_miscut_rad"], np.deg2rad([5.56512, 200.0]), rtol=0)
    assert case["name"].endswith("cut=(1,0,1) roll=17 miscut=5.56512@200")
    # Stored line grids belong to the catalog orientation; a cut resolves its own.
    assert "line_grid_policy" in case
    assert "line_grid_policy" in build_case(_scene())
    assert (CATALOG.crystal("hopg").surface_hkl, CATALOG.crystal("hopg").beam_uvw) == before


def test_distinct_cuts_have_distinct_case_identities():
    cuts = [
        None,
        pr.geometry.CrystalCut(beam_uvw=(1, 0, 0)),
        pr.geometry.CrystalCut(roll_deg=10.0),
        pr.geometry.CrystalCut(miscut_deg=5.56512),
        pr.geometry.CrystalCut(miscut_deg=5.565121),
        pr.geometry.CrystalCut(miscut_deg=5.56512, miscut_azim_deg=90.0),
        CUT,
    ]
    cases = [build_case(_scene(cut)) for cut in cuts]
    assert len({case["name"] for case in cases}) == len(cuts)
    payloads = [repr(sorted(case.to_dict().items(), key=lambda kv: kv[0])) for case in cases]
    assert len(set(payloads)) == len(cuts)


def test_target_role_lowers_the_same_cut():
    body = pr.geometry.RectangularBody(
        pr.geometry.SubstanceRef("crystal", "hopg"), (5.0, 5.0), 0.00005
    )
    target = pr.Target(body, tilt_deg=84.5, lattice_cut=CUT)
    assert target.to_slab().cut == CUT
    assert pr.Target(body).to_slab().cut is None


@pytest.mark.parametrize(
    "kwargs,error",
    [
        (dict(surface_hkl=(0, 0, 1), beam_uvw=(0, 0, 1)), ValueError),
        (dict(surface_hkl=(0, 0, 0)), ValueError),
        (dict(surface_hkl=(0.0, 0.0, 1.0)), ValueError),
        (dict(beam_uvw=(1, 0)), ValueError),
        (dict(miscut_deg=90.0), ValueError),
        (dict(miscut_deg=-1.0), ValueError),
        (dict(roll_deg=np.nan), ValueError),
        (dict(miscut_azim_deg=np.inf), ValueError),
        (dict(roll_deg=(1.0, 2.0)), TypeError),
        (dict(miscut_deg=True), TypeError),
    ],
)
def test_invalid_cuts_are_rejected(kwargs, error):
    with pytest.raises(error):
        pr.geometry.CrystalCut(**kwargs)


def test_cut_is_flat_single_slab_only():
    with pytest.raises(ValueError, match="flat entrance face"):
        _slab(CUT, tilt=30.0, azimuth=180.0, entrance_face=pr.geometry.BlazedGrooves(1e4))
    with pytest.raises(TypeError, match="CrystalCut"):
        _slab("(001)")
    with pytest.raises(ValueError, match="single-slab"):
        target_from_flat("hopg", substrate="silicon", cut=CUT)
    with pytest.raises(ValueError, match="lattice_cut"):
        pr.Target(
            pr.geometry.RectangularBody(
                pr.geometry.SubstanceRef("crystal", "hopg"), (5.0, 5.0), 0.00005
            ),
            lattice_cut="custom",  # type: ignore[arg-type]
        )


def test_sweep_axis_and_cut_axis_conflict():
    sweep = Sweep(
        target=_slab(pr.geometry.CrystalCut(surface_hkl=(1, 0, 1))),
        beam=BeamSpec(energy_keV=30.0),
        beam_uvw=(0, 0, 1),
    )
    with pytest.raises(ValueError, match="not both"):
        build_cases(sweep)


def test_targeted_longitudinal_timing_rejects_a_miscut():
    sweep = Sweep(
        target=_slab(pr.geometry.CrystalCut(miscut_deg=3.0), tilt=45.0, azimuth=180.0),
        beam=BeamSpec(
            energy_keV=30.0,
            transverse_fwhm_x_mm=0.1,
            transverse_fwhm_y_mm=0.1,
            bunch_charge_pc=1.0,
            rep_rate_hz=5000.0,
            longitudinal=LongitudinalDistribution("microtrain", envelope_rms_fs=200.0),
        ),
    )
    with pytest.raises(ValueError, match="miscut"):
        build_cases(sweep)


def test_flat_projection_carries_the_cut_only_when_set():
    assert "cut" not in target_flat_fields(_slab())
    assert target_flat_fields(_slab(CUT))["cut"] == CUT
    assert target_replace(_slab(CUT), tilt_deg=30.0).cut == CUT
    assert target_replace(_slab(), cut=CUT).cut == CUT


def test_case_rotation_matches_an_independent_construction():
    case = build_case(_scene(CUT))
    runtime = _orientation_R(
        HOPG.lattice,
        case["beam_uvw"],
        case.get("azimuth_rad", 0.0),
        case.get("recip_miscut_rad"),
        surface_hkl=case["surface_hkl"],
    )
    np.testing.assert_allclose(runtime, _independent_lattice_rotation(CUT), atol=1e-13)
    # The cut axis lands on the miscut direction, off the face normal.
    axis, norm = reciprocal_g_vector((1, 0, 1), HOPG.lattice)
    theta, phi = np.deg2rad([CUT.miscut_deg, CUT.miscut_azim_deg])
    np.testing.assert_allclose(
        runtime @ axis / norm,
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        atol=1e-13,
    )


@pytest.mark.parametrize("tilt,azimuth", [(30.0, 0.0), (84.5, 145.0)])
@pytest.mark.parametrize("energy", [None, 600.0])
def test_cut_front_cancels_independent_ray_phases(tilt, azimuth, energy):
    scene = _scene(CUT, tilt, azimuth)
    direction = np.array([0.2, -0.4, 0.7])
    hkl = (1, 0, 2)
    front = scene.phase_matched_bunch_front_tilt(
        hkl, photon_energy_eV=energy, observation_direction_lab=direction
    )
    # Independent sample rotation, lattice rotation and ray/face intersections.
    theta, phi = np.deg2rad([tilt, azimuth])
    axis = np.array([-np.sin(phi), np.cos(phi), 0.0])
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * np.outer(axis, axis)
    R += np.sin(theta) * cross
    positions_lab = np.array([[100.0, -200.0, 0.0], [-300.0, 150.0, 0.0], [0.0, 0.0, 0.0]])
    origins = positions_lab @ R
    b = R.T @ np.array([0.0, 0.0, 1.0])
    flight = -origins[:, 2] / b[2]
    face = origins + flight[:, None] * b
    beta = float(beta_from_Ee(30_000.0))
    n = R.T @ (direction / np.linalg.norm(direction))
    g_crystal, _ = reciprocal_g_vector(hkl, HOPG.lattice)
    g = _independent_lattice_rotation(CUT) @ g_crystal
    omega = beta * (b @ g) / (1 - beta * (b @ n)) if energy is None else energy / HBARC_EV_ANG
    assert omega > 0.0
    delay = positions_lab[:, :2] @ front.gradient(30.0) + flight / beta
    phase = omega * delay - face @ (omega * n + g)
    np.testing.assert_allclose(phase - phase[-1], 0.0, atol=1e-9)


def test_uncut_front_is_unchanged():
    assert _scene().phase_matched_bunch_front_tilt((0, 0, 2)).tilt_deg == pytest.approx(
        85.7217094848694, abs=1e-12
    )
    assert replace(_scene(), target=_slab(pr.geometry.CrystalCut())).phase_matched_bunch_front_tilt(
        (0, 0, 2)
    ) == _scene().phase_matched_bunch_front_tilt((0, 0, 2))


@pytest.mark.parametrize("cut", [pr.geometry.CrystalCut(surface_hkl=(1, 0, 1)), CUT])
@pytest.mark.parametrize("tilt", [0.0, 20.0, -35.0])
def test_tilt_is_measured_from_the_cut_face_normal(cut, tilt):
    from pyrite.montecarlo.geometry import tilted_geometry

    scene = replace(_scene(), target=_slab(cut, tilt=tilt, allow_normal_incidence=True))
    case = build_case(scene)
    beam, n_hat = tilted_geometry(case["theta_obs_rad"], np.deg2rad(case["tilt_deg"]), 0.0)
    rotation = _orientation_R(
        HOPG.lattice,
        case["beam_uvw"],
        case.get("azimuth_rad", 0.0),
        case.get("recip_miscut_rad"),
        surface_hkl=case["surface_hkl"],
    )
    axis, norm = reciprocal_g_vector((1, 0, 1), HOPG.lattice)
    # The beam meets the cut face normal at the target tilt, and the face sits
    # miscut_deg off the selected (101) plane regardless of tilt.
    np.testing.assert_allclose(beam @ np.array([0.0, 0.0, 1.0]), np.cos(np.deg2rad(tilt)))
    plane_normal = rotation @ axis / norm
    face_normal = np.array([0.0, 0.0, 1.0])
    # atan2 stays conditioned at zero miscut; acos amplifies dot-product roundoff.
    face_to_plane = np.rad2deg(
        np.arctan2(np.linalg.norm(np.cross(plane_normal, face_normal)), plane_normal @ face_normal)
    )
    assert face_to_plane == pytest.approx(cut.miscut_deg, abs=1e-9)
    if cut.miscut_deg == 0.0:
        np.testing.assert_allclose(beam @ (rotation @ axis / norm), np.cos(np.deg2rad(tilt)))
