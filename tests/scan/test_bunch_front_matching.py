"""Scene inverse tested against independent full ray phases, not its algebra."""

from dataclasses import replace

import numpy as np
import pytest

import pyrite as pr
from pyrite.api import build_case
from pyrite.materials import CATALOG
from pyrite.materials.crystal import HBARC_EV_ANG, beta_from_Ee, reciprocal_g_vector


def _scene(tilt=84.5, azimuth=0.0):
    return pr.Scene(
        beam=pr.Beam(
            energy_keV=30.0,
            transverse=pr.TransverseSpot(0.001, 0.002),
            longitudinal=pr.BunchLength(0.001),
        ),
        target=pr.Slab("hopg", 500.0, tilt_deg=tilt, tilt_azim_deg=azimuth, footprint=None),
        detector=pr.Detector(observation_angle_deg=145.0),
    )


def test_notebook_geometry_and_explicit_peak_energy_agree():
    scene = _scene()
    front = scene.phase_matched_bunch_front_tilt((0, 0, 2))
    assert front.tilt_deg == pytest.approx(85.7217094848694, abs=1e-12)
    assert front.azimuth_deg == pytest.approx(0.0, abs=1e-12)
    assert scene.phase_matched_bunch_front_tilt((0, 0, 2), photon_energy_eV=91.2) == front
    assert scene.beam.bunch_front_tilt is None
    assert replace(scene.beam, bunch_front_tilt=front).longitudinal == scene.beam.longitudinal


def test_default_off_axis_line_matches_independently_derived_reference():
    scene = _scene(30.0, 75.0)
    direction = (0.6, -0.4, 0.7)
    front = scene.phase_matched_bunch_front_tilt((1, 0, 2), observation_direction_lab=direction)
    # Independent reciprocal-basis, Rodrigues and vacuum-line derivation;
    # recorded in the linear-bunch-front-tilt verification write-up.
    assert front.tilt_deg == pytest.approx(77.90231131695, rel=0.0, abs=1e-10)
    assert front.azimuth_deg == pytest.approx(35.00764031289, rel=0.0, abs=1e-10)
    explicit = scene.phase_matched_bunch_front_tilt(
        (1, 0, 2), photon_energy_eV=486.718502128692, observation_direction_lab=direction
    )
    np.testing.assert_allclose(front.gradient(30.0), explicit.gradient(30.0), atol=1e-13)


@pytest.mark.parametrize("raw", [(30.24, 75.24), (30.25, -75.25)])
def test_off_grid_angles_match_the_actual_case_geometry(raw):
    direction = (0.6, -0.4, 0.7)
    front = _scene(*raw).phase_matched_bunch_front_tilt(
        (1, 0, 2), photon_energy_eV=1800.0, observation_direction_lab=direction
    )
    scene = _scene(*raw)
    case = build_case(replace(scene, beam=replace(scene.beam, bunch_front_tilt=front)))
    assert (case["tilt_deg"], case["tilt_azim_deg"]) == raw


@pytest.mark.parametrize("tilt,azimuth", [(30.24, 0.0), (-30.25, 75.24), (84.5, 145.0)])
@pytest.mark.parametrize("energy", [100.0, 600.0])
def test_off_axis_reflection_cancels_independent_ray_phases(tilt, azimuth, energy):
    scene = _scene(tilt, azimuth)
    direction = np.array([0.2, -0.4, 0.7])
    front = scene.phase_matched_bunch_front_tilt(
        (1, 0, 2), photon_energy_eV=energy, observation_direction_lab=direction
    )
    # Independent Rodrigues rotation and explicit lab ray/face intersections.
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
    delay = positions_lab[:, :2] @ front.gradient(30.0) + flight / beta
    n = R.T @ (direction / np.linalg.norm(direction))
    g, _ = reciprocal_g_vector((1, 0, 2), CATALOG.crystal("hopg").lattice)
    omega = energy / HBARC_EV_ANG
    phase = omega * delay - face @ (omega * n + g)
    np.testing.assert_allclose(phase, 0.0, atol=1e-10)


@pytest.mark.parametrize("hkl", [(0, 0, 0), (0.0, 0.0, 2.0), (0, 2), (0, 0, -2)])
def test_invalid_or_negative_default_resonance_is_rejected(hkl):
    with pytest.raises(ValueError, match="hkl|Miller"):
        _scene().phase_matched_bunch_front_tilt(hkl)


@pytest.mark.parametrize("energy", [0.0, -1.0, np.inf, np.nan])
def test_invalid_photon_energy_is_rejected(energy):
    with pytest.raises(ValueError, match="photon energy"):
        _scene().phase_matched_bunch_front_tilt((0, 0, 2), photon_energy_eV=energy)


@pytest.mark.parametrize("direction", [(0, 0, 0), (1, 0), (1, np.nan, 0)])
def test_invalid_observation_direction_is_rejected(direction):
    with pytest.raises(ValueError, match="finite nonzero 3-vector"):
        _scene().phase_matched_bunch_front_tilt((0, 0, 2), observation_direction_lab=direction)


def test_unsupported_source_and_stack_are_rejected():
    scene = _scene()
    spread = replace(scene, beam=replace(scene.beam, energy_spread_frac=0.01))
    with pytest.raises(ValueError, match="monoenergetic"):
        spread.phase_matched_bunch_front_tilt((0, 0, 2))
    stack = replace(
        scene, target=pr.Stack((pr.Layer("hopg", 500), pr.Layer("hopg", 500)), footprint=None)
    )
    with pytest.raises(ValueError, match="single Slab"):
        stack.phase_matched_bunch_front_tilt((0, 0, 2))


def test_planar_detector_requires_an_explicit_ray():
    scene = _scene()
    planar = replace(
        scene,
        detector=pr.PlanarDetector(
            pose=pr.PlanarPose(
                center_mm=(0.0, 0.0, 100.0), normal=(0.0, 0.0, 1.0), x_axis=(1.0, 0.0, 0.0)
            ),
            size_mm=(10.0, 10.0),
        ),
    )
    with pytest.raises(ValueError, match="requires observation_direction_lab"):
        planar.phase_matched_bunch_front_tilt((0, 0, 2))
    theta = np.deg2rad(145.0)
    front = planar.phase_matched_bunch_front_tilt(
        (0, 0, 2), observation_direction_lab=(np.sin(theta), 0.0, np.cos(theta))
    )
    assert front.tilt_deg == pytest.approx(
        scene.phase_matched_bunch_front_tilt((0, 0, 2)).tilt_deg, abs=1e-12
    )
