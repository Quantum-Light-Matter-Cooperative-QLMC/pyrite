"""Scene-camera geometry for the Plotly 3D viewers.

The viewers resolve an EXPLICIT camera because marimo drops `scene.camera`
relayout events, so a dragged camera never reaches Python and a snapshot or an
offscreen render would otherwise fall back to the figure's default view. These
checks pin the geometry those controls rest on.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from pyrite.plots.plotly.camera import (
    CAMERA_PRESETS,
    DEFAULT_ANGLES,
    DEFAULT_EYE,
    camera_eye,
    data_aspect_ratio,
    eye_angles,
    resolve_scene_camera,
    scene_camera,
    scene_extents,
)
from pyrite.plots.plotly.trajectories import _CAMERA_EYE, VOLUME_CAMERA_ANGLES


def _eye_vector(camera):
    return np.array([camera["eye"]["x"], camera["eye"]["y"], camera["eye"]["z"]])


def test_camera_eye_places_the_eye_on_the_requested_sphere() -> None:
    eye = camera_eye(30.0, 20.0, 2.5)

    assert np.linalg.norm(list(eye.values())) == pytest.approx(2.5)
    # Azimuth sweeps +x toward +y, elevation lifts toward +z.
    assert eye["x"] > 0 and eye["y"] > 0 and eye["z"] > 0
    assert camera_eye(0.0, 0.0, 1.0)["x"] == pytest.approx(1.0)
    assert camera_eye(90.0, 0.0, 1.0)["y"] == pytest.approx(1.0)


def test_eye_angles_round_trips_through_camera_eye() -> None:
    for eye in [(1.25, 1.25, 1.25), (-0.8, 1.5, -0.4), (0.0, -2.0, 0.5)]:
        assert _eye_vector({"eye": camera_eye(*eye_angles(eye))}) == pytest.approx(eye)


def test_volume_figure_publishes_its_own_hand_tuned_view_exactly() -> None:
    # The trace tab seeds its controls from this, so "default" must reproduce
    # the long-standing eye rather than a rounded approximation of it.
    assert _eye_vector({"eye": camera_eye(*VOLUME_CAMERA_ANGLES)}) == pytest.approx(_CAMERA_EYE)
    assert _eye_vector({"eye": camera_eye(*DEFAULT_ANGLES)}) == pytest.approx(DEFAULT_EYE)


def test_elevation_is_clamped_short_of_the_degenerate_pole() -> None:
    # At exactly +/-90 deg the eye is collinear with the +z up vector and
    # Plotly's view matrix degenerates, so the poles are approached, not hit.
    top = camera_eye(0.0, 90.0, 1.0)

    assert top["z"] == pytest.approx(1.0, abs=1e-4)
    assert top["z"] < 1.0
    assert camera_eye(0.0, 120.0, 1.0) == camera_eye(0.0, 90.0, 1.0)
    assert camera_eye(0.0, -120.0, 1.0) == camera_eye(0.0, -90.0, 1.0)


def test_scene_camera_names_its_projection_in_both_directions() -> None:
    assert scene_camera(0.0, 0.0, 1.0)["projection"]["type"] == "perspective"
    assert scene_camera(0.0, 0.0, 1.0, orthographic=True)["projection"]["type"] == "orthographic"
    assert scene_camera(0.0, 0.0, 1.0)["up"] == {"x": 0.0, "y": 0.0, "z": 1.0}


def test_resolve_scene_camera_default_keeps_the_hosts_own_view() -> None:
    camera = resolve_scene_camera("default", 0.0, 0.0, default_angles=VOLUME_CAMERA_ANGLES)

    assert _eye_vector(camera) == pytest.approx(_CAMERA_EYE)


def test_resolve_scene_camera_custom_takes_the_slider_angles() -> None:
    camera = resolve_scene_camera("custom", 90.0, 0.0, default_angles=DEFAULT_ANGLES)
    distance = DEFAULT_ANGLES[2]

    assert _eye_vector(camera) == pytest.approx([0.0, distance, 0.0], abs=1e-12)


def test_resolve_scene_camera_presets_look_along_their_named_axis() -> None:
    distance = DEFAULT_ANGLES[2]
    along = {
        "down x": [distance, 0.0, 0.0],
        "down y": [0.0, distance, 0.0],
        "down z": [0.0, 0.0, distance],
    }
    for preset, expected in along.items():
        camera = resolve_scene_camera(preset, 0.0, 0.0, default_angles=DEFAULT_ANGLES)
        # Each preset carries the orbit axis that reaches it head-on, so "down
        # z" lands ON the z axis instead of against a z-orbit's pole clamp.
        assert _eye_vector(camera) == pytest.approx(expected, abs=1e-12)


def test_orbit_axis_repoles_the_sphere_so_every_direction_is_reachable() -> None:
    # Orbiting about z can never bring the eye over the z pole (the clamp), but
    # orbiting about x or y sweeps straight through it.
    over_the_pole = camera_eye(90.0, 0.0, 1.0, "x")

    assert over_the_pole == pytest.approx({"x": 0.0, "y": 0.0, "z": 1.0}, abs=1e-12)
    assert camera_eye(0.0, 90.0, 1.0, "z")["z"] < 1.0  # clamped, for contrast
    # The clamp only ever blocks the axis being orbited about.
    for axis in ("x", "y", "z"):
        assert camera_eye(0.0, 90.0, 1.0, axis)[axis] < 1.0


def test_up_vector_follows_the_orbit_axis() -> None:
    # Plotly's view degenerates when the eye is collinear with `up`, so the up
    # vector has to move with the pole rather than stay pinned to +z.
    for axis in ("x", "y", "z"):
        camera = scene_camera(0.0, 0.0, 1.0, orbit_axis=axis)
        assert camera["up"][axis] == 1.0
        assert sum(camera["up"].values()) == 1.0


def test_orbit_axis_keeps_the_eye_on_its_sphere() -> None:
    for axis in ("x", "y", "z"):
        for azimuth, elevation in ((0.0, 0.0), (37.0, -22.0), (180.0, 61.0)):
            eye = camera_eye(azimuth, elevation, 1.7, axis)
            assert np.linalg.norm(list(eye.values())) == pytest.approx(1.7)


def test_unknown_preset_falls_back_to_the_default_view() -> None:
    # A stale persisted control value must not throw or produce a blank scene.
    assert "nonsense" not in CAMERA_PRESETS
    fallback = resolve_scene_camera("nonsense", 0.0, 0.0, default_angles=DEFAULT_ANGLES)

    assert _eye_vector(fallback) == pytest.approx(DEFAULT_EYE)


def test_zoom_scales_the_eye_distance_under_every_preset() -> None:
    for preset in ("default", "custom", *CAMERA_PRESETS):
        near = resolve_scene_camera(preset, 45.0, 20.0, 2.0, default_angles=DEFAULT_ANGLES)
        far = resolve_scene_camera(preset, 45.0, 20.0, 0.5, default_angles=DEFAULT_ANGLES)
        # zoom > 1 moves the eye closer; the direction is untouched.
        assert np.linalg.norm(_eye_vector(near)) < np.linalg.norm(_eye_vector(far))
        assert np.linalg.norm(_eye_vector(near)) == pytest.approx(DEFAULT_ANGLES[2] / 2.0)


def test_data_aspect_ratio_matches_the_data_proportions() -> None:
    # aspectmode="data" proportions, rebuilt server-side: an axis spanning half
    # of the longest one gets half its ratio component.
    fig = go.Figure(go.Scatter3d(x=[0.0, 10.0], y=[0.0, 5.0], z=[0.0, 2.5]))

    assert data_aspect_ratio(fig) == pytest.approx({"x": 1.0, "y": 0.5, "z": 0.25})


def test_data_aspect_ratio_is_how_an_orthographic_scene_zooms() -> None:
    # An orthographic projection box is fixed, so the camera distance cannot
    # zoom it; plotly scales the aspect ratio instead. Uniform scaling means
    # magnification without distortion.
    fig = go.Figure(go.Scatter3d(x=[0.0, 10.0], y=[0.0, 5.0], z=[0.0, 2.5]))
    base = data_aspect_ratio(fig)
    zoomed = data_aspect_ratio(fig, 2.0)

    for axis in ("x", "y", "z"):
        assert zoomed[axis] == pytest.approx(2.0 * base[axis])


def test_scene_extents_spans_every_trace_and_ignores_segment_breaks() -> None:
    # Cell-edge traces are None-separated line segments; None must not poison
    # the extents (nor produce a NaN ratio).
    fig = go.Figure(
        [
            go.Scatter3d(x=[0.0, 1.0, None, 2.0], y=[0.0, 1.0, None, 1.0], z=[0.0, 1.0, None, 3.0]),
            go.Scatter3d(x=[-4.0], y=[0.5], z=[0.5]),
        ]
    )

    assert scene_extents(fig) == [(-4.0, 2.0), (0.0, 1.0), (0.0, 3.0)]
    assert all(np.isfinite(list(data_aspect_ratio(fig).values())))


def test_data_aspect_ratio_survives_a_flat_or_empty_scene() -> None:
    # A degenerate axis must not divide by zero or hand Plotly a 0 component.
    flat = go.Figure(go.Scatter3d(x=[0.0, 2.0], y=[1.0, 1.0], z=[0.0, 1.0]))

    assert all(value > 0.0 for value in data_aspect_ratio(flat).values())
    assert all(value > 0.0 for value in data_aspect_ratio(go.Figure()).values())
