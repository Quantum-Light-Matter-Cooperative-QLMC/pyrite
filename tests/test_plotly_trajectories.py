"""Regression checks for browser-native 3D penetration plots."""

import numpy as np
import plotly.graph_objects as go

from cxr_mc.plots.plotly_trajectories import track_vertices_3d


def test_track_vertices_3d_separates_every_physical_segment():
    data = {
        "start_xyz": np.array([[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]]),
        "end_xyz": np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]),
        "E": np.array([30.0, 12.0]),
        "elec_id": np.array([0, 1]),
    }
    xyz, energy, elec_id = track_vertices_3d(data)

    assert xyz.shape == (6, 3)
    expected = np.vstack(
        (
            data["start_xyz"][0],
            data["end_xyz"][0],
            data["start_xyz"][1],
            data["end_xyz"][1],
        )
    )
    np.testing.assert_array_equal(xyz[[0, 1, 3, 4]], expected)
    assert np.isnan(xyz[[2, 5]]).all()
    np.testing.assert_array_equal(energy[[0, 1, 3, 4]], [30.0, 30.0, 12.0, 12.0])
    np.testing.assert_array_equal(elec_id[[0, 1, 3, 4]], [0.0, 0.0, 1.0, 1.0])


def test_trajectory_volume_figure_contains_volume_tracks_and_direction_arrows():
    from cxr_mc.config import default_settings, trajectory_sweep
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_figure
    from cxr_mc.sweep import build_cases

    settings = default_settings()
    case = build_cases(
        trajectory_sweep("hopg", energies=(30.0,), tilts=(0.0,)),
        settings.n_electrons,
        settings.n_electrons_brem,
    )[0]
    fig = trajectory_volume_figure(case, Ne=6)

    assert isinstance(fig, go.Figure)
    assert any(trace.type == "mesh3d" and trace.name == "crystal volume" for trace in fig.data)
    tracks = next(trace for trace in fig.data if trace.name == "electron tracks")
    assert tracks.type == "scatter3d"
    assert tracks.line.colorscale
    assert {trace.name for trace in fig.data} >= {"beam", "detector direction"}
    assert fig.layout.scene.aspectmode == "data"
    fig.to_json()  # browser/export payload remains serializable
