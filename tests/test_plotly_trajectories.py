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
    # beam is now a particle BUNDLE (one stub per electron), not a single arrow
    assert {trace.name for trace in fig.data} >= {"incident beam", "detector direction"}
    beam = next(trace for trace in fig.data if trace.name == "incident beam")
    assert beam.type == "scatter3d"
    assert fig.layout.scene.aspectmode == "data"
    fig.to_json()  # browser/export payload remains serializable


def _hopg_thin_slab_case():
    """30 keV / hopg / 4 um slab at normal incidence: with Ne=80, seed=0 (the
    ``trajectory_volume_figure`` defaults) this mix produces BOTH backscattered
    (top-face) and transmitted (bottom-face) terminal segments, so it exercises
    both exit-path branches."""
    from cxr_mc.config import default_settings, trajectory_sweep
    from cxr_mc.sweep import build_cases

    settings = default_settings()
    return build_cases(
        trajectory_sweep("hopg", energies=(30.0,), tilts=(0.0,), thickness_ang=40000.0),
        settings.n_electrons,
        settings.n_electrons_brem,
    )[0]


def test_exit_paths_drawn_once_for_backscatter_and_transmission():
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_figure

    case = _hopg_thin_slab_case()
    fig = trajectory_volume_figure(case, Ne=80)

    exit_traces = [trace for trace in fig.data if trace.name == "exit path"]
    assert len(exit_traces) == 1  # one legend entry for every electron's exit dash
    exit_trace = exit_traces[0]
    assert exit_trace.type == "scatter3d"
    assert exit_trace.line.dash == "dash"
    # NaN-separated: at least one break between per-electron dashes, i.e. more
    # than one exit path drawn (both a backscattered and a transmitted electron).
    assert np.isnan(np.asarray(exit_trace.z, dtype=float)).sum() >= 2
    fig.to_json()


def test_beam_fwhm_mm_override_changes_footprint_extent():
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_figure

    case = _hopg_thin_slab_case()
    fig_narrow = trajectory_volume_figure(case, Ne=6, realistic=True, beam_fwhm_mm=0.2)
    fig_wide = trajectory_volume_figure(case, Ne=6, realistic=True, beam_fwhm_mm=4.0)

    narrow = next(trace for trace in fig_narrow.data if trace.name == "beam footprint")
    wide = next(trace for trace in fig_wide.data if trace.name == "beam footprint")
    narrow_extent = float(np.max(np.abs(narrow.x)))
    wide_extent = float(np.max(np.abs(wide.x)))
    assert wide_extent > narrow_extent  # override, not the case's own beam_fwhm_mm


def test_ne_is_transported_literally_not_scaled():
    """``Ne`` used to be silently multiplied by a hidden ``_NE_SCALE`` factor;
    it is now the literal electron count handed to transport."""
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_figure

    case = _hopg_thin_slab_case()
    fig = trajectory_volume_figure(case, Ne=10, seed=0)
    tracks = next(trace for trace in fig.data if trace.name == "electron tracks")
    elec_id = np.asarray(tracks.customdata)[:, 1]
    n_electrons = len(np.unique(elec_id[np.isfinite(elec_id)]))
    assert n_electrons <= 10  # never inflated by a hidden multiplier
