"""Regression checks for browser-native 3D penetration plots."""

import numpy as np
import plotly.graph_objects as go

from cxr_mc.plots.plotly_trajectories import (
    N_FRAMES,
    _exit_paths_3d,
    advance_frame,
    case_t_max,
    dataset_t_max,
    frame_reveal_fs,
    track_vertices_3d,
)


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
    assert exit_trace.line.dash == "dot"  # tight dotted: first mark flush to exit point
    # dashes are colored by exit energy on the tracks' Turbo scale (per-vertex
    # color array), not one flat muted color, and don't add a second colorbar.
    assert exit_trace.line.colorscale is not None
    assert exit_trace.line.showscale is False
    assert np.asarray(exit_trace.line.color).size == np.asarray(exit_trace.z).size

    # exit_trace.{x,y,z} are NaN-separated triples per electron:
    # [face-exit point, extrapolated dash endpoint, NaN, ...]. The face-exit
    # point (index 0 of each triple) is the one guaranteed by _exit_paths_3d
    # to sit within tol of z=0 (backscattered, top face) or z=thick
    # (transmitted, bottom face); the extrapolated endpoint continues past the
    # face and is not face-adjacent, so only the face-exit points are checked.
    z = np.asarray(exit_trace.z, dtype=float)
    assert np.isnan(z[2::3]).all()  # every third point is the NaN separator
    face_z = z[0::3]
    assert np.isfinite(face_z).all()

    exit_face_trace = next(trace for trace in fig.data if trace.name == "exit face")
    thick = float(np.asarray(exit_face_trace.z, dtype=float)[0])
    tol = max(1e-6 * thick, np.finfo(float).eps)

    assert np.any(np.abs(face_z) <= tol)  # at least one backscattered (top-face) exit
    assert np.any(np.abs(face_z - thick) <= tol)  # at least one transmitted (bottom-face) exit
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


def test_track_vertices_3d_reveal_until_fs_filters_by_start_age():
    data = {
        "start_xyz": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        "end_xyz": np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]]),
        "E": np.array([30.0, 20.0, 10.0]),
        "elec_id": np.array([0, 0, 1]),
        "t_fs": np.array([0.0, 5.0, 9.0]),
    }
    xyz_full, _, _ = track_vertices_3d(data)
    xyz_cut, energy_cut, _ = track_vertices_3d(data, t_fs=data["t_fs"], reveal_until_fs=5.0)

    assert xyz_cut.shape[0] < xyz_full.shape[0]  # strictly fewer vertices
    assert xyz_cut.shape[0] == 6  # 2 of 3 segments (t_fs 0.0, 5.0 <= 5.0) survive
    # revealed segment's energy (30, 20) is the pair with t_fs <= cutoff, not (10,)
    np.testing.assert_array_equal(energy_cut[[0, 1, 3, 4]], [30.0, 30.0, 20.0, 20.0])


def test_track_vertices_3d_reveal_until_fs_none_is_unfiltered():
    data = {
        "start_xyz": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        "end_xyz": np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]),
        "E": np.array([30.0, 20.0]),
        "elec_id": np.array([0, 1]),
        "t_fs": np.array([0.0, 9.0]),
    }
    xyz_default, _, _ = track_vertices_3d(data)
    xyz_explicit_none, _, _ = track_vertices_3d(data, t_fs=data["t_fs"], reveal_until_fs=None)
    np.testing.assert_array_equal(xyz_default, xyz_explicit_none)


def test_exit_paths_3d_reveal_until_fs_gates_by_terminal_start_age():
    data = {
        "elec_id": np.array([0, 1]),
        "start_xyz": np.array([[0.0, 0.0, 0.9], [0.0, 0.0, 0.1]]),
        "end_xyz": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        "E": np.array([20.0, 5.0]),
        "t_fs": np.array([2.0, 8.0]),
        "thick": 1.0,
    }
    full = _exit_paths_3d(data, 0.1, cmax=30.0)
    gated = _exit_paths_3d(data, 0.1, cmax=30.0, reveal_until_fs=5.0)

    assert len(full.x) == 6  # 2 electrons x (start, end, NaN)
    assert len(gated.x) == 3  # only electron 0 (age 2.0 <= 5.0) survives
    assert np.isnan(gated.z[2])


def test_trajectory_volume_figure_reveal_until_fs_reduces_track_vertices():
    from cxr_mc.plots.plotly_trajectories import (
        _ZOOM_BEAM_FWHM_MM,
        _trajectory_data,
        trajectory_volume_figure,
    )
    from cxr_mc.plots.trajectories import _case_of

    case = _hopg_thin_slab_case()
    data = _trajectory_data(_case_of(case), 80, 0, beam_fwhm_mm=_ZOOM_BEAM_FWHM_MM)
    t_max = dataset_t_max(data)
    cutoff = frame_reveal_fs(N_FRAMES // 2, t_max)
    assert 0.0 < cutoff < t_max  # midpoint strictly inside range, else test proves nothing

    fig_full = trajectory_volume_figure(case, Ne=80, seed=0)
    fig_cut = trajectory_volume_figure(case, Ne=80, seed=0, reveal_until_fs=cutoff)

    tracks_full = next(t for t in fig_full.data if t.name == "electron tracks")
    tracks_cut = next(t for t in fig_cut.data if t.name == "electron tracks")
    n_full = np.count_nonzero(np.isfinite(np.asarray(tracks_full.x)))
    n_cut = np.count_nonzero(np.isfinite(np.asarray(tracks_cut.x)))
    assert n_cut < n_full


def test_frame_reveal_fs_endpoints_and_midpoint():
    t_max = 120.0
    assert frame_reveal_fs(0, t_max) == 0.0
    assert frame_reveal_fs(N_FRAMES - 1, t_max) == t_max
    mid = frame_reveal_fs((N_FRAMES - 1) // 2, t_max)
    assert 0.0 < mid < t_max


def test_dataset_t_max_from_synthetic_data():
    data = {"t_fs": np.array([1.0, np.nan, 7.5, 3.0])}
    assert dataset_t_max(data) == 7.5
    assert dataset_t_max({"t_fs": np.array([])}) == 0.0


def test_advance_frame_wraps_when_repeat():
    next_index, playing = advance_frame(N_FRAMES - 2, speed=5, repeat=True)
    assert next_index == 0
    assert playing is True


def test_advance_frame_clamps_when_not_repeat():
    next_index, playing = advance_frame(N_FRAMES - 2, speed=5, repeat=False)
    assert next_index == N_FRAMES - 1
    assert playing is False


def test_advance_frame_speed_scaling():
    next_index, playing = advance_frame(0, speed=3, repeat=False)
    assert next_index == 3
    assert playing is True


def test_advance_frame_slow_speed_never_stalls():
    # speed=0.25 -> round(0.25) == 0, which must be floored to a 1-frame
    # step, not a no-op (see advance_frame stall bug fix).
    next_index, playing = advance_frame(0, speed=0.25, repeat=False)
    assert next_index == 1
    assert playing is True


def test_case_t_max_positive_for_hopg_case():
    t_max = case_t_max(_hopg_thin_slab_case(), Ne=10, seed=0)
    assert isinstance(t_max, float)
    assert t_max > 0.0


def test_case_t_max_matches_dataset_t_max_of_same_transport():
    from cxr_mc.plots.plotly_trajectories import _ZOOM_BEAM_FWHM_MM, _trajectory_data
    from cxr_mc.plots.trajectories import _case_of

    case = _hopg_thin_slab_case()
    data = _trajectory_data(_case_of(case), 10, 0, beam_fwhm_mm=_ZOOM_BEAM_FWHM_MM)
    assert case_t_max(case, Ne=10, seed=0) == dataset_t_max(data)


def test_case_t_max_zero_safe_on_trivial_input():
    # An electron count small enough (or a degenerate case) that transport
    # yields no finite ages must not raise -- dataset_t_max's 0.0-safe default
    # propagates through unchanged.
    assert case_t_max(_hopg_thin_slab_case(), Ne=1, seed=0) >= 0.0
