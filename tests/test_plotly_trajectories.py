"""Regression checks for browser-native 3D penetration plots."""

import numpy as np
import plotly.graph_objects as go

from cxr_mc.plots.plotly_trajectories import (
    N_FRAMES,
    _exit_paths_3d,
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
    assert exit_trace.line.dash == "solid"  # solid: continuous from exit point, no gaps
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


def test_exit_paths_3d_empty_ok_returns_empty_trace_instead_of_none():
    # A dataset with zero exits at this cutoff: the default (empty_ok=False)
    # keeps returning None (unchanged contract); empty_ok=True instead
    # returns a same-styled, zero-length trace -- what trajectory_volume_
    # animation needs so a frame can always update this trace's fixed index.
    data = {
        "elec_id": np.array([0]),
        "start_xyz": np.array([[0.0, 0.0, 0.4]]),
        "end_xyz": np.array([[0.0, 0.0, 0.5]]),
        "E": np.array([10.0]),
        "t_fs": np.array([1.0]),
        "thick": 1.0,
    }
    assert _exit_paths_3d(data, 0.1, cmax=30.0) is None

    empty = _exit_paths_3d(data, 0.1, cmax=30.0, empty_ok=True)
    assert empty is not None
    assert empty.type == "scatter3d"
    assert len(empty.x) == 0
    assert empty.line.colorscale is not None  # still styled like a real exit-path trace


def _hopg_thin_slab_data(Ne=80, seed=0):
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_data

    return trajectory_volume_data(_hopg_thin_slab_case(), Ne=Ne, seed=seed)


def test_trajectory_volume_animation_returns_expected_frame_count():
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_animation

    case = _hopg_thin_slab_case()
    data = _hopg_thin_slab_data()
    fig = trajectory_volume_animation(case, data)

    assert len(fig.frames) == N_FRAMES
    assert [frame.name for frame in fig.frames] == [str(k) for k in range(N_FRAMES)]


def test_trajectory_volume_animation_n_frames_override():
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_animation

    case = _hopg_thin_slab_case()
    data = _hopg_thin_slab_data()
    fig = trajectory_volume_animation(case, data, n_frames=5)

    assert len(fig.frames) == 5


def test_trajectory_volume_animation_base_traces_match_full_reveal_figure():
    # Base figure (before any frame is selected) must be the SAME full-reveal
    # content trajectory_volume_figure_from_data has always produced -- the
    # animation only ADDS frames + playback controls on top of it.
    from cxr_mc.plots.plotly_trajectories import (
        trajectory_volume_animation,
        trajectory_volume_figure_from_data,
    )

    case = _hopg_thin_slab_case()
    data = _hopg_thin_slab_data()
    fig = trajectory_volume_animation(case, data)
    full_fig = trajectory_volume_figure_from_data(case, data)

    names = {trace.name for trace in fig.data}
    assert names == {trace.name for trace in full_fig.data}
    assert {"crystal volume", "electron tracks", "incident beam", "detector direction"} <= names

    tracks = next(t for t in fig.data if t.name == "electron tracks")
    full_tracks = next(t for t in full_fig.data if t.name == "electron tracks")
    n_revealed = np.count_nonzero(np.isfinite(np.asarray(tracks.x, dtype=float)))
    n_full = np.count_nonzero(np.isfinite(np.asarray(full_tracks.x, dtype=float)))
    assert n_revealed == n_full


def test_trajectory_volume_animation_frames_grow_monotonically_revealed():
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_animation

    case = _hopg_thin_slab_case()
    data = _hopg_thin_slab_data()
    fig = trajectory_volume_animation(case, data)

    def n_finite(frame):
        return np.count_nonzero(np.isfinite(np.asarray(frame.data[0].x, dtype=float)))

    counts = [n_finite(frame) for frame in fig.frames]
    assert counts == sorted(counts)  # non-decreasing reveal across the pass
    assert counts[0] < counts[-1]  # frame 0 reveals strictly less than the last frame
    # frame trace omits per-frame customdata (only the base trace carries it,
    # see trajectory_volume_animation's PAYLOAD note) -- confirms the trim.
    assert fig.frames[0].data[0].customdata is None


def test_trajectory_volume_animation_exit_path_traced_every_frame_when_present():
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_animation

    case = _hopg_thin_slab_case()
    data = _hopg_thin_slab_data()
    fig = trajectory_volume_animation(case, data)

    names = [trace.name for trace in fig.data]
    assert "exit path" in names  # this fixture always has both exit kinds
    exit_idx = names.index("exit path")
    assert all(exit_idx in frame.traces for frame in fig.frames)


def test_trajectory_volume_animation_updatemenus_and_slider_present():
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_animation

    case = _hopg_thin_slab_case()
    data = _hopg_thin_slab_data()
    fig = trajectory_volume_animation(case, data)

    assert fig.layout.updatemenus
    labels = [button.label for button in fig.layout.updatemenus[0].buttons]
    assert any("Play" in label for label in labels)
    assert any("Pause" in label for label in labels)
    for button in fig.layout.updatemenus[0].buttons:
        assert button.args[1]["frame"]["redraw"] is True  # scatter3d needs redraw

    assert fig.layout.sliders
    assert len(fig.layout.sliders[0].steps) == N_FRAMES
    fig.to_json()  # the whole animated payload remains serializable


def test_trajectory_volume_animation_speed_scales_frame_duration():
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_animation

    case = _hopg_thin_slab_case()
    data = _hopg_thin_slab_data()
    fig_1x = trajectory_volume_animation(case, data, speed=1.0)
    fig_2x = trajectory_volume_animation(case, data, speed=2.0)

    duration_1x = fig_1x.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]
    duration_2x = fig_2x.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"]
    assert duration_2x == duration_1x / 2


# ---- blazed groove profile (Task 6) ------------------------------------------
def _hopg_grooved_case(spacing_ang=2.0e4, tilt_deg=45.0, energy=30.0, thickness_ang=2.0e5):
    """One grooved hopg case: azim=180, 0<tilt<90, theta_obs=90 -- the restricted
    geometry blazed_groove_spec / build_cases accept."""
    from cxr_mc.config import default_settings, trajectory_sweep
    from cxr_mc.sweep import build_cases

    settings = default_settings()
    sweep = trajectory_sweep(
        "hopg",
        energies=(energy,),
        tilts=(tilt_deg,),
        thickness_ang=thickness_ang,
        azim_deg=180.0,
        groove_spacing_ang=spacing_ang,
    )
    return build_cases(sweep, settings.n_electrons, settings.n_electrons_brem)[0]


def _groove_spec_45(spacing_ang=2.0e4):
    from cxr_mc.montecarlo.groove import blazed_groove_spec

    return blazed_groove_spec(spacing_ang, np.pi / 2, np.deg2rad(45.0), np.pi)


def test_groove_profile_knots_depth_spans_zero_to_groove_depth():
    """Sample-frame sawtooth geometry, tested BEFORE any lab-frame rotation:
    apexes touch z=0, valley floors reach the groove depth, x strictly increases."""
    from cxr_mc.plots.trajectories import groove_profile_knots

    spec = _groove_spec_45()
    xs, zs = groove_profile_knots(-3 * spec.spacing_ang, 3 * spec.spacing_ang, spec)
    assert zs.min() < 1e-6
    assert abs(zs.max() - spec.depth_ang) < 1e-6 * spec.depth_ang
    assert np.all(np.diff(xs) > 0)


def test_groove_profile_knots_falls_back_over_period_cap():
    from cxr_mc.plots.trajectories import groove_profile_knots

    spec = _groove_spec_45()
    assert groove_profile_knots(0.0, 500 * spec.spacing_ang, spec, max_periods=200) is None


def test_grooved_case_draws_groove_profile_surface():
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_figure

    case = _hopg_grooved_case()
    assert case.get("groove_spacing_ang") == 2.0e4
    fig = trajectory_volume_figure(case, Ne=12)
    grooves = [trace for trace in fig.data if trace.name == "groove profile"]
    assert len(grooves) == 1
    assert grooves[0].type == "mesh3d"
    assert len(grooves[0].x) > 0
    fig.to_json()  # payload stays serializable


def test_ungrooved_case_has_no_groove_profile_surface():
    from cxr_mc.config import default_settings, trajectory_sweep
    from cxr_mc.plots.plotly_trajectories import trajectory_volume_figure
    from cxr_mc.sweep import build_cases

    settings = default_settings()
    case = build_cases(
        trajectory_sweep("hopg", energies=(30.0,), tilts=(0.0,)),
        settings.n_electrons,
        settings.n_electrons_brem,
    )[0]
    fig = trajectory_volume_figure(case, Ne=12)
    assert not any(trace.name == "groove profile" for trace in fig.data)


def test_grooved_trajectory_entries_land_on_relief_facets():
    """Drawn tracks for a grooved case genuinely start on the relief facets: the
    first segment of each electron enters at depth z in [0, groove depth), and the
    uniform-phase fallback spreads entries across many phases."""
    from cxr_mc.plots.trajectories import _trajectory_data

    case = _hopg_grooved_case()
    data = _trajectory_data(case, 60, 0)
    h = _groove_spec_45().depth_ang
    elec = np.asarray(data["elec_id"])
    _, first = np.unique(elec, return_index=True)
    entry_z = np.asarray(data["start_xyz"])[first, 2] * data["u"]  # display -> Ang
    assert np.all(entry_z >= -1e-6)
    assert np.all(entry_z < h + 1e-3)
    assert entry_z.max() > 0.1 * h  # on the facets, not the flat face
    assert np.unique(np.round(entry_z, 3)).size > 3  # multiple lateral phases
