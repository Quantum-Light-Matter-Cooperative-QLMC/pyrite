"""Guard tests for the 2D electron-penetration views (Plotly) and their frames.

The renderer-neutral frame builders (``survival_frame``, ``tracks_frame``,
``track_segments_frame``, ``vacuum_segments_frame``) take already-simulated
``_trajectory_data`` dicts, so they are tested on synthetic data with no Monte
Carlo. The ``*_chart`` entry points run the electron transport; small-Ne
integration smokes exercise that plumbing on a real case.
"""

import numpy as np
import plotly.graph_objects as go

from pyrite.plots._frames import survival_frame
from pyrite.plots._trajectory_frames import (
    track_segments_frame,
    tracks_frame,
    vacuum_segments_frame,
)
from pyrite.plots.plotly import penetration
from pyrite.plots.plotly.penetration import penetration_survival_chart, trajectory_chart
from pyrite.plots.plotly.trajectories import _GROOVE


# ---- pure frame builders (synthetic, no MC) ----------------------------------
def _survival_data(max_depths, thick=1.0):
    """A minimal _trajectory_data-like dict: one segment per electron whose depth
    is its max depth."""
    n = len(max_depths)
    return {
        "Ne": n,
        "elec_id": np.arange(n),
        "z_u": np.asarray(max_depths, dtype=float),
        "thick": thick,
    }


def test_survival_frame_monotone_and_full_at_surface():
    data_by_energy = {
        30.0: _survival_data([0.1, 0.3, 0.5, 0.7]),
        60.0: _survival_data([0.2, 0.5, 0.8, 0.95]),
    }
    df = survival_frame(data_by_energy, n_bins=40)
    assert list(df.columns) == ["depth", "survival", "energy"]
    assert set(df["energy"]) == {"30 keV", "60 keV"}
    for lbl in ("30 keV", "60 keV"):
        s = df[df.energy == lbl].sort_values("depth")["survival"].to_numpy()
        assert s[0] == 100.0  # everyone survives at the surface
        assert np.all(np.diff(s) <= 1e-9)  # monotonically non-increasing


def test_survival_frame_absolute_depth():
    df = survival_frame(
        {30.0: _survival_data([0.5, 1.0, 1.5], thick=2.0)}, n_bins=20, depth_frac=False
    )
    assert float(df["depth"].max()) == 2.0  # absolute depth spans the thickness


def test_tracks_frame_splits_on_nan():
    data = {
        "px": np.array([1.0, 2.0, np.nan, 3.0, 4.0]),
        "py": np.array([0.0, 0.1, np.nan, 0.2, 0.3]),
        "pE": np.array([30.0, 20.0, np.nan, 25.0, 15.0]),
    }
    df = tracks_frame(data)
    assert list(df.columns) == ["x", "y", "E", "track"]
    assert df["track"].nunique() == 2  # two electrons
    assert not df[["x", "y"]].isna().any().any()  # NaN break rows dropped


def test_track_segments_frame_no_cross_electron_bridge():
    # two electrons (2 verts each) -> exactly ONE segment per electron, never a
    # segment bridging the NaN gap between them.
    data = {
        "px": np.array([1.0, 2.0, np.nan, 3.0, 4.0]),
        "py": np.array([0.0, 0.1, np.nan, 0.2, 0.3]),
        "pE": np.array([30.0, 20.0, np.nan, 25.0, 15.0]),
    }
    seg = track_segments_frame(data)
    assert list(seg.columns) == ["x", "y", "x2", "y2", "E"]
    assert len(seg) == 2
    assert seg[(seg["x"] == 2.0) & (seg["x2"] == 3.0)].empty
    # sorted by E ascending so high-energy strokes last (on top)
    assert list(seg["E"]) == sorted(seg["E"])


def _vacuum_data():
    return {
        "px": np.array([0.0, 1.0]),
        "py": np.array([0.0, 0.5]),
        "pE": np.array([30.0, 20.0]),
        "pts": np.array([[0.0, 0.0], [1.0, 0.5]]),
        "beam": np.array([1.0, 0.0, 0.0]),
        "detector": np.array([0.0, 1.0, 0.0]),
        "nslab": np.array([0.0, 1.0]),
        "ndet": np.array([0.0, 1.0]),
        "u": 10.0,
        "ulab": "nm",
        "thick": 1.0,
        "layer_bounds": [],
        "Ne": 1,
        "eta": 0.0,
        "thru": 0.0,
        "vacuum_start_xyz": np.array([[1.0, 2.0, 3.0]]),
        "vacuum_end_xyz": np.array([[4.0, 5.0, 6.0]]),
        "vacuum_E": np.array([17.0]),
        "vacuum_t_fs": np.array([0.75]),
        "vacuum_elec_id": np.array([3], dtype=np.int64),
    }


def test_vacuum_segments_frame_stays_separate_from_radiating_tracks():
    data = _vacuum_data()
    material = track_segments_frame(data)
    vacuum = vacuum_segments_frame(data)

    assert len(material) == 1
    assert list(vacuum.columns) == ["x", "y", "x2", "y2", "E", "t_fs", "elec_id"]
    assert len(vacuum) == 1
    np.testing.assert_allclose(
        vacuum.loc[0, ["x", "y", "x2", "y2"]].to_numpy(dtype=float),
        [1.0, 2.0, 4.0, 5.0],
    )
    assert vacuum.loc[0, "E"] == 17.0
    assert vacuum.loc[0, "t_fs"] == 0.75
    assert vacuum.loc[0, "elec_id"] == 3


# ---- chart plumbing (small-Ne integration on a real case) --------------------
def _real_cases(material="hopg"):
    from pyrite.campaign.config import default_settings, trajectory_sweep
    from pyrite.campaign.sweep import build_cases

    s = default_settings()
    sweep = trajectory_sweep(material)
    return build_cases(sweep, s.n_electrons, s.n_electrons_brem)


def _grooved_case(spacing_ang=2.0e4, tilt_deg=45.0, energy=30.0, thickness_ang=2.0e5):
    from pyrite.campaign.config import default_settings, trajectory_sweep
    from pyrite.campaign.sweep import build_cases

    s = default_settings()
    sweep = trajectory_sweep(
        "hopg",
        energies=(energy,),
        tilts=(tilt_deg,),
        thickness_ang=thickness_ang,
        azim_deg=180.0,
        groove_spacing_ang=spacing_ang,
    )
    return build_cases(
        sweep,
        s.n_electrons,
        s.n_electrons_brem,
        inelastic_model="continuous",
        straggling=False,
        radiative_model="uncoupled",
    )[0]


def _named(fig, name):
    return [trace for trace in fig.data if trace.name == name]


def test_penetration_survival_chart_one_bounded_line_per_energy():
    fig = penetration_survival_chart(_real_cases(), Ne=6, n_bins=20)
    assert isinstance(fig, go.Figure)
    assert fig.data
    energies = [trace.name for trace in fig.data]
    assert len(energies) == len(set(energies))
    for trace in fig.data:
        survival = np.asarray(trace.y, dtype=float)
        assert survival[0] == 100.0
        assert np.all(np.diff(survival) <= 1e-9)
    assert list(fig.layout.yaxis.range) == [0, 100]
    assert fig.layout.yaxis.title.text == "surviving electrons (% of N0)"


def test_penetration_survival_chart_none_on_empty():
    assert penetration_survival_chart([]) is None


def test_trajectory_chart_draws_energy_coloured_segments_with_colourbar():
    fig = trajectory_chart(_real_cases()[0], Ne=6)
    assert isinstance(fig, go.Figure)
    tracks = _named(fig, "tracks")
    assert len(tracks) == 1
    marker = tracks[0].marker
    assert marker.colorscale is not None and marker.colorbar.title.text
    assert marker.cmin == 5.0 and marker.cmax == _real_cases()[0]["E0_keV"]
    bands = [t for t in fig.data if t.mode == "lines" and t.name is None]
    assert bands, "no energy-band line traces"
    # Each band draws independent segments: NaN after every segment end point.
    for band in bands:
        xs = np.asarray(band.x, dtype=float)
        assert np.isnan(xs[2::3]).all() and np.isfinite(xs[0::3]).all()
    fig.to_json()  # serializable


def test_trajectory_chart_segments_never_bridge_electrons(monkeypatch):
    data = _vacuum_data()
    data["px"] = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    data["py"] = np.array([0.0, 0.1, np.nan, 0.2, 0.3])
    data["pE"] = np.array([30.0, 20.0, np.nan, 25.0, 15.0])
    monkeypatch.setattr(penetration, "_trajectory_data", lambda *_args, **_kw: data)
    fig = trajectory_chart(_real_cases()[0], Ne=2, frame=(-1.0, 5.0, -1.0, 5.0))
    pairs = set()
    for band in (t for t in fig.data if t.mode == "lines" and t.name is None):
        xs = np.asarray(band.x, dtype=float)
        pairs |= set(zip(xs[0::3], xs[1::3], strict=True))
    assert pairs == {(1.0, 2.0), (3.0, 4.0)}


def test_trajectory_chart_grooved_has_profile_trace():
    fig = trajectory_chart(_grooved_case(), Ne=8)
    groove = _named(fig, "groove profile")
    assert len(groove) == 1
    assert groove[0].line.color == _GROOVE


def test_trajectory_chart_ungrooved_has_no_profile_trace():
    assert not _named(trajectory_chart(_real_cases()[0], Ne=8), "groove profile")


def test_vacuum_legs_stay_separate_from_radiating_track_data():
    from pyrite.plots.mpl.trajectories import _trajectory_data

    data = _trajectory_data(_grooved_case(), Ne=40, seed=42)

    assert len(data["vacuum_start_xyz"]) == len(data["vacuum_end_xyz"])
    assert len(data["vacuum_start_xyz"]) == len(data["vacuum_E"])
    assert len(data["vacuum_start_xyz"]) > 0
    assert len(data["start_xyz"]) == len(data["E"])
    assert len(data["start_xyz"]) != len(data["E"]) + len(data["vacuum_E"])


def test_flat_trajectory_data_has_typed_empty_vacuum_arrays():
    from pyrite.plots.mpl.trajectories import _trajectory_data

    data = _trajectory_data(_real_cases()[0], Ne=4, seed=7)

    assert data["vacuum_start_xyz"].shape == (0, 3)
    assert data["vacuum_end_xyz"].shape == (0, 3)
    assert data["vacuum_E"].shape == (0,)
    assert data["vacuum_t_fs"].shape == (0,)
    assert data["vacuum_elec_id"].shape == (0,)
    assert np.issubdtype(data["vacuum_start_xyz"].dtype, np.floating)
    assert np.issubdtype(data["vacuum_end_xyz"].dtype, np.floating)
    assert np.issubdtype(data["vacuum_E"].dtype, np.floating)
    assert np.issubdtype(data["vacuum_t_fs"].dtype, np.floating)
    assert data["vacuum_elec_id"].dtype == np.int64


def test_trajectory_chart_draws_vacuum_as_separate_faint_trace(monkeypatch):
    data = _vacuum_data()
    monkeypatch.setattr(penetration, "_trajectory_data", lambda *_args, **_kw: data)

    fig = trajectory_chart(_grooved_case(), Ne=1)
    vacuum = _named(fig, "vacuum flight")
    assert len(vacuum) == 1
    np.testing.assert_allclose(np.asarray(vacuum[0].x, dtype=float)[:2], [1.0, 4.0])
    assert len(_named(fig, "tracks")) == 1
    assert "0.35" in vacuum[0].line.color  # faint, distinct from energy-coloured tracks
