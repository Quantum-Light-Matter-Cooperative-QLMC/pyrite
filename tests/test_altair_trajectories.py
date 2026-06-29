"""Guard tests for the Altair trajectory renderers (cxr_mc.plots.altair_trajectories).

The pure frame builders (survival_frame / tracks_frame) take already-simulated
``_trajectory_data`` dicts, so they're tested on synthetic data with NO Monte
Carlo. The ``*_chart`` entry points run the electron transport; one small-Ne
integration smoke each exercises that plumbing on a real case. Charts'
``to_dict()`` serializes under Vega-Lite's row cap.
"""

import altair as alt
import numpy as np

from cxr_mc.plots.altair_trajectories import (
    survival_frame,
    track_segments_frame,
    tracks_frame,
)


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
    assert len(seg) == 2  # one segment per 2-vertex electron, no cross-bridge
    # the would-be bridge (2.0 -> 3.0) must NOT appear as a segment
    bridges = seg[(seg["x"] == 2.0) & (seg["x2"] == 3.0)]
    assert bridges.empty
    # sorted by E ascending so high-energy strokes last (on top)
    assert list(seg["E"]) == sorted(seg["E"])


# ---- chart plumbing (small-Ne integration on a real case) --------------------
def _real_cases(material="hopg"):
    from cxr_mc.config import default_settings, trajectory_sweep
    from cxr_mc.sweep import build_cases

    s = default_settings()
    sweep = trajectory_sweep(material)
    return build_cases(sweep, s.n_electrons, s.n_electrons_brem)


def test_penetration_survival_chart_builds_valid_spec():
    from cxr_mc.plots.altair_trajectories import penetration_survival_chart

    chart = penetration_survival_chart(_real_cases(), Ne=6, n_bins=20)
    assert isinstance(chart, alt.Chart)
    chart.to_dict()  # raises if malformed


def test_trajectory_chart_builds_valid_spec():
    from cxr_mc.plots.altair_trajectories import trajectory_chart

    chart = trajectory_chart(_real_cases()[0], Ne=6)
    assert isinstance(chart, alt.LayerChart)
    chart.to_dict()


def test_trajectory_chart_draws_tracks_as_rule_segments():
    """Regression guard for the invisible-tracks bug: the energy-coloured tracks
    must be drawn as ``rule`` segments (x2/y2), NOT a ``line`` with a quantitative
    ``color`` -- which Vega-Lite renders as nothing. A valid-spec/data test cannot
    catch this; the mark type is the property that distinguishes working from
    broken."""
    from cxr_mc.plots.altair_trajectories import trajectory_chart

    spec = trajectory_chart(_real_cases()[0], Ne=6).to_dict()
    track_layers = [
        layer
        for layer in spec["layer"]
        if layer.get("encoding", {}).get("color", {}).get("field") == "E"
    ]
    assert track_layers, "no energy-coloured track layer found"
    for layer in track_layers:
        mark = layer["mark"]
        mark_type = mark["type"] if isinstance(mark, dict) else mark
        assert mark_type == "rule", f"tracks drawn as {mark_type!r}, not 'rule'"
        assert "x2" in layer["encoding"] and "y2" in layer["encoding"]


def test_penetration_survival_chart_none_on_empty():
    from cxr_mc.plots.altair_trajectories import penetration_survival_chart

    assert penetration_survival_chart([]) is None
