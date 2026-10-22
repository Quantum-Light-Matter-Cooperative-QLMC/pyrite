"""Regression tests for the opt-in #622 Plotly cell-picking experiment.

No transport, detector response, checkpoint, or browser is needed: the frame
under test is the same prepared map contract used by the Altair renderer.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pyrite.apps.analysis_ui.map_selection import (
    map_scope,
    restore_selection,
    selection_for_scope,
)
from pyrite.plots.plotly.map_select import (
    cell_from_points,
    selectable_map_figure,
    selected_cell_frame,
    single_panel_frame,
)


def _frame():
    return pd.DataFrame(
        {
            "x": [0.0, 1.5, 0.0, 1.5, 0.0],
            "y": [2.0, 2.0, 4.0, 4.0, 2.0],
            "panel": ["30 keV"] * 4 + ["50 keV"],
            "panel_raw": [30.0] * 4 + [50.0],
            "value": [0.0, 4.0, 8.0, np.nan, 100.0],
            "name": ["a", "b", "c", "masked", "other-energy"],
        }
    )


def test_one_energy_and_missing_energy():
    selected = single_panel_frame(_frame(), 30.0)
    assert len(selected) == 4
    assert selected["name"].tolist() == ["a", "b", "c", "masked"]
    assert single_panel_frame(_frame(), 40.0).empty


def test_map_mask_zero_orientation_click_targets_and_units():
    selected = single_panel_frame(_frame(), 30.0)
    orig = selected.copy(deep=True)
    fig = selectable_map_figure(selected, label="line flux", color_domain=(0.0, 10.0))
    heatmap, target = fig.data
    z = np.asarray(heatmap.z)
    assert z.shape == (2, 2)
    assert z[0, 0] == 0.0
    assert z[1, 0] == 8.0
    assert np.isnan(z[1, 1])
    assert list(heatmap.x) == [0.0, 1.5]
    assert list(heatmap.y) == [2.0, 4.0]
    assert len(target.x) == 3  # no target over masked cell
    assert list(target.customdata) == ["a", "b", "c"]
    assert fig.layout.clickmode == "event+select"
    assert fig.layout.dragmode == "pan"
    assert fig.layout.coloraxis.cmin == 0.0
    assert fig.layout.coloraxis.cmax == 10.0
    assert "tilt" in fig.layout.yaxis.title.text.lower()
    pd.testing.assert_frame_equal(selected, orig)


def test_decoder_only_accepts_single_valid_coordinate():
    frame = single_panel_frame(_frame(), 30.0)
    point = {"curveNumber": 1, "pointIndex": 1, "x": 1.5, "y": 2.0, "customdata": "b"}
    assert cell_from_points([point], frame) == {
        "x": 1.5,
        "y": 2.0,
        "name": "b",
        "panel_raw": 30.0,
    }
    assert cell_from_points([point, dict(point)], frame)["name"] == "b"
    assert cell_from_points([point, {"x": 0.0, "y": 2.0}], frame) is None
    assert cell_from_points([{"x": 1.5, "y": 4.0}], frame) is None  # NaN
    assert cell_from_points([{"x": 17, "y": 2.0}], frame) is None
    assert cell_from_points([{"x": "nan", "y": 2.0}], frame) is None
    assert cell_from_points([{"x": "inf", "y": 2.0}], frame) is None
    assert cell_from_points([{}], frame) is None
    assert cell_from_points(None, frame) is None


def test_decoder_does_not_trust_browser_case_name():
    frame = single_panel_frame(_frame(), 30.0)
    point = {"x": 0.0, "y": 4.0, "customdata": "spoofed"}
    assert cell_from_points([point], frame)["name"] == "c"


def test_session_scope_and_reactivity_preserve_same_cell_across_theme_or_quantity():
    frame = single_panel_frame(_frame(), 30.0)
    ctx = SimpleNamespace(selected_material="Si", selected_face="(111)", checkpoint_stem="run-42")
    scope = map_scope(ctx, energy=30.0, thickness=1_000_000.0)
    stored = selection_for_scope([{"x": 0.0, "y": 2.0}], frame, scope)
    assert stored["scope"] == scope
    assert restore_selection(frame, stored, scope)["name"].tolist() == ["a"]
    # Theme changes do not change the geometry. A new quantity can preserve a
    # record if its cell is still available at the same coordinate.
    refreshed = frame.copy()
    refreshed["value"] *= 2.0
    assert restore_selection(refreshed, stored, scope)["name"].tolist() == ["a"]
    new_scope = map_scope(ctx, energy=50.0, thickness=1_000_000.0)
    assert restore_selection(frame, stored, new_scope).empty
    newer_case = frame.copy()
    newer_case.loc[0, "name"] = "different-best-record"
    assert restore_selection(newer_case, stored, scope).empty
    missing = frame.drop(index=0)
    assert restore_selection(missing, stored, scope).empty


def test_empty_grids_and_nonuniform_axes():
    frame = single_panel_frame(_frame(), 30.0)
    assert selectable_map_figure(frame.iloc[:0], label="test") is None
    assert selected_cell_frame(frame, None).empty
    assert selected_cell_frame(frame, {"x": 0.0, "y": 2.0, "panel_raw": 999.0, "name": "a"}).empty
    assert len(selectable_map_figure(frame, label="test").data) == 2
    all_nan = frame.copy()
    all_nan["value"] = np.nan
    assert selectable_map_figure(all_nan, label="test") is None


def test_validated_selection_from_marimo_point_list():
    frame = single_panel_frame(_frame(), 30.0)
    scope = ("Si", "111", "checkpoint", 30.0, 1000000.0)
    stored = selection_for_scope([{"x": 0.0, "y": 4.0}], frame, scope)
    chosen = restore_selection(frame, stored, scope)
    assert chosen.iloc[0]["name"] == "c"
    assert chosen.iloc[0]["value"] == pytest.approx(8.0)


def test_dense_grid_targets_are_bounded_and_serializable():
    xs, ys = np.meshgrid(np.linspace(-4.0, 6.0, 25), np.linspace(1.0, 11.0, 25))
    frame = pd.DataFrame(
        {
            "x": xs.ravel(),
            "y": ys.ravel(),
            "value": np.arange(625, dtype=float),
            "panel": ["30 keV"] * 625,
            "panel_raw": np.full(625, 30.0),
            "name": [f"case-{i}" for i in range(625)],
        }
    )
    fig = selectable_map_figure(frame, label="dense map")
    assert len(fig.data[1].x) == 625
    assert 3 <= fig.data[1].marker.size <= 14
    assert np.asarray(fig.data[0].z).shape == (25, 25)
    assert len(fig.to_json()) < 200_000
