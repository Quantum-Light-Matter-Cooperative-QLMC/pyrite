"""Saved trace views must not call transport, and reject oversized selections."""

import numpy as np
import pytest

from pyrite.apps.viewer import load_saved_capture
from pyrite.montecarlo.trajectories import TrajectoryArtifactError, write_trajectory_artifact
from pyrite.montecarlo.trajectory_selection import select_trajectories
from pyrite.plots import _frames
from pyrite.plots.altair.trajectories import penetration_survival_chart, trajectory_chart
from pyrite.plots.altair.trajectory_summary import summary_charts
from pyrite.plots.plotly.trajectories import trajectory_volume_data, trajectory_volume_figure
from tests.montecarlo.test_trajectory_summary import CASE, summary, transport


@pytest.fixture
def capture(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("saved view must not transport")

    monkeypatch.setattr(_frames, "simulate_trajectories", fail)
    return write_trajectory_artifact(tmp_path / "saved.h5", transport(), case=CASE)


def test_all_existing_builders_accept_selection_without_transport(capture):
    selection, case, data = load_saved_capture(str(capture), first=3)
    assert selection.histories == 3
    assert len(data["E"]) == 9
    assert data["Ne"] == 3
    assert len(data["initial_E_keV"]) == 3
    np.testing.assert_array_equal(data["elec_id"], np.repeat(np.arange(3), 3))
    assert trajectory_volume_data(selection)["Ne"] == 3
    assert trajectory_volume_figure(selection).data
    assert trajectory_chart(selection).to_dict()
    assert penetration_survival_chart(selection).to_dict()


def test_read_budget_and_track_only_selection_are_rejected(capture):
    with pytest.raises(TrajectoryArtifactError, match="max_segments"):
        load_saved_capture(str(capture), first=3, max_segments=8)
    selection = select_trajectories(capture, tracks=(1,))
    with pytest.raises(TrajectoryArtifactError, match="whole-history"):
        trajectory_volume_data(selection)


def test_summary_charts_keep_explicit_units(capture):
    charts = summary_charts(summary(capture, 4))
    assert set(charts) == {
        "depth_loss",
        "fates",
        "penetration",
        "range",
        "exit_energy",
        "exit_angle",
        "hard_events",
        "generations",
    }
    for chart in charts.values():
        assert chart.to_dict()


def test_saved_app_panel_uses_saved_rows(capture):
    import marimo as mo

    from pyrite.apps.viewer import saved_capture_panel

    loaded = load_saved_capture(str(capture), first=3)
    panel = saved_capture_panel(loaded, None, mo=mo, theme="light", camera={})
    assert panel is not None
    assert saved_capture_panel(None, "missing file", mo=mo, theme="light", camera={}) is not None
