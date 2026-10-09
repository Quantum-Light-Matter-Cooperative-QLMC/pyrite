"""Saved viewer preserves full identities and segment associations."""

import numpy as np
import pytest
from click.testing import CliRunner

from pyrite.cli.commands.app_trajectories import command
from pyrite.montecarlo.trajectories import TrajectoryArtifactError, write_trajectory_artifact
from pyrite.trajectory_viewer.selection import load_selection, plan_selection


def _capture(tmp_path, *, legacy=False):
    transport = dict(
        r_mid=np.array([[i, i % 2, i + 1] for i in range(6)], dtype=float),
        v_hat=np.tile([2**-0.5, 0.0, 2**-0.5], (6, 1)),
        L_ang=np.ones(6),
        electron_id=np.array([1, 0, 1, 0, 1, 0]),
        E_start_keV=np.array([20.0, 18.0, 12.0, 9.0, 4.0, 2.0]),
        t_start_ang=np.arange(6.0) * 2997.924580,
        t0_ang=np.zeros(6),
        hard_channel=np.array([0, 0, 1, 0, 0, 1]),
        track_id=np.array([2, 0, 3, 0, 3, 1]),
        parent_id=np.array([-1, -1, 2, -1, 2, 0]),
        generation=np.array([0, 0, 1, 0, 1, 1]),
    )
    if legacy:
        for key in ("track_id", "parent_id", "generation"):
            del transport[key]
    return write_trajectory_artifact(
        tmp_path / "saved.h5",
        transport,
        case=dict(
            name="saved",
            crystal="Si",
            E0_keV=20.0,
            thickness_ang=10.0,
            tilt_deg=0.0,
            tilt_azim_deg=0.0,
        ),
    )


def test_whole_track_order_and_events(tmp_path):
    artifact = _capture(tmp_path)
    plan = plan_selection(artifact, tracks=(3,), attributes=("hard_channel",))
    assert plan.segments == 2
    data = load_selection(plan)
    np.testing.assert_array_equal(data["segment_id"], [2, 4])
    np.testing.assert_array_equal(data["parent_id"], [2, 2])
    np.testing.assert_array_equal(data["hard_channel"], [1, 0])
    with pytest.raises(TrajectoryArtifactError, match="no segments"):
        plan_selection(artifact, histories=(0,), tracks=(3,))


def test_budget_and_changed_capture(tmp_path):
    artifact = _capture(tmp_path)
    with pytest.raises(TrajectoryArtifactError, match="exceeds 1 segments"):
        plan_selection(artifact, tracks=(3,), max_segments=1)
    plan = plan_selection(artifact, histories=(1,))
    with pytest.raises(TrajectoryArtifactError, match="exceeds budget"):
        load_selection(plan, memory_budget_bytes=1)
    artifact.touch()
    with pytest.raises(TrajectoryArtifactError, match="changed"):
        load_selection(plan)


def test_legacy_unknown_ancestry_and_invalid_attribute(tmp_path):
    artifact = _capture(tmp_path, legacy=True)
    data = load_selection(plan_selection(artifact, histories=(0,)))
    np.testing.assert_array_equal(data["segment_id"], [1, 3, 5])
    assert np.all(data["track_id"] == -1)
    assert np.all(data["parent_id"] == -1)
    with pytest.raises(TrajectoryArtifactError, match="no track_id"):
        plan_selection(artifact, tracks=(0,))
    with pytest.raises(TrajectoryArtifactError, match="available segment"):
        plan_selection(artifact, attributes=("made_up",))


def test_cli_preflight(tmp_path):
    artifact = _capture(tmp_path)
    runner = CliRunner()
    result = runner.invoke(command, ["inspect", str(artifact), "--track", "3"])
    assert result.exit_code == 0, result.output
    assert "2 segments; fields:" in result.stdout
    assert "estimated peak" in result.stderr
    output = tmp_path / "frame.png"
    output.write_text("existing")
    result = runner.invoke(command, ["export", str(artifact), str(output)])
    assert result.exit_code == 1
    assert output.read_text() == "existing"
    result = runner.invoke(command, ["export", str(artifact), str(tmp_path / "frame.pdf")])
    assert result.exit_code == 2


def test_optional_render_identity_units_and_exports(tmp_path):
    pytest.importorskip("pyvista")
    from PIL import Image

    from pyrite.trajectory_viewer.native import show
    from pyrite.trajectory_viewer.render import CaptureViewer

    artifact = _capture(tmp_path)
    plan = plan_selection(artifact, attributes=("hard_channel",))
    viewer = CaptureViewer(plan, load_selection(plan), off_screen=True)
    try:
        assert viewer.visible.n_cells == 6
        show(viewer, output_dir=tmp_path, interactive=False)
        assert viewer.visible.n_cells == 6
        assert viewer.apply_filters(energy_min=10.0) == 3
        index = np.flatnonzero(viewer.visible.cell_data["track_id"] == 3)[0]
        info = viewer.inspect_cell(index)
        assert info["parent_id"] == 2
        assert info["clipped"]
        assert info["full_segments"] == 2
        assert info["hard_channel"] == 1
        assert len(viewer.full_track(3)["segment_id"]) == 2
        viewer.apply_filters(energy_min=100.0)
        assert viewer.select_parent(info)["visible_segments"] == 0
        assert viewer.apply_filters(time_max=0.0) == 1
        viewer.apply_filters(clip_x=0.9)
        assert viewer.inspect_track(0)["clipped"]
        assert viewer.inspect_track(0)["visible_segments"] == 2
        viewer.apply_filters()
        near = viewer.visible.points.copy()
        viewer.show_scale("instrument")
        np.testing.assert_allclose(viewer.visible.points, near / 1e7)
        viewer.show_scale("closeup")
        image = viewer.screenshot(tmp_path / "frame.png")
        with Image.open(image) as rendered:
            assert rendered.size == (1000, 700)
            assert np.asarray(rendered).std() > 5
        movie = viewer.movie(tmp_path / "orbit.gif", frames=3)
        with Image.open(movie) as rendered:
            assert rendered.n_frames == 3
        assert image.with_suffix(".png.json").exists()
    finally:
        viewer.close()


def test_optional_server_controls_without_browser(tmp_path, monkeypatch):
    """Server state/controller contract behind the browser controls.

    The connected-browser path (event serialization, widgets, downloads) is
    covered by checks/saved_trajectory_browser_probe.py.
    """
    pytest.importorskip("pyvista")
    pytest.importorskip("trame.app")
    pytest.importorskip("trame.ui.vuetify3")
    pytest.importorskip("trame_vtk")
    import json

    from PIL import Image

    from pyrite.trajectory_viewer.render import CaptureViewer
    from pyrite.trajectory_viewer.server import build_server

    artifact = _capture(tmp_path)
    plan = plan_selection(artifact)
    viewer = CaptureViewer(plan, load_selection(plan), off_screen=True)
    try:
        server = build_server(viewer, output_dir=tmp_path / "exports")
        state, ctrl = server.state, server.controller
        state.ready()
        # A literal list would bind a state variable named "closeup".
        assert state.scale_options == ["closeup", "instrument"]
        assert state.clip_units == "angstrom"

        # Isolate segment 2 (track 3, parent 2): energy >= 10 keV, lab x >= 1.9.
        with state:
            state.energy_min = 10.0
            state.clipping = True
            state.clip_x = 1.9
        assert viewer.visible.n_cells == 1
        renderer = viewer.plotter.renderer
        renderer.SetWorldPoint(*viewer.visible.get_cell(0).center, 1.0)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        ctrl.pick_track({"mode": "remote", "position": {"x": x, "y": y, "z": 0}})
        picked = json.loads(state.picked)
        assert (picked["segment_id"], picked["track_id"], picked["parent_id"]) == (2, 3, 2)
        assert picked["clipped"] and picked["E_start_keV"] == 12.0
        assert viewer.highlighted == (3, 1)
        ctrl.pick_track({"position": {"x": 1, "y": 1}})
        assert json.loads(state.picked)["segment_id"] == 2
        assert "unchanged" in state.pick_status

        # Refreshing after filter/scale changes keeps the picked segment.
        with state:
            state.energy_min = 0.0
            state.clipping = False
        picked = json.loads(state.picked)
        assert picked["segment_id"] == 2 and not picked["clipped"]
        with state:
            state.scale = "instrument"
        assert viewer.scale == "instrument" and state.clip_units == "mm"
        low, high = viewer.x_range()
        assert state.clip_min == low and state.clip_max == high
        assert json.loads(state.picked)["segment_id"] == 2
        with state:
            state.energy_min = 10.0

        ctrl.inspect_parent()
        assert json.loads(state.picked)["track_id"] == 2
        ctrl.inspect_parent()
        assert state.pick_status == "Primary track: no parent."
        ctrl.clear_selection()
        ctrl.inspect_parent()
        assert state.pick_status == "Pick a track first."

        iso = np.array(viewer.plotter.camera_position.to_list())
        viewer.plotter.camera.azimuth += 40
        assert not np.allclose(viewer.plotter.camera_position.to_list(), iso)
        ctrl.reset_view()
        np.testing.assert_allclose(viewer.plotter.camera_position.to_list(), iso, atol=1e-5)

        payload = ctrl.trigger_fn("export_png")()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        (png,) = (tmp_path / "exports").glob("*.png")
        assert png.read_bytes() == payload
        meta = json.loads(png.with_suffix(".png.json").read_text())
        assert meta["scale"] == "instrument" and meta["filters"]["energy_min"] == 10.0
        assert meta["image_size"] == list(viewer.plotter.window_size)
        with Image.open(png) as image:
            assert list(image.size) == meta["image_size"]
        with monkeypatch.context() as patch:

            def fail_export(*args, **kwargs):
                raise OSError("export destination unavailable")

            patch.setattr(viewer, "screenshot", fail_export)
            assert ctrl.trigger_fn("export_png")() is None
            assert state.message == "Export failed: export destination unavailable"
        assert ctrl.trigger_fn("export_png")()[:8] == b"\x89PNG\r\n\x1a\n"
        assert state.message.startswith("Export ready:")
    finally:
        viewer.close()
