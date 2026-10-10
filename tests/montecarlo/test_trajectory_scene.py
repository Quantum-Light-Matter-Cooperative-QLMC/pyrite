"""Scene interchange preserves coordinates, scale separation and safe publication."""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from pyrite.cli.commands.trajectories import command
from pyrite.instrument import FilterPlate, PixelGrid, PlanarDetector, PlanarPose
from pyrite.instrument.scene import scene_payload
from pyrite.montecarlo.trajectories import (
    TrajectoryArtifactError,
    TrajectoryCapture,
    preflight_capture,
    read_trajectory_artifact,
    write_trajectory_artifact,
)
from pyrite.montecarlo.trajectory_scene import export_trajectory_scene
from tests.montecarlo.test_trajectory_capture import _read_vtp


def _capture(tmp_path, *, scene=None, **case_overrides):
    case = dict(
        name="scene",
        crystal="Si",
        E0_keV=200.0,
        thickness_ang=100.0,
        tilt_deg=60.0,
        tilt_azim_deg=0.0,
        beam_fwhm_mm=0.1,
        crystal_width_mm=1.0,
        crystal_height_mm=2.0,
    )
    case.update(case_overrides)
    transport = dict(
        r_mid=np.array([[0.0, 0.0, 1.0]]),
        v_hat=np.array([[0.0, 0.0, 1.0]]),
        L_ang=np.array([2.0]),
        E_start_keV=np.array([200.0]),
        electron_id=np.array([0]),
        track_id=np.array([1]),
        parent_id=np.array([0]),
        generation=np.array([1]),
    )
    return write_trajectory_artifact(tmp_path / "a.h5", transport, case=case, scene=scene)


def _blocks(manifest):
    return {
        node.get("name"): manifest.parent / node.get("file")
        for node in ET.parse(manifest).getroot().iter("DataSet")
    }


def _strings(path):
    data = path.read_bytes().split(b'<AppendedData encoding="raw">')[0]
    if b"</VTKFile>" not in data:
        data += b"</VTKFile>"
    return {
        node.get("Name"): bytes(int(code) for code in node.text.split()).rstrip(b"\0").decode()
        for node in ET.fromstring(data).iter("Array")
    }


def _physical_scene():
    plate = FilterPlate(
        material="hopg",
        thickness_mm=0.1,
        size_mm=(3.0, 4.0),
        pose=PlanarPose.from_observation(10.0, 90.0),
        name="plate & window",
    )
    detector = PlanarDetector(
        pose=PlanarPose.from_observation(30.0, 90.0, roll_deg=35.0),
        pixels=PixelGrid(shape=(2, 3), pitch_mm=(0.2, 0.1)),
    )
    return scene_payload((plate,), detector)


def test_scene_roundtrip_and_scale_groups(tmp_path):
    scene = _physical_scene()
    scene["sample_origin_lab_mm"] = [0.2, -0.3, 0.4]
    artifact = _capture(tmp_path, scene=scene, abs_layers=[([], 40.0, None), ([], 120.0, None)])
    assert read_trajectory_artifact(artifact).scene == scene
    near, far = export_trajectory_scene(artifact, tmp_path / "a.vtp")
    close = _blocks(near)
    distant = _blocks(far)
    assert {"tracks", "crystal", "entry beam", "layer-1"} <= close.keys()
    assert {
        "beam axis",
        "beam footprint",
        "target footprint",
        "detector",
        "filters/0:plate & window",
    } <= distant.keys()
    assert "tracks" not in distant and "detector" not in close
    assert all(path.is_file() for path in [*close.values(), *distant.values()])
    _, arrays = _read_vtp(close["tracks"])
    expected = np.array([[0.0, 0.0, 0.0], [np.sqrt(3), 0.0, 1.0]]) + np.array([2e6, -3e6, 4e6])
    np.testing.assert_allclose(arrays["Points"], expected, atol=1e-9)
    np.testing.assert_allclose(arrays["v_hat"], [[np.sqrt(3) / 2, 0.0, 0.5]], atol=1e-14)
    np.testing.assert_allclose(arrays["r_mid"], expected.mean(axis=0)[None, :], atol=1e-9)
    assert _strings(close["tracks"])["units"] == "angstrom"
    assert _strings(distant["detector"])["units"] == "mm"
    assert _strings(close["crystal"])["thickness_ang"] == "120.0"
    points = ET.parse(distant["detector"]).find(".//Points/DataArray")
    np.testing.assert_allclose(
        np.fromstring(points.text, sep=" ").reshape(-1, 3), scene["detector"]["corners_mm"]
    )
    # Scene snapshots survive catalog-free export and caller mutations.
    scene["detector"]["corners_mm"][0][0] += 99
    assert read_trajectory_artifact(artifact).scene != scene


def test_legacy_scene_does_not_invent_detector_or_footprint(tmp_path):
    artifact = _capture(tmp_path, crystal_width_mm=None, crystal_height_mm=None, beam_fwhm_mm=None)
    assert read_trajectory_artifact(artifact).scene is None
    near, far = export_trajectory_scene(artifact, tmp_path / "a.vtp")
    assert set(_blocks(far)) == {"beam axis"}
    assert _strings(_blocks(near)["tracks"])["downstream_scene"] == "unavailable"


def test_closeup_crop_follows_selected_off_axis_history(tmp_path):
    from pyrite.montecarlo.trajectory_selection import select_trajectories

    transport = dict(
        r_mid=np.array([[1e6, -2e6, 10.0], [-1e6, 2e6, 10.0]]),
        v_hat=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        L_ang=np.array([20.0, 20.0]),
        E_start_keV=np.array([30.0, 30.0]),
        electron_id=np.array([0, 1]),
    )
    artifact = write_trajectory_artifact(
        tmp_path / "spot.h5",
        transport,
        case=dict(name="spot", crystal="Si", E0_keV=30.0, thickness_ang=100.0),
    )
    near, _ = export_trajectory_scene(
        artifact, tmp_path / "spot.vtp", selection=select_trajectories(artifact, histories=(0,))
    )
    points = ET.parse(_blocks(near)["crystal"]).find(".//Points/DataArray")
    xyz = np.fromstring(points.text, sep=" ").reshape(-1, 3)
    # A micron-scale shower landing 0.1 mm off-axis must not crop back to the origin
    # or include the unselected history on the opposite side of the beam spot.
    np.testing.assert_allclose(xyz[:, :2].mean(axis=0), [1e6, -2e6])
    assert np.max(np.ptp(xyz[:, :2], axis=0)) < 100.0


def test_scene_grooves_use_bounded_shared_mesh(tmp_path):
    artifact = _capture(
        tmp_path,
        tilt_deg=45.0,
        tilt_azim_deg=180.0,
        theta_obs_rad=np.pi / 2,
        groove_spacing_ang=10.0,
    )
    near, _ = export_trajectory_scene(artifact, tmp_path / "a.vtp")
    assert "grooves" in _blocks(near)
    groove = ET.parse(_blocks(near)["grooves"]).find(".//Piece")
    assert int(groove.get("NumberOfPolys")) > 0


def test_scene_cli_checks_all_collisions_before_writing(tmp_path):
    artifact = _capture(tmp_path)
    (tmp_path / "a.instrument.vtm").write_text("keep me")
    result = CliRunner().invoke(command, [str(artifact), "--scene"])
    assert result.exit_code == 1 and "--overwrite" in result.output
    assert not (tmp_path / "a.vtp").exists()
    assert (tmp_path / "a.instrument.vtm").read_text() == "keep me"
    # A second artifact's close-up manifest collides with the first instrument manifest.
    other = tmp_path / "a.instrument.h5"
    other.write_bytes(artifact.read_bytes())
    result = CliRunner().invoke(command, [str(artifact), str(other), "--scene", "--overwrite"])
    assert result.exit_code == 2
    assert not (tmp_path / "a.vtp").exists()


def test_scene_cli_default_stays_vtp_only_and_scene_mirrors_directory(tmp_path):
    artifact = _capture(tmp_path)
    default = CliRunner().invoke(command, [str(artifact)])
    assert default.exit_code == 0, default.output
    assert not (tmp_path / "a.vtm").exists()
    scene = CliRunner().invoke(
        command, [str(tmp_path), "--scene", "--out-dir", str(tmp_path / "out")]
    )
    assert scene.exit_code == 0, scene.output
    assert (tmp_path / "out/a.vtm").is_file()
    assert (tmp_path / "out/a.instrument.vtm").is_file()


def test_scene_failed_overwrite_keeps_previous_references(tmp_path, monkeypatch):
    import pyrite.montecarlo.trajectory_scene as writer

    artifact = _capture(tmp_path)
    near, far = export_trajectory_scene(artifact, tmp_path / "a.vtp")
    original = near.read_bytes(), far.read_bytes()
    directories = set(tmp_path.glob("*.scene-*"))

    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(writer, "_manifest", fail)
    with pytest.raises(OSError, match="injected"):
        export_trajectory_scene(artifact, tmp_path / "a.vtp", overwrite=True)
    assert (near.read_bytes(), far.read_bytes()) == original
    assert all(path.exists() for path in [*_blocks(near).values(), *_blocks(far).values()])
    assert set(tmp_path.glob("*.scene-*")) == directories


def test_scene_resume_rejects_changed_snapshot_without_changing_case_digest(tmp_path):
    scene = _physical_scene()
    artifact = read_trajectory_artifact(_capture(tmp_path, scene=scene))
    capture = TrajectoryCapture(str(tmp_path / "capture"), scene=scene)
    capture.write(artifact.case, artifact.transport)
    assert preflight_capture(capture, [artifact.case], []).kept == 1
    scene["sample_origin_lab_mm"] = [1.0, 0.0, 0.0]
    with pytest.raises(TrajectoryArtifactError, match="scene"):
        preflight_capture(capture, [artifact.case], [])


def test_run_scene_handles_scalar_defaults():
    from pyrite.campaign.observation import profile_trajectory_scene

    scene = profile_trajectory_scene("standard")
    assert scene["detector"] is None


def test_run_scene_snapshots_physical_geometry_without_acquisition(monkeypatch):
    import pyrite.materials as materials
    from pyrite.campaign.observation import profile_trajectory_scene

    row = dict(distance_mm=30.0, shape=(2, 3), pitch_mm=(0.2, 0.1))
    fake = SimpleNamespace(
        profile_detector_set=lambda profile: {"camera": row},
        profile_filters={
            "standard": [
                dict(
                    material="hopg",
                    thickness_mm=0.1,
                    size_mm=(3.0, 4.0),
                    distance_mm=10.0,
                    name="window",
                )
            ]
        },
    )
    monkeypatch.setattr(materials, "CATALOG", fake)
    scene = profile_trajectory_scene("standard", "camera")
    assert scene["detector"]["pixels"]["shape"] == [2, 3]
    assert scene["filters"][0]["name"] == "window"


def test_scene_interrupted_manifest_replacement_keeps_all_references(tmp_path, monkeypatch):
    import pyrite.montecarlo.trajectory_scene as writer

    artifact = _capture(tmp_path)
    near, far = export_trajectory_scene(artifact, tmp_path / "a.vtp")
    old_blocks = [*_blocks(near).values(), *_blocks(far).values()]
    replace = writer.os.replace

    def fail_second(source, destination):
        if destination == far:
            raise OSError("interrupted publication")
        replace(source, destination)

    monkeypatch.setattr(writer.os, "replace", fail_second)
    with pytest.raises(OSError, match="interrupted"):
        export_trajectory_scene(artifact, tmp_path / "a.vtp", overwrite=True)
    assert all(
        path.exists() for path in [*old_blocks, *_blocks(near).values(), *_blocks(far).values()]
    )


def test_scene_omits_unresolved_nonround_beam_footprint(tmp_path):
    artifact = _capture(tmp_path, beam_fwhm_y_mm=0.2)
    _, far = export_trajectory_scene(artifact, tmp_path / "a.vtp")
    assert "beam footprint" not in _blocks(far)
    assert "unavailable" in _strings(far)["beam_footprint"]


def test_scene_rejects_artifact_replacement_before_publication(tmp_path, monkeypatch):
    import pyrite.montecarlo.trajectory_scene as writer

    artifact = _capture(tmp_path)
    original = read_trajectory_artifact(artifact)
    export = writer.export_segments_vtp

    def replace_after_export(*args, **kwargs):
        result = export(*args, **kwargs)
        write_trajectory_artifact(
            artifact,
            original.transport,
            case=dict(original.case, E0_keV=201.0),
            overwrite=True,
        )
        return result

    monkeypatch.setattr(writer, "export_segments_vtp", replace_after_export)
    with pytest.raises(TrajectoryArtifactError, match="changed during"):
        export_trajectory_scene(artifact, tmp_path / "a.vtp")
    assert not list(tmp_path.glob("*.vtm"))
    assert not list(tmp_path.glob("*.scene-*"))


def test_scene_empty_capture_has_finite_context(tmp_path):
    artifact = _capture(tmp_path)
    original = read_trajectory_artifact(artifact)
    write_trajectory_artifact(
        artifact,
        {key: value[:0] for key, value in original.transport.items()},
        case=original.case,
        overwrite=True,
    )
    near, _ = export_trajectory_scene(artifact, tmp_path / "a.vtp")
    piece, _ = _read_vtp(_blocks(near)["tracks"])
    assert piece.get("NumberOfLines") == "0"
    points = ET.parse(_blocks(near)["crystal"]).find(".//Points/DataArray")
    assert np.all(np.isfinite(np.fromstring(points.text, sep=" ")))


@pytest.mark.filterwarnings("error:The VTK reader")
def test_scene_opens_in_vtk(tmp_path):
    pv = pytest.importorskip("pyvista")
    artifact = _capture(tmp_path, scene=_physical_scene())
    near, far = export_trajectory_scene(artifact, tmp_path / "a.vtp")
    close, distant = pv.read(near), pv.read(far)
    assert close["tracks"].n_cells == 1
    assert close["tracks"].field_data["units"][0] == "angstrom"
    assert distant["detector"].field_data["units"][0] == "mm"
    assert close["tracks"].cell_data["track_id"][0] == 1
    assert close.field_data["units"][0] == "angstrom"
    assert distant.field_data["units"][0] == "mm"
