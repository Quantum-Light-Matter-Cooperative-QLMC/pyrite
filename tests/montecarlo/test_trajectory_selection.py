"""Whole-history/track selection of trajectory artifacts and subset VTK export."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from pyrite.cli.commands.trajectories import command
from pyrite.montecarlo import trajectory_selection
from pyrite.montecarlo.trajectories import TrajectoryArtifactError, write_trajectory_artifact
from pyrite.montecarlo.trajectory_export import export_segments_vtp
from pyrite.montecarlo.trajectory_scene import export_trajectory_scene
from pyrite.montecarlo.trajectory_selection import read_selection, select_trajectories
from tests.montecarlo.test_trajectory_capture import _grooved_transport, _read_vtp

_CASE = dict(
    name="saved",
    crystal="Si",
    E0_keV=20.0,
    thickness_ang=10.0,
    tilt_deg=0.0,
    tilt_azim_deg=0.0,
)


def _secondaries():
    """Two interleaved showers; history 1 has secondary track 3 (parent 2)."""
    return dict(
        r_mid=np.array([[i, i % 2, i + 1] for i in range(6)], dtype=float),
        v_hat=np.tile([2**-0.5, 0.0, 2**-0.5], (6, 1)),
        L_ang=np.ones(6),
        electron_id=np.array([1, 0, 1, 0, 1, 0]),
        E_start_keV=np.array([20.0, 18.0, 12.0, 9.0, 4.0, 2.0]),
        hard_channel=np.array([0, 0, 1, 0, 0, 1]),
        track_id=np.array([2, 0, 3, 0, 3, 1]),
        parent_id=np.array([-1, -1, 2, -1, 2, 0]),
        generation=np.array([0, 0, 1, 0, 1, 1]),
        initial_E_keV=np.full(3, 20.0),  # history 2 missed the target
    )


@pytest.fixture
def capture(tmp_path):
    return write_trajectory_artifact(tmp_path / "saved.h5", _secondaries(), case=_CASE)


def test_history_selection_keeps_whole_showers_in_transport_order(capture):
    selection = select_trajectories(capture, histories=(1,))
    selected = read_selection(selection, ("electron_id", "track_id", "E_start_keV"))

    assert selection.segments == 3 and selection.histories == 1
    np.testing.assert_array_equal(selected.transport["segment_id"], [0, 2, 4])
    np.testing.assert_array_equal(selected.transport["track_id"], [2, 3, 3])
    np.testing.assert_array_equal(selected.transport["E_start_keV"], [20.0, 12.0, 4.0])
    assert selected.units["E_start_keV"] == "keV" and selected.units["segment_id"] == "1"


def test_tracks_intersect_histories(capture):
    tracks = read_selection(select_trajectories(capture, tracks=(3,)))
    both = select_trajectories(capture, histories=(1,), tracks=(3, 1))

    np.testing.assert_array_equal(tracks.transport["segment_id"], [2, 4])
    np.testing.assert_array_equal(tracks.transport["hard_channel"], [1, 0])
    assert "E_keV" not in tracks.transport  # aliases are not separate fields
    assert both.segments == 2
    with pytest.raises(TrajectoryArtifactError, match=r"2 histories of 3 sampled .*IDs 0\.\.1"):
        select_trajectories(capture, histories=(0,), tracks=(3,))
    with pytest.raises(TrajectoryArtifactError, match="no segments"):
        select_trajectories(capture, histories=(2,))


def test_sorted_history_lookup_does_not_scan(capture, monkeypatch):
    def _scan(*_args, **_kwargs):
        raise AssertionError("electron-major artifacts bisect history IDs")

    monkeypatch.setattr(trajectory_selection, "_match_runs", _scan)
    assert select_trajectories(capture, histories=(0, 1)).segments == 6


def test_unsorted_legacy_rows_scan_for_histories(capture):
    with h5py.File(capture, "r+") as handle:
        del handle.attrs["row_order"]
    assert select_trajectories(capture, histories=(0,)).segments == 3


def test_first_and_sample_select_reproducible_whole_histories(capture):
    first = select_trajectories(capture, first=1)
    sample = select_trajectories(capture, sample=1, seed=5)

    assert first.request["history_ids"] == [0] and first.segments == 3
    assert sample.request == select_trajectories(capture, sample=1, seed=5).request
    assert select_trajectories(capture, sample=10).histories == 2
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_trajectories(capture, histories=(0,), first=1)
    with pytest.raises(ValueError, match="positive"):
        select_trajectories(capture, sample=0)


def test_read_selection_rejects_unknown_fields_and_budget(capture):
    selection = select_trajectories(capture, histories=(1,))
    with pytest.raises(TrajectoryArtifactError, match=r"initial_E_keV.*available: .*r_mid"):
        read_selection(selection, ("initial_E_keV",))
    with pytest.raises(TrajectoryArtifactError, match="max_segments=2"):
        read_selection(selection, max_segments=2)


def test_subset_vtp_holds_selected_rows_and_record(capture, tmp_path):
    selection = select_trajectories(capture, tracks=(3,))
    summary = export_segments_vtp(capture, tmp_path / "s.vtp", selection=selection)
    piece, arrays = _read_vtp(tmp_path / "s.vtp")
    record = (tmp_path / "s.vtp").read_bytes().split(b"</FieldData>")[0]

    assert summary["cells"] == int(piece.get("NumberOfLines")) == 2
    np.testing.assert_array_equal(arrays["segment_id"], [2, 4])
    np.testing.assert_array_equal(arrays["track_id"], [3, 3])
    np.testing.assert_allclose(
        0.5 * (arrays["Points"][0::2] + arrays["Points"][1::2])[:, 0], [2, 4]
    )
    assert b"selection" in record


def test_full_export_is_unchanged_without_selection(capture, tmp_path):
    summary = export_segments_vtp(capture, tmp_path / "f.vtp")
    assert "segment_id" not in summary["fields"] and summary["cells"] == 6
    assert b"<FieldData>" not in (tmp_path / "f.vtp").read_bytes().split(b"<AppendedData")[0]


def test_grooved_history_selection_carries_its_vacuum_legs(tmp_path):
    segs = _grooved_transport()
    path = write_trajectory_artifact(tmp_path / "g.h5", segs, case=_CASE)
    history = int(np.bincount(segs["vacuum_elec_id"]).argmax())
    selection = select_trajectories(path, histories=(history,))
    summary = export_segments_vtp(path, tmp_path / "g.vtp", selection=selection)
    _, arrays = _read_vtp(tmp_path / "g.vtp")

    n_seg = int(np.sum(segs["electron_id"] == history))
    n_vac = int(np.sum(segs["vacuum_elec_id"] == history))
    assert n_vac and summary == dict(summary, segments=n_seg, vacuum_legs=n_vac)
    assert set(arrays["electron_id"].tolist()) == {history}
    np.testing.assert_array_equal(arrays["is_vacuum"], np.repeat([0, 1], [n_seg, n_vac]))
    assert (arrays["segment_id"][n_seg:] == -1).all()


def test_scene_export_limits_tracks_to_selection(capture, tmp_path):
    selection = select_trajectories(capture, histories=(0,))
    closeup, _ = export_trajectory_scene(capture, tmp_path / "s.vtp", selection=selection)
    tracks = next(closeup.parent.glob("s.scene-*/tracks.vtp"))
    _, arrays = _read_vtp(tracks)
    np.testing.assert_array_equal(arrays["segment_id"], [1, 3, 5])


def test_cli_selection_options(capture, tmp_path):
    cli = CliRunner()
    out = tmp_path / "out"

    ok = cli.invoke(command, [str(capture), "--out-dir", str(out), "--sample", "1", "--seed", "3"])
    conflict = cli.invoke(command, [str(capture), "--history", "0", "--first", "1"])
    seedless = cli.invoke(command, [str(capture), "--seed", "1"])
    empty = cli.invoke(command, [str(capture), "--out-dir", str(tmp_path / "e"), "--history", "9"])

    assert ok.exit_code == 0, ok.output
    assert "1 histories selected" in ok.output
    record = (out / "saved.vtp").read_bytes().split(b"</FieldData>")[0]
    codes = record.split(b'format="ascii">')[1].split(b"</Array>")[0].split()
    assert json.loads(bytes(int(c) for c in codes[:-1]))["seed"] == 3
    assert conflict.exit_code == 2 and "mutually exclusive" in conflict.output
    assert seedless.exit_code == 2 and "--seed requires --sample" in seedless.output
    assert empty.exit_code == 1 and "IDs 0..1" in empty.output
    assert not (tmp_path / "e").exists()


def test_paraview_script_path_is_bundled_and_compiles():
    result = CliRunner().invoke(command, ["--paraview-script"])
    script = Path(result.output.strip())

    assert result.exit_code == 0 and script.name == "pyrite_trajectories.py"
    compile(script.read_text(), str(script), "exec")
