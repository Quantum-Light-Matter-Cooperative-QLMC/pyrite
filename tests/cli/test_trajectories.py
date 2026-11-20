"""`pyrite trajectories`: capture discovery, description, and compatibility aliases (#397)."""

import json

import h5py
import numpy as np
import pytest

from pyrite.cli import command as root_command
from pyrite.montecarlo import trajectory_inventory
from pyrite.montecarlo.trajectories import write_trajectory_artifact
from tests.helpers.cli import invoke
from tests.helpers.trajectory_schema import downgrade
from tests.montecarlo.runner.test_trajectory_selection import _CASE, _secondaries

_PROVENANCE = {"checkpoint_stem": "hopg@demo", "profile": "demo", "fidelity": "quick"}


@pytest.fixture
def root(tmp_path):
    """A capture root with complete, partial, incomplete, and foreign files."""
    stem = tmp_path / "hopg@demo"
    write_trajectory_artifact(
        stem / "saved" / "E0_20keV.h5",
        _secondaries(),
        case=_CASE,
        provenance=_PROVENANCE,
        scene={"detector": "pixel"},
    )
    write_trajectory_artifact(
        stem / "saved" / "E0_30keV.h5",
        _secondaries(),
        case={**_CASE, "E0_keV": 30.0},
        provenance=_PROVENANCE,
    )
    (stem / "saved" / "E0_40keV.h5.partial").write_bytes(b"interrupted")
    incomplete = write_trajectory_artifact(
        stem / "saved" / "E0_50keV.h5", _secondaries(), case={**_CASE, "E0_keV": 50.0}
    )
    with h5py.File(incomplete, "r+") as handle:
        handle.attrs["complete"] = False
    with h5py.File(stem / "foreign.h5", "w") as handle:
        handle["x"] = np.arange(3)
    return tmp_path


def test_root_lists_trajectories_group():
    result = invoke(root_command, ["--help"])

    assert "trajectories" in result.stdout


def test_ls_lists_mixed_capture_root_and_exits_zero(root):
    result = invoke(root_command, ["trajectories", "ls", str(root)])

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0].split() == [
        "STEM",
        "CASE",
        "E0_KEV",
        "SCHEMA",
        "STATUS",
        "SEGMENTS",
        "HISTORIES",
        "MAX_GEN",
        "BYTES",
        "PATH",
    ]
    statuses = {line.split()[-1].rsplit("/", 1)[-1]: line.split()[4] for line in lines[1:6]}
    assert statuses == {
        "foreign.h5": "foreign",
        "E0_20keV.h5": "complete",
        "E0_30keV.h5": "complete",
        "E0_40keV.h5.partial": "partial",
        "E0_50keV.h5": "incomplete",
    }
    assert "hopg@demo: 2 artifact(s), 12 segments" in result.stdout


def test_ls_json_is_one_envelope(root):
    result = invoke(root_command, ["trajectories", "ls", str(root), "-o", "json"])

    envelope = json.loads(result.stdout)
    assert result.exit_code == 0 and result.stderr == ""
    assert envelope["schema"] == "pyrite.trajectories.ls.v1"
    complete = [row for row in envelope["payload"]["artifacts"] if row["status"] == "complete"]
    assert [(row["E0_keV"], row["histories"], row["max_generation"]) for row in complete] == [
        (20.0, 2, 1),
        (30.0, 2, 1),
    ]
    assert envelope["payload"]["stems"] == [
        {
            "stem": "hopg@demo",
            "artifacts": 2,
            "segments": 12,
            "bytes": sum(row["bytes"] for row in complete),
        }
    ]


def test_ls_without_captures_reports_none(tmp_path):
    result = invoke(root_command, ["trajectories", "ls", str(tmp_path)])

    assert result.exit_code == 0 and "No trajectory artifacts" in result.stdout


class _ReadGuard:
    """Fail any whole-dataset `/transport` read or any read of a non-ID column."""

    def __init__(self, monkeypatch):
        original = h5py.Dataset.__getitem__
        self.reads = []

        def guarded(dataset, key):
            if dataset.name.startswith("/transport/"):
                self.reads.append(dataset.name)
                assert dataset.dtype.kind in "iu", f"read non-ID column {dataset.name}"
                assert key != () and key is not Ellipsis, f"whole read of {dataset.name}"
                if isinstance(key, slice):
                    start, stop, _ = key.indices(dataset.shape[0])
                    assert stop - start <= trajectory_inventory._BLOCK, dataset.name
            return original(dataset, key)

        monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_info_reads_only_id_columns(tmp_path, monkeypatch, schema):
    path = write_trajectory_artifact(
        tmp_path / "a.h5", _secondaries(), case=_CASE, provenance=_PROVENANCE
    )
    if schema < 3:
        downgrade(path, schema)
    guard = _ReadGuard(monkeypatch)

    result = invoke(root_command, ["trajectories", "info", str(path), "-o", "json"])

    assert result.exit_code == 0, result.output
    info = json.loads(result.stdout)["payload"]
    assert info["schema_version"] == schema
    assert info["histories"] == 2 and info["history_id_range"] == [0, 1]
    assert info["tracks"] == 4 and info["segments_by_generation"] == {"0": 3, "1": 3}
    assert info["provenance"]["checkpoint_stem"] == "hopg@demo" and info["scene"] is False
    roles = {record["name"]: record["role"] for record in info["fields"]}
    assert roles["r_mid"] == "segment" and roles["initial_E_keV"] == "electron"
    assert set(guard.reads) <= {
        "/transport/electron_id",
        "/transport/track_id",
        "/transport/generation",
    }


def test_info_human_lists_fields_with_units(root):
    path = root / "hopg@demo" / "saved" / "E0_20keV.h5"

    result = invoke(root_command, ["trajectories", "info", str(path)])

    assert result.exit_code == 0, result.output
    assert "scene" in result.stdout and "recorded" in result.stdout
    assert "Per-segment fields:" in result.stdout
    # Pre-#1079 captures still record the retired tier; it is never shown.
    assert "fidelity" not in result.stdout
    assert any(
        line.split()[:1] == ["E_start_keV"] and line.split()[-1] == "keV"
        for line in result.stdout.splitlines()
    )


def test_info_refuses_directory_and_foreign_file(root):
    directory = invoke(root_command, ["trajectories", "info", str(root)])
    foreign = invoke(root_command, ["trajectories", "info", str(root / "hopg@demo" / "foreign.h5")])

    assert directory.exit_code == 2 and "pyrite trajectories ls" in directory.stderr
    assert foreign.exit_code == 1 and "not a PyRITE trajectory artifact" in foreign.stderr
    assert "Traceback" not in directory.output + foreign.output


def test_export_skips_foreign_files_with_a_warning(root, tmp_path):
    result = invoke(
        root_command,
        ["trajectories", "export", str(root), "--out-dir", str(tmp_path / "vtp")],
    )

    assert result.exit_code == 0, result.output
    assert "warning: skipping foreign file" in result.stderr
    assert "warning: skipping partial file" in result.stderr
    assert sorted(p.name for p in (tmp_path / "vtp").rglob("*.vtp")) == [
        "E0_20keV.vtp",
        "E0_30keV.vtp",
    ]


@pytest.mark.parametrize(
    ("old", "new"),
    [("export-trajectories", "export"), ("score-trajectories", "score")],
)
def test_checkpoint_aliases_are_removed_and_canonical_paths_remain(old, new):
    alias_help = invoke(root_command, ["checkpoint", old, "--help"])
    new_help = invoke(root_command, ["trajectories", new, "--help"])
    alias_usage = invoke(root_command, ["checkpoint", old])
    new_usage = invoke(root_command, ["trajectories", new])

    assert alias_help.exit_code == 2
    assert "No such command" in alias_help.stderr
    assert new_help.exit_code == 0
    assert alias_usage.exit_code == new_usage.exit_code == 2
    assert "No such command" in alias_usage.stderr
    assert "is deprecated" not in new_usage.stderr
    assert old not in invoke(root_command, ["checkpoint", "--help"]).stdout


# --- `score` on API-saved artifacts (#1169) ---------------------------------


@pytest.fixture(scope="module")
def saved_artifact(tmp_path_factory):
    """One `save_trajectories` artifact (no run identity) and its recorded scene."""
    import pyrite
    from pyrite.beams import Beam, PointBunch, Spot
    from pyrite.config import FixedSampling, Numerics
    from pyrite.detectors import Detector
    from pyrite.geometry import Footprint, Slab
    from pyrite.storage import save_trajectories
    from pyrite.transport import simulate_trajectories

    del pyrite
    numerics = Numerics(sampling=FixedSampling(cxr_histories=8, brem_histories=4), seed=3)
    slab = Slab("hopg", 1000.0, tilt_deg=30.0, footprint=Footprint(5.0, 5.0))
    beam = Beam(30.0, transverse=Spot(), longitudinal=PointBunch())
    trajectories = simulate_trajectories(beam, slab, numerics=numerics, detector=Detector())
    return save_trajectories(trajectories, tmp_path_factory.mktemp("api") / "api.h5")


def _npz_metadata(path):
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata"]))


def test_score_matches_score_spectra_on_an_api_saved_artifact(saved_artifact, tmp_path):
    import pyrite
    from pyrite.storage import load_trajectories

    out = tmp_path / "scored"
    result = invoke(
        root_command,
        ["trajectories", "score", str(saved_artifact), "--out-dir", str(out)],
    )

    assert result.exit_code == 0, result.output
    target = out / "api.npz"
    assert str(target) in result.stdout and "segments" in result.stdout
    expected = pyrite.score_spectra(load_trajectories(saved_artifact, stream=True))
    with np.load(target, allow_pickle=False) as data:
        for name, part in (
            ("cxr_incoherent", expected.cxr.incoherent),
            ("characteristic", expected.characteristic),
            ("brem", expected.brem),
        ):
            np.testing.assert_array_equal(data[f"{name}_photon_energy_eV"], part.photon_energy_eV)
            np.testing.assert_allclose(data[f"{name}_density"], part.density, rtol=1e-12, atol=0)
        assert "cxr_coherent_density" not in data
    metadata = _npz_metadata(target)
    assert metadata["format"] == "pyrite-trajectory-spectra"
    assert metadata["source_trajectory"]["path"] == str(saved_artifact)
    assert metadata["identity_digest"] == expected.provenance["identity_digest"]


def test_score_writes_beside_the_artifact_and_keeps_existing_unless_overwrite(
    saved_artifact, tmp_path
):
    import shutil

    copy = str(shutil.copy(saved_artifact, tmp_path / "api.h5"))
    target = tmp_path / "api.npz"

    first = invoke(root_command, ["trajectories", "score", copy])
    again = invoke(root_command, ["trajectories", "score", copy])
    redo = invoke(root_command, ["trajectories", "score", copy, "--overwrite"])

    assert first.exit_code == again.exit_code == redo.exit_code == 0
    assert "already scored (--overwrite to redo)" in again.stdout
    assert not list(tmp_path.glob("*.partial"))
    assert "segments" in first.stdout and "segments" in redo.stdout
    assert target.exists()


def test_score_refuses_a_saved_artifact_that_fails_its_fingerprint(saved_artifact, tmp_path):
    import shutil

    tampered = shutil.copy(saved_artifact, tmp_path / "bad.h5")
    with h5py.File(tampered, "r+") as handle:
        settings = json.loads(handle["settings"].attrs["json"])
        settings["segment_fingerprint"] = "17:" + "0" * 32
        handle["settings"].attrs["json"] = json.dumps(settings)

    result = invoke(root_command, ["trajectories", "score", str(tampered)])

    assert result.exit_code == 1 and "fingerprint" in result.stderr
    assert "Traceback" not in result.output
    assert not (tmp_path / "bad.npz").exists()


def test_score_flags_are_refused_when_no_input_uses_them(saved_artifact, tmp_path):
    from tests.checkpoint.test_score_trajectories import _captured_run

    only_saved = invoke(
        root_command,
        ["trajectories", "score", str(saved_artifact), "--checkpoint-dir", str(tmp_path / "ck")],
    )
    _captured_run(tmp_path)
    only_runs = invoke(
        root_command,
        ["trajectories", "score", str(tmp_path / "traj"), "--out-dir", str(tmp_path / "npz")],
    )

    assert only_saved.exit_code == only_runs.exit_code == 2
    assert "--checkpoint-dir applies" in only_saved.stderr
    assert "--out-dir applies" in only_runs.stderr
    assert not (tmp_path / "ck").exists() and not (tmp_path / "npz").exists()


def test_score_mixed_inputs_route_each_kind_and_leave_run_captures_unchanged(
    saved_artifact, tmp_path
):
    import shutil

    from pyrite.checkpoints.persistence import _checkpoint_load
    from tests.checkpoint.test_score_trajectories import _captured_run

    cases, _capture, _live = _captured_run(tmp_path)
    saved = shutil.copy(saved_artifact, tmp_path / "traj" / "api.h5")
    checkpoints, npz = tmp_path / "s", tmp_path / "npz"

    result = invoke(
        root_command,
        [
            "trajectories",
            "score",
            str(tmp_path / "traj"),
            "--checkpoint-dir",
            str(checkpoints),
            "--out-dir",
            str(npz),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (npz / "api.npz").exists() and saved
    scored = _checkpoint_load(checkpoints / "hopg")
    assert {case["name"] for case in cases} == set(scored)
    assert f"{checkpoints / 'hopg'}: scored 2" in result.stdout


def test_score_help_documents_the_saved_artifact_contract():
    text = " ".join(invoke(root_command, ["trajectories", "score", "--help"]).stdout.split())

    for phrase in (
        "--out-dir",
        "save_trajectories",
        "<component>_density",
        "photons / incident electron / eV / sr",
        "recorded fingerprint",
    ):
        assert phrase in text, phrase
