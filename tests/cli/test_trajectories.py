"""`pyrite trajectories`: capture discovery, description, and compatibility aliases (#397)."""

import json

import h5py
import numpy as np
import pytest

from pyrite.cli import command as root_command
from pyrite.cli._deprecations import message
from pyrite.montecarlo import trajectory_inventory
from pyrite.montecarlo.trajectories import write_trajectory_artifact
from tests.helpers.cli import invoke
from tests.montecarlo.test_trajectory_selection import _CASE, _secondaries

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


def _schema_1(path):
    with h5py.File(path, "r+") as handle:
        handle.attrs["schema_version"] = 1
        del handle["case/payload"]
        del handle.attrs["row_order"]
    return path


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


@pytest.mark.parametrize("schema", [1, 2])
def test_info_reads_only_id_columns(tmp_path, monkeypatch, schema):
    path = write_trajectory_artifact(
        tmp_path / "a.h5", _secondaries(), case=_CASE, provenance=_PROVENANCE
    )
    if schema == 1:
        _schema_1(path)
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
def test_checkpoint_aliases_warn_and_keep_help_and_exit_codes(old, new):
    alias_help = invoke(root_command, ["checkpoint", old, "--help"])
    new_help = invoke(root_command, ["trajectories", new, "--help"])
    alias_usage = invoke(root_command, ["checkpoint", old])
    new_usage = invoke(root_command, ["trajectories", new])

    assert alias_help.exit_code == new_help.exit_code == 0
    assert alias_help.stdout.split("\n", 1)[1] == new_help.stdout.split("\n", 1)[1]
    assert alias_usage.exit_code == new_usage.exit_code == 2
    assert message(f"checkpoint {old}") in alias_usage.stderr
    assert "is deprecated" not in new_usage.stderr
    assert old not in invoke(root_command, ["checkpoint", "--help"]).stdout
