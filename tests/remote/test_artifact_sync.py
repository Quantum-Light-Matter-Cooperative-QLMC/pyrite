import tarfile

import pytest

from pyrite import _energy_grid_artifacts as artifacts
from pyrite.remote import config, transport


def _identity(material: str, stop: float):
    return artifacts.artifact_identity(
        material,
        [{"energy_keV": 30, "start_eV": 10, "stop_eV": stop, "num": 4}],
        {"stop_eV": 100, "step_eV": 10},
        [30],
    )


def test_remote_artifact_inventory_accepts_only_content_matching_names(monkeypatch):
    digest = "a" * 64
    monkeypatch.setattr(config, "REMOTE_DIR", "/remote")
    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        lambda command: (
            f"{digest}  /remote/src/pyrite/data/catalog/energy-grid-artifacts/aa/{digest}.json\n"
            + "b" * 64
            + "  /remote/src/pyrite/data/catalog/energy-grid-artifacts/cc/"
            + "c" * 64
            + ".json\n"
        ),
    )

    assert transport._remote_energy_grid_artifacts() == frozenset({digest})


def test_remote_artifact_inventory_fails_closed_on_malformed_output(monkeypatch):
    monkeypatch.setattr(transport, "_ssh_capture", lambda command: "not sha256sum output\n")
    with pytest.raises(SystemExit, match="invalid remote energy-grid artifact inventory"):
        transport._remote_energy_grid_artifacts()


def test_sync_code_skips_remote_hashes_and_sends_only_missing_objects(tmp_path, monkeypatch):
    local_root = tmp_path / "repo"
    source = local_root / "src"
    source.mkdir(parents=True)
    (source / "regular.py").write_text("VALUE = 1\n")
    store = source / "pyrite" / "data" / "catalog" / "energy-grid-artifacts"
    present = artifacts.write_artifact(store, _identity("hopg", 40))
    missing = artifacts.write_artifact(store, _identity("hbn", 50))

    monkeypatch.setattr(config, "LOCAL_ROOT", local_root)
    monkeypatch.setattr(config, "SYNC_PATHS", ["src"])
    monkeypatch.setattr(config, "REMOTE_DIR", "/remote")
    inventory_commands = []

    def remote_inventory(command):
        if "squeue" in command:  # code-sync live-job guard, not the inventory
            return ""
        inventory_commands.append(command)
        return (
            f"{present.digest}  /remote/src/pyrite/data/catalog/energy-grid-artifacts/"
            f"{present.digest[:2]}/{present.digest}.json\n"
        )

    monkeypatch.setattr(
        transport,
        "_ssh_capture",
        remote_inventory,
    )
    archived = []

    def fake_run(argv, **kwargs):
        if argv[0] == "scp":
            with tarfile.open(argv[1], "r:gz") as bundle:
                archived.extend(bundle.getnames())

    monkeypatch.setattr(transport, "_run", fake_run)

    transport.sync_code()

    assert "src/regular.py" in archived
    assert any(missing.digest in name for name in archived)
    assert not any(present.digest in name for name in archived)
    assert len(inventory_commands) == 1
    assert "/remote/src/pyrite/data/catalog/energy-grid-artifacts" in inventory_commands[0]
