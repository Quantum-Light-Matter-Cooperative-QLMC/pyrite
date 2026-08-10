import hashlib
from copy import deepcopy

import pytest

from pyrite.energy_grid import artifacts


def _identity(*, annotations=None):
    return artifacts.artifact_identity(
        " HOPG ",
        [
            {"energy_keV": 100, "start_eV": 50, "stop_eV": 4600, "num": 1518},
            {"energy_keV": 30, "start_eV": 10, "stop_eV": 2600, "num": 864},
        ],
        {"stop_eV": 136500, "step_eV": 25},
        [100, 30, 30.0],
        annotations=annotations,
    )


def test_identity_field_contract_and_exact_hash_are_frozen():
    identity = _identity()
    assert artifacts.IDENTITY_FIELDS == (
        "schema",
        "material",
        "line_rows",
        "brem_grid",
        "beam_energies_keV",
    )
    assert artifacts.EXCLUDED_ANNOTATION_FIELDS == (
        "provenance",
        "notes",
        "timestamp",
        "ref_name",
        "orphaned_at",
    )
    assert artifacts.canonical_bytes(identity) == (
        b'{"beam_energies_keV":[30.0,100.0],"brem_grid":{"start_eV":0.0,'
        b'"step_eV":25.0,"stop_eV":136500.0},"line_rows":[{"energy_keV":30.0,'
        b'"num":864,"start_eV":10.0,"stop_eV":2600.0},{"energy_keV":100.0,'
        b'"num":1518,"start_eV":50.0,"stop_eV":4600.0}],"material":"hopg",'
        b'"schema":"cxr.energy-grid-artifact.v1"}'
    )
    assert (
        artifacts.artifact_digest(identity)
        == "c701b3119c83ec199933f65b35c6c5efddb47a55e05513f94306e04deab18f55"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("material", "hbn"),
        ("line_energy", 31.0),
        ("line_start", 11.0),
        ("line_stop", 2601.0),
        ("line_num", 865),
        ("line_rows", 1),
        ("brem_start", 1.0),
        ("brem_stop", 136501.0),
        ("brem_step", 26.0),
        ("beam_energies_keV", [30.0, 90.0, 100.0]),
    ],
)
def test_each_identity_input_changes_hash(field, value):
    identity = _identity()
    changed = deepcopy(identity)
    if field == "line_energy":
        changed["line_rows"][0]["energy_keV"] = value
    elif field == "line_start":
        changed["line_rows"][0]["start_eV"] = value
    elif field == "line_stop":
        changed["line_rows"][0]["stop_eV"] = value
    elif field == "line_num":
        changed["line_rows"][0]["num"] = value
    elif field == "line_rows":
        changed["line_rows"] = changed["line_rows"][:value]
    elif field.startswith("brem_"):
        changed["brem_grid"][f"{field.removeprefix('brem_')}_eV"] = value
    else:
        changed[field] = value
    assert artifacts.artifact_digest(identity) != artifacts.artifact_digest(changed)


def test_identity_normalizes_orders_and_excludes_annotations():
    plain = _identity()
    annotated = _identity(annotations={"notes": "rerun after calibration", "timestamp": "later"})
    assert annotated == plain
    assert artifacts.artifact_digest(annotated) == artifacts.artifact_digest(plain)
    assert plain["line_rows"][0]["energy_keV"] == 30.0
    assert plain["beam_energies_keV"] == [30.0, 100.0]


def test_write_deduplicates_and_loads_verified_bytes(tmp_path):
    identity = _identity()
    first = artifacts.write_artifact(tmp_path, identity)
    second = artifacts.write_artifact(tmp_path, identity)
    assert second == first
    assert first.path == tmp_path / first.digest[:2] / f"{first.digest}.json"
    assert artifacts.load_artifact(tmp_path, first.digest) == first
    assert artifacts.inventory_artifacts(tmp_path) == (first.digest,)


def test_load_detects_missing_corrupt_and_mismatched_artifacts(tmp_path):
    identity = _identity()
    digest = artifacts.artifact_digest(identity)
    with pytest.raises(artifacts.ArtifactMissingError, match="is missing"):
        artifacts.verify_artifact(tmp_path, digest)

    path = artifacts.artifact_path(tmp_path, digest)
    path.parent.mkdir()
    path.write_bytes(b"not json")
    with pytest.raises(artifacts.ArtifactCorruptError, match="bytes hash"):
        artifacts.load_artifact(tmp_path, digest)

    replacement = artifacts.canonical_bytes(_identity())
    mismatched_digest = "0" * 64
    mismatch_path = artifacts.artifact_path(tmp_path, mismatched_digest)
    mismatch_path.parent.mkdir(exist_ok=True)
    mismatch_path.write_bytes(replacement)
    with pytest.raises(artifacts.ArtifactCorruptError, match="bytes hash"):
        artifacts.load_artifact(tmp_path, mismatched_digest)


def test_write_rejects_existing_bytes_that_do_not_match_digest(tmp_path):
    identity = _identity()
    digest = artifacts.artifact_digest(identity)
    path = artifacts.artifact_path(tmp_path, digest)
    path.parent.mkdir()
    path.write_bytes(b"corrupt")
    with pytest.raises(artifacts.ArtifactCorruptError, match="bytes hash"):
        artifacts.write_artifact(tmp_path, identity)


def test_inventory_ignores_unsafe_or_malformed_entries(tmp_path):
    digest = hashlib.sha256(b"ok").hexdigest()
    valid = tmp_path / digest[:2] / f"{digest}.json"
    valid.parent.mkdir()
    valid.write_bytes(b"ok")
    (tmp_path / "junk").mkdir()
    bad_shard = tmp_path / "ab"
    bad_shard.mkdir()
    (bad_shard / "not-a-digest.json").write_text("x")
    assert artifacts.inventory_artifacts(tmp_path) == (digest,)
