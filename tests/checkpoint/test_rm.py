import json

from pyrite.checkpoints import _checkpoint_store
from pyrite.cli import command as root_command
from tests.helpers.cli import assert_clean_result, invoke


def _dataset(root, stem, material, keys, *, profile="standard"):
    directory = root / stem
    directory.mkdir(parents=True)
    (directory / "line.pkl").touch()
    (directory / "cases.json").write_text(
        json.dumps(
            {
                "schema": "cxr.case-manifest.v1",
                "material": material,
                "catalog_profile": profile,
                "cases": [
                    {"name": f"case-{index}", "E0_keV": 30.0, "content_key": key}
                    for index, key in enumerate(keys)
                ],
            }
        )
    )
    return directory


def _archive(root, label, material, keys, *, profile="standard"):
    return _dataset(root / "archive", label, material, keys, profile=profile)


def _blob(root, material, key):
    path = _checkpoint_store.cas_blob_path(material, key, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"case")
    return path


def test_rm_preview_lists_datasets_and_newly_unreachable_blobs(tmp_path):
    key = "a" * 64
    dataset = _dataset(tmp_path, "hopg", "hopg", [key])
    blob = _blob(tmp_path, "hopg", key)

    result = invoke(
        root_command,
        ["checkpoint", "rm", "hopg", "--checkpoint-dir", str(tmp_path)],
    )

    assert_clean_result(result)
    assert str(dataset) in result.stdout
    assert str(blob) in result.stdout
    assert "preview only" in result.stdout
    assert dataset.exists()
    assert blob.exists()


def test_rm_deletes_selected_datasets_but_preserves_archive_reachable_blob(tmp_path):
    retained_key = "a" * 64
    dropped_key = "b" * 64
    primary = _dataset(tmp_path, "hopg", "hopg", [retained_key])
    variant = _dataset(
        tmp_path,
        "hopg@sub_100keV",
        "hopg",
        [dropped_key],
        profile="sub_100keV",
    )
    archive = _archive(tmp_path, "keeper", "hopg", [retained_key])
    retained_blob = _blob(tmp_path, "hopg", retained_key)
    dropped_blob = _blob(tmp_path, "hopg", dropped_key)

    result = invoke(
        root_command,
        ["checkpoint", "rm", "hopg", "--checkpoint-dir", str(tmp_path), "--yes"],
    )

    assert_clean_result(result)
    assert "deleted 2 dataset(s) and 1 unreachable CAS blob(s)" in result.stdout
    assert not (primary / "line.pkl").exists()
    assert not variant.exists()
    assert archive.exists()
    assert retained_blob.exists()
    assert not dropped_blob.exists()


def test_rm_profile_selects_only_matching_dataset_identity(tmp_path):
    standard = _dataset(tmp_path, "hopg", "hopg", [], profile="standard")
    survey = _dataset(tmp_path, "hopg@survey", "hopg", [], profile="survey")

    result = invoke(
        root_command,
        [
            "checkpoint",
            "rm",
            "--profile",
            "survey",
            "--checkpoint-dir",
            str(tmp_path),
            "--yes",
        ],
    )

    assert_clean_result(result)
    assert standard.exists()
    assert not survey.exists()


def test_rm_fails_closed_on_malformed_archive_manifest(tmp_path):
    dataset = _dataset(tmp_path, "hopg", "hopg", [])
    archive = tmp_path / "archive" / "broken"
    archive.mkdir(parents=True)
    (archive / "line.pkl").touch()
    (archive / "cases.json").write_text("not json")

    result = invoke(
        root_command,
        ["checkpoint", "rm", "hopg", "--checkpoint-dir", str(tmp_path), "--yes"],
    )

    assert result.exit_code == 1
    assert "cannot safely clear: invalid manifest" in result.stderr
    assert dataset.exists()


def test_rm_requires_exactly_one_selector(tmp_path):
    missing = invoke(
        root_command,
        ["checkpoint", "rm", "--checkpoint-dir", str(tmp_path)],
    )
    conflicting = invoke(
        root_command,
        ["checkpoint", "rm", "hopg", "--all", "--checkpoint-dir", str(tmp_path)],
    )

    assert missing.exit_code == 2
    assert conflicting.exit_code == 2
    assert "exactly one" in missing.stderr
    assert "exactly one" in conflicting.stderr
