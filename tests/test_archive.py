"""Tests for the local checkpoint shelf (checkpoint lifecycle, component 2):
cxr archive / restore / archives. Pure local file ops on a temp checkpoints/
tree -- no ssh, CPU-only, fast."""

import pickle

import pytest

from cxr_mc import archive


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _store(n_configs=2, n_energies=2):
    return {
        f"cfg{i}": {30.0 + j: {"case": {}} for j in range(n_energies)} for i in range(n_configs)
    }


def test_archive_default_label_is_stem_dated(tmp_path):
    _write(tmp_path / "hopg.pkl", _store())
    dst = archive.archive_checkpoint("hopg", root=str(tmp_path))
    # default label: hopg-YYYYMMDD
    name = dst.rsplit("/", 1)[-1].replace("\\", "/").rsplit("/", 1)[-1]
    assert name.startswith("hopg-") and name.endswith(".pkl")
    assert (tmp_path / "archive" / name).is_file()


def test_archive_explicit_label(tmp_path):
    _write(tmp_path / "hopg.pkl", _store())
    archive.archive_checkpoint("hopg", "good-thickness", root=str(tmp_path))
    assert (tmp_path / "archive" / "good-thickness.pkl").is_file()


def test_archive_missing_active_slot_errors(tmp_path):
    with pytest.raises(SystemExit, match="no such active checkpoint"):
        archive.archive_checkpoint("nope", root=str(tmp_path))


def test_archive_refuses_overwrite_without_force(tmp_path):
    _write(tmp_path / "hopg.pkl", _store())
    archive.archive_checkpoint("hopg", "snap", root=str(tmp_path))
    with pytest.raises(SystemExit, match="already exists"):
        archive.archive_checkpoint("hopg", "snap", root=str(tmp_path))
    archive.archive_checkpoint("hopg", "snap", force=True, root=str(tmp_path))  # no raise


def test_roundtrip_preserves_contents(tmp_path):
    payload = _store(n_configs=3)
    _write(tmp_path / "hopg.pkl", payload)
    archive.archive_checkpoint("hopg", "snap", root=str(tmp_path))
    (tmp_path / "hopg.pkl").unlink()  # lose the active slot
    archive.restore_checkpoint("snap", "hopg", root=str(tmp_path))
    with open(tmp_path / "hopg.pkl", "rb") as f:
        assert set(pickle.load(f)) == set(payload)


def test_restore_infers_stem_from_dated_label(tmp_path):
    _write(tmp_path / "archive" / "hopg-20260704.pkl", _store())
    archive.restore_checkpoint("hopg-20260704", root=str(tmp_path))
    assert (tmp_path / "hopg.pkl").is_file()  # stem 'hopg' inferred (date stripped)


def test_restore_missing_label_errors(tmp_path):
    with pytest.raises(SystemExit, match="no such archive"):
        archive.restore_checkpoint("ghost", root=str(tmp_path))


def test_restore_refuses_overwrite_without_force(tmp_path):
    _write(tmp_path / "hopg.pkl", _store())
    _write(tmp_path / "archive" / "snap.pkl", _store(n_configs=5))
    with pytest.raises(SystemExit, match="already exists"):
        archive.restore_checkpoint("snap", "hopg", root=str(tmp_path))
    archive.restore_checkpoint("snap", "hopg", force=True, root=str(tmp_path))  # no raise
    with open(tmp_path / "hopg.pkl", "rb") as f:
        assert len(pickle.load(f)) == 5  # the forced restore won


def test_archives_lists_labels_and_counts(tmp_path, capsys):
    _write(tmp_path / "archive" / "a.pkl", _store(n_configs=2, n_energies=2))  # 4 records
    _write(tmp_path / "archive" / "b.pkl", _store(n_configs=1, n_energies=1))  # 1 record
    labels = archive.list_archives(root=str(tmp_path))
    assert labels == ["a", "b"]
    out = capsys.readouterr().out
    assert "a" in out and "4 records" in out and "1 records" in out


def test_archives_empty_shelf(tmp_path, capsys):
    assert archive.list_archives(root=str(tmp_path)) == []
    assert "(no archives)" in capsys.readouterr().out


def test_archive_and_restore_prints_use_forward_slashes(tmp_path, capsys):
    """The confirmation lines mix 'checkpoints/<stem>.pkl' with an os.path.join'd
    archive path, which prints 'archive\\label.pkl' on Windows -- keep both sides
    forward-slashed."""
    _write(tmp_path / "hopg.pkl", _store())
    archive.archive_checkpoint("hopg", "snap", root=str(tmp_path))
    archive.restore_checkpoint("snap", "hopg", force=True, root=str(tmp_path))
    out = capsys.readouterr().out
    assert "\\" not in out
    assert "archive/snap.pkl" in out


def test_default_root_is_repo_anchored():
    """cxr archive must hit the same checkpoints/ dir no matter the cwd, and it
    must be the exact dir run.load_checkpoint reads from (repo-root anchored)."""
    import os

    from cxr_mc import run

    assert os.path.isabs(archive.DEFAULT_ROOT)
    assert archive.DEFAULT_ROOT == run._DEFAULT_CHECKPOINT_DIR


def test_stem_from_label_strips_only_date_suffix():
    assert archive._stem_from_label("hopg-20260704") == "hopg"
    assert archive._stem_from_label("good-thickness") == "good-thickness"
    assert archive._stem_from_label("mose2_quick-20260101") == "mose2_quick"
