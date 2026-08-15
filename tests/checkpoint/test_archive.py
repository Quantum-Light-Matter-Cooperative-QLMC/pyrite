"""Tests for the local checkpoint shelf (checkpoint lifecycle, component 2):
cxr archive / restore / archives / union. Pure local file ops on a temp
checkpoints/ tree -- no ssh, CPU-only, fast."""

import json
import pickle

import pytest

from pyrite.checkpoints import _checkpoint_io, _checkpoint_store, archive


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _store(n_configs=2, n_energies=2):
    return {
        f"cfg{i}": {30.0 + j: {"case": {}} for j in range(n_energies)} for i in range(n_configs)
    }


def _material_store(material, config_names, n_energies=2):
    """A results store like ``_store`` but with a real ``case["crystal"]`` (union
    needs it for the material-match check) and caller-chosen config names, so
    tests can control which (name, E0) points overlap between two stores."""
    return {
        name: {30.0 + j: {"case": {"crystal": material}} for j in range(n_energies)}
        for name in config_names
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


def test_component_checkpoint_archive_restore_roundtrip(tmp_path):
    payload = _material_store("hopg", ["cfgA", "cfgB"])
    _checkpoint_store.save("hopg", tmp_path, payload)

    archive.archive_checkpoint("hopg", "snap", root=str(tmp_path))
    assert (tmp_path / "archive" / "snap" / "line.h5").is_file()
    assert (tmp_path / "archive" / "snap" / "brem.h5").is_file()

    for path in (tmp_path / "hopg").iterdir():
        path.unlink()
    (tmp_path / "hopg").rmdir()
    archive.restore_checkpoint("snap", "hopg", root=str(tmp_path))

    assert set(_checkpoint_store.load("hopg", tmp_path)) == {"cfgA", "cfgB"}


def test_shards_only_checkpoint_archive_restore_roundtrip(tmp_path):
    expected = _material_store("hopg", ["cfg"])
    _checkpoint_store.save_part("hopg", tmp_path, "cfg", expected["cfg"])

    archive.archive_checkpoint("hopg", "paused", root=str(tmp_path))
    assert (tmp_path / "archive" / "paused" / "parts").is_dir()
    _checkpoint_store.clear_parts("hopg", tmp_path)
    archive.restore_checkpoint("paused", "hopg", force=True, root=str(tmp_path))

    assert _checkpoint_store.load("hopg", tmp_path) == expected


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

    from pyrite.runs import run

    assert os.path.isabs(archive.DEFAULT_ROOT)
    assert archive.DEFAULT_ROOT == run.DEFAULT_CHECKPOINT_DIR


def test_stem_from_label_strips_only_date_suffix():
    assert archive._stem_from_label("hopg-20260704") == "hopg"
    assert archive._stem_from_label("good-thickness") == "good-thickness"
    assert archive._stem_from_label("mose2_quick-20260101") == "mose2_quick"


# ---- union --------------------------------------------------------------------


def test_union_happy_path_merges_and_pre_archives(tmp_path):
    live = _material_store("hopg", ["cfgA", "cfgB"])
    incoming = _material_store("hopg", ["cfgB", "cfgC"])
    _write(tmp_path / "hopg.pkl", live)
    _write(tmp_path / "archive" / "snap.pkl", incoming)

    archive.union_checkpoint("hopg", "snap", root=str(tmp_path))

    merged = _checkpoint_io.load(str(tmp_path / "hopg.pkl"))
    assert set(merged) == {"cfgA", "cfgB", "cfgC"}
    # default: undoable -- the pre-union live checkpoint was archived first
    predate_label = archive._default_label("hopg")
    assert (tmp_path / "archive" / f"{predate_label}.pkl").is_file()
    # default: the source archive is left intact
    assert (tmp_path / "archive" / "snap.pkl").is_file()


def test_union_refuses_material_mismatch(tmp_path):
    _write(tmp_path / "hopg.pkl", _material_store("hopg", ["cfgA"]))
    _write(tmp_path / "archive" / "snap.pkl", _material_store("mos2", ["cfgB"]))
    with pytest.raises(SystemExit, match="material mismatch"):
        archive.union_checkpoint("hopg", "snap", root=str(tmp_path))
    # refused before any mutation: the active slot is untouched
    live = _checkpoint_io.load(str(tmp_path / "hopg.pkl"))
    assert set(live) == {"cfgA"}


def test_union_refuses_resolved_dataset_identity_mismatch(tmp_path):
    live = _material_store("hopg", ["cfgA"])
    incoming = _material_store("hopg", ["cfgB"])
    _checkpoint_store.save("hopg", tmp_path, live)
    _checkpoint_store.save("snap", tmp_path / "archive", incoming)
    for path, profile, digest in (
        (tmp_path / "hopg" / "meta.json", "full", "a" * 64),
        (tmp_path / "archive" / "snap" / "meta.json", "survey", "b" * 64),
    ):
        path.write_text(
            json.dumps(
                {
                    "dataset_identity": {
                        "profile": profile,
                        "parameter_sha256": digest,
                    }
                }
            )
        )

    with pytest.raises(SystemExit, match="dataset identity mismatch"):
        archive.union_checkpoint("hopg", "snap", pre_archive=False, root=str(tmp_path))

    assert set(_checkpoint_store.load("hopg", tmp_path)) == {"cfgA"}


def test_union_live_wins_on_collision(tmp_path):
    live = {"cfgA": {30.0: {"case": {"crystal": "hopg"}, "tag": "live"}}}
    incoming = {"cfgA": {30.0: {"case": {"crystal": "hopg"}, "tag": "archived"}}}
    _write(tmp_path / "hopg.pkl", live)
    _write(tmp_path / "archive" / "snap.pkl", incoming)

    archive.union_checkpoint("hopg", "snap", root=str(tmp_path))

    merged = _checkpoint_io.load(str(tmp_path / "hopg.pkl"))
    assert merged["cfgA"][30.0]["tag"] == "live"  # live wins the (name, E0) collision


def test_union_no_archive_skips_pre_archive(tmp_path):
    _write(tmp_path / "hopg.pkl", _material_store("hopg", ["cfgA"]))
    _write(tmp_path / "archive" / "snap.pkl", _material_store("hopg", ["cfgB"]))

    archive.union_checkpoint("hopg", "snap", pre_archive=False, root=str(tmp_path))

    predate_label = archive._default_label("hopg")
    assert not (tmp_path / "archive" / f"{predate_label}.pkl").exists()
    merged = _checkpoint_io.load(str(tmp_path / "hopg.pkl"))
    assert set(merged) == {"cfgA", "cfgB"}


def test_union_delete_archive_removes_source(tmp_path):
    _write(tmp_path / "hopg.pkl", _material_store("hopg", ["cfgA"]))
    _write(tmp_path / "archive" / "snap.pkl", _material_store("hopg", ["cfgB"]))

    archive.union_checkpoint("hopg", "snap", delete_archive=True, root=str(tmp_path))

    assert not (tmp_path / "archive" / "snap.pkl").exists()


def test_union_missing_archive_errors(tmp_path):
    _write(tmp_path / "hopg.pkl", _material_store("hopg", ["cfgA"]))
    with pytest.raises(SystemExit, match="no such archive"):
        archive.union_checkpoint("hopg", "ghost", root=str(tmp_path))


def test_union_missing_active_slot_errors(tmp_path):
    _write(tmp_path / "archive" / "snap.pkl", _material_store("hopg", ["cfgA"]))
    with pytest.raises(SystemExit, match="no such active checkpoint"):
        archive.union_checkpoint("hopg", "snap", root=str(tmp_path))


def test_union_refuses_pre_archive_collision_with_source_even_with_force(tmp_path):
    """If the union source archive happens to sit at the pre-archive step's
    default label (e.g. a same-day round trip: archive this morning, union that
    label back in this afternoon), the pre-archive step must not be allowed to
    overwrite it -- not even with --force -- since that would silently destroy
    the very archive the union is reading from while claiming (by leaving
    delete_archive=False) to keep it intact."""
    predate_label = archive._default_label("hopg")
    _write(tmp_path / "hopg.pkl", _material_store("hopg", ["cfgA"]))
    _write(tmp_path / "archive" / f"{predate_label}.pkl", _material_store("hopg", ["cfgB"]))

    with pytest.raises(SystemExit, match="refusing to union"):
        archive.union_checkpoint("hopg", predate_label, root=str(tmp_path))
    with pytest.raises(SystemExit, match="refusing to union"):
        archive.union_checkpoint("hopg", predate_label, force=True, root=str(tmp_path))

    # refused before any mutation: both the active slot and the source archive
    # are byte-for-byte untouched
    live = _checkpoint_io.load(str(tmp_path / "hopg.pkl"))
    assert set(live) == {"cfgA"}
    archived = _checkpoint_io.load(str(tmp_path / "archive" / f"{predate_label}.pkl"))
    assert set(archived) == {"cfgB"}

    # --no-archive still lets the same union go through
    archive.union_checkpoint("hopg", predate_label, pre_archive=False, root=str(tmp_path))
    merged = _checkpoint_io.load(str(tmp_path / "hopg.pkl"))
    assert set(merged) == {"cfgA", "cfgB"}
