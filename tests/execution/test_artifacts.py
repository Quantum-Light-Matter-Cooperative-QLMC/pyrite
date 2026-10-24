"""Worker artifact publication is atomic, immutable and independently verifiable."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from pyrite.execution.artifacts import (
    ArtifactIntegrityError,
    CaseArtifactStore,
)

KEY = "a" * 64
OTHER_KEY = "b" * 64


def _output(value=1.0):
    return {"spec": np.array([value, value + 1]), "E_grid": np.array([100., 200.])}


def test_worker_publication_is_immutable_and_returns_lightweight_reference(tmp_path):
    store = CaseArtifactStore(tmp_path)
    first = store.publish("hopg", KEY, _output())
    assert first.material == "hopg"
    assert first.content_key == KEY
    assert first.size_bytes > 0
    assert store.path_for("hopg", KEY).is_file()
    assert not (tmp_path / "hopg" / "meta.json").exists()
    assert not (tmp_path / "hopg" / "parts").exists()
    assert not (tmp_path / "hopg" / "line.h5").exists()

    # Same physics key: first complete result wins, not last writer.
    second = store.publish("hopg", KEY, _output(9.0))
    assert second == first
    np.testing.assert_array_equal(store.load(first)["spec"], [1., 2.])


def test_concurrent_workers_do_not_clobber_same_content_key(tmp_path):
    store = CaseArtifactStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        refs = list(pool.map(lambda i: store.publish("hopg", KEY, _output(float(i))), range(4)))
    assert len(set(refs)) == 1
    assert len(list((tmp_path / "hopg" / ".execution-artifacts").rglob("*.h5"))) == 1
    assert list((tmp_path / "hopg" / ".execution-artifacts").rglob("*.tmp")) == []


def test_crashed_writer_has_no_completed_artifact(tmp_path, monkeypatch):
    store = CaseArtifactStore(tmp_path)
    from pyrite.checkpoints import _checkpoint_io

    def truncated_dump(_obj, path):
        with open(path, "wb") as stream:
            stream.write(b"partial")
        raise RuntimeError("worker crashed while serializing")

    monkeypatch.setattr(_checkpoint_io, "dump", truncated_dump)
    with pytest.raises(RuntimeError, match="worker crashed"):
        store.publish("hopg", KEY, _output())
    assert store.recover("hopg", KEY) is None
    assert list((tmp_path / "hopg" / ".execution-artifacts").rglob("*.tmp")) == []


def test_missing_unfinished_and_corrupt_are_distinct(tmp_path):
    store = CaseArtifactStore(tmp_path)
    assert store.recover("hopg", KEY) is None
    path = store.path_for("hopg", KEY)
    path.parent.mkdir(parents=True)
    (path.parent / ".abandoned.tmp").write_bytes(b"incomplete")
    assert store.recover("hopg", KEY) is None
    ref = store.publish("hopg", KEY, _output())
    recovered_ref, recovered = store.recover("hopg", KEY)
    assert recovered_ref == ref
    np.testing.assert_array_equal(recovered["spec"], [1., 2.])
    with path.open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ArtifactIntegrityError, match="digest or length"):
        store.load(ref)
    with pytest.raises(ArtifactIntegrityError):
        store.recover("hopg", KEY)
    with pytest.raises(ArtifactIntegrityError):
        store.publish("hopg", KEY, _output())


def test_case_identity_validation_is_independent_of_worker_claim(tmp_path):
    store = CaseArtifactStore(tmp_path)
    ref = store.publish("hopg", KEY, _output())
    with pytest.raises(ArtifactIntegrityError, match="content key"):
        store.load(ref, expected_content_key=OTHER_KEY)
    with pytest.raises(ArtifactIntegrityError, match="material"):
        store.load(ref, expected_material="silicon")
    with pytest.raises(ArtifactIntegrityError, match="digest"):
        store.load(replace(ref, sha256=OTHER_KEY))
    with pytest.raises(ValueError, match="path component"):
        store.path_for("../escape", KEY)
    with pytest.raises(ValueError, match="SHA-256"):
        store.publish("hopg", "not a digest", _output())


def test_crash_between_data_link_and_receipt_is_resumable(tmp_path, monkeypatch):
    import pyrite.execution.artifacts as artifacts

    store = CaseArtifactStore(tmp_path)
    original_link = artifacts.os.link

    def fail_receipt_link(source, destination):
        if str(destination).endswith(".json"):
            raise RuntimeError("coordinator connection died before commit")
        return original_link(source, destination)

    with monkeypatch.context() as scope:
        scope.setattr(artifacts.os, "link", fail_receipt_link)
        with pytest.raises(RuntimeError, match="before commit"):
            store.publish("hopg", KEY, _output())

    # Linked data alone is not a committed result. A later retry repairs the
    # receipt using that exact immutable data rather than replacing it.
    assert store.path_for("hopg", KEY).is_file()
    assert store.recover("hopg", KEY) is None
    ref = store.publish("hopg", KEY, _output(999.0))
    np.testing.assert_array_equal(store.load(ref)["spec"], [1., 2.])
