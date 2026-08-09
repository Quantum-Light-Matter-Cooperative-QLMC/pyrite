"""Tests for cxr_mc.checkpoints._checkpoint_io: zstd checkpoint compression (P2 #8) with
transparent backward-compat reads of gzip and legacy plain pickles."""

import gzip
import io
import pickle

import numpy as np

from cxr_mc.checkpoints import _checkpoint_io as ckio


def _payload():
    return {"cfg_a": {30.0: {"case": {"crystal": "hopg"}, "spec": np.ones(1000)}}}


def test_dump_writes_zstd_magic_header(tmp_path):
    path = tmp_path / "hopg.pkl"
    ckio.dump(_payload(), str(path))
    assert path.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"


def test_dump_is_byte_stable_across_paths_and_write_times(tmp_path):
    """Remote slim exports must compare equal to equivalent local checkpoints.

    zstd frames carry no timestamp or filename, so equivalent dumps match with
    no header pinning (gzip needed an mtime override for this).
    """
    local_path = tmp_path / "hopg.pkl"
    remote_path = tmp_path / "hopg.grid.pkl"
    ckio.dump(_payload(), str(local_path))
    ckio.dump(_payload(), str(remote_path))

    assert remote_path.read_bytes() == local_path.read_bytes()


def test_dump_stream_matches_dump_and_leaves_stream_open(tmp_path):
    """`cxr slim -o -` pipes the same bytes a file dump would produce, and must
    not close the caller's stdout."""
    path = tmp_path / "hopg.pkl"
    ckio.dump(_payload(), str(path))
    buffer = io.BytesIO()
    ckio.dump_stream(_payload(), buffer)
    assert buffer.getvalue() == path.read_bytes()
    assert not buffer.closed


def test_dump_honours_compresslevel(tmp_path):
    payload = {f"cfg_{i}": {30.0: {"spec": np.arange(4000.0)}} for i in range(40)}
    low = tmp_path / "low.pkl"
    high = tmp_path / "high.pkl"
    ckio.dump(payload, str(low), compresslevel=ckio.LEVEL_RANGE[0])
    ckio.dump(payload, str(high), compresslevel=ckio.MAX_LEVEL)
    assert high.stat().st_size <= low.stat().st_size
    assert ckio.load(str(high)).keys() == payload.keys()


def test_dump_load_roundtrips(tmp_path):
    path = tmp_path / "hopg.pkl"
    payload = _payload()
    ckio.dump(payload, str(path))
    loaded = ckio.load(str(path))
    assert set(loaded) == set(payload)
    assert np.allclose(loaded["cfg_a"][30.0]["spec"], payload["cfg_a"][30.0]["spec"])


def test_dump_is_smaller_than_plain_pickle(tmp_path):
    payload = _payload()
    compressed = tmp_path / "compressed.pkl"
    plain = tmp_path / "plain.pkl"
    ckio.dump(payload, str(compressed))
    with open(plain, "wb") as f:
        pickle.dump(payload, f)
    assert compressed.stat().st_size < plain.stat().st_size


def test_load_reads_legacy_plain_pickle(tmp_path):
    """A checkpoint written before this change (or copied verbatim from the
    box) is a plain pickle -- load() must read it with no conversion step."""
    path = tmp_path / "legacy.pkl"
    payload = _payload()
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    loaded = ckio.load(str(path))
    assert set(loaded) == set(payload)
    assert np.allclose(loaded["cfg_a"][30.0]["spec"], payload["cfg_a"][30.0]["spec"])


def test_load_reads_gzip_pickle(tmp_path):
    """Checkpoints written by the previous (gzip) format -- on the laptop or
    left on the box -- must keep loading with no conversion step; detection is
    content-based, not a marker we control end-to-end."""
    path = tmp_path / "ext.pkl"
    payload = _payload()
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f)
    loaded = ckio.load(str(path))
    assert set(loaded) == set(payload)
