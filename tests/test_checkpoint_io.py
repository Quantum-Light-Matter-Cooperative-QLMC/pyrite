"""Tests for cxr_mc._checkpoint_io: gzip checkpoint compression (P2 #8) with
transparent backward-compat reads of legacy plain pickles."""

import gzip
import pickle

import numpy as np

from cxr_mc import _checkpoint_io as ckio


def _payload():
    return {"cfg_a": {30.0: {"case": {"crystal": "hopg"}, "spec": np.ones(1000)}}}


def test_dump_writes_gzip_magic_header(tmp_path):
    path = tmp_path / "hopg.pkl"
    ckio.dump(_payload(), str(path))
    assert path.read_bytes()[:2] == b"\x1f\x8b"


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


def test_load_reads_externally_gzipped_pickle(tmp_path):
    """Any gzip-compressed pickle sniffs as the new format, not just ones
    written by ckio.dump -- the detection is content-based, not a marker we
    control end-to-end."""
    path = tmp_path / "ext.pkl"
    payload = _payload()
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f)
    loaded = ckio.load(str(path))
    assert set(loaded) == set(payload)
