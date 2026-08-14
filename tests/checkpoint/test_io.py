"""Versioned HDF5 checkpoint encoding and permanent pickle compatibility."""

import gzip
import io
import pickle

import h5py
import numpy as np
from compression import zstd

from pyrite.checkpoints import _checkpoint_io as ckio


def _payload():
    return {"cfg_a": {30.0: {"case": {"crystal": "hopg"}, "spec": np.ones(1000)}}}


def test_dump_writes_independently_readable_hdf5(tmp_path):
    path = tmp_path / "hopg.pkl"
    ckio.dump(_payload(), str(path))
    assert path.read_bytes()[:8] == b"\x89HDF\r\n\x1a\n"
    with h5py.File(path, "r") as h5:
        assert h5.attrs["schema"] == "pyrite.result"
        assert h5.attrs["schema_version"] == 1
        assert h5.attrs["identity_version"] == 1
        assert h5["value/items/00000000/value/items/00000000/value/items/00000001/value"][...].shape == (
            1000,
        )


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


def test_spectrum_bytes_roundtrip_bit_for_bit(tmp_path):
    path = tmp_path / "spectrum.pkl"
    spectrum = np.array([0.0, -0.0, np.nan, np.inf, -np.inf, np.nextafter(1.0, 2.0)])
    ckio.dump({"spec": spectrum}, str(path))
    loaded = ckio.load(str(path))["spec"]
    assert loaded.dtype == spectrum.dtype
    assert loaded.tobytes() == spectrum.tobytes()


def test_dump_load_preserves_container_kinds_nulls_and_numpy_dtypes(tmp_path):
    path = tmp_path / "shapes.pkl"
    payload = {
        "absent-is-distinct": None,
        "sequence": ([np.int16(2)], ("x", np.float32(3.5))),
        30.0: {"unicode": "MoS₂", "logical": True, "complex": 1 + 2j},
    }
    ckio.dump(payload, str(path))
    loaded = ckio.load(str(path))
    assert loaded.keys() == payload.keys()
    assert loaded["absent-is-distinct"] is None
    assert isinstance(loaded["sequence"], tuple)
    assert isinstance(loaded["sequence"][0], list)
    assert isinstance(loaded["sequence"][0][0], np.int16)
    assert isinstance(loaded["sequence"][1][1], np.float32)
    assert loaded[30.0] == payload[30.0]


def test_dump_uses_portable_gzip_filter_for_numeric_arrays(tmp_path):
    path = tmp_path / "compressed.pkl"
    ckio.dump(_payload(), str(path))
    with h5py.File(path, "r") as h5:
        spec = h5["value/items/00000000/value/items/00000000/value/items/00000001/value"]
        assert spec.compression == "gzip"
        assert spec.shuffle


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


def test_load_reads_zstd_pickle(tmp_path):
    path = tmp_path / "zstd.pkl"
    payload = _payload()
    with zstd.ZstdFile(path, "wb") as f:
        pickle.dump(payload, f)
    loaded = ckio.load(str(path))
    assert set(loaded) == set(payload)
    assert np.array_equal(loaded["cfg_a"][30.0]["spec"], payload["cfg_a"][30.0]["spec"])
