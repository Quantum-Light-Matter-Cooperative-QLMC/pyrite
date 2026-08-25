"""Versioned HDF5 checkpoint encoding and permanent pickle compatibility."""

import gzip
import io
import pickle
import time
from pathlib import Path

import h5py
import numpy as np
from compression import zstd

from pyrite.checkpoints import _checkpoint_io as ckio

_DATA = Path(__file__).parent / "data"


def _payload():
    return {"cfg_a": {30.0: {"case": {"crystal": "hopg"}, "spec": np.ones(1000)}}}


def test_dump_writes_independently_readable_hdf5(tmp_path):
    """ADR-0009: a stored artifact opens in stock h5py, with no pyrite code and
    no unpickling, and its arrays are reachable under readable names."""
    path = tmp_path / "hopg.pkl"
    ckio.dump(_payload(), str(path))
    assert path.read_bytes()[:8] == b"\x89HDF\r\n\x1a\n"
    with h5py.File(path, "r") as h5:
        assert h5.attrs["schema"] == "pyrite.result"
        assert h5.attrs["schema_version"] == 2
        assert h5.attrs["identity_version"] == 1
        assert h5["value/cfg_a/_00000000/spec"][...].shape == (1000,)
        assert h5["value/cfg_a/_00000000/case"].attrs["v:crystal"] == "hopg"


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


def test_dump_stream_frames_the_container_and_leaves_stream_open(tmp_path):
    """`pyrite slim -o -` is a wire format, not a stored artifact: it frames the
    same container in zstd so redundant HDF5 metadata is not sent raw. It must
    not close the caller's stdout, and load() must take the frame back."""
    path = tmp_path / "hopg.pkl"
    ckio.dump(_payload(), str(path))
    buffer = io.BytesIO()
    ckio.dump_stream(_payload(), buffer)
    framed = buffer.getvalue()
    assert not buffer.closed
    assert framed[:4] == b"\x28\xb5\x2f\xfd"
    assert zstd.decompress(framed)[:8] == b"\x89HDF\r\n\x1a\n"

    piped = tmp_path / "piped.pkl"
    piped.write_bytes(framed)
    loaded = ckio.load(str(piped))
    assert np.array_equal(loaded["cfg_a"][30.0]["spec"], _payload()["cfg_a"][30.0]["spec"])


def test_compresslevel_selects_the_frame_and_never_the_stored_artifact(tmp_path):
    """The 1--22 interface survives for CLI compatibility, but deflate is
    counterproductive once array content is deduplicated, so nothing on disk is
    compressed. The level now only picks the transfer frame's strength."""
    payload = {f"cfg_{i}": {30.0: {"spec": np.arange(4000.0)}} for i in range(40)}
    low = tmp_path / "low.pkl"
    high = tmp_path / "high.pkl"
    ckio.dump(payload, str(low), compresslevel=ckio.LEVEL_RANGE[0])
    ckio.dump(payload, str(high), compresslevel=ckio.MAX_LEVEL)
    assert high.read_bytes() == low.read_bytes()
    assert ckio.load(str(high)).keys() == payload.keys()

    cheap, dear = io.BytesIO(), io.BytesIO()
    ckio.dump_stream(payload, cheap, compresslevel=ckio.LEVEL_RANGE[0])
    ckio.dump_stream(payload, dear, compresslevel=ckio.MAX_LEVEL)
    assert len(dear.getvalue()) <= len(cheap.getvalue())


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


def test_stored_arrays_carry_no_filters_and_wire_arrays_carry_shuffle(tmp_path):
    """Measured on a real store: gzip costs ~25x the write time to save nothing
    once arrays are deduplicated, so no stored dataset is filtered. Its
    companion byte-transposition filter is what pays, and only in front of the
    frame, so it appears on the wire and nowhere else."""
    path = tmp_path / "stored.pkl"
    ckio.dump(_payload(), str(path))
    with h5py.File(path, "r") as h5:
        spec = h5["value/cfg_a/_00000000/spec"]
        assert spec.compression is None
        assert not spec.shuffle

    buffer = io.BytesIO()
    ckio.dump_stream(_payload(), buffer)
    wire = tmp_path / "wire.h5"
    wire.write_bytes(zstd.decompress(buffer.getvalue()))
    with h5py.File(wire, "r") as h5:
        spec = h5["value/cfg_a/_00000000/spec"]
        assert spec.compression is None
        assert spec.shuffle


def test_dump_load_roundtrips_empty_containers(tmp_path):
    """`pyrite slim` trims records down to empty mappings and arrays; an empty set
    of names has no inferable element type, so it needs an explicit one."""
    path = tmp_path / "empty.pkl"
    payload = {"n": {30.0: {"case": {}, "hkl_list": [], "tags": (), "spec": np.zeros(0)}}}
    ckio.dump(payload, str(path))
    record = ckio.load(str(path))["n"][30.0]
    assert record["case"] == {}
    assert record["hkl_list"] == []
    assert record["tags"] == ()
    assert record["spec"].shape == (0,)


def test_load_reads_schema_version_1_hdf5_fixture():
    """Version 1 spent one HDF5 object per Python leaf. It is never written
    again, but every artifact already on a laptop or box is one, and there is no
    migration step -- so this committed fixture must read forever."""
    payload = ckio.load(str(_DATA / "schema_v1.pkl"))
    assert set(payload) == {"config-a", "config-b", "meta"}
    record = payload["config-a"][30.0]
    assert set(record) == {"E_grid", "spec", "eta", "case"}
    assert record["E_grid"].shape == record["spec"].shape
    assert isinstance(record["eta"], float)
    assert record["case"]["crystal"] == "hopg"
    assert record["case"]["hkl_list"] == [(0, 0, 2), (0, 0, -2)]
    assert record["case"]["surface_hkl"] is None
    assert record["case"]["coherent_emission"] is True
    assert payload["config-b"][40.0]["spec"][1] == 2.0
    assert payload["meta"] == ("v1", b"\x00\x01", ["x", 2, None])


def test_encoding_cost_per_record_stays_bounded(tmp_path):
    """Regression guard for the 26-minute pull. The bounds are ~2x the measured
    cost of this store and ~10x below version 1's, so machine noise passes and a
    return to per-leaf objects (which was ~90 kB and ~15 ms per record) fails."""
    rng = np.random.default_rng(0)
    energies = (20.0, 30.0, 40.0)
    grids = {e: np.linspace(1.0, e, 900) for e in energies}
    store = {}
    for i in range(60):
        for energy in energies:
            store.setdefault(f"cfg_{i:03d}", {})[energy] = {
                # A real case mapping is wide and mostly scalar; that width is
                # what version 1 charged an object apiece for.
                "case": {f"k{j}": (j * 0.5 if j % 2 else f"s{j}") for j in range(32)},
                "E_grid": grids[energy],  # shared across configs: must dedup
                "spec": rng.random(900),
                "eta": i / 60,
            }
    n_records = 60 * len(energies)

    path = tmp_path / "census.pkl"
    started = time.perf_counter()
    ckio.dump(store, str(path))
    write_seconds = time.perf_counter() - started
    started = time.perf_counter()
    loaded = ckio.load(str(path))
    read_seconds = time.perf_counter() - started

    assert loaded["cfg_000"][30.0]["case"]["k1"] == 0.5
    assert path.stat().st_size / n_records < 16_000
    assert write_seconds / n_records < 3e-3
    assert read_seconds / n_records < 3e-3


def test_shared_arrays_are_stored_once_but_never_aliased(tmp_path):
    """Duplicate array content dominates a real store (measured 3.5x), so the
    blob pool holds one copy. Decoding must still hand every reference its own
    array: callers mutate spectra in place."""
    grid = np.linspace(1.0, 30.0, 256)
    store = {f"cfg_{i}": {30.0: {"E_grid": grid, "spec": np.full(256, float(i))}} for i in range(8)}
    path = tmp_path / "shared.pkl"
    ckio.dump(store, str(path))
    with h5py.File(path, "r") as h5:
        assert len(h5["blobs"]) == 9  # eight distinct spectra, one shared grid

    loaded = ckio.load(str(path))
    loaded["cfg_0"][30.0]["E_grid"][0] = -1.0
    assert loaded["cfg_1"][30.0]["E_grid"][0] == grid[0]


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


def test_dump_load_roundtrips_a_long_uniform_list_payload(tmp_path):
    """A cross-material comparison cache is a flat list of thousands of record
    dicts, not a `{config: {E0: record}}` mapping, so it takes the generic
    typed-tree path rather than the record-table encoder. That path used to
    pack each entry's kind tag into one `kinds` *attribute*; past a few
    thousand entries the attribute's encoded size exceeds HDF5's ~64 KiB
    object-header message ceiling and raises ``OSError: Unable to
    synchronously create attribute (object header message is too large)``."""
    path = tmp_path / "comparison.pkl"
    payload = [{"E0_keV": float(i), "line_eV": i * 1.5, "quality": 0.5} for i in range(6000)]
    ckio.dump(payload, str(path))
    loaded = ckio.load(str(path))
    assert loaded == payload


def test_load_reads_legacy_attribute_packed_kinds(tmp_path):
    """``kinds`` moved from an attribute to a dataset (see above); an archive
    written before the change stored it as an attribute and must still load."""
    path = tmp_path / "legacy_kinds.pkl"
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = "pyrite.result"
        h5.attrs["schema_version"] = 2
        h5.attrs["identity_version"] = 1
        node = h5.create_group("value")
        node.attrs["kind"] = "list"
        node.attrs["kinds"] = np.array(["int", "int"], dtype=object)
        node.attrs["v:00000000"] = 1
        node.attrs["v:00000001"] = 2
    loaded = ckio.load(str(path))
    assert loaded == [1, 2]


def test_load_reads_zstd_pickle(tmp_path):
    path = tmp_path / "zstd.pkl"
    payload = _payload()
    with zstd.ZstdFile(path, "wb") as f:
        pickle.dump(payload, f)
    loaded = ckio.load(str(path))
    assert set(loaded) == set(payload)
    assert np.array_equal(loaded["cfg_a"][30.0]["spec"], payload["cfg_a"][30.0]["spec"])
