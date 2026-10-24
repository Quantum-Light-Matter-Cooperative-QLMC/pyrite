"""Selective reads must not decode spectra belonging to unrelated cases."""

import numpy as np
import pytest

from pyrite.checkpoints import _checkpoint_io as ckio
from pyrite.checkpoints import _checkpoint_store as store
from pyrite.checkpoints import _result_v2


def _records(count):
    return {
        f"cfg_{i}": {
            energy: {
                "case": {"crystal": "hopg", "E0_keV": energy, "sequence": [i, i + 1]},
                "E_grid": np.arange(8.0),
                "spec": np.full(8, i + energy),
                "brem": np.full(8, i),
                "spec_characteristic": np.full(8, 2 * i),
            }
            for energy in (20.0, 30.0)
        }
        for i in range(count)
    }


@pytest.mark.parametrize("count", [1, 20])
def test_selective_read_only_decodes_requested_spectra(tmp_path, monkeypatch, count):
    path = tmp_path / "records.h5"
    records = _records(count)
    ckio.dump(records, str(path))
    decoded = []
    original = _result_v2._Reader.blob

    def track(self, index):
        decoded.append(index)
        return original(self, index)

    monkeypatch.setattr(_result_v2._Reader, "blob", track)
    selected = ckio.load_selected(str(path), {(f"cfg_{count - 1}", 30.0)})
    assert list(selected) == [f"cfg_{count - 1}"]
    assert list(selected[f"cfg_{count - 1}"]) == [30.0]
    np.testing.assert_array_equal(selected[f"cfg_{count - 1}"][30.0]["spec"], count + 29.0)
    assert len(decoded) <= 4
    assert ckio.load_selected(str(path), {("absent", 1.0)}) == {}


def test_selective_components_and_shards(tmp_path):
    records = _records(20)
    store.save("hopg", tmp_path, records)
    replacement = _records(1)["cfg_0"]
    replacement[30.0]["spec"][:] = 999
    store.save_part("hopg", tmp_path, "cfg_0", {30.0: replacement[30.0]})
    selected = store.load_selected("hopg", tmp_path, {("cfg_0", 30.0)}, components=("line",))
    record = selected["cfg_0"][30.0]
    np.testing.assert_array_equal(record["spec"], 999)
    assert "brem" not in record
    assert "spec_characteristic" not in record
    with pytest.raises(ValueError, match="unknown checkpoint component"):
        store.load_selected("hopg", tmp_path, set(), components=("unknown",))


def test_selective_read_empty_request(tmp_path):
    assert ckio.load_selected(str(tmp_path / "missing.h5"), set()) == {}


def test_selected_column_encodings_and_independent_arrays(tmp_path):
    records = _records(20)
    for i, by_energy in enumerate(records.values()):
        for record in by_energy.values():
            record.update(
                null=None,
                optional=None if i % 2 else i,
                ragged=list(range(i % 4)),
                fixed=(i, i + 1),
                tree={"mixed": [i, "value"]},
            )
            if i % 3:
                record["sparse"] = np.full(3, i)
    path = tmp_path / "records.h5"
    ckio.dump(records, str(path))
    requested = {("cfg_2", 20.0), ("cfg_7", 30.0), ("cfg_9", 20.0)}
    selected = ckio.load_selected(str(path), requested)
    for name, energy in requested:
        record = selected[name][energy]
        expected = records[name][energy]
        assert record.keys() == expected.keys()
        for key, value in expected.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(record[key], value)
            else:
                assert record[key] == value
    selected["cfg_2"][20.0]["E_grid"][:] = -1
    np.testing.assert_array_equal(selected["cfg_7"][30.0]["E_grid"], np.arange(8.0))


@pytest.mark.parametrize("framed", [False, True])
def test_selected_legacy_and_transfer_frames(tmp_path, framed):
    import io
    import pickle

    records = _records(2)
    path = tmp_path / "legacy.pkl"
    if framed:
        stream = io.BytesIO()
        ckio.dump_stream(records, stream)
        path.write_bytes(stream.getvalue())
    else:
        path.write_bytes(pickle.dumps(records))
    selected = ckio.load_selected(str(path), {("cfg_1", 30.0)})
    assert list(selected) == ["cfg_1"]
    assert list(selected["cfg_1"]) == [30.0]


def test_missing_optional_component_and_absent_cases(tmp_path):
    store.save("hopg", tmp_path, _records(20))
    selected = store.load_selected("hopg", tmp_path, {("cfg_1", 20.0)})
    assert "spec_channeling" not in selected["cfg_1"][20.0]
    assert store.load_selected("hopg", tmp_path, {("missing", 1.0)}) == {}


def test_table_nested_under_single_config(tmp_path):
    records = {"cfg": {float(i): _records(1)["cfg_0"][20.0] for i in range(20)}}
    path = tmp_path / "nested.h5"
    ckio.dump(records, str(path))
    selected = ckio.load_selected(str(path), {("cfg", 17.0)})
    assert list(selected["cfg"]) == [17.0]
    np.testing.assert_array_equal(selected["cfg"][17.0]["spec"], 20.0)


def test_selective_store_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        store.load_selected("absent", tmp_path, {("cfg", 20.0)})


def test_selected_companion_grid_mismatch_is_rejected(tmp_path):
    records = _records(1)
    store.save("hopg", tmp_path, records)
    records["cfg_0"][30.0]["E_grid"] += 1
    store.save("hopg", tmp_path, records, components=("characteristic",))
    with pytest.raises(ValueError, match="grid does not match"):
        store.load_selected("hopg", tmp_path, {("cfg_0", 30.0)})


@pytest.mark.parametrize("legacy", [False, True])
def test_selected_store_matches_full_loader_with_legacy_shard(tmp_path, legacy):
    import pickle

    records = _records(2)
    if legacy:
        store.legacy_path("hopg", tmp_path).write_bytes(pickle.dumps(records))
    else:
        store.save("hopg", tmp_path, records)
    store.save_part("hopg", tmp_path, "cfg_1", {30.0: records["cfg_1"][30.0]})
    shard = next(store.parts_dir("hopg", tmp_path).glob("*.h5"))
    shard.rename(shard.with_suffix(".pkl"))
    selected = store.load_selected("hopg", tmp_path, {("cfg_0", 20.0), ("cfg_1", 30.0)})
    full = store.load("hopg", tmp_path)
    for name, by_energy in selected.items():
        for energy, record in by_energy.items():
            expected = full[name][energy]
            assert record.keys() == expected.keys()
            for key, value in record.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, expected[key])
                else:
                    assert value == expected[key]


@pytest.mark.parametrize("count", [1, 20])
def test_metadata_reads_skip_spectral_columns(tmp_path, monkeypatch, count):
    path = tmp_path / "records.h5"
    ckio.dump(_records(count), str(path))

    def forbidden(*args, **kwargs):
        pytest.fail("metadata reader decoded spectral arrays")

    monkeypatch.setattr(_result_v2._Reader, "blob", forbidden)
    monkeypatch.setattr(ckio, "load", forbidden)
    metadata = ckio.load_metadata(str(path))
    assert sum(map(len, metadata.values())) == count * 2
    assert all(
        set(record) == {"case"} for values in metadata.values() for record in values.values()
    )
