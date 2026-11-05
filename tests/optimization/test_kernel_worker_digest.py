"""Transport digest of the kernel tuning worker (#776)."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.optimization import _kernel_worker as worker


def _transported(**segments):
    base = {"L_ang": np.arange(12.0).reshape(3, 4), "empty": np.zeros((0, 2))}
    return SimpleNamespace(_transport={"segs": base | segments, "E_grid": np.array([1.0, 2.0])})


def test_digest_is_deterministic_across_checksum_chunks(monkeypatch):
    first = worker.transport_digest(_transported())
    assert worker.transport_digest(_transported()) == first
    monkeypatch.setattr(worker, "_CHECKSUM_CHUNK_WORDS", 5)
    assert worker.transport_digest(_transported()) == first


@pytest.mark.parametrize(
    "changed",
    [
        {"L_ang": np.arange(12.0).reshape(4, 3)},
        {"L_ang": np.arange(12.0, dtype=np.float32).reshape(3, 4)},
        {"L_ang": np.arange(12.0).reshape(3, 4) + 1e-9},
        {"nested": {"channeling": [np.array([1])]}},
    ],
)
def test_digest_sees_shape_dtype_values_and_nested_arrays(changed):
    assert worker.transport_digest(_transported()) != worker.transport_digest(
        _transported(**changed)
    )


def test_digest_sees_a_change_inside_nested_arrays():
    one = _transported(nested={"channeling": [np.array([1])]})
    two = _transported(nested={"channeling": [np.array([2])]})
    assert worker.transport_digest(one) != worker.transport_digest(two)


@pytest.mark.parametrize("bit", [0, 31, 52, 63])
def test_checksum_sees_a_single_bit_flip(bit):
    values = np.arange(1000.0)
    flipped = values.copy()
    flipped.view(np.uint64)[417] ^= np.uint64(1) << np.uint64(bit)
    first, second = worker._array_checksum(np, values), worker._array_checksum(np, flipped)
    assert first[0] != second[0] and first[1] != second[1]


def test_checksum_sees_swapped_words():
    values = np.arange(1000.0)
    swapped = values.copy()
    swapped[[3, 900]] = swapped[[900, 3]]
    assert worker._array_checksum(np, values) != worker._array_checksum(np, swapped)


@pytest.mark.parametrize("dtype", [bool, np.int8, np.int16, np.float32, np.complex128])
def test_checksum_handles_each_word_size(dtype):
    assert worker._array_checksum(np, np.zeros(4, dtype=dtype)) != worker._array_checksum(
        np, np.ones(4, dtype=dtype)
    )
