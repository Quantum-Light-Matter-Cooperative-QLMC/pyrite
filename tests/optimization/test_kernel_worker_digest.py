"""Transport digest of the kernel tuning worker (#776)."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyrite.optimization import _kernel_worker as worker


def _transported(**segments):
    base = {"L_ang": np.arange(12.0).reshape(3, 4), "empty": np.zeros((0, 2))}
    return SimpleNamespace(_transport={"segs": base | segments, "E_grid": np.array([1.0, 2.0])})


def test_digest_is_deterministic_when_arrays_span_chunks(monkeypatch):
    first = worker.transport_digest(_transported())
    assert worker.transport_digest(_transported()) == first
    monkeypatch.setattr(worker, "_DIGEST_CHUNK_BYTES", 7)  # arrays span many chunks
    small = worker.transport_digest(_transported())
    assert worker.transport_digest(_transported()) == small


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


def test_digest_chunks_do_not_copy():
    array = np.arange(8.0)
    assert all(np.shares_memory(chunk, array) for chunk in worker._byte_chunks(array))
