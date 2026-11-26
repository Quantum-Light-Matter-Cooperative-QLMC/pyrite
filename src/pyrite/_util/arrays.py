"""Backend-neutral NumPy array conversions."""

import numpy as np


def host_array(array: object) -> np.ndarray:
    """NumPy view or copy of ``array``; objects with ``.get()`` (CuPy) are copied back."""
    get = getattr(array, "get", None)
    return np.asarray(get() if get is not None else array)


def readonly_array(value: object, *, dtype=None) -> np.ndarray:
    """``np.asarray(value, dtype=dtype)`` with writes disabled.

    When no conversion is needed the returned array is ``value`` itself, so the
    caller's array becomes read-only too.
    """
    array = np.asarray(value, dtype=dtype)
    array.setflags(write=False)
    return array
