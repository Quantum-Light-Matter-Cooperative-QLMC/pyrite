"""Helpers for comparing case dictionaries whose grids are exact arrays (#278)."""

import numpy as np


def assert_same_case(actual, expected):
    """Compare two case dictionaries key by key; array values compare elementwise."""
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
        else:
            assert actual[key] == value, key
