"""Regression contracts for the shared grid descriptor decoder."""

import numpy as np
import pytest

from pyrite.materials._catalog_decode import _Errors, _grid, resolve_grid_descriptor


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3, [3.0]),
        ({"values": [3, -1, 3]}, [3.0, -1.0, 3.0]),
        ({"arange": {"start": 3, "stop": 0, "step": -1}}, [3.0, 2.0, 1.0]),
        ({"linspace": {"start": 0, "stop": 3, "num": 3, "endpoint": False}}, [0.0, 1.0, 2.0]),
        ({"logspace": {"start": 0, "stop": 3, "num": 4, "base": 2}}, [1.0, 2.0, 4.0, 8.0]),
        (
            {"logspace": {"start": 0, "stop": 3, "num": 3, "endpoint": False}},
            [1.0, 10.0, 100.0],
        ),
    ],
)
def test_grid_coordinates_are_ordered_and_immutable(value, expected) -> None:
    grid = resolve_grid_descriptor(value)

    np.testing.assert_array_equal(grid, expected)
    assert grid.dtype == np.float64
    assert not grid.flags.writeable
    with pytest.raises(ValueError, match="WRITEABLE"):
        grid.setflags(write=True)


@pytest.mark.parametrize(
    ("value", "messages"),
    [
        (
            {"bad": 1, "values": [1]},
            [
                "grid.bad: unknown grid descriptor",
                "grid: must contain exactly one of values/arange/linspace/logspace",
            ],
        ),
        (
            {"arange": {"start": True, "stop": False, "step": 1}},
            [
                "grid.arange.start: must be a finite number (not bool)",
                "grid: invalid arange descriptor (start must be a finite number (not bool))",
            ],
        ),
        (
            {"linspace": {"start": 0, "stop": 2, "num": 3, "endpoint": 1}},
            ["grid: invalid linspace descriptor (endpoint must be a boolean)"],
        ),
        (
            {"logspace": {"start": 0, "stop": 2, "num": 0, "base": False}},
            ["grid: invalid logspace descriptor (num must be a positive integer)"],
        ),
        (
            {"arange": {"start": 0, "stop": 1, "step": 0}},
            ["grid: invalid arange descriptor (step must be nonzero)"],
        ),
        ({"values": []}, ["grid: grid must be nonempty"]),
    ],
)
def test_grid_error_paths_and_order(value, messages) -> None:
    errors = _Errors()

    assert _grid(value, "grid", errors) is None
    assert errors.items == messages


def test_grid_rejects_nonfinite_generated_coordinates() -> None:
    errors = _Errors()
    with np.errstate(over="ignore"):
        grid = _grid({"logspace": {"start": 400, "stop": 401, "num": 2}}, "grid", errors)

    assert grid is None
    assert errors.items == ["grid: grid values must be finite"]


def test_range_descriptor_normalizes_each_coordinate() -> None:
    errors = _Errors()
    grid = _grid(
        {"arange": {"start": "1 keV", "stop": "0.004 MeV", "step": "1 keV"}},
        "grid",
        errors,
        unit="keV",
        expected="energy",
    )

    assert errors.items == []
    assert grid is not None
    np.testing.assert_array_equal(grid, [1.0, 2.0, 3.0])
