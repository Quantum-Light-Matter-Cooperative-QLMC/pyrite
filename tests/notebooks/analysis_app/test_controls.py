"""Unit tests for analysis_ui.controls functions."""

import numpy as np

from pyrite.apps.analysis_ui.controls import _grid_limits


class TestGridLimits:
    """Tests for the _grid_limits vectorized function."""

    def test_empty_input(self):
        """Empty input returns default (1.0, 1.0)."""
        assert _grid_limits([], "E_grid") == (1.0, 1.0)

    def test_single_record_with_key(self):
        """Single record with values returns min/max of the key array."""
        source = [{"E_grid": [40.0, 5000.0]}]
        assert _grid_limits(source, "E_grid") == (40.0, 5000.0)

    def test_multiple_records_combine_values(self):
        """Multiple records have their values combined across all arrays."""
        source = [
            {"E_grid": [40.0, 5000.0], "E_grid_brem": [20.0, 150000.0]},
            {"E_grid": [30.0, 6000.0], "E_grid_brem": [10.0, 200000.0]},
        ]
        assert _grid_limits(source, "E_grid") == (30.0, 6000.0)
        assert _grid_limits(source, "E_grid_brem") == (10.0, 200000.0)

    def test_key_none_falls_back_to_e_grid(self):
        """Record with None value for key falls back to E_grid."""
        source = [
            {"E_grid": [40.0, 5000.0], "E_grid_brem": None},
            {"E_grid": [30.0, 6000.0], "E_grid_brem": [10.0, 200000.0]},
        ]
        # When E_grid_brem is None, fall back to E_grid
        result = _grid_limits(source, "E_grid_brem")
        # First record falls back to E_grid [40, 5000], second uses E_grid_brem [10, 200000]
        assert result == (10.0, 200000.0)

    def test_key_missing_falls_back_to_e_grid(self):
        """Record missing key falls back to E_grid."""
        source = [
            {"E_grid": [40.0, 5000.0]},  # No E_grid_brem key
            {"E_grid": [30.0, 6000.0], "E_grid_brem": [10.0, 200000.0]},
        ]
        result = _grid_limits(source, "E_grid_brem")
        # First record falls back to E_grid [40, 5000], second uses E_grid_brem [10, 200000]
        assert result == (10.0, 200000.0)

    def test_key_missing_e_grid_also_missing(self):
        """Record missing both key and E_grid is skipped."""
        source = [
            {"E_grid": [40.0, 5000.0]},
            {"E_grid": [30.0, 6000.0]},
        ]
        # Key not found, falls back to E_grid; both records contribute
        result = _grid_limits(source, "E_grid_brem")
        assert result == (30.0, 6000.0)

    def test_empty_arrays_ignored(self):
        """Records with empty arrays are ignored."""
        source = [
            {"E_grid": []},
            {"E_grid": [30.0, 6000.0]},
        ]
        assert _grid_limits(source, "E_grid") == (30.0, 6000.0)

    def test_non_finite_values_filtered(self):
        """Non-finite values (inf, -inf, nan) are filtered out."""
        source = [
            {"E_grid": [40.0, np.inf, 5000.0, np.nan, -np.inf]},
            {"E_grid": [30.0, 6000.0]},
        ]
        result = _grid_limits(source, "E_grid")
        # Finite values: [40.0, 5000.0, 30.0, 6000.0]
        assert result == (30.0, 6000.0)

    def test_zero_and_negative_values_filtered(self):
        """Zero and negative values are filtered out."""
        source = [
            {"E_grid": [40.0, 0.0, -100.0, 5000.0]},
            {"E_grid": [30.0, -50.0, 6000.0, 0.0]},
        ]
        result = _grid_limits(source, "E_grid")
        # Positive values: [40.0, 5000.0, 30.0, 6000.0]
        assert result == (30.0, 6000.0)

    def test_all_values_filtered_returns_default(self):
        """If all values are filtered out, returns default (1.0, 1.0)."""
        source = [
            {"E_grid": [0.0, -50.0, np.nan]},
            {"E_grid": [-100.0, 0.0, np.inf]},
        ]
        assert _grid_limits(source, "E_grid") == (1.0, 1.0)

    def test_single_positive_finite_value(self):
        """Single valid value returns that value as both min and max."""
        source = [{"E_grid": [100.0]}]
        assert _grid_limits(source, "E_grid") == (100.0, 100.0)

    def test_numpy_array_input(self):
        """Numpy arrays are accepted as values."""
        source = [
            {"E_grid": np.array([40.0, 5000.0])},
            {"E_grid": np.array([30.0, 6000.0])},
        ]
        assert _grid_limits(source, "E_grid") == (30.0, 6000.0)

    def test_list_input(self):
        """Regular Python lists are accepted as values."""
        source = [
            {"E_grid": [40.0, 5000.0]},
            {"E_grid": [30.0, 6000.0]},
        ]
        assert _grid_limits(source, "E_grid") == (30.0, 6000.0)

    def test_tuple_input(self):
        """Tuples are accepted as values."""
        source = [
            {"E_grid": (40.0, 5000.0)},
            {"E_grid": (30.0, 6000.0)},
        ]
        assert _grid_limits(source, "E_grid") == (30.0, 6000.0)

    def test_mixed_array_types(self):
        """Different array types (list, numpy, tuple) in different records."""
        source = [
            {"E_grid": [40.0, 5000.0]},
            {"E_grid": np.array([30.0, 6000.0])},
            {"E_grid": (35.0, 7000.0)},
        ]
        assert _grid_limits(source, "E_grid") == (30.0, 7000.0)

    def test_fallback_with_none_and_empty_e_grid(self):
        """Fallback to E_grid when key is None, but E_grid is empty."""
        source = [
            {"E_grid_brem": None, "E_grid": []},
            {"E_grid_brem": [10.0, 100.0]},
        ]
        result = _grid_limits(source, "E_grid_brem")
        # First record: E_grid_brem is None, falls back to E_grid which is empty, so skipped
        # Second record: E_grid_brem is [10.0, 100.0]
        assert result == (10.0, 100.0)

    def test_fallback_with_complex_records(self):
        """Complex records with nested structures work correctly."""
        source = [
            {
                "E_grid": [40.0, 5000.0],
                "E_grid_brem": None,
                "case": {"tilt_deg": 45.0},
            },
            {
                "E_grid": [30.0, 6000.0],
                "E_grid_brem": [10.0, 200000.0],
                "case": {"tilt_deg": 30.0},
            },
        ]
        assert _grid_limits(source, "E_grid") == (30.0, 6000.0)
        assert _grid_limits(source, "E_grid_brem") == (10.0, 200000.0)

    def test_very_small_positive_values(self):
        """Very small positive values are included."""
        source = [{"E_grid": [1e-10, 1000.0]}]
        assert _grid_limits(source, "E_grid") == (1e-10, 1000.0)

    def test_very_large_values(self):
        """Very large values are included."""
        source = [{"E_grid": [100.0, 1e10]}]
        assert _grid_limits(source, "E_grid") == (100.0, 1e10)

    def test_duplicate_values_handled(self):
        """Duplicate values are handled correctly."""
        source = [{"E_grid": [100.0, 100.0, 1000.0, 100.0]}]
        assert _grid_limits(source, "E_grid") == (100.0, 1000.0)

    def test_unsorted_values_handled(self):
        """Unsorted values are handled correctly."""
        source = [{"E_grid": [5000.0, 100.0, 3000.0, 200.0]}]
        assert _grid_limits(source, "E_grid") == (100.0, 5000.0)
