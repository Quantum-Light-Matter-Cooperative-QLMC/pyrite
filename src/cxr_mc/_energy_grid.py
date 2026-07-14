"""Case-dictionary encoding helpers for photon-energy grids."""

import numpy as np

type EnergyGridEncoding = tuple[float, float, float] | np.ndarray


def encode_energy_grid(grid: object) -> EnergyGridEncoding:
    """Keep legacy triples for uniform grids and exact arrays otherwise."""
    values = np.asarray(grid, dtype=float)
    if values.size >= 2:
        differences = np.diff(values)
        step = float(differences[0])
        tolerance = 1e-12 * max(1.0, abs(step))
        if np.allclose(differences, step, rtol=1e-12, atol=tolerance):
            start = float(values[0])
            stop = float(values[-1]) + step
            reconstructed = np.arange(start, stop, step, dtype=float)
            if reconstructed.shape != values.shape or not np.allclose(
                reconstructed, values, rtol=1e-12, atol=tolerance
            ):
                # ``last + step`` can admit one extra point for endpoint-inclusive
                # linspace grids. A midpoint between the last value and its
                # successor remains an exclusive arange stop without changing
                # existing arange-derived legacy triples.
                stop = float(values[-1]) + 0.5 * step
            return (start, stop, step)
    return values.copy()


def decode_energy_grid(encoded: object) -> np.ndarray:
    """Decode a legacy ``(start, stop, step)`` tuple or an exact array."""
    if isinstance(encoded, tuple):
        return np.arange(*encoded, dtype=float)
    return np.asarray(encoded, dtype=float)


__all__ = ["EnergyGridEncoding", "decode_energy_grid", "encode_energy_grid"]
