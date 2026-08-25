"""Case-dictionary encoding helpers for photon-energy grids.

This package-root leaf is shared by campaign, checkpoint, and Monte Carlo
drivers without making any of them depend on the ``energy_grid`` driver
package.
"""

from typing import cast

import numpy as np

type EnergyGridEncoding = tuple[float, float, float] | np.ndarray


def encode_energy_grid(grid: object) -> EnergyGridEncoding:
    """Keep legacy triples for uniform grids and exact arrays otherwise."""
    values = np.atleast_1d(np.asarray(grid, dtype=float))
    if values.size >= 2:
        step = float(values[1] - values[0])
        if step != 0.0:
            start = float(values[0])
            # Prefer the historical stop for arange-derived grids. A midpoint
            # stop avoids admitting an extra sample for some endpoint-inclusive
            # linspace grids. Only encode either candidate when np.arange
            # reproduces every declared float exactly; near-uniform grids must
            # remain explicit rather than being silently rounded.
            for stop in (float(values[-1]) + step, float(values[-1]) + 0.5 * step):
                reconstructed = np.arange(start, stop, step, dtype=float)
                if np.array_equal(reconstructed, values):
                    return (start, stop, step)
    exact = values.copy()
    exact.setflags(write=False)
    return exact


def decode_energy_grid(encoded: object) -> np.ndarray:
    """Decode a legacy ``(start, stop, step)`` tuple or an exact array."""
    if isinstance(encoded, tuple):
        start, stop, step = cast(tuple[float, float, float], encoded)
        return np.arange(start, stop, step, dtype=float)
    return np.asarray(encoded, dtype=float)


__all__ = ["EnergyGridEncoding", "decode_energy_grid", "encode_energy_grid"]
