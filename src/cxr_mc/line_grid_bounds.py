"""Pure helpers for scripts/analyze_line_grid_bounds.py: turn a simulated
coherent-line spectrum into a coverage energy, and a coverage energy into a
catalog-ready line-grid ``stop``/``num`` pair.

See docs/superpowers/specs/2026-07-16-line-grid-max-energy-design.md.
"""

import numpy as np


def coverage_energy(E_grid: np.ndarray, spec: np.ndarray, coverage: float = 0.99) -> float:
    """The smallest energy on ``E_grid`` at or below which the trapezoidal-
    integrated ``spec`` reaches ``coverage`` (0-1) of its total integral over
    ``E_grid``.

    Returns ``E_grid[0]`` when ``spec`` integrates to zero: a geometry with no
    coherent-line intensity places no requirement on the grid width, and must
    not be mistaken for the widest requirement when candidates are ranked by
    this value.
    """
    E_grid = np.asarray(E_grid, dtype=float)
    spec = np.asarray(spec, dtype=float)
    if E_grid.size < 2:
        raise ValueError("E_grid must have at least two points")
    increments = np.diff(E_grid) * (spec[:-1] + spec[1:]) / 2.0
    total = increments.sum()
    if total <= 0.0:
        return float(E_grid[0])
    cumulative = np.cumsum(increments) / total
    index = int(np.searchsorted(cumulative, coverage))
    index = min(index, cumulative.size - 1)
    return float(E_grid[index + 1])


def margined_stop(raw_energy_eV: float, margin: float = 0.15, round_to: float = 100.0) -> float:
    """Add a safety margin above ``raw_energy_eV`` and round up to the
    nearest ``round_to`` eV, so the catalog ``stop`` clears the measured
    coverage energy even for materials/geometries the empirical scan didn't
    sample exactly."""
    margined = raw_energy_eV * (1.0 + margin)
    return float(np.ceil(margined / round_to) * round_to)


def spacing_num(start_eV: float, stop_eV: float, target_spacing_eV: float = 3.0) -> int:
    """The endpoint-inclusive ``num`` for ``linspace(start_eV, stop_eV, num)``
    closest to ``target_spacing_eV`` uniform spacing -- the convention already
    used by every ``E_grid_line_by_energy`` row in materials.toml."""
    return int(round((stop_eV - start_eV) / target_spacing_eV)) + 1
