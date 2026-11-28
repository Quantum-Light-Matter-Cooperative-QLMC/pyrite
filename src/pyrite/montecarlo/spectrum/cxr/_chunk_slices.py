"""Memory-bounded flight, energy and spectral-window slicing."""

from ...._backend import _to_cpu, xp


def _flight_blocks(bounds, chunk):
    """Split flight groups into row blocks of at most ``chunk`` rows.

    ``bounds`` holds the group start offsets plus the total row count. Yields
    ``(first_group, last_group_exclusive)`` pairs whose row spans never split a
    group, which is what keeps a flight's substeps inside one coherent sum. A
    single group larger than ``chunk`` is emitted whole rather than split.
    """
    n_groups = bounds.size - 1
    ka = 0
    for kb in range(1, n_groups + 1):
        if bounds[kb] - bounds[ka] >= chunk:
            yield ka, kb
            ka = kb
    if ka < n_groups:
        yield ka, n_groups


def _energy_slices(n_rows, chunk, n_energy):
    """Energy slices keeping a ``(n_rows, slice)`` block within ``chunk`` full rows.

    :func:`_flight_blocks` never splits a group, so one long group can exceed
    ``chunk`` rows and its dense ``(rows, E)`` block the memory ``chunk`` was
    sized for. Group reductions run along rows only, so slicing the energy axis
    is exact: each slice's per-energy result is the full block's.
    """
    width = n_energy if n_rows <= chunk else max(1, (chunk * n_energy) // n_rows)
    for e0 in range(0, n_energy, width):
        yield slice(e0, min(e0 + width, n_energy))


def _sinc_window_bounds(E_grid, lo, hi):
    """Return the padded node slice intersecting ``[lo, hi]``.

    Include the node immediately below ``lo`` and the node immediately above
    ``hi``, matching the historical uniform-grid arithmetic while deriving the
    bounds from the coordinates themselves.  ``E_grid`` must be ascending.
    """
    i0 = max(int(_to_cpu(xp.searchsorted(E_grid, lo, side="right"))) - 1, 0)
    i1 = min(int(_to_cpu(xp.searchsorted(E_grid, hi, side="right"))) + 1, E_grid.size)
    return i0, i1
