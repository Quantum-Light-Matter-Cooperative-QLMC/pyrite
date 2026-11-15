"""Reflection-batched construction of the line chi/U coupling tables (#1167).

The tables are ``(N_hkl, N_tab)`` and independent of the segment chunk, so
thousands of reflections on an edge-resolved grid reach gigabytes. Building
them in reflection batches straight into the REAL outputs keeps the float64
transient near :data:`TABLE_BATCH_BYTES`; rows are independent and the cast is
elementwise, so the result is bit-identical to one unbatched call.
"""

from ...._backend import REAL, xp

#: Float64 working-set cap of one tabulation batch.
TABLE_BATCH_BYTES = 256 << 20
# Four float64 tables plus emission_coupling_tables' two negated copies.
_BATCH_TABLES = 6


def _batch_rows(n_hkl, n_tab):
    return min(int(n_hkl), max(1, TABLE_BATCH_BYTES // (_BATCH_TABLES * 8 * max(1, n_tab))))


def fill_tables_in_batches(build, hkl_list, n_tab):
    """Fill four ``(N_hkl, n_tab)`` REAL tables from ``build(hkl_batch)``.

    ``build`` returns the four float64 tables of one reflection batch.
    """
    n_hkl = len(hkl_list)
    out = tuple(xp.empty((n_hkl, n_tab), dtype=REAL) for _ in range(4))
    step = max(1, _batch_rows(n_hkl, n_tab))
    for i0 in range(0, n_hkl, step):
        i1 = min(i0 + step, n_hkl)
        for dst, table in zip(out, build(hkl_list[i0:i1]), strict=True):
            dst[i0:i1] = table
    return out


def line_table_bytes(n_hkl, n_tab, itemsize):
    """Peak device bytes of the tables: four outputs plus one float64 batch."""
    n_tab = max(1, int(n_tab))
    return (
        4 * int(n_hkl) * n_tab * int(itemsize)
        + _BATCH_TABLES * 8 * _batch_rows(n_hkl, n_tab) * n_tab
    )


#: Table bytes one measured-selection line call may hold, at the backend's
#: precision. Fixed rather than read from the device budget so that a
#: selection's chunking, hence its float summation order, is the same on every
#: device of a backend (#1200). Measured on the #642 Si 300 keV automatic axis
#: (3.6e5 tabulation points, CUDA): 128 / 256 / 512 reflections per call took
#: 43.6 / 37.5 / 34.8 s, with device peaks near 1.15 x the tables.
SELECTION_CALL_TABLE_BYTES = 2 << 30

#: Bounds on reflections per selection call. Fewer calls amortise the per-call
#: prologue and bin-mean flush; the floor keeps a very wide axis from
#: degenerating into single-reflection calls.
SELECTION_CALL_REFLECTIONS = (64, 8192)


def selection_reflections_per_call(n_tab, itemsize=None):
    """Reflections per measured-selection line call for ``n_tab`` tabulation points.

    The largest count whose four table rows of ``itemsize`` bytes (``REAL``
    by default) fit :data:`SELECTION_CALL_TABLE_BYTES`, clamped to
    :data:`SELECTION_CALL_REFLECTIONS`. It depends on the case, the axis and
    the backend precision, never on the device memory; admission against the
    device budget is separate.
    """
    if itemsize is None:
        import numpy as np

        itemsize = np.dtype(REAL).itemsize
    lo, hi = SELECTION_CALL_REFLECTIONS
    per_reflection = 4 * int(itemsize) * max(1, int(n_tab))
    return int(min(hi, max(lo, SELECTION_CALL_TABLE_BYTES // per_reflection)))
