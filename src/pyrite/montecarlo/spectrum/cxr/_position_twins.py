"""Float64 twins of segment positions and ages across a float32 upload.

The flight-grouped phase (``_formation.expand_escape_pieces(keep_f64=True)``,
``_setup._grouped_phase64``) needs each row's midpoint and start age in
float64. Device staging (``_kernels._segments_on_device`` and the energy
cutoff's upload) narrows host rows to float32 REAL, which quantizes |r| near
1e7 Angstrom (beam footprints, bunch offsets) to ~1 Angstrom per row: ~0.7 rad
of ``n_hat.r`` phase noise inside a flight. ``r_mid_f64``/``t_ang_f64`` are
registered segment rows (``_kernels._SEG_ARRAYS``), so every row subset keeps
them aligned, and staging uploads them unnarrowed.

Validation: coherent-formation-absorption
"""

import numpy as np

from ...._backend import REAL, _to_cpu, xp

F64_TWINS = {"r_mid_f64": "r_mid", "t_ang_f64": "t_ang"}


def _rows_may_share_flights(segments):
    if segments.get("flight_id") is None:
        return False
    substep = segments.get("substep_id")
    return substep is None or bool((_to_cpu(xp.asarray(substep)) > 0).any())


def with_position_twins(segments):
    """``segments`` plus float64 twins of its host position/age rows, when needed.

    Only float32 REAL builds whose rows can share a flight pay for the copies;
    rows that already carry twins, or are device resident, are left as they are.
    """
    if np.dtype(REAL) == np.float64 or not _rows_may_share_flights(segments):
        return segments
    out = dict(segments)
    for twin, key in F64_TWINS.items():
        row = out.get(key)
        if twin not in out and isinstance(row, np.ndarray) and row.dtype == np.float64:
            out[twin] = row
    return out


def stage_row(key, row, stage):
    """Stage one segment row: twins upload at float64, every other row via ``stage``."""
    return xp.asarray(row) if key in F64_TWINS else stage(row)


def shorten_position_twin(out, old_L, new_L, direction):
    """Move the midpoint twin of rows the cutoff shortened (same start, new length)."""
    if "r_mid_f64" in out:
        half = 0.5 * (new_L - old_L).astype(np.float64)
        out["r_mid_f64"] = out["r_mid_f64"] + half[:, None] * direction.astype(np.float64)
