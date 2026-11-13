"""Float64 twins of segment positions and ages across a float32 upload.

The flight-grouped phase (``_formation.expand_escape_pieces(keep_f64=True)``,
``_setup._grouped_phase64``) needs each row's midpoint and start age in
float64. The energy-cutoff upload (``_kernels._clip_segments_to_cutoff``)
narrows host rows to float32 REAL, which quantizes |r| near 1e7 Angstrom
(beam footprints, bunch offsets) to ~1 Angstrom per row and scrambles the
intra-flight phase. These helpers keep ``<key>_f64`` selections beside the
staged rows; ``expand_escape_pieces`` reads and drops them.

Validation: coherent-formation-absorption
"""

import numpy as np

from ...._backend import REAL, _to_cpu, xp

F64_TWIN_KEYS = ("r_mid", "t_ang")


def keep_position_twins(out, segments, host_idx, keep_idx):
    """Store ``<key>_f64`` float64 selections of the host rows a float32 stage narrows.

    Only flight-grouping candidates (rows carrying ``flight_id``) on a float32
    REAL build pay for the copies.
    """
    if np.dtype(REAL) == np.float64 or segments.get("flight_id") is None:
        return
    for key in F64_TWIN_KEYS:
        row = segments.get(key)
        if isinstance(row, np.ndarray) and row.dtype == np.float64:
            idx = _to_cpu(keep_idx) if host_idx is None else host_idx
            out[f"{key}_f64"] = xp.asarray(row[idx])


def shorten_position_twin(out, old_L, new_L, direction):
    """Move the midpoint twin of rows the cutoff shortened (same start, new length)."""
    if "r_mid_f64" in out:
        half = 0.5 * (new_L - old_L).astype(np.float64)
        out["r_mid_f64"] = out["r_mid_f64"] + half[:, None] * direction.astype(np.float64)
