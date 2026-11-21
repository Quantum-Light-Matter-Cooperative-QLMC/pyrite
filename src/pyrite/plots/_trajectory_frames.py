"""Renderer-neutral 2D trajectory tables for the penetration cross-section.

Builders take an already-simulated ``_trajectory_data`` dict (or a saved
capture's equivalent), so they are unit-testable without the Monte Carlo and
shared by every renderer. Coordinates are in the trajectory display unit
(``data["u"]`` Angstrom per unit) in the beam-detector plane.
"""

import numpy as np
import pandas as pd

from ._common import _beam_detector_basis, _groove_spec, groove_profile_knots

# Groove-period cap for the 2D overlay; larger than the 3D mesh cap because a
# flat polyline is far cheaper than a surface.
GROOVE_MAX_PERIODS = 400


def tracks_frame(data):
    """Tidy per-vertex track table: the NaN-separated polylines (``px``/``py``/``pE``)
    split into one ``track`` group per electron. Columns ``x, y, E, track``
    (NaN break rows dropped)."""
    px, py, pE = data["px"], data["py"], data["pE"]
    track = np.cumsum(np.isnan(px))  # increments at each NaN separator
    df = pd.DataFrame({"x": px, "y": py, "E": pE, "track": track})
    return df.dropna(subset=["x", "y"]).reset_index(drop=True)


def track_segments_frame(data):
    """One row per consecutive vertex pair within a single electron's polyline.

    A segment runs from ``(x, y)`` to ``(x2, y2)`` and carries its start-vertex
    energy ``E`` (keV). Saved captures with explicit segment endpoints also carry
    ``track_id``, ``parent_id`` and ``generation``. Rows are sorted by ``E``
    ascending so the highest-energy segments draw last (on top), matching the
    datashader ``ds.max("E")`` raster.
    """
    if "track_id" in data:
        e1, e2 = _beam_detector_basis(data["beam"], data["detector"])
        start = np.asarray(data["start_xyz"])
        end = np.asarray(data["end_xyz"])
        return (
            pd.DataFrame(
                {
                    "x": start @ e1,
                    "y": start @ e2,
                    "x2": end @ e1,
                    "y2": end @ e2,
                    "E": data["E"],
                    "track_id": data["track_id"],
                    "parent_id": data["parent_id"],
                    "generation": data["generation"],
                }
            )
            .sort_values("E")
            .reset_index(drop=True)
        )
    df = tracks_frame(data)
    v = df[["x", "y", "E", "track"]].to_numpy()
    if len(v) < 2:
        return pd.DataFrame(columns=["x", "y", "x2", "y2", "E"])
    same = v[:-1, 3] == v[1:, 3]  # consecutive vertices share an electron
    seg = pd.DataFrame(
        {
            "x": v[:-1, 0][same],
            "y": v[:-1, 1][same],
            "x2": v[1:, 0][same],
            "y2": v[1:, 1][same],
            "E": v[:-1, 2][same],
        }
    )
    return seg.sort_values("E").reset_index(drop=True)


def vacuum_segments_frame(data):
    """Non-radiating groove-gap flights as separate 2D segments.

    Vacuum endpoints already use the trajectory display unit; they are projected
    through the same beam/detector basis as material tracks but never added to
    :func:`tracks_frame` or :func:`track_segments_frame`.
    """
    columns = ["x", "y", "x2", "y2", "E", "t_fs", "elec_id"]
    start = np.asarray(data.get("vacuum_start_xyz", np.empty((0, 3))), dtype=float)
    end = np.asarray(data.get("vacuum_end_xyz", np.empty((0, 3))), dtype=float)
    if not len(start):
        return pd.DataFrame(columns=columns)
    e1, e2 = _beam_detector_basis(data["beam"], data["detector"])
    return pd.DataFrame(
        {
            "x": start @ e1,
            "y": start @ e2,
            "x2": end @ e1,
            "y2": end @ e2,
            "E": np.asarray(data["vacuum_E"], dtype=float),
            "t_fs": np.asarray(data["vacuum_t_fs"], dtype=float),
            "elec_id": np.asarray(data["vacuum_elec_id"], dtype=np.int64),
        },
        columns=columns,
    )


def groove_profile_xy(case, data, frame, *, max_periods=GROOVE_MAX_PERIODS):
    """Blazed-groove sawtooth in cross-section coordinates, or ``None``.

    ``None`` when the case is ungrooved or the visible extent needs more than
    ``max_periods`` teeth. The cross-section is the beam-detector plane, so a
    sample-frame surface point ``(x, 0, groove_profile_z(x))`` maps to
    ``(P.e1, P.e2) / u``; the profile therefore rides the entrance face at any
    tilt. Apexes touch the origin face line; valleys bulge one groove depth into
    the slab.
    """
    spec = _groove_spec(case)
    if spec is None:
        return None
    e1, e2 = _beam_detector_basis(data["beam"], data["detector"])
    u = data["u"]
    xlo, xhi, ylo, yhi = frame
    # The sample x-axis projects into the chart with magnitude |(e1[0], e2[0])|
    # display units per Angstrom; invert that to reach the far frame corner.
    r_frame = max(abs(xlo), abs(xhi), abs(ylo), abs(yhi))
    ax_mag = float(np.hypot(e1[0], e2[0]))
    x_max_ang = 1.6 * r_frame * u / max(ax_mag, 1e-6)
    knots = groove_profile_knots(-x_max_ang, x_max_ang, spec, max_periods=max_periods)
    if knots is None:
        return None
    xk, zk = knots
    return (xk * e1[0] + zk * e1[2]) / u, (xk * e2[0] + zk * e2[2]) / u
