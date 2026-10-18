"""Bounded saved-capture adapter for the existing trajectory views."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..montecarlo.geometry import tilted_geometry
from ..montecarlo.trajectories import TrajectoryArtifactError
from ..montecarlo.trajectory_selection import TrajectorySelection, read_selection
from ._frames import _trajectory_from_transport


def saved_trajectory_data(
    selection: TrajectorySelection,
    *,
    max_segments: int = 100_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return saved case and display data without transport; reject oversized reads.

    Fractions and survival describe selected material-entering histories only.
    Track-only selections are unsuitable for primary survival and are rejected.
    Source beam coordinates are read for selected histories, never the full bunch.
    """
    import h5py

    if selection.request.get("tracks") is not None:
        raise TrajectoryArtifactError("trajectory views require whole-history selections")
    vacuum_count = sum(b - a for a, b in selection.vacuum_ranges)
    if selection.segments + vacuum_count > max_segments:
        raise TrajectoryArtifactError(
            f"selection exceeds max_segments={max_segments:,}, including vacuum legs"
        )
    segs: dict[str, Any] = dict(read_selection(selection, max_segments=max_segments).transport)
    ids = np.asarray(segs["electron_id"], dtype=np.int64)
    histories, dense = np.unique(ids, return_inverse=True)
    segs["electron_id"] = segs["elec_id"] = dense
    segs["E_keV"] = segs["E_start_keV"]
    segs["t_ang"] = segs["t_start_ang"]
    segs["Ne"] = len(histories)
    from ..montecarlo.trajectory_summary import _history_rows

    fates = [row[1] for row in _history_rows(segs)]
    segs["n_backscattered"] = fates.count("backscattered")
    segs["n_transmitted"] = fates.count("transmitted")
    with h5py.File(selection.path, "r") as handle:
        case = json.loads(handle["case"].attrs["json"])
        group = handle["transport"]
        for key in ("initial_r_ang", "initial_v_hat", "initial_E_keV", "initial_t0_ang"):
            if key not in group:
                raise TrajectoryArtifactError(f"saved view requires {key}")
            segs[key] = group[key][histories]
        for a, b in selection.vacuum_ranges:
            for key in (
                "vacuum_start_ang",
                "vacuum_end_ang",
                "vacuum_E_keV",
                "vacuum_t_ang",
                "vacuum_elec_id",
            ):
                segs.setdefault(key, []).append(group[key][a:b])
        for key in list(segs):
            if key.startswith("vacuum_"):
                segs[key] = np.concatenate(segs[key])
        if "vacuum_elec_id" in segs:
            segs["vacuum_elec_id"] = np.searchsorted(histories, segs["vacuum_elec_id"])
    beam, detector = tilted_geometry(
        case["theta_obs_rad"],
        np.deg2rad(case.get("tilt_deg", 0)),
        np.deg2rad(case.get("tilt_azim_deg", 0)),
    )
    layers = case.get("abs_layers")
    thickness = float(layers[-1][1] if layers else case["thickness_ang"])
    return case, _trajectory_from_transport(case, segs, beam, -detector, thickness)
