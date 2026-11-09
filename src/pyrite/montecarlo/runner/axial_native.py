"""Pure axial fibre-fate to native-primary launch adapter (#994, milestone 4).

The fibre instrument owns the *single* terminal draw for each line primary.
This adapter neither samples a new fate nor re-draws the photon history. It
turns its already sampled outgoing records into the same :class:`LaunchState`
used by native secondary passes, preserving primary IDs and RNG stream keys.
Retained electrons cross the rear face without native transport. Unresolved
weight is neither called transmitted nor silently sent into amorphous
transport; its incident IDs remain explicit for bounded provenance.

This is a preparation seam, not production admission. The axial admission gate
remains closed until the runner, scorer and persisted-channel claim are joined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...quantum._admission import AxialAdmissionError
from ..transport.secondaries import LaunchState

if TYPE_CHECKING:
    from ..channeling.axial_outgoing_launch import AxialHandoffRecord, AxialIncident


@dataclass(frozen=True)
class AxialNativePlan:
    """Stable per-primary partition of one admitted fibre population.

    ``launch_ids`` index ``launch`` rows and MUST be used to remap the native
    local ``electron_id`` back to the original primary. ``exit_ids`` crossed
    the rear face while retained (or dechanneled exactly at the rear face).
    ``unresolved_ids`` carry a numerical-fate uncertainty and are never counted
    as physically transmitted or native-transported electrons.
    """

    launch: LaunchState | None
    launch_ids: np.ndarray
    exit_ids: np.ndarray
    unresolved_ids: np.ndarray
    n_primaries: int

    def provenance(self) -> dict[str, int]:
        """Counts to persist with the *calibrated* depth/free-free budgets."""
        return {
            "n_fibre_primaries": self.n_primaries,
            "n_fibre_outgoing_native": int(self.launch_ids.size),
            "n_fibre_retained_exit": int(self.exit_ids.size),
            "n_fibre_unresolved": int(self.unresolved_ids.size),
        }


def _validated_launch(record: AxialHandoffRecord, incident: AxialIncident, depth: float):
    """Check a native handoff without modifying its jointly sampled state."""
    launch = record.launch
    if launch is None or launch.incident_id != incident.incident_id:
        raise AxialAdmissionError("axial outgoing fate has no matching launch")
    if launch.stream_key != incident.stream_key:
        raise AxialAdmissionError("axial launch changed its primary RNG stream key")
    if not np.isclose(launch.depth_ang, record.depth_ang, rtol=0, atol=1e-8):
        raise AxialAdmissionError("axial launch and terminal event disagree on depth")
    if not 0 <= launch.depth_ang <= depth:
        raise AxialAdmissionError("axial launch lies outside the calibrated slab")
    pos = np.asarray(launch.r_ang, dtype=float)
    vec = np.asarray(launch.v_hat, dtype=float)
    quantities = (launch.E_keV, launch.t_ang, launch.t0_ang)
    if (
        pos.shape != (3,)
        or vec.shape != (3,)
        or not np.all(np.isfinite(pos))
        or not np.all(np.isfinite(vec))
        or not np.all(np.isfinite(quantities))
        or launch.E_keV <= 0
        or launch.t_ang < incident.clock_at_surface_ang
        or launch.t0_ang != incident.t0_ang
        or not np.isclose(float(vec @ vec), 1.0, rtol=1e-12, atol=1e-12)
    ):
        raise AxialAdmissionError("axial native launch has invalid phase-space or clock")
    return pos, vec


def axial_native_plan(
    records: list[AxialHandoffRecord],
    incidents: list[AxialIncident],
    *,
    thickness_ang: float,
) -> AxialNativePlan:
    """Partition exactly one jointly drawn fate per native line primary.

    Incident IDs must be the complete zero-based line population, irrespective
    of the ordering of either input. Native launch rows are sorted by original
    ID, so reducing/chunking the caller's records does not permute the native
    RNG streams. A physical-outgoing event at the *rear face* needs no native
    continuation and counts as a rear-face exit. Unknown numerical fates stay
    explicit and must be reported/bounded by the later persistence adapter.

    No native trajectories are executed and no spectra are scored here.
    """
    if not np.isfinite(thickness_ang) or thickness_ang <= 0:
        raise ValueError("axial thickness must be finite and positive")
    n = len(incidents)
    if n == 0 or len(records) != n:
        raise AxialAdmissionError("axial fibre needs one fate per line primary")
    by_id = {i.incident_id: i for i in incidents}
    fates = {r.incident_id: r for r in records}
    expected = set(range(n))
    if len(by_id) != n or len(fates) != n or set(by_id) != expected or set(fates) != expected:
        raise AxialAdmissionError("axial fates must cover each zero-based primary exactly once")
    launches = []
    launch_ids, exit_ids, unresolved_ids = [], [], []
    for identity in range(n):
        incident, record = by_id[identity], fates[identity]
        if not np.isfinite(record.depth_ang) or not 0 <= record.depth_ang <= thickness_ang:
            raise AxialAdmissionError("axial terminal depth lies outside the slab")
        if record.fate == "physical_outgoing":
            pos, vec = _validated_launch(record, incident, thickness_ang)
            if record.depth_ang == thickness_ang:
                exit_ids.append(identity)
                continue
            launch_ids.append(identity)
            launches.append((record.launch, pos, vec))
        elif record.fate == "retained":
            if record.launch is not None or not np.isclose(
                record.depth_ang, thickness_ang, atol=1e-8, rtol=0
            ):
                raise AxialAdmissionError("retained axial primary must exit at the back face")
            exit_ids.append(identity)
        elif record.fate == "unresolved":
            if record.launch is not None:
                raise AxialAdmissionError("unresolved axial weight cannot launch native")
            unresolved_ids.append(identity)
        else:
            raise AxialAdmissionError(f"unknown axial terminal fate {record.fate!r}")
    native = None
    if launches:
        # Use the exact sampled lab/slab position, direction, clock and seed
        # stream: never resample or re-randomize a dechanneling electron.
        native = LaunchState(
            r_ang=np.stack([x[1] for x in launches]),
            v_hat=np.stack([x[2] for x in launches]),
            E_keV=np.asarray([x[0].E_keV for x in launches], dtype=float),
            t_ang=np.asarray([x[0].t_ang for x in launches], dtype=float),
            t0_ang=np.asarray([x[0].t0_ang for x in launches], dtype=float),
            stream_keys=np.asarray([x[0].stream_key for x in launches], dtype=np.uint64),
            # The same explicit-launch convention as planar channeling.
            radiative_keys=np.zeros(len(launches), dtype=np.uint64),
        )
    return AxialNativePlan(
        native,
        np.asarray(launch_ids, dtype=np.int64),
        np.asarray(exit_ids, dtype=np.int64),
        np.asarray(unresolved_ids, dtype=np.int64),
        n,
    )
