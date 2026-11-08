"""Unit-weight terminal fates of the Si fibre model for axial depth budgets (#993). Research only.

Samples the #883 physical outgoing instrument
(:func:`pyrite.montecarlo.channeling.axial_outgoing_launch.sample_physical_handoff`)
on the channels that ``checks/axial_fiber_source.py`` builds from the same
options, one record per independent incident. Only ``(incident_id, fate,
depth)`` are kept: the depth-budget fate bounds need nothing else.

Usage::

    python checks/axial_fiber_fates.py chunk '<json kwargs>' START COUNT OUT.json
    python checks/axial_fiber_fates.py merge OUT.json PART.json [PART.json ...]

Kwargs are those of ``checks/axial_fiber_source.py`` plus ``stopping_eV_per_ang``
(CSDA zone stopping; it moves launch energies, never fates) and
``matching_window_eV``: the declared half-width around the target within which
every selected cut must lie. The gap-midpoint cut rule of
``axial_outgoing_maps.gap_cut`` can land farther than the 2 eV default from
the target when the meeting gap is wide; the declaration only states it.

``events ARM.json OUT.json`` converts the primary events kept by a
``checks/axial_fiber_source.py`` checkpoint run into fate records (``tail``
counts as physical outgoing, as in the instrument). Every entry weight must be
one, which holds for ``reference`` (P = R) arms; the instrument itself needs
a finite matching cut and cannot run them.

Validation: cr-axial-zone-handoff-physical
"""

from __future__ import annotations

import dataclasses
import json
import runpy
import sys
import time
from pathlib import Path

import numpy as np

_SOURCE = runpy.run_path(str(Path(__file__).with_name("axial_fiber_source.py")))


def chunk(options: dict, start: int, count: int) -> dict:
    """Fates of incidents ``start .. start + count - 1``."""
    from pyrite.montecarlo.channeling.axial_fiber_unraveling import _Engine
    from pyrite.montecarlo.channeling.axial_outgoing_launch import (
        AxialIncident,
        AxialMatchingDeclaration,
        AxialZoneKinematics,
        sample_physical_handoff,
    )

    extra = ("stopping_eV_per_ang", "matching_window_eV")
    stopping = float(options.get("stopping_eV_per_ang", 0.037))
    window_eV = float(options.get("matching_window_eV", 2.0))
    channels, entry, hbar_v, info, o = _SOURCE["build"](
        {k: v for k, v in options.items() if k not in extra}
    )
    u_min, u_max = info["potential_extrema_eV"]
    seen = [0.0]
    rule = channels.cut

    def cut(energies):
        value = float(rule(energies))
        seen[0] = max(seen[0], abs(value - float(o["target_eV"])))
        return value

    channels = dataclasses.replace(channels, cut=cut)
    window = np.zeros(_Engine(channels, hbar_v).box_shape, dtype=complex)
    window[0, 0] = 1.0
    clock = time.perf_counter()
    records = sample_physical_handoff(
        channels,
        entry,
        carrier_inv_ang=tuple(o["carrier_inv_ang"]),
        hbar_v_eV_ang=hbar_v,
        window=window,
        max_depth_ang=float(o["thickness_ang"]),
        incidents=[AxialIncident(i, (0.0, 0.0), 0.0, 0.0, i) for i in range(start, start + count)],
        zone=AxialZoneKinematics((0.0, 0.0, 0.0), np.eye(3), float(o["energy_eV"]), stopping),
        matching=AxialMatchingDeclaration(
            float(o["target_eV"]), float(u_max), float(u_min), window_eV=window_eV
        ),
        seed=int(o["seed"]),
        search_step_ang=200.0,
    )
    return {
        "status": "research-only; #883 instrument on the fibred absorbing Si model",
        "settings": {**o, "stopping_eV_per_ang": stopping, "matching_window_eV": window_eV},
        "start": start,
        "count": len(records),
        "seconds": time.perf_counter() - clock,
        "max_cut_offset_eV": seen[0],
        "records": [[int(r.incident_id), r.fate, float(r.depth_ang)] for r in records],
    }


def merge(parts: list[dict]) -> dict:
    """Concatenate chunks after checking settings and contiguous incident IDs."""
    first = parts[0]
    if any(p["settings"] != first["settings"] for p in parts[1:]):
        raise ValueError("chunks were run with different settings")
    records = sorted((r for p in parts for r in p["records"]), key=lambda r: r[0])
    ids = [r[0] for r in records]
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError("chunks overlap or leave a gap")
    fates: dict[str, int] = {}
    for _, fate, _ in records:
        fates[fate] = fates.get(fate, 0) + 1
    return {
        "status": first["status"],
        "settings": first["settings"],
        "incidents": len(records),
        "cpu_seconds": float(sum(p["seconds"] for p in parts)),
        "max_cut_offset_eV": max(p["max_cut_offset_eV"] for p in parts),
        "fate_counts": fates,
        "records": records,
    }


def from_events(arm: dict) -> dict:
    """Unit-weight fate records from a merged checkpoint arm's primary events."""
    records = []
    for c in arm["chunks"]:
        for i, kind, depth, weight in c["events"]:
            if abs(weight - 1.0) > 1e-9:
                raise ValueError("entry weight is not one: events are not unit-weight fates")
            fate = {"outgoing": "physical_outgoing", "tail": "physical_outgoing"}.get(kind, kind)
            records.append([int(i), fate, float(depth)])
    records.sort(key=lambda r: r[0])
    fates: dict[str, int] = {}
    for _, fate, _ in records:
        fates[fate] = fates.get(fate, 0) + 1
    return {
        "status": "research-only; primary events of a P = R fibre-source arm",
        "settings": arm["settings"],
        "incidents": len(records),
        "fate_counts": fates,
        "records": records,
    }


def main(argv: list[str]) -> None:
    if argv[0] == "chunk":
        options, start, count, out = json.loads(argv[1]), int(argv[2]), int(argv[3]), argv[4]
        Path(out).write_text(json.dumps(chunk(options, start, count)))
    elif argv[0] == "merge":
        merged = merge([json.loads(Path(p).read_text()) for p in argv[2:]])
        Path(argv[1]).write_text(json.dumps(merged))
    elif argv[0] == "events":
        arm = json.loads(Path(argv[1]).read_text())
        Path(argv[2]).write_text(json.dumps(from_events(arm)))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
