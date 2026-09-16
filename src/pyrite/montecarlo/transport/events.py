"""Canonical row-end event codes for transport trajectory rows.

Every row that ``energy_model="midpoint"`` transport emits carries an
``event_kind`` naming what ended it. The codes are plain ``np.int8`` scalars so
the Numba and ``cupyx.jit`` cores read the same values this module documents;
keep this module free of accelerator imports.

A row's event sits at the row's far end: the row's ``v_hat`` is the pre-event
direction, ``E_end_keV``/``t_end_ang`` the pre-event state, and the next row of
the same electron the post-event state. See
``docs/physics/beam-transport/transport-outputs.md`` for the contract.
"""

from enum import IntEnum

import numpy as np

EVENT_ELASTIC = np.int8(0)
EVENT_SUBSTEP = np.int8(1)
EVENT_LAYER_BOUNDARY = np.int8(2)
EVENT_GROOVE_SURFACE = np.int8(3)
EVENT_EXIT_TOP = np.int8(4)
EVENT_EXIT_BOTTOM = np.int8(5)
EVENT_EXIT_SIDE = np.int8(6)
EVENT_CUTOFF = np.int8(7)
# Reserved: no current core emits these. They fix the codes, and the contract
# below, that hard inelastic (#93), hard radiative (#95), and fictitious
# (delta) interactions must honour when they are added.
EVENT_HARD_INELASTIC = np.int8(8)
EVENT_HARD_RADIATIVE = np.int8(9)
EVENT_DELTA = np.int8(10)


class SegmentEvent(IntEnum):
    """Named view of the ``event_kind`` row codes."""

    ELASTIC = int(EVENT_ELASTIC)
    SUBSTEP = int(EVENT_SUBSTEP)
    LAYER_BOUNDARY = int(EVENT_LAYER_BOUNDARY)
    GROOVE_SURFACE = int(EVENT_GROOVE_SURFACE)
    EXIT_TOP = int(EVENT_EXIT_TOP)
    EXIT_BOTTOM = int(EVENT_EXIT_BOTTOM)
    EXIT_SIDE = int(EVENT_EXIT_SIDE)
    CUTOFF = int(EVENT_CUTOFF)
    HARD_INELASTIC = int(EVENT_HARD_INELASTIC)
    HARD_RADIATIVE = int(EVENT_HARD_RADIATIVE)
    DELTA = int(EVENT_DELTA)


RESERVED_EVENTS = frozenset(
    {SegmentEvent.HARD_INELASTIC, SegmentEvent.HARD_RADIATIVE, SegmentEvent.DELTA}
)
# Physical interactions with the medium. Each one is a physical segment
# boundary: it closes the flight, and default radiation reductions add the
# flights on either side incoherently.
PHYSICAL_INTERACTIONS = frozenset(
    {SegmentEvent.ELASTIC, SegmentEvent.HARD_INELASTIC, SegmentEvent.HARD_RADIATIVE}
)
GEOMETRY_EVENTS = frozenset(
    {
        SegmentEvent.LAYER_BOUNDARY,
        SegmentEvent.GROOVE_SURFACE,
        SegmentEvent.EXIT_TOP,
        SegmentEvent.EXIT_BOTTOM,
        SegmentEvent.EXIT_SIDE,
    }
)
# Integration nodes, not events: the next row resumes the same flight with the
# same direction, energy, clock, position, and collision budget.
FLIGHT_CONTINUING = frozenset({SegmentEvent.SUBSTEP, SegmentEvent.DELTA})
TERMINAL_EVENTS = frozenset(
    {
        SegmentEvent.EXIT_TOP,
        SegmentEvent.EXIT_BOTTOM,
        SegmentEvent.EXIT_SIDE,
        SegmentEvent.CUTOFF,
    }
)
# Events after which the electron keeps its direction. Hard radiative events
# only lower the energy (photon recoil is not modelled).
DIRECTION_PRESERVING = frozenset(
    {
        SegmentEvent.SUBSTEP,
        SegmentEvent.DELTA,
        SegmentEvent.LAYER_BOUNDARY,
        SegmentEvent.GROOVE_SURFACE,
        SegmentEvent.HARD_RADIATIVE,
    }
)
# Events that change kinetic energy discontinuously at the event point.
ENERGY_DISCONTINUOUS = frozenset({SegmentEvent.HARD_INELASTIC, SegmentEvent.HARD_RADIATIVE})

_KNOWN = np.array(sorted(int(k) for k in SegmentEvent), dtype=np.int8)


def closes_flight(event_kind):
    """Boolean mask: which rows end their physical flight."""
    kind = np.asarray(event_kind)
    return ~np.isin(kind, [int(k) for k in FLIGHT_CONTINUING])


_REQUIRED = (
    "r_mid",
    "v_hat",
    "L_ang",
    "E_start_keV",
    "E_end_keV",
    "t_start_ang",
    "t_end_ang",
    "electron_id",
    "flight_id",
    "substep_id",
    "event_kind",
)


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def check_segment_event_contract(segments, *, rtol=1e-12, atol_ang=1e-5):
    """Raise ``ValueError`` if ``segments`` violate the row-event contract.

    Rows may arrive in any order (lockstep output is step-major); they are
    ordered by ``(electron_id, flight_id, substep_id)`` first. Checks:

    - every ``event_kind`` is a known code;
    - each electron's first row opens flight 0 at substep 0;
    - after a flight-continuing row the next row keeps ``flight_id``, advances
      ``substep_id`` by one, and keeps direction, energy, clock, and position;
    - after any other row the next row opens ``flight_id + 1`` at substep 0,
      with energy continuous unless the event is energy-discontinuous (then
      non-increasing), direction unchanged for direction-preserving events,
      and clock and position continuous except across a groove surface, whose
      vacuum leg moves both forward;
    - a terminal event is its electron's last row, and every electron's last
      row is terminal or a groove surface (escape through a facet).

    ``atol_ang`` absorbs the interface nudge transport applies after a layer
    crossing. Frozen rows have no ``event_kind``; the check requires midpoint
    rows.
    """
    missing = [k for k in _REQUIRED if segments.get(k) is None]
    if missing:
        raise ValueError(f"segments lack event-contract fields: {', '.join(missing)}")
    kind = _host(segments["event_kind"]).astype(np.int8, copy=False)
    n = kind.size
    if n == 0:
        return
    unknown = ~np.isin(kind, _KNOWN)
    if unknown.any():
        raise ValueError(f"unknown event_kind codes: {sorted(set(kind[unknown].tolist()))}")

    eid = _host(segments["electron_id"]).astype(np.int64, copy=False)
    fid = _host(segments["flight_id"]).astype(np.int64, copy=False)
    sid = _host(segments["substep_id"]).astype(np.int64, copy=False)
    order = np.lexsort((sid, fid, eid))
    eid, fid, sid, kind = eid[order], fid[order], sid[order], kind[order]
    L = _host(segments["L_ang"]).astype(float, copy=False)[order]
    v = _host(segments["v_hat"]).astype(float, copy=False)[order]
    mid = _host(segments["r_mid"]).astype(float, copy=False)[order]
    E0 = _host(segments["E_start_keV"]).astype(float, copy=False)[order]
    E1 = _host(segments["E_end_keV"]).astype(float, copy=False)[order]
    t0 = _host(segments["t_start_ang"]).astype(float, copy=False)[order]
    t1 = _host(segments["t_end_ang"]).astype(float, copy=False)[order]
    entry = mid - 0.5 * L[:, None] * v
    exit_ = mid + 0.5 * L[:, None] * v

    def fail(mask, what):
        if mask.any():
            j = int(np.flatnonzero(mask)[0])
            raise ValueError(
                f"{what}: electron {int(eid[j])} flight {int(fid[j])} "
                f"substep {int(sid[j])} (event {SegmentEvent(int(kind[j])).name})"
            )

    first = np.empty(n, dtype=bool)
    first[0] = True
    first[1:] = eid[1:] != eid[:-1]
    last = np.empty(n, dtype=bool)
    last[-1] = True
    last[:-1] = first[1:]
    fail(first & ((fid != 0) | (sid != 0)), "electron does not open at flight 0 substep 0")

    terminal = np.isin(kind, [int(k) for k in TERMINAL_EVENTS])
    fail(terminal & ~last, "terminal event is not the electron's last row")
    fail(
        last & ~terminal & (kind != SegmentEvent.GROOVE_SURFACE),
        "electron's last row is not a terminal or groove-surface event",
    )

    # Pairs (a -> b) of consecutive rows of one electron.
    a = np.flatnonzero(~last)
    b = a + 1
    ka = kind[a]

    def close(x, y, scale):
        return np.abs(x - y) <= rtol * np.maximum(scale, 1.0)

    def at(mask):
        out = np.zeros(n, dtype=bool)
        out[a[mask]] = True
        return out

    cont = np.isin(ka, [int(k) for k in FLIGHT_CONTINUING])
    fail(
        at(cont & ((fid[b] != fid[a]) | (sid[b] != sid[a] + 1))),
        "substep does not continue its flight",
    )
    fail(
        at(~cont & ((fid[b] != fid[a] + 1) | (sid[b] != 0))),
        "physical event does not open a new flight",
    )

    keeps_dir = np.isin(ka, [int(k) for k in DIRECTION_PRESERVING])
    same_dir = np.all(np.abs(v[b] - v[a]) <= 1e-12, axis=1)
    fail(at(keeps_dir & ~same_dir), "direction changed across a direction-preserving event")

    jumps = np.isin(ka, [int(k) for k in ENERGY_DISCONTINUOUS])
    E_scale = np.abs(E1[a])
    fail(at(~jumps & ~close(E0[b], E1[a], E_scale)), "energy is discontinuous across the event")
    fail(at(jumps & (E0[b] > E1[a] * (1.0 + rtol))), "energy rises across a hard event")

    surface = ka == SegmentEvent.GROOVE_SURFACE
    t_scale = np.abs(t1[a])
    fail(at(~surface & ~close(t0[b], t1[a], t_scale)), "clock is discontinuous across the event")
    fail(
        at(surface & (t0[b] < t1[a] * (1.0 - rtol) - atol_ang)),
        "clock runs backward across a surface",
    )
    gap = np.linalg.norm(entry[b] - exit_[a], axis=1)
    fail(at(~surface & (gap > atol_ang)), "position is discontinuous across the event")
