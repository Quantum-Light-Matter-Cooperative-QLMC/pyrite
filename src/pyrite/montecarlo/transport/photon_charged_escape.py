"""Charged boundary phase space for explicit photon-feedback continuation.

This adapter retains every escaping cascade track, including shell daughters.
It does not change a flight, sample a draw, or debit escaped energy.
"""

import numpy as np

from .events import EVENT_EXIT_BOTTOM, EVENT_EXIT_SIDE, EVENT_EXIT_TOP
from .photon_histories import ELECTRON, daughter_stream_key

_ESCAPE_SALT = np.uint64(0xCBBB9D5DC1059ED8)


def escaped_charged_rows(result, rows):
    """Terminal exit rows in Å/eV, owned by the original charged launch.

    Exit events have no collision: the recorded segment direction is the
    outgoing direction and its endpoint is the boundary position. Last rows
    are selected by (track, flight, substep), independently of storage order.
    Generation-zero tracks retain their launch species; shell daughters are
    electrons (primary pair conversion is owned by the photon queue). Keys
    derive from the input lineage and track ID under a separate salt.

    Source: the shell-secondary-transport terminal-row energy partition and
    PENELOPE-2024 §1.4.4 interface continuation. An empty cascade has no exits;
    summed kinetic energy equals the cascade's escaped kinetic term.
    Validation: photon-charged-reentry
    """
    from .secondaries import _host

    track = _host(result["track_id"]).astype(np.int64)
    order = np.lexsort((_host(result["substep_id"]), _host(result["flight_id"]), track))
    last = np.r_[track[order][1:] != track[order][:-1], True] if order.size else np.empty(0, bool)
    end = order[last]
    end = end[
        np.isin(
            _host(result["event_kind"])[end], [EVENT_EXIT_TOP, EVENT_EXIT_BOTTOM, EVENT_EXIT_SIDE]
        )
    ]
    owner = _host(result["electron_id"])[end].astype(np.int64)
    tracks = result["secondary_tracks"]
    primary = np.asarray(tracks["generation"])[track[end]] == 0
    species = np.where(primary, rows["species"][owner], ELECTRON)
    direction = _host(result["v_hat"])[end]
    position = _host(result["r_mid"])[end] + 0.5 * _host(result["L_ang"])[end, None] * direction
    keys = np.array(
        [
            daughter_stream_key(rows["key"][p] ^ _ESCAPE_SALT, int(t))
            for p, t in zip(owner, track[end], strict=True)
        ],
        dtype=np.uint64,
    )
    return dict(
        position_ang=position,
        direction=direction,
        kinetic_eV=_host(result["E_end_keV"])[end] * 1e3,
        species=species,
        key=keys,
        parent=owner,
    )


def join_charged_exits(parts):
    """Concatenate phase-space parts with an explicit empty-table schema."""
    empty = dict(
        position_ang=np.empty((0, 3)),
        direction=np.empty((0, 3)),
        kinetic_eV=np.empty(0),
        species=np.empty(0, dtype=np.int8),
        key=np.empty(0, dtype=np.uint64),
        parent=np.empty(0, dtype=np.int64),
    )
    return {
        name: np.concatenate([p[name] for p in parts]) if parts else value
        for name, value in empty.items()
    }
