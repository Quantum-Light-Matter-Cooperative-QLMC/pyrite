from __future__ import annotations

import numpy as np


def fake_segments(
    count: int = 1,
    *,
    ne: int | None = None,
    elec_id=None,
):
    if elec_id is None:
        elec_id = np.arange(count, dtype=np.int64)

    elec_id = np.asarray(elec_id, dtype=np.int64)

    if elec_id.shape != (count,):
        raise ValueError("elec_id must have one entry per segment")

    if ne is None:
        ne = int(elec_id.max()) + 1 if count else 0

    return {
        "r_mid": np.tile([4.0, 0.0, 5.0], (count, 1)),
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 10.0),
        "E_keV": np.full(count, 30.0),
        "t_ang": np.zeros(count),
        "t0_ang": np.zeros(count),
        "elec_id": elec_id,
        "layer": np.zeros(count, dtype=np.int16),
        "Ne": ne,
        "thickness_ang": 10.0,
        "n_backscattered": 0,
        "n_missed": 0,
    }


def runner_transport_payload(
    segs,
    *,
    E_grid,
    E_brem=None,
    n_hat=None,
    groove=None,
    ne_lines=None,
    ne_brem=None,
):
    ne = int(segs["Ne"])

    return {
        "E_grid": E_grid,
        "E_brem": E_grid if E_brem is None else E_brem,
        "n_hat": n_hat,
        "segs": segs,
        "Ne_lines": ne if ne_lines is None else ne_lines,
        "Ne_brem": ne if ne_brem is None else ne_brem,
        "groove": groove,
    }
