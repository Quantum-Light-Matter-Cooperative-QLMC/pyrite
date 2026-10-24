"""Runner glue for the planar channeling zone (#506, ADR-0018 D4-A).

:func:`transport_channeling_case` samples the primaries' entrance states with
the same pure functions of the seed as amorphous transport, solves the case's
ensemble zone, samples each electron's handoff, builds the zone rows, launches
every handed-off (or never-channeled) electron through the
``transport.secondaries`` launch seam with its own primary stream key, and
joins the zone rows in front of the native rows. A never-channeled electron
therefore runs exactly the trajectory amorphous transport gives it.

The joined result is an ordinary transport result plus a ``channeling_zone``
row tag and a ``channeling`` diagnostics record (fate counts, ensemble
probabilities, balance defect, angles).

Validation: cr-zone-handoff-accounting
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..channeling.handoff import (
    CHANNELING_MODEL,
    FATE_BOUND,
    FATE_ENTRY,
    FATE_NAMES,
    beta_from_keV,
    handoff_directions,
    sample_handoffs,
    solve_zone,
    warn_overflow,
)
from ..transport.events import (
    EVENT_CUTOFF,
    EVENT_DECHANNEL,
    EVENT_EXIT_BOTTOM,
    EVENT_EXIT_SIDE,
)

# Row keys every midpoint transport result carries, and their aliases.
_ALIASES = {"E_keV": "E_start_keV", "t_ang": "t_start_ang", "elec_id": "electron_id"}
_COUNTERS = (
    "n_backscattered",
    "n_transmitted",
    "n_side_exited",
    "n_missed",
    "n_cutoff_stopped",
    "n_step_limited",
    "n_stopped",
)
_LAUNCH_ONLY = ("hard_secondary_v_hat",)
# Beam-sampling arguments a launched pass must not see (cf. secondaries._BEAM_ARGS).
_LAUNCH_BEAM_ARGS = {
    "beam_fwhm_mm": None,
    "beam_fwhm_y_mm": None,
    "bunch_length_fs": None,
    "long_offsets_fs": None,
    "longitudinal_distribution": None,
    "transverse_distribution": None,
    "energy_spread_frac": None,
    "bunch_front_tilt": None,
}


def _exit_lengths(r0, v0, thickness_ang, width_ang, height_ang):
    """Path length to the first slab face along ``v0`` and that face's event."""
    length = (thickness_ang - r0[:, 2]) / v0[:, 2]
    event = np.full(r0.shape[0], EVENT_EXIT_BOTTOM, dtype=np.int8)
    if width_ang is not None:
        for axis, half in ((0, 0.5 * width_ang), (1, 0.5 * height_ang)):
            v = v0[:, axis]
            with np.errstate(divide="ignore", invalid="ignore"):
                side = np.where(v > 0, (half - r0[:, axis]) / v, (-half - r0[:, axis]) / v)
            side = np.where(v != 0.0, side, np.inf)
            hit = side < length
            length = np.where(hit, side, length)
            event = np.where(hit, EVENT_EXIT_SIDE, event).astype(np.int8)
    return length, event


def _zone_rows(template, ids, r0, v0, E0, t_start, t0, length, kind):
    """Zone rows shaped like ``template``'s row arrays (missing fields zero)."""
    n = ids.size
    E_end = E0 - template["_S"] * length
    beta = beta_from_keV(0.5 * (E0 + E_end))
    rows: dict[str, Any] = {
        "r_mid": r0 + 0.5 * length[:, None] * v0,
        "v_hat": v0.copy(),
        "L_ang": length.copy(),
        "E_start_keV": E0.copy(),
        "E_end_keV": E_end,
        "E_repr_keV": 0.5 * (E0 + E_end),
        "t_start_ang": t_start.copy(),
        "t_end_ang": t_start + length / beta,
        "t0_ang": t0.copy(),
        "electron_id": ids.astype(np.int64),
        "layer": np.zeros(n, dtype=np.int16),
        "flight_id": np.zeros(n, dtype=np.int64),
        "substep_id": np.zeros(n, dtype=np.int64),
        "event_kind": kind.astype(np.int8),
    }
    if "hard_channel" in template["rows"]:
        rows["hard_channel"] = np.full(n, -1, dtype=template["rows"]["hard_channel"].dtype)
    for key, value in template["rows"].items():
        if key not in rows:
            rows[key] = np.zeros((n, *value.shape[1:]), dtype=value.dtype)
    return rows


def _row_arrays(result) -> dict[str, np.ndarray]:
    n = int(np.asarray(result["L_ang"]).shape[0])
    return {
        key: np.asarray(value)
        for key, value in result.items()
        if key not in _ALIASES
        and isinstance(value, np.ndarray)
        and value.ndim >= 1
        and value.shape[0] == n
        and not key.startswith(("initial_", "vacuum_"))
        and key not in _LAUNCH_ONLY
    }


def transport_channeling_case(
    case,
    keep,
    *,
    simulate,
    E_cut_by_electrons,
    beam_dir,
    beam_kw,
    tilt_polar_rad,
    tilt_azim_rad,
    stopping_tables,
):
    """Transport one channeling case: zone, handoff and native continuation.

    ``simulate(keep, max_steps, start, stop, block_kw, cut)`` is the runner's
    ``simulate_trajectories`` closure; it is called once for the launched
    electrons on the per-electron CPU or CUDA core, with the primaries' table range so
    every table matches the amorphous run.

    Validation: cr-zone-handoff-accounting
    """
    from ..transport.beam_entry import sample_primary_entry, table_energy_range
    from ..transport.kinematics import stream_keys
    from ..transport.secondaries import LaunchState, SecondaryPass
    from ..transport.stopping import prepare_sbethe_stopping_table, sbethe_stopping_keV_per_ang
    from .step_budget import retry_step_budget

    Ne = int(E_cut_by_electrons.size)
    seed = int(case["seed"])
    E0 = float(case["E0_keV"])
    thickness = float(case["thickness_ang"])
    r0, v0, E_keV, clock, t0, record = sample_primary_entry(
        E0,
        Ne,
        seed,
        beam_dir=beam_dir,
        tilt_polar_rad=tilt_polar_rad,
        tilt_azim_rad=tilt_azim_rad,
        **{key: beam_kw.get(key) for key in _LAUNCH_BEAM_ARGS},
        long_shape=beam_kw.get("long_shape", "gaussian"),
    )
    source_bins = None
    if case.get("channeling_source") is not None:
        from ..channeling.planar_injection import source_momentum_bins

        v0, source_bins = source_momentum_bins(case, v0, seed)
        if record is not None:
            record = {**record, "quantum_source": case["channeling_source"]["model"]}
    log_energy, log_stopping = prepare_sbethe_stopping_table(stopping_tables[0])
    S = -float(sbethe_stopping_keV_per_ang(log_energy, log_stopping, E0))
    ensemble = solve_zone(case, np.asarray(beam_dir, dtype=float), v0, S)
    warn_overflow(ensemble, str(case.get("name", "case")))
    width = case.get("crystal_width_mm")
    height = case.get("crystal_height_mm")
    width_ang = None if width is None else float(width) * 1e7
    height_ang = None if height is None else float(height) * 1e7
    ids = np.arange(Ne)
    if source_bins is None:
        sample = sample_handoffs(ensemble, seed, ids)
    else:
        from ..channeling.planar_injection import conditional_handoffs

        sample = conditional_handoffs(case, ensemble, seed, ids, source_bins)
    fate = sample.fate.copy()
    if width_ang is not None and height_ang is not None:  # a missed electron never enters
        missed = (np.abs(r0[:, 0]) > width_ang / 2.0) | (np.abs(r0[:, 1]) > height_ang / 2.0)
        fate[missed] = FATE_ENTRY
    exit_length, exit_kind = _exit_lengths(r0, v0, thickness, width_ang, height_ang)
    zoned = fate != FATE_ENTRY
    length = np.where(zoned, np.minimum(sample.z_ang, exit_length), 0.0)
    kind = np.where(sample.z_ang < exit_length, EVENT_DECHANNEL, exit_kind).astype(np.int8)
    # The continuous zone loss may reach the electron's cutoff first.
    to_cut = np.maximum(E_keV - E_cut_by_electrons, 0.0) / S
    cut = zoned & (to_cut <= length)
    length = np.where(cut, to_cut, length)
    kind = np.where(cut, EVENT_CUTOFF, kind).astype(np.int8)
    exited = zoned & (kind != EVENT_DECHANNEL)
    outcome = fate.copy()
    outcome[exited & (fate != FATE_BOUND)] = FATE_BOUND  # reached a face still channeled
    launched = ~exited
    handed = zoned & launched
    direction = v0.copy()
    direction[handed] = handoff_directions(
        v0[handed], ensemble.plane_normal, sample.psi_rad[handed], sample.sign[handed]
    )
    E_launch = E_keV - S * length
    beta = beta_from_keV(0.5 * (E_keV + E_launch))
    t_launch = clock + length / beta
    r_launch = r0 + length[:, None] * v0
    table_range = table_energy_range(E_cut_by_electrons, E_keV)
    keys = stream_keys(seed, Ne)
    sel = np.flatnonzero(launched)
    run_ids = sel if sel.size else np.array([0])  # schema template when none launch
    launch = LaunchState(
        r_ang=(r_launch if sel.size else r0)[run_ids],
        v_hat=(direction if sel.size else v0)[run_ids],
        E_keV=(E_launch if sel.size else E_keV)[run_ids],
        t_ang=(t_launch if sel.size else clock)[run_ids],
        t0_ang=t0[run_ids],
        stream_keys=keys[run_ids],
        radiative_keys=np.zeros(run_ids.size, dtype=np.uint64),
    )
    block_kw = {
        **_LAUNCH_BEAM_ARGS,
        "_secondary": SecondaryPass(launch=launch, table_range_keV=table_range, photon_seed=seed),
        "_energy_range_keV": table_range,
    }
    native = retry_step_budget(
        simulate,
        keep,
        stop=run_ids.size,
        block_kw=block_kw,
        cut=E_cut_by_electrons[run_ids],
    )
    native_rows = _row_arrays(native)
    if not sel.size:
        native_rows = {key: value[:0] for key, value in native_rows.items()}
    local = native_rows["electron_id"]
    native_rows["electron_id"] = run_ids[local].astype(np.int64)
    native_rows["flight_id"] = native_rows["flight_id"] + zoned[run_ids][local]
    z_ids = np.flatnonzero(zoned)
    template = {"rows": native_rows, "_S": S}
    zone = _zone_rows(
        template, z_ids, r0[z_ids], v0[z_ids], E_keV[z_ids], clock[z_ids], t0[z_ids],
        length[z_ids], kind[z_ids],
    )  # fmt: skip
    joined = {key: np.concatenate([zone[key], native_rows[key]]) for key in native_rows}
    joined["channeling_zone"] = np.concatenate(
        [np.ones(z_ids.size, dtype=bool), np.zeros(native_rows["L_ang"].size, dtype=bool)]
    )
    order = np.lexsort((joined["substep_id"], joined["flight_id"], joined["electron_id"]))
    joined = {key: value[order] for key, value in joined.items()}
    # A launched pass also records shell secondary directions; channeling runs
    # transport no secondaries, so the result keeps the primary-run schema.
    out = {
        key: value
        for key, value in native.items()
        if key not in native_rows and key not in _LAUNCH_ONLY
    }
    out.update(joined)
    for alias, canonical in _ALIASES.items():
        if alias in native:
            out[alias] = out[canonical]
    out.update(initial_r_ang=r0, initial_v_hat=v0, initial_E_keV=E_keV, initial_t0_ang=t0, Ne=Ne)
    if record is not None:
        out["beam_entry"] = record
    for counter in _COUNTERS:
        out[counter] = int(native.get(counter, 0)) if sel.size else 0
    out["n_transmitted"] += int(np.count_nonzero(exited & (kind == EVENT_EXIT_BOTTOM)))
    out["n_side_exited"] += int(np.count_nonzero(exited & (kind == EVENT_EXIT_SIDE)))
    zone_cut = int(np.count_nonzero(exited & (kind == EVENT_CUTOFF)))
    out["n_cutoff_stopped"] += zone_cut
    out["n_stopped"] += zone_cut
    counts = np.bincount(outcome, minlength=len(FATE_NAMES))
    probabilities = ensemble.probabilities
    handed_psi = sample.psi_rad[handed & (fate != FATE_ENTRY)]
    out["channeling"] = {
        "model": CHANNELING_MODEL,
        "fates": dict(zip(FATE_NAMES, (int(c) for c in counts), strict=True)),
        "probabilities": dict(zip(FATE_NAMES, (float(p) for p in probabilities), strict=True)),
        "balance_defect": ensemble.balance_defect,
        "zone_cutoffs": zone_cut,
        "stopping_keV_per_ang": S,
        "path_ang": ensemble.path_ang,
        "incidence_rad": ensemble.incidence_rad,
        "divergence_rad": ensemble.divergence_rad,
        "psi_critical_rad": ensemble.psi_critical_rad,
        "psi_edge_rad": ensemble.psi_edge_rad,
        "mean_handoff_psi_rad": float(handed_psi.mean()) if handed_psi.size else 0.0,
        "margin_eV": float(case["channeling"]["margin_eV"]),
    }
    return out
