"""Runner glue for an admitted axial fibre case (#994, milestone 4).

One fibre history per line electron (``fiber_stream(seed, id)``) gives both
its on-axis photon estimate and its terminal fate; nothing is drawn twice.

* Photon source: :func:`.fiber_on_axis_source` on the case's line grid with
  the admitted entry's numerics (``boundary="max"``: bound→bound and capture,
  no free→free), in photons/(eV sr incident electron). It is the mean over
  the line electrons of signed per-primary estimates and is not clipped.
* Fates: :func:`.sample_physical_handoff` on the same events with the delta
  Weyl window of the #993 calibration (exact in-cell position, momentum
  uniform over the outgoing box). Outgoing electrons launch native at the
  drawn depth, position and direction with their own primary stream key;
  retained electrons leave through the back face along the beam; unresolved
  electrons (numerical out-of-box weight) end at their terminal depth, are
  transported no further and are counted in the diagnostics.
* Zone rows: each line electron's straight segment along the beam from the
  entrance face to its terminal depth, with the constant-stopping energy and
  clock of :func:`.zone_energy_and_clock`. They score no native PXR/CBS lines
  (:func:`native_line_rows`): the fibre source owns the line-band emission of
  the zone, and the in-cut free→free block it omits is carried as the entry's
  declared one-sided bound. Characteristic and incoherent bremsstrahlung
  scoring on zone rows is unchanged.
* Bremsstrahlung-only electrons (IDs from ``Ne`` on) run native amorphous
  transport from the entrance face, as in planar channeling.

The beam must be one plane wave: any spread in entrance direction or energy
refuses. The observation direction ``theta_obs = 0`` is the beam direction,
which for a tilted case differs from the axis by the tilt (1.5 mrad for
si-fibre/A, against 1/gamma = 128 mrad); the source is evaluated on axis.

Validation: cr-axial-persisted-channel
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ...quantum._admission import AxialAdmissionError
from ..transport.events import EVENT_DECHANNEL, EVENT_EXIT_BOTTOM
from .axial_native import axial_native_plan
from .channeling import _COUNTERS, _LAUNCH_BEAM_ARGS, _row_arrays, _zone_rows, join_zone_rows

_FATES = ("physical_outgoing", "retained", "unresolved")


def _plane_wave(v0, E_keV, beam_dir, E0) -> None:
    beam = np.asarray(beam_dir, dtype=float)
    if not np.allclose(v0, beam, rtol=0, atol=1e-12) or not np.all(E_keV == E0):
        raise AxialAdmissionError(
            "axial fibre requires a beam without angular or energy spread (#994)"
        )


def fibre_source_and_fates(case, admitted, r0, clock, t0, keys, stopping_keV_per_ang):
    """Fibre photon source and one handoff record per line electron.

    ``r0``, ``clock``, ``t0`` and ``keys`` are the entrance states and native
    stream keys of line electrons ``0..n-1``. Returns
    ``(FiberOnAxisSource, records, incidents, zone)``.

    Validation: cr-axial-persisted-channel
    """
    from ..._energy_grid_encoding import decode_energy_grid
    from ...materials.attenuation import linear_attenuation_inv_mm
    from ...quantum._axial_envelope import check_axial_line_band
    from ..channeling.axial_case import axial_zone_frame
    from ..channeling.axial_fiber_case import MATCHING_WINDOW_EV, axial_fiber_inputs
    from ..channeling.axial_fiber_regression import fiber_on_axis_source
    from ..channeling.axial_fiber_unraveling import _Engine
    from ..channeling.axial_outgoing_launch import (
        AxialIncident,
        AxialMatchingDeclaration,
        AxialZoneKinematics,
        matching_checked_channels,
        sample_physical_handoff,
    )

    energies = np.asarray(decode_energy_grid(case["E_grid_line"]), dtype=float)
    check_axial_line_band(admitted, energies)
    frame = axial_zone_frame(case, admitted.axis_uvw)
    inputs = axial_fiber_inputs(admitted)
    matching = AxialMatchingDeclaration(
        inputs.target_eV, inputs.boundary_eV, inputs.potential_min_eV, window_eV=MATCHING_WINDOW_EV
    )
    channels = matching_checked_channels(inputs.channels, matching)
    window = np.zeros(_Engine(channels, inputs.hbar_v_eV_ang).box_shape, dtype=complex)
    window[0, 0] = 1.0
    depths = inputs.depths_ang
    seed = int(case["seed"])
    count = int(r0.shape[0])
    attenuation = {}
    if admitted.numerics.absorption:  # exit along the axis: path T - z
        attenuation = dict(
            absorption_inv_ang=linear_attenuation_inv_mm(admitted.crystal, energies) * 1e-7,
            exit_path_ang=depths[-1] - depths,
        )
    shared: dict[str, Any] = dict(
        hbar_v_eV_ang=inputs.hbar_v_eV_ang,
        carrier_inv_ang=inputs.carrier_inv_ang,
        window=window,
        seed=seed,
        search_step_ang=inputs.search_step_ang,
    )
    source = fiber_on_axis_source(
        channels,
        inputs.entry_orders,
        kinetic_energy_eV=admitted.kinetic_energy_eV,
        depths_ang=depths,
        photon_energy_eV=energies,
        incident_ids=range(count),
        boundary_eV=inputs.boundary_eV,
        **attenuation,
        **shared,
    )
    impact = np.asarray(r0, dtype=float) @ frame[:, :2]
    incidents = [
        AxialIncident(i, (float(impact[i, 0]), float(impact[i, 1])), float(clock[i]),
                      float(t0[i]), int(keys[i]))
        for i in range(count)
    ]  # fmt: skip
    zone = AxialZoneKinematics(
        entrance_origin_lab_ang=(0.0, 0.0, 0.0),
        frame_lab=frame,
        kinetic_energy_eV=admitted.kinetic_energy_eV,
        stopping_eV_per_ang=1e3 * float(stopping_keV_per_ang),
    )
    records = sample_physical_handoff(
        inputs.channels,
        inputs.entry_orders,
        max_depth_ang=float(depths[-1]),
        incidents=incidents,
        zone=zone,
        matching=matching,
        events=source.events,
        **shared,
    )
    return source, records, incidents, zone


def _axial_zone_rows(template, records, zone, r0, v0, E_keV, clock, t0):
    """Straight beam-direction rows from the entrance face to each terminal depth."""
    from ..channeling.axial_outgoing_launch import zone_energy_and_clock

    kept = [r for r in records if r.depth_ang > 0.0]
    ids = np.asarray([r.incident_id for r in kept], dtype=np.int64)
    depth = np.asarray([r.depth_ang for r in kept], dtype=float)
    kind = np.asarray(
        [EVENT_EXIT_BOTTOM if r.fate == "retained" else EVENT_DECHANNEL for r in kept],
        dtype=np.int8,
    )
    length = depth / v0[ids, 2]
    rows = _zone_rows(
        template, ids, r0[ids], v0[ids], E_keV[ids], clock[ids], t0[ids], length, kind
    )
    state = [zone_energy_and_clock(zone, z) for z in depth]
    rows["E_end_keV"] = np.asarray([e for e, _ in state], dtype=float).reshape(-1) / 1e3
    rows["E_repr_keV"] = 0.5 * (rows["E_start_keV"] + rows["E_end_keV"])
    rows["t_end_ang"] = clock[ids] + np.asarray([t for _, t in state], dtype=float).reshape(-1)
    return rows, ids


def native_line_rows(case, segs):
    """Rows of ``segs`` that native PXR/CBS line scoring may read.

    For an axial fibre case the zone rows are removed, so the line spectrum
    ``spec`` is the sum over native rows only:
    ``spec = (1/Ne) sum_{rows not in zone} dN/(dE dOmega)``. The scalars
    (``Ne``, thickness) are kept, so the per-incident-electron normalization
    is unchanged. Every other case returns ``segs`` itself. An axial transport
    without the ``channeling_zone`` tag refuses.

    The native line kernel treats a row as a free plane-wave electron on a
    straight path. Inside the zone the electron is in the fibre model's
    retained transverse states, whose bound→bound and capture emission is
    already in ``spec_channeling``; scoring the same path natively counts that
    population twice (measured for si-fibre/A: 2.33e-4 /sr in band against the
    4.17e-5 /sr in-cut free→free block, #994). The omitted in-cut free→free
    block is declared as ``free_free_upper``; above-cut free→free on zone rows
    is unbounded and declared with results.

    Validation: cr-axial-zone-row-native-lines
    """
    record = case.get("channeling")
    if not isinstance(record, Mapping) or record.get("model") != "axial":
        return segs
    from ..spectrum.cxr._kernels import _SEG_ARRAYS

    zone = segs.get("channeling_zone")
    if zone is None:
        raise AxialAdmissionError("axial transport carries no zone-row tag (#994)")
    keep = ~np.asarray(zone, dtype=bool)
    out = dict(segs)
    for key in (*_SEG_ARRAYS, "channeling_zone"):
        if out.get(key) is not None:
            out[key] = out[key][keep]
    return out


def transport_axial_case(
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
    """Transport one admitted axial fibre case; see the module docstring.

    ``simulate`` is the runner's ``simulate_trajectories`` closure on the
    per-electron CPU core. The result is an ordinary transport result with a
    ``channeling_zone`` row tag and a ``channeling`` record holding the fate
    counts, the declared budgets and the fibre source on the line grid.

    Validation: cr-axial-persisted-channel
    """
    from ..channeling.axial_case import admitted_axial_case
    from ..transport.beam_entry import sample_primary_entry, table_energy_range
    from ..transport.kinematics import stream_keys
    from ..transport.secondaries import LaunchState, SecondaryPass
    from ..transport.stopping import prepare_sbethe_stopping_table, sbethe_stopping_keV_per_ang
    from .step_budget import retry_step_budget

    admitted = admitted_axial_case(case)
    Ne = int(E_cut_by_electrons.size)
    lines = int(case["Ne"])
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
    _plane_wave(v0, E_keV, beam_dir, E0)
    log_energy, log_stopping = prepare_sbethe_stopping_table(stopping_tables[0])
    S = -float(sbethe_stopping_keV_per_ang(log_energy, log_stopping, E0))
    keys = stream_keys(seed, Ne)
    source, records, incidents, zone = fibre_source_and_fates(
        case, admitted, r0[:lines], clock[:lines], t0[:lines], keys[:lines], S
    )
    plan = axial_native_plan(records, incidents, thickness_ang=thickness)
    extra = np.arange(lines, Ne)  # bremsstrahlung-only electrons: native from the face
    parts = [] if plan.launch is None else [plan.launch]
    if extra.size:
        parts.append(
            LaunchState(
                r_ang=r0[extra],
                v_hat=v0[extra],
                E_keV=E_keV[extra],
                t_ang=clock[extra],
                t0_ang=t0[extra],
                stream_keys=keys[extra],
                radiative_keys=np.zeros(extra.size, dtype=np.uint64),
            )
        )
    run_ids = np.concatenate([plan.launch_ids, extra]).astype(np.int64)
    launched = bool(run_ids.size)
    if not launched:  # schema template when nothing launches
        run_ids = np.array([0])
        parts = [
            LaunchState(r0[:1], v0[:1], E_keV[:1], clock[:1], t0[:1], keys[:1],
                        np.zeros(1, dtype=np.uint64))
        ]  # fmt: skip
    launch = LaunchState(
        *(
            np.concatenate([getattr(part, name) for part in parts])
            for name in (
                "r_ang", "v_hat", "E_keV", "t_ang", "t0_ang", "stream_keys", "radiative_keys"
            )
        )
    )  # fmt: skip
    table_range = table_energy_range(E_cut_by_electrons, E_keV)
    block_kw = {
        **_LAUNCH_BEAM_ARGS,
        "_secondary": SecondaryPass(launch=launch, table_range_keV=table_range, photon_seed=seed),
        "_energy_range_keV": table_range,
    }
    native = retry_step_budget(
        simulate, keep, stop=run_ids.size, block_kw=block_kw, cut=E_cut_by_electrons[run_ids]
    )
    native_rows = _row_arrays(native)
    if not launched:
        native_rows = {key: value[:0] for key, value in native_rows.items()}
    local = native_rows["electron_id"]
    native_rows["electron_id"] = run_ids[local].astype(np.int64)
    zoned = np.zeros(Ne, dtype=bool)
    template = {"rows": native_rows, "_S": S}
    zone_rows, z_ids = _axial_zone_rows(template, records, zone, r0, v0, E_keV, clock, t0)
    zoned[z_ids] = True
    native_rows["flight_id"] = native_rows["flight_id"] + zoned[run_ids][local]
    out = join_zone_rows(native, native_rows, zone_rows)
    out.update(initial_r_ang=r0, initial_v_hat=v0, initial_E_keV=E_keV, initial_t0_ang=t0, Ne=Ne)
    if record is not None:
        out["beam_entry"] = record
    for counter in _COUNTERS:
        out[counter] = int(native.get(counter, 0)) if launched else 0
    out["n_transmitted"] += int(plan.exit_ids.size)
    weights = {fate: 0.0 for fate in _FATES}
    for item in records:
        weights[item.fate] += 1.0 / lines
    out["channeling"] = {
        "model": case["channeling"]["model_marker"],
        "case_key": admitted.key,
        **plan.provenance(),
        "fate_fractions": weights,
        "entrance_weight": float(source.events[0].weight),
        "stopping_keV_per_ang": S,
        "depth_budget": admitted.depth_budget,
        "joint_confidence": admitted.joint_confidence,
        "free_free": admitted.free_free,
        "free_free_upper": admitted.free_free_upper,
        "source": {
            "E_eV": source.photon_energy_eV,
            "spectrum": source.spectrum,
            "sem": source.sem,
        },
    }
    return out


def case_axial_spectrum(case, transport) -> np.ndarray:
    """Persisted fibre source [photons/(eV sr incident electron)] on the line grid.

    The transport phase already scored it from the same histories that gave
    the fates; a transport result without it, or on another grid, refuses.

    Validation: cr-axial-persisted-channel
    """
    info = transport["segs"].get("channeling")
    source = info.get("source") if isinstance(info, dict) else None
    grid = np.asarray(transport["E_grid"], dtype=float)
    if source is None or info.get("model") != case["channeling"]["model_marker"]:
        raise AxialAdmissionError("axial transport carries no fibre source (#994)")
    if source["E_eV"].shape != grid.shape or not np.array_equal(source["E_eV"], grid):
        raise AxialAdmissionError("axial fibre source was scored on another line grid (#994)")
    spectrum = np.asarray(source["spectrum"], dtype=float)
    if not np.all(np.isfinite(spectrum)):
        raise FloatingPointError("axial fibre source is not finite")
    return spectrum.copy()
