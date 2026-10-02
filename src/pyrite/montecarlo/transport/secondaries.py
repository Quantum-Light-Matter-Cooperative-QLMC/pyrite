"""Generation-batched transport of shell hard-collision secondaries (#94).

``simulate_trajectories(secondary_threshold_eV=T_s)`` hands its arguments to
:func:`transport_secondary_cascade`. The primaries run first; the secondaries
their hard collisions launch above ``T_s`` form the next generation, which the
same transport runs from an explicit :class:`LaunchState` instead of beam
sampling, until a generation launches nothing. Rows of every generation are
joined into one result with ``track_id``/``parent_id``/``generation`` fields.
See ``docs/physics/beam-transport/shell-soft-hard-transport.md``.

RNG. A secondary's stream key hashes (seed, parent track, parent hard ordinal)
under its own salt, so it depends on neither batch size nor processing order.
Its hard, soft-straggling and coupled-radiative streams derive from that key
exactly as a primary's derive from its own.

With ``pair_production_model`` each generation's coupled hard photons above
``2 m_e c^2`` also get one first-interaction step (``pair_production.py``); a
pair's electron above ``T_s`` joins the next generation keyed on (seed, parent
track, photon ordinal) under a salt of its own, and its positron is recorded,
not transported (#276).

Validation: shell-secondary-transport, photon-pair-first-interaction
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .batching import resolve_transport_core
from .events import EVENT_CUTOFF, EVENT_EXIT_BOTTOM, EVENT_EXIT_SIDE, EVENT_EXIT_TOP
from .kinematics import (
    _SM64_GOLDEN,
    _SM64_MIX1,
    _SM64_MIX2,
    _SM64_ONE,
    _SM64_S27,
    _SM64_S30,
    _SM64_S31,
)

__all__ = [
    "LaunchState",
    "SecondaryPass",
    "secondary_energy_balance",
    "secondary_stream_keys",
    "transport_secondary_cascade",
]

# Distinct from the hard, Urban and radiative salts and the photon salt below.
_SECONDARY_STREAM_SALT = np.uint64(0x7F4A7C159E3779B9)
_PHOTON_SEED_SALT = np.uint64(0x2545F4914F6CDD1D)
_PAIR_ELECTRON_STREAM_SALT = np.uint64(0x9B05688C2B3E6C1F)
#: ``secondary_tracks["launch_kind"]`` values.
LAUNCH_PRIMARY, LAUNCH_SHELL, LAUNCH_PAIR = 0, 1, 2
DEFAULT_MAX_SECONDARY_GENERATIONS = 64
DEFAULT_SECONDARY_TRACKS_PER_PRIMARY = 1000
# Per-row result arrays joined across generations. Aliases are re-pointed after.
_ROW_KEYS = (
    "r_mid",
    "v_hat",
    "L_ang",
    "E_start_keV",
    "t_start_ang",
    "t0_ang",
    "electron_id",
    "layer",
    "E_end_keV",
    "t_end_ang",
    "E_repr_keV",
    "flight_id",
    "substep_id",
    "event_kind",
    "hard_W_keV",
    "hard_channel",
    "hard_secondary_v_hat",
    "hard_radiative_k_eV",
    "hard_radiative_Z",
    "hard_radiative_direction",
    "hard_radiative_target_momentum_eV_c",
)
_ALIASES = {"E_keV": "E_start_keV", "t_ang": "t_start_ang", "elec_id": "electron_id"}
# Beam-sampling arguments a launched generation must not see.
_BEAM_ARGS = {
    "beam_dir": None,
    "beam_fwhm_mm": None,
    "beam_fwhm_y_mm": None,
    "bunch_length_fs": None,
    "long_offsets_fs": None,
    "longitudinal_distribution": None,
    "transverse_distribution": None,
    "energy_spread_frac": None,
    "tilt_polar_rad": 0.0,
    "tilt_azim_rad": 0.0,
    "E_cut_by_electrons": None,
}
_EXITS = (int(EVENT_EXIT_TOP), int(EVENT_EXIT_BOTTOM), int(EVENT_EXIT_SIDE))


@dataclass(frozen=True)
class LaunchState:
    """Explicit per-particle start state replacing beam sampling.

    Positions and directions ``(n, 3)`` in the slab frame [Angstrom], kinetic
    energy [keV], start clock and bunch offset [Angstrom, c=1], and the
    primary-domain stream keys the transport and its derived streams use.
    The layer follows from the position, as for every transport row.
    """

    r_ang: np.ndarray
    v_hat: np.ndarray
    E_keV: np.ndarray
    t_ang: np.ndarray
    t0_ang: np.ndarray
    stream_keys: np.ndarray
    radiative_keys: np.ndarray


@dataclass(frozen=True)
class SecondaryPass:
    """One generation's hooks into :func:`simulate_trajectories`.

    ``launch`` is None for the primaries. ``table_range_keV`` makes every
    generation build identical shell tables; ``photon_seed`` keys the host
    completion of coupled hard-photon directions per generation.
    """

    launch: LaunchState | None
    table_range_keV: tuple[float, float]
    photon_seed: int


def _mix(x):
    x = (x ^ (x >> _SM64_S30)) * _SM64_MIX1
    x = (x ^ (x >> _SM64_S27)) * _SM64_MIX2
    return x ^ (x >> _SM64_S31)


def secondary_stream_keys(seed, parent_track, parent_ordinal, *, salt=_SECONDARY_STREAM_SALT):
    """Stream keys of secondaries from (seed, parent track, parent hard ordinal)."""
    with np.errstate(over="ignore"):
        x = np.uint64(seed) ^ np.uint64(salt)
        parent = np.asarray(parent_track, dtype=np.uint64)
        ordinal = np.asarray(parent_ordinal, dtype=np.uint64)
        x = _mix(x + _SM64_GOLDEN * (parent + _SM64_ONE))
        return _mix(x + _SM64_GOLDEN * (ordinal + _SM64_ONE))


def _radiative_keys(keys):
    from ._jit_radiative import _RADIATIVE_STREAM_SALT

    with np.errstate(over="ignore"):
        return _mix(np.asarray(keys, dtype=np.uint64) ^ np.uint64(_RADIATIVE_STREAM_SALT))


def _photon_seed(seed, generation):
    if generation == 0:
        return seed
    with np.errstate(over="ignore"):
        x = np.uint64(seed) ^ _PHOTON_SEED_SALT
        return int(_mix(x + _SM64_GOLDEN * np.uint64(generation)))


def _is_device(a):
    return hasattr(a, "__cuda_array_interface__")


def _host(a):
    return a.get() if _is_device(a) else np.asarray(a)


def _take(a, index):
    """Host copy of ``a[index]``; gathers on the device before the download."""
    if _is_device(a):
        import cupy

        return a[cupy.asarray(index)].get()
    return np.asarray(a)[index]


def _concat(parts):
    if any(_is_device(p) for p in parts):
        import cupy

        return cupy.concatenate([cupy.asarray(p) for p in parts])
    return np.concatenate(parts)


def _validate(kw):
    """Threshold [keV] and caps; raise on every unsupported combination."""
    threshold_eV = kw["secondary_threshold_eV"]
    if (
        isinstance(threshold_eV, bool)
        or not np.isfinite(threshold_eV)
        or not float(threshold_eV) > 0.0
    ):
        raise ValueError("secondary_threshold_eV must be finite and positive")
    if kw["inelastic_model"] != "shell-soft-hard":
        raise ValueError("secondary_threshold_eV requires inelastic_model='shell-soft-hard'")
    if kw["collect_diagnostics"]:
        raise ValueError("collect_diagnostics assumes one track per electron; not with secondaries")
    if kw["_secondary"] is not None:
        raise ValueError("a secondary generation cannot launch its own cascade")
    generations = kw["max_secondary_generations"]
    tracks = kw["max_secondary_tracks"]
    if isinstance(generations, bool) or int(generations) != generations or generations < 1:
        raise ValueError("max_secondary_generations must be a positive integer")
    if tracks is None:
        tracks = DEFAULT_SECONDARY_TRACKS_PER_PRIMARY * int(kw["Ne"])
    if isinstance(tracks, bool) or int(tracks) != tracks or tracks < 1:
        raise ValueError("max_secondary_tracks must be a positive integer")
    threshold_keV = float(threshold_eV) * 1e-3
    _validate_coverage(kw, threshold_keV)
    _validate_pair_model(kw)
    return threshold_keV, int(generations), int(tracks)


def _validate_pair_model(kw):
    """Pair conversion needs coupled hard photons: BremsLib tables, no groove."""
    from .pair_production import PAIR_PRODUCTION_MODELS

    model = kw.get("pair_production_model")
    if model is None:
        return
    if model not in PAIR_PRODUCTION_MODELS:
        raise ValueError(f"pair_production_model must be one of {PAIR_PRODUCTION_MODELS} or None")
    if (
        kw["radiative_model"] == "uncoupled"
        or kw["bremslib_tables"] is None
        or kw["groove"] is not None
    ):
        raise ValueError(
            "pair_production_model requires coupled BremsLib radiative transport "
            "(bremslib_tables, no groove)"
        )


def _stack_geometry(kw):
    """Layer stack and finite footprint [Angstrom] the photons cross."""
    from ...materials.attenuation import _normalize_composition

    layers = kw["layers"]
    if layers is None:
        layers = [
            (
                0.0,
                float(kw["thickness_ang"]),
                _normalize_composition(kw["element"], kw["n_atoms_per_ang3"], kw["composition"]),
            )
        ]
    width, height = kw["crystal_width_mm"], kw["crystal_height_mm"]
    return (
        layers,
        None if width is None else float(width) * 1.0e7,
        None if height is None else float(height) * 1.0e7,
    )


def _pair_rows(result):
    """Host copies of the row fields the photon step reads."""
    keys = (
        "electron_id",
        "flight_id",
        "substep_id",
        "r_mid",
        "v_hat",
        "L_ang",
        "t_end_ang",
        "hard_radiative_k_eV",
        "hard_radiative_direction",
    )
    return {k: _host(result[k]) for k in keys}


def _validate_coverage(kw, threshold_keV):
    """Reject a threshold outside any table the secondaries use; never clamp.

    Primaries' own range is checked by the transport. Here only the new lower
    end ``T_s`` is: SBETHE (and the shell tables on its nodes), ELSEPA, and the
    coupled BremsLib partition, whose photon cutoff must not exceed ``T_s``.
    """
    from .scattering import check_elsepa_coverage
    from .stopping import prepare_sbethe_stopping_tables

    E_max = float(kw["E0_keV"])
    tables = kw["stopping_tables"]
    if tables is not None:
        n_layers = len(tables) if kw["layers"] is None else len(kw["layers"])
        for log_e, _ in prepare_sbethe_stopping_tables(tables, n_layers):
            lower = float(np.nextafter(np.exp(log_e[0]), -np.inf))
            if threshold_keV < lower:
                raise ValueError(
                    f"secondary_threshold_eV={threshold_keV * 1e3:g} is below the SBETHE "
                    f"table floor {lower * 1e3:g} eV"
                )
    check_elsepa_coverage(kw["elastic_tables"], threshold_keV, threshold_keV)
    if kw["radiative_model"] == "bremslib-soft-hard":
        cutoff_eV = kw["radiative_cutoff_eV"]
        if cutoff_eV is not None and float(cutoff_eV) > threshold_keV * 1e3:
            raise ValueError("radiative_cutoff_eV must not exceed secondary_threshold_eV")
        from .hard_radiative import pack_radiative_layer_tables

        layers = kw["layers"]
        if layers is None:
            from ...materials.attenuation import _normalize_composition

            comps = [
                _normalize_composition(kw["element"], kw["n_atoms_per_ang3"], kw["composition"])
            ]
        else:
            comps = [layer[2] for layer in layers]
        pack_radiative_layer_tables(comps, kw["bremslib_tables"], threshold_keV * 1e3, E_max * 1e3)


def _harvest(result, threshold_keV):
    """Launch rows of one generation: local parent, hard ordinal, kinetic energy.

    Every row with ``hard_channel >= 0`` is a hard collision; its ordinal counts
    the parent's hard rows in flight order. Returns the hard rows' accounting
    and the launched subset ordered by (parent, ordinal).
    """
    from .shell_transport import hard_event_energy_accounting

    channel = _host(result["hard_channel"])
    rows = np.flatnonzero(channel >= 0)
    parent = _take(result["electron_id"], rows).astype(np.int64)
    flight = _take(result["flight_id"], rows).astype(np.int64)
    order = np.lexsort((flight, parent))
    rows, parent = rows[order], parent[order]
    first = np.ones(rows.size, dtype=bool)
    first[1:] = parent[1:] != parent[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(rows.size), 0))
    ordinal = np.arange(rows.size) - start
    accounting = hard_event_energy_accounting(
        {
            "inelastic": result["inelastic"],
            "hard_channel": channel[rows],
            "layer": _take(result["layer"], rows),
            "hard_W_keV": _take(result["hard_W_keV"], rows),
        }
    )
    kinetic = accounting["secondary_keV"]
    launched = kinetic > threshold_keV
    sel = rows[launched]
    L = _take(result["L_ang"], sel)
    return {
        "kinetic_keV": kinetic,
        "binding_keV": accounting["binding_keV"],
        "launched": launched,
        "parent": parent[launched],
        "ordinal": ordinal[launched],
        "r_ang": _take(result["r_mid"], sel) + 0.5 * L[:, None] * _take(result["v_hat"], sel),
        "v_hat": _take(result["hard_secondary_v_hat"], sel),
        "E_keV": kinetic[launched],
        "t_ang": _take(result["t_end_ang"], sel),
    }


def transport_secondary_cascade(simulate, kw):
    """Run ``simulate`` over primaries and every launched generation; join rows.

    ``kw`` is the caller's full :func:`simulate_trajectories` argument mapping.
    Validation: shell-secondary-transport, photon-pair-first-interaction
    """
    threshold_keV, max_generations, max_tracks = _validate(kw)
    seed = kw["seed"]
    base = {
        k: v
        for k, v in kw.items()
        if k not in ("_secondary", "pair_production_model")
        and not k.startswith(("secondary_", "max_secondary_"))
    }
    pair_model = kw.get("pair_production_model")
    if pair_model is not None:
        from .pair_production import convert_hard_photons

        pair_layers, pair_width, pair_height = _stack_geometry(kw)
    pair_events = []
    pair_counts = []
    E_cut = kw["E_cut_by_electrons"]
    cut_min = float(kw["E_cut_keV"] if E_cut is None else np.min(E_cut))
    lo = min(cut_min, threshold_keV)
    Ne = int(kw["Ne"])
    gen0 = simulate(
        **base, _secondary=SecondaryPass(None, (lo, float(kw["E0_keV"])), _photon_seed(seed, 0))
    )
    # The transport widened gen 0's range to its sampled energies; match it.
    table_range = (lo, max(float(kw["E0_keV"]), float(np.max(gen0["initial_E_keV"]))))
    first_core = resolve_transport_core(kw["transport_core"], Ne, kw["groove"])
    later_core = "cuda" if first_core == "cuda" else "per-electron"

    # Per-track tables, indexed by track id.
    tracks = {
        "track_id": [np.arange(Ne, dtype=np.int64)],
        "parent_id": [np.full(Ne, -1, dtype=np.int64)],
        "generation": [np.zeros(Ne, dtype=np.int16)],
        "electron_id": [np.arange(Ne, dtype=np.int64)],
        "parent_hard_ordinal": [np.full(Ne, -1, dtype=np.int64)],
        "launch_kind": [np.full(Ne, LAUNCH_PRIMARY, dtype=np.int8)],
        "launch_E_keV": [np.asarray(gen0["initial_E_keV"], dtype=float)],
        "launch_r_ang": [np.asarray(gen0["initial_r_ang"], dtype=float)],
        "launch_v_hat": [np.asarray(gen0["initial_v_hat"], dtype=float)],
        "launch_t_ang": [np.zeros(Ne)],
    }
    generations = [gen0]
    track_offset = [0]
    counts = [_counts(gen0)]
    n_tracks = Ne
    electron_of = tracks["electron_id"][0]
    t0_of = np.asarray(gen0["initial_t0_ang"], dtype=float)
    current = gen0
    row_offset = 0
    while True:
        g = len(generations)
        harvest = _harvest(current, threshold_keV)
        harvest["kind"] = np.full(harvest["parent"].size, LAUNCH_SHELL, dtype=np.int8)
        harvest["keys"] = secondary_stream_keys(
            seed, track_offset[-1] + harvest["parent"], harvest["ordinal"]
        )
        if pair_model is not None:
            events, photon_counts = convert_hard_photons(
                _pair_rows(current),
                seed=seed,
                parent_track_offset=track_offset[-1],
                layers=pair_layers,
                width_ang=pair_width,
                height_ang=pair_height,
            )
            events["electron_launched"] = events["electron_keV"] > threshold_keV
            events["global_row"] = row_offset + events["row"]
            events["parent_track"] = track_offset[-1] + events["parent"]
            events["electron_id"] = electron_of[events["parent"]]
            events["electron_track"] = np.full(events["row"].size, -1, dtype=np.int64)
            pair_events.append(events)
            pair_counts.append(photon_counts)
            harvest = _merge_pair_launches(harvest, events, seed)
        row_offset += _host(current["electron_id"]).size
        n_new = int(harvest["parent"].size)
        if n_new == 0:
            break
        if g > max_generations:
            raise RuntimeError(
                f"secondary cascade exceeded max_secondary_generations={max_generations}"
            )
        if n_tracks - Ne + n_new > max_tracks:
            raise RuntimeError(
                f"secondary cascade exceeded max_secondary_tracks={max_tracks} "
                f"(generation {g} would add {n_new})"
            )
        parent_track = track_offset[-1] + harvest["parent"]
        keys = harvest["keys"]
        if pair_model is not None:
            launched = np.flatnonzero(pair_events[-1]["electron_launched"])
            pair_events[-1]["electron_track"][launched] = n_tracks + harvest["pair_index"]
        electrons = electron_of[harvest["parent"]]
        t0 = t0_of[harvest["parent"]]
        launch = LaunchState(
            harvest["r_ang"],
            harvest["v_hat"],
            harvest["E_keV"],
            harvest["t_ang"],
            t0,
            keys,
            _radiative_keys(keys),
        )
        step = dict(base, **_BEAM_ARGS)
        step.update(
            E0_keV=float(np.max(harvest["E_keV"])),
            Ne=n_new,
            E_cut_keV=threshold_keV,
            transport_core=later_core,
        )
        current = simulate(
            **step, _secondary=SecondaryPass(launch, table_range, _photon_seed(seed, g))
        )
        tracks["track_id"].append(n_tracks + np.arange(n_new, dtype=np.int64))
        tracks["parent_id"].append(parent_track.astype(np.int64))
        tracks["generation"].append(np.full(n_new, g, dtype=np.int16))
        tracks["electron_id"].append(electrons)
        tracks["parent_hard_ordinal"].append(harvest["ordinal"].astype(np.int64))
        tracks["launch_kind"].append(harvest["kind"])
        tracks["launch_E_keV"].append(harvest["E_keV"])
        tracks["launch_r_ang"].append(harvest["r_ang"])
        tracks["launch_v_hat"].append(harvest["v_hat"])
        tracks["launch_t_ang"].append(harvest["t_ang"])
        generations.append(current)
        track_offset.append(n_tracks)
        counts.append(_counts(current))
        n_tracks += n_new
        electron_of, t0_of = electrons, t0
    track_table = {k: np.concatenate(v) for k, v in tracks.items()}
    out = _join(generations, track_offset, track_table, counts, threshold_keV)
    if pair_model is not None:
        out["pair_production"] = _pair_summary(pair_model, pair_events, pair_counts)
    return out


def _merge_pair_launches(harvest, events, seed):
    """Append launched pair electrons after the shell launches, same fields.

    ``ordinal`` is the photon ordinal on the parent track; ``pair_index`` is
    each launched pair electron's position within this generation's launches.
    Validation: photon-pair-first-interaction
    """
    launched = events["electron_launched"]
    n_shell = harvest["parent"].size
    n_pair = int(np.count_nonzero(launched))
    keys = secondary_stream_keys(
        seed,
        events["parent_track"][launched],
        events["photon_ordinal"][launched],
        salt=_PAIR_ELECTRON_STREAM_SALT,
    )
    merged = dict(harvest)
    for name, extra in (
        ("parent", events["parent"][launched]),
        ("ordinal", events["photon_ordinal"][launched]),
        ("r_ang", events["r_ang"][launched]),
        ("v_hat", events["electron_v_hat"][launched]),
        ("E_keV", events["electron_keV"][launched]),
        ("t_ang", events["t_ang"][launched]),
        ("kind", np.full(n_pair, LAUNCH_PAIR, dtype=np.int8)),
        ("keys", keys),
    ):
        merged[name] = np.concatenate(
            [np.asarray(harvest[name]), extra.astype(np.asarray(harvest[name]).dtype)]
        )
    merged["pair_index"] = n_shell + np.arange(n_pair)
    return merged


def _pair_summary(model, pair_events, pair_counts):
    """Joined pair events and photon outcome counts; warns on untransported e+."""
    from .pair_production import _empty_events

    fields = list(_empty_events()) + [
        "electron_launched",
        "global_row",
        "parent_track",
        "electron_id",
        "electron_track",
    ]
    events = {name: np.concatenate([e[name] for e in pair_events]) for name in fields}
    events["row"] = events.pop("global_row")
    counts = {
        name: int(sum(c[name] for c in pair_counts))
        for name in ("photons", "pair", "other", "escaped")
    }
    if counts["pair"]:
        warnings.warn(
            f"{counts['pair']} pair-production positrons are recorded but not "
            "transported (#276); their kinetic energy is reported, not deposited",
            UserWarning,
            stacklevel=3,
        )
    return {
        "model": model,
        "positrons_transported": False,
        "photon_counts": counts,
        "events": events,
    }


def _counts(result):
    return {
        k: int(result[k])
        for k in ("n_backscattered", "n_transmitted", "n_side_exited", "n_cutoff_stopped")
    }


def _join(generations, offsets, track_table, counts, threshold_keV):
    """One result: gen-0 metadata and counts, rows of every generation."""
    out = dict(generations[0])
    parents = track_table["parent_id"]
    rows = {k: [] for k in _ROW_KEYS if generations[0].get(k) is not None}
    rows.update(track_id=[], parent_id=[], generation=[])
    for g, (result, offset) in enumerate(zip(generations, offsets, strict=True)):
        local = result["electron_id"]
        n_local = _host(local).size
        for k in rows:
            if k in _ROW_KEYS:
                rows[k].append(result[k])
        local_host = _host(local).astype(np.int64)
        track = offset + local_host
        rows["track_id"].append(track)
        rows["parent_id"].append(parents[track] if n_local else np.empty(0, dtype=np.int64))
        rows["generation"].append(np.full(n_local, g, dtype=np.int16))
        if g:
            # Local indices name this generation's tracks; rows carry the
            # primary history instead.
            rows["electron_id"][-1] = track_table["electron_id"][track]
    for k, parts in rows.items():
        out[k] = _concat(parts)
    for alias, key in _ALIASES.items():
        out[alias] = out[key]
    out["secondary_tracks"] = track_table
    out["secondaries"] = {
        "threshold_eV": threshold_keV * 1e3,
        "n_generations": len(generations),
        "tracks_per_generation": [
            int(np.sum(track_table["generation"] == g)) for g in range(len(generations))
        ],
        "counts_per_generation": counts,
    }
    return out


def secondary_energy_balance(result, *, per_history=False):
    """All-generation energy balance of a secondary-transport result [keV].

    ``incident = escaped + deposited + binding_reserved + radiated`` where
    deposited is the continuous row loss, every cutoff residual and every
    secondary at or below the threshold; ``residual`` is the difference.
    In the coupled radiative mode the continuous loss includes the soft
    radiative share. With ``per_history`` every term is an array over the
    primary histories (``electron_id``), each closing on its own.

    With pair production a converted photon leaves ``radiated`` and splits
    exactly into ``pair_rest_mass`` (``2 m_e c^2``), the untransported
    ``positron`` kinetic energy, and its electron: a launched track's own rows,
    or ``pair_subthreshold`` (part of ``deposited``) at or below the threshold.
    Validation: shell-secondary-transport, photon-pair-first-interaction
    """
    from .shell_transport import hard_event_energy_accounting

    host = {k: _host(result[k]) for k in _ROW_KEYS if result.get(k) is not None}
    host["inelastic"] = result["inelastic"]
    threshold_keV = result["secondaries"]["threshold_eV"] * 1e-3
    track = _host(result["track_id"])
    history = host["electron_id"].astype(np.int64)
    kind = host["event_kind"]
    E0, E1 = host["E_start_keV"], host["E_end_keV"]
    acc = hard_event_energy_accounting(host)
    kinetic = acc["secondary_keV"]
    photon = host.get("hard_radiative_k_eV", np.zeros(E0.size)) * 1e-3
    W = host["hard_W_keV"]
    order = np.lexsort((host["substep_id"], host["flight_id"], track))
    last = np.ones(order.size, dtype=bool)
    last[:-1] = track[order][1:] != track[order][:-1]
    end = order[last]
    exits = end[np.isin(kind[end], _EXITS)]
    cut = end[kind[end] == int(EVENT_CUTOFF)]
    tracks = result["secondary_tracks"]
    entered = np.zeros(tracks["track_id"].size, dtype=bool)
    entered[np.unique(track)] = True
    primary = tracks["generation"] == 0
    below = kinetic <= threshold_keV
    n_history = int(np.count_nonzero(primary))

    def total(where, values):
        if per_history:
            return np.bincount(where, weights=values, minlength=n_history)
        return float(np.sum(values))

    terms = {
        "incident_keV": total(
            tracks["electron_id"][primary & entered].astype(np.int64),
            tracks["launch_E_keV"][primary & entered],
        ),
        "escaped_keV": total(history[exits], E1[exits]),
        "continuous_keV": total(history, E0 - E1),
        "cutoff_residual_keV": total(history[cut], (E1 - W - photon)[cut]),
        "subthreshold_keV": total(history[acc["row"][below]], kinetic[below]),
        "binding_reserved_keV": total(history[acc["row"]], acc["binding_keV"]),
    }
    pair = result.get("pair_production")
    radiated = photon
    if pair is not None:
        from .pair_production import PAIR_THRESHOLD_EV

        events = pair["events"]
        radiated = photon.copy()
        radiated[events["row"]] = 0.0
        owner = events["electron_id"].astype(np.int64)
        kept = ~events["electron_launched"]
        terms["pair_rest_mass_keV"] = total(owner, np.full(owner.size, PAIR_THRESHOLD_EV * 1e-3))
        terms["positron_keV"] = total(owner, events["positron_keV"])
        terms["pair_subthreshold_keV"] = total(owner[kept], events["electron_keV"][kept])
    terms["radiated_keV"] = total(history, radiated)
    terms["deposited_keV"] = (
        terms["continuous_keV"]
        + terms["cutoff_residual_keV"]
        + terms["subthreshold_keV"]
        + terms.get("pair_subthreshold_keV", 0.0)
    )
    terms["residual_keV"] = terms["incident_keV"] - (
        terms["escaped_keV"]
        + terms["deposited_keV"]
        + terms["binding_reserved_keV"]
        + terms["radiated_keV"]
        + terms.get("pair_rest_mass_keV", 0.0)
        + terms.get("positron_keV", 0.0)
    )
    return terms
