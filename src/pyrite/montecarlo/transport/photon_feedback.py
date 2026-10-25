"""CPU emission adapters and charged feedback for the experimental photon queue.

Reuses the shell cascade, Bhabha/positron tables, Heitler annihilation and
sampled EADL relaxation. No production transport selector is enabled here.
Validation: photon-secondary-feedback
"""

from dataclasses import dataclass, replace

import numpy as np
import xraydb

from ..eadl_relaxation import load_eadl_relaxation
from ..eedl_ionization import EEDL_SUBSHELL_LABELS
from .annihilation import (
    ANNIHILATION_AT_REST,
    ANNIHILATION_IN_FLIGHT,
    ELECTRON_REST_KEV,
    sample_at_rest_directions,
    sample_heitler,
)
from .pair_production import PAIR_THRESHOLD_EV
from .photon_coupling import (
    CascadeChargedStage,
    ChargedStageResult,
    _hard_photons,
    _transport_kwargs,
    cascade_charged_stage,
    stage_seed,
)
from .photon_histories import ELECTRON, POSITRON, _isotropic, daughter_stream_key
from .relaxation import sample_relaxation_cascade

_REST_EV = PAIR_THRESHOLD_EV
_VACANCY_SALT = np.uint64(0xA54FF53A5F1D36F1)
_ANNIHILATION_SALT = np.uint64(0x510E527FADE682D1)
_DESIGNATORS = {label: designator for designator, label in EEDL_SUBSHELL_LABELS.items()}


def annihilation_emissions(kinds, kinetic_keV, directions, origins, keys):
    """Unattenuated at-rest/Heitler photon rows, two per annihilation event.

    Inputs follow ``score_annihilation_photons`` but no first-interaction sink
    is applied. Energies sum to T + 2mc² per event; at rest each is mc² and
    directions are opposite. Independent per-event Philox keys are supplied
    by the caller. Source: the existing Heitler and at-rest samplers (#295).
    Validation: photon-secondary-feedback
    """
    kinds, keys = np.asarray(kinds), np.asarray(keys, dtype=np.uint64)
    n = kinds.size
    if keys.shape != (n,) or np.shape(kinetic_keV) != (n,):
        raise ValueError("annihilation energies and keys must match kinds")
    kinetic_keV = np.asarray(kinetic_keV, dtype=float)
    if not np.isin(kinds, [ANNIHILATION_AT_REST, ANNIHILATION_IN_FLIGHT]).all():
        raise ValueError("unknown annihilation kind")
    if not np.isfinite(kinetic_keV).all() or np.any(kinetic_keV < 0):
        raise ValueError("annihilation kinetic energies must be finite and nonnegative")
    if np.any((kinds == ANNIHILATION_AT_REST) & (kinetic_keV != 0)):
        raise ValueError("at-rest annihilation requires zero kinetic energy")
    origins = np.asarray(origins, dtype=float).reshape(n, 3)
    directions = np.asarray(directions, dtype=float).reshape(n, 3)
    energies, emitted = np.empty((n, 2)), np.empty((n, 2, 3))
    for i in range(n):
        rng = np.random.Generator(np.random.Philox(key=int(keys[i])))
        if kinds[i] == ANNIHILATION_IN_FLIGHT:
            energies[i], emitted[i] = sample_heitler(kinetic_keV[i], directions[i], rng)
        else:
            energies[i], emitted[i] = ELECTRON_REST_KEV, sample_at_rest_directions(rng)
    return dict(
        position_ang=np.repeat(origins, 2, axis=0),
        direction=emitted.reshape(-1, 3),
        energy_eV=energies.ravel() * 1e3,
        parent=np.repeat(np.arange(n), 2),
        ordinal=np.tile([0, 1], n),
    )


def impact_relaxation_emissions(result, keys, cutoff_eV):
    """Sample characteristic/Auger sources from recorded charged-impact holes.

    Inner-shell metadata provide atom and EADL shell label; events are placed
    at hard-collision endpoints in sample-frame Å. One independent lineage
    stream per hole samples EADL transitions and isotropic directions. Outer
    free-electron proxy channels create no hole. ``released_eV`` is the sum
    of emitted photon and Auger energies per input history: subtract it from
    reserved binding, retaining terminal binding and signed EADL/transport
    energy defects separately from emitted kinetic energy. No attenuation or
    primary-electron debit is applied. Source: ENDF-102 MF=28; shell hard-event
    energy partition. With no inner-shell collision the source is empty.
    Validation: photon-secondary-feedback
    """
    from .secondaries import _host

    keys = np.asarray(keys, dtype=np.uint64)
    channel = _host(result["hard_channel"])
    owner = _host(result["electron_id"]).astype(np.int64)
    track = _host(result["track_id"])
    order = np.lexsort((_host(result["flight_id"]), track, owner))
    lookup = [{c["code"]: c for c in layer} for layer in result["inelastic"]["channels"]]
    released, ordinals = np.zeros(keys.size), np.zeros(keys.size, dtype=np.int64)
    columns = {
        k: [] for k in ("position_ang", "direction", "energy_eV", "parent", "species", "key")
    }
    cache = {}
    for row in order[channel[order] >= 0]:
        ch = lookup[int(_host(result["layer"])[row])][int(channel[row])]
        if not ch["inner"]:
            continue
        parent = owner[row]
        key = daughter_stream_key(keys[parent] ^ _VACANCY_SALT, ordinals[parent])
        ordinals[parent] += 1
        rng = np.random.Generator(np.random.Philox(key=int(key)))
        draws = iter(rng.random, None)
        symbol = xraydb.atomic_symbol(ch["atomic_number"])
        if symbol not in cache:
            cache[symbol] = load_eadl_relaxation(symbol)
        cascade = sample_relaxation_cascade(
            cache[symbol], _DESIGNATORS[ch["shell"]], cutoff_eV, draws
        )
        position = _host(result["r_mid"])[row] + (
            0.5 * _host(result["L_ang"])[row] * _host(result["v_hat"])[row]
        )
        for ordinal, transition in enumerate(cascade.transitions):
            columns["position_ang"].append(position)
            columns["direction"].append(_isotropic(draws))
            columns["energy_eV"].append(transition.energy_eV)
            columns["parent"].append(parent)
            columns["species"].append(-1 if transition.kind == "photon" else ELECTRON)
            columns["key"].append(daughter_stream_key(key, ordinal))
            released[parent] += transition.energy_eV
    out = {name: np.asarray(values) for name, values in columns.items()}
    for name in ("position_ang", "direction"):
        out[name] = out[name].reshape(-1, 3)
    out["parent"] = out["parent"].astype(np.int64)
    out["species"] = out["species"].astype(np.int8)
    out["key"] = np.asarray(columns["key"], dtype=np.uint64)
    return out, released


def _launch_cascade(stage, rows, seed, *, positron=False):
    """Run explicit charged launches; truncate positrons before harvesting daughters.

    The wrapper changes only generation 0. Subsequent charged-impact
    secondaries are electrons and use the ordinary electron tables.
    Validation: photon-secondary-feedback
    """
    from .annihilation import electron_density_per_ang3
    from .api import simulate_trajectories
    from .secondaries import (
        LaunchState,
        _annihilate_positrons,
        _init_positron_fates,
        _mix,
        _positron_overrides,
        _radiative_keys,
        transport_secondary_cascade,
    )

    n = len(rows["kinetic_eV"])
    seed = stage_seed(seed, int(positron))
    keys = _mix(np.asarray(rows["key"], dtype=np.uint64) ^ _VACANCY_SALT)
    launch = LaunchState(
        rows["position_ang"],
        rows["direction"],
        rows["kinetic_eV"] * 1e-3,
        np.zeros(n),
        np.zeros(n),
        keys,
        _radiative_keys(keys),
    )
    kw = _transport_kwargs(stage.transport, seed, stage.threshold_eV)
    events = dict(
        row=np.arange(n),
        positron_keV=launch.E_keV,
        r_ang=launch.r_ang,
        t_ang=launch.t_ang,
        parent_track=np.asarray(rows["key"], dtype=np.uint64),
        photon_ordinal=np.zeros(n, dtype=np.int64),
    )
    _init_positron_fates(events, -1.0)  # every row in this batch is transported
    first = True

    def simulate(**call):
        nonlocal first
        if not positron or not first:
            return simulate_trajectories(**call)
        first = False
        layers = stage.transport["layers"]
        call.update(_positron_overrides(kw, layers))
        call["_secondary"] = replace(call["_secondary"], projectile="positron")
        result = simulate_trajectories(**call)
        return _annihilate_positrons(result, events, seed, electron_density_per_ang3(layers))

    return transport_secondary_cascade(simulate, kw, launch=launch), events


def _annihilation_return(events, keys):
    """Emission rows and escaped-rest partition of transported positrons.

    Validation: photon-secondary-feedback
    """
    from .annihilation import ANNIHILATION_AT_REST
    from .secondaries import FATE_ESCAPED, FATE_IN_FLIGHT

    escaped = events["positron_fate"] == FATE_ESCAPED
    index = np.flatnonzero(~escaped)
    kinds = np.where(
        events["positron_fate"][index] == FATE_IN_FLIGHT,
        ANNIHILATION_IN_FLIGHT,
        ANNIHILATION_AT_REST,
    )
    photons = annihilation_emissions(
        kinds,
        events["annihilation_keV"][index],
        events["annihilation_v_hat"][index],
        events["annihilation_r_ang"][index],
        keys[index] ^ _ANNIHILATION_SALT,
    )
    photons["parent"] = index[photons["parent"]]
    return photons, escaped


def _charged_rows(rows):
    """Validate the public charged-stage phase-space and explicit stream boundary."""
    energy = np.asarray(rows["kinetic_eV"], dtype=float)
    if energy.ndim != 1 or not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError("charged kinetic energies must be finite, nonnegative and one-dimensional")
    n = energy.size
    position, direction = (
        np.asarray(rows[name], dtype=float) for name in ("position_ang", "direction")
    )
    if position.shape != (n, 3) or direction.shape != (n, 3):
        raise ValueError("charged positions and directions must have shape (N, 3)")
    if not np.isfinite(position).all() or not np.isfinite(direction).all():
        raise ValueError("charged positions and directions must be finite")
    if not np.allclose(np.linalg.norm(direction, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("charged directions must be unit vectors")
    keys = np.asarray(rows["key"])
    if keys.shape != (n,) or not np.issubdtype(keys.dtype, np.integer) or np.any(keys < 0):
        raise ValueError("charged keys must be nonnegative integers matching the population")
    if np.unique(keys).size != n:
        raise ValueError("charged keys must be unique")
    species = np.asarray(rows.get("species", np.full(n, ELECTRON)))
    if species.shape != (n,) or not np.isin(species, [ELECTRON, POSITRON]).all():
        raise ValueError("charged species must be electron or positron")
    return dict(
        position_ang=position,
        direction=direction,
        kinetic_eV=energy,
        key=keys.astype(np.uint64),
        species=species,
    )


@dataclass(frozen=True)
class FeedbackChargedStage:
    """Photon-born electron/positron cascade with vacancy and annihilation return.

    A launched positron owns its pair's 2mc² reservoir; escape retains it,
    otherwise two unattenuated annihilation photons return it to the queue.
    Auger daughters above threshold re-enter charged transport; smaller ones
    deposit. Binding remainders retain signed tabulation defects. Geometry is
    the base stage's contiguous stack; filter-born charged transport is not
    represented. Existing shell/annihilation approximations are inherited.
    Validation: photon-secondary-feedback
    """

    base: CascadeChargedStage
    photon_cutoff_eV: float
    max_relaxation_rounds: int = 64
    transports_positrons = True
    requires_stack_only = True

    def __post_init__(self):
        if not np.isfinite(self.photon_cutoff_eV) or self.photon_cutoff_eV <= 0:
            raise ValueError("photon_cutoff_eV must be finite and positive")
        if type(self.max_relaxation_rounds) is not int or self.max_relaxation_rounds < 1:
            raise ValueError("max_relaxation_rounds must be a positive integer")
        if getattr(self.base, "transport", {}).get("elastic_model") == "mott":
            raise ValueError("positron feedback requires sr or elsepa elastic transport")

    @property
    def threshold_eV(self):
        return self.base.threshold_eV

    def __call__(self, rows, seed):
        """Close each input's kinetic plus owned rest energy across charged rounds.

        Validation: photon-secondary-feedback
        """
        pending = _charged_rows(rows)
        n = pending["kinetic_eV"].size
        rest = np.where(pending["species"] == POSITRON, _REST_EV, 0.0)
        owners, totals, emitted = np.arange(n), np.zeros((4, n)), []
        low = (pending["species"] == POSITRON) & (pending["kinetic_eV"] <= self.threshold_eV)
        if low.any():
            from .annihilation import ANNIHILATION_AT_REST

            index = np.flatnonzero(low)
            photons = annihilation_emissions(
                np.full(index.size, ANNIHILATION_AT_REST),
                np.zeros(index.size),
                pending["direction"][low],
                pending["position_ang"][low],
                pending["key"][low] ^ _ANNIHILATION_SALT,
            )
            photons["parent"] = index[photons["parent"]]
            emitted.append(photons)
            totals[1, low], totals[3, low] = pending["kinetic_eV"][low], _REST_EV
            pending = {name: value[~low] for name, value in pending.items()}
            owners = owners[~low]
        for round_index in range(self.max_relaxation_rounds):
            next_parts, next_owners = [], []
            for species in (ELECTRON, POSITRON):
                select = np.flatnonzero(pending["species"] == species)
                if not select.size:
                    continue
                part = {name: value[select] for name, value in pending.items()}
                owner = owners[select]
                values, photons, auger, parents = _feedback_batch(
                    self, part, stage_seed(seed, round_index)
                )
                emitted.append({**photons, "parent": owner[photons["parent"]]})
                next_parts.append(auger)
                next_owners.append(owner[parents])
                for term in range(4):
                    np.add.at(totals[term], owner, values[term])
            if not next_owners or not sum(len(v) for v in next_owners):
                return ChargedStageResult(*totals, _join_emissions(emitted), rest)
            pending = {
                name: np.concatenate([p[name] for p in next_parts], axis=0) for name in pending
            }
            owners = np.concatenate(next_owners)
        raise RuntimeError("charged relaxation feedback exceeded max_relaxation_rounds")


def _join_emissions(parts):
    """Stable per-parent emission ordinals across charged and relaxation rounds."""
    empty = dict(
        position_ang=np.empty((0, 3)),
        direction=np.empty((0, 3)),
        energy_eV=np.empty(0),
        parent=np.empty(0, dtype=np.int64),
    )
    photons = {
        name: np.concatenate([p[name] for p in parts], axis=0) if parts else value
        for name, value in empty.items()
    }
    order = np.argsort(photons["parent"], kind="stable")
    photons = {name: value[order] for name, value in photons.items()}
    parent = photons["parent"]
    first = np.r_[True, parent[1:] != parent[:-1]] if parent.size else np.empty(0, dtype=bool)
    start = np.maximum.accumulate(np.where(first, np.arange(parent.size), 0))
    photons["ordinal"] = np.arange(parent.size) - start
    return photons


def _feedback_batch(stage, part, seed):
    """One species' shell cascade, annihilation return and vacancy-energy split.

    Validation: photon-secondary-feedback
    """
    from .secondaries import secondary_energy_balance

    n = len(part["kinetic_eV"])
    positron = part["species"][0] == POSITRON
    result, events = _launch_cascade(stage.base, part, seed, positron=positron)
    terms = secondary_energy_balance(result, per_history=True)
    values = np.array(
        [
            terms[name] * 1e3
            for name in ("escaped_keV", "deposited_keV", "binding_reserved_keV", "radiated_keV")
        ]
    )
    emitted = [_hard_photons(result)]
    if positron:
        ann, escaped = _annihilation_return(events, part["key"])
        values[0] += np.where(escaped, _REST_EV, 0.0)
        values[3] += np.bincount(ann["parent"], weights=ann["energy_eV"], minlength=n)
        emitted.append(ann)
    particles, released = impact_relaxation_emissions(
        result, part["key"], min(stage.photon_cutoff_eV, stage.threshold_eV)
    )
    values[2] -= released
    light = particles["species"] == -1
    values[3] += np.bincount(
        particles["parent"][light], weights=particles["energy_eV"][light], minlength=n
    )
    emitted.append(
        {
            name: particles[name][light]
            for name in ("position_ang", "direction", "energy_eV", "parent")
        }
    )
    low = ~light & (particles["energy_eV"] <= stage.threshold_eV)
    values[1] += np.bincount(
        particles["parent"][low], weights=particles["energy_eV"][low], minlength=n
    )
    keep = ~light & ~low
    auger = dict(
        position_ang=particles["position_ang"][keep],
        direction=particles["direction"][keep],
        kinetic_eV=particles["energy_eV"][keep],
        key=particles["key"][keep],
        species=np.full(np.count_nonzero(keep), ELECTRON),
    )
    return values, _join_emissions(emitted), auger, particles["parent"][keep]


def cascade_feedback_stage(layers, *, photon_cutoff_eV, **transport):
    """Build the CPU coupled feedback stage on the photon navigator's stack.

    Validation: photon-secondary-feedback
    """
    if not layers:
        raise ValueError("charged feedback requires a nonempty contiguous stack from z=0")
    previous = 0.0
    for top, bottom, composition in layers:
        if (
            not np.isfinite([top, bottom]).all()
            or top != previous
            or bottom <= top
            or not composition
        ):
            raise ValueError("charged feedback requires a contiguous material stack from z=0")
        previous = bottom
    return FeedbackChargedStage(cascade_charged_stage(layers, **transport), photon_cutoff_eV)
