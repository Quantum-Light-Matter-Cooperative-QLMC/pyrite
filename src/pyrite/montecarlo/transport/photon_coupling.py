"""Host reference coupling of photon histories and their charged descendants (#341).

Alternates the photon queue (``photon_histories``) with a charged stage that
transports photon-born electrons and returns their hard bremsstrahlung
photons, until no launch remains. :func:`cascade_charged_stage` runs the
existing shell soft/hard cascade from an explicit launch. Electron-only
stages retain pending positrons; ``photon_feedback`` supplies the optional
electron/positron, vacancy and annihilation stage. No production caller,
identity or device path. Coordinates are sample-frame Å, energies eV.
"""

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum
from typing import Any, Protocol

import numpy as np

from .pair_production import _mix
from .photon_histories import (
    ELECTRON,
    POSITRON,
    PhotonHistories,
    daughter_stream_key,
    track_photon_histories,
)

_STAGE_SALT = np.uint64(0x1F83D9ABFB41BD6B)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_ANG_PER_MM = 1.0e7

#: Photon-queue ledger terms that end there; ``charged_kinetic_eV`` is split below.
_PHOTON_TERMS = (
    "escaped_eV",
    "detected_eV",
    "deposited_eV",
    "terminal_binding_eV",
    "transition_defect_eV",
    "positron_rest_eV",
    "unresolved_eV",
)
#: Where launched or held charged kinetic energy ends.
_CHARGED_TERMS = (
    "charged_escaped_eV",
    "charged_deposited_eV",
    "charged_binding_eV",
    "charged_local_eV",
    "radiated_local_eV",
    "positron_pending_eV",
    "charged_unresolved_eV",
)
LEDGER_TERMS = _PHOTON_TERMS + _CHARGED_TERMS


@dataclass(frozen=True, slots=True)
class ChargedStageResult:
    """Per launched row energy split [eV] and the photons the rows emitted.

    ``escaped + deposited + binding + radiated`` equals each row's launch
    kinetic energy plus ``rest_energy_eV`` when a positron-capable stage
    releases the pair reservoir. Escaped energy then includes escaped rest
    energy. Binding retains terminal holes and signed table-energy defects.
    ``photons`` holds ``position_ang (M, 3)``, ``direction``,
    ``energy_eV``, ``parent`` (launched-row index) and ``ordinal`` (that
    row's emission order, independent of batch composition); their energies
    sum to ``radiated_eV`` per parent.
    """

    escaped_eV: np.ndarray
    deposited_eV: np.ndarray
    binding_eV: np.ndarray
    radiated_eV: np.ndarray
    photons: Mapping[str, np.ndarray]
    rest_energy_eV: np.ndarray | None = None


class ChargedStage(Protocol):
    """Transport charged rows; ``threshold_eV`` is the electron launch boundary.

    Optional ``transports_positrons=True`` also accepts in-stack positrons at
    every kinetic energy and owns the returned ``rest_energy_eV`` reservoir.
    """

    threshold_eV: float

    def __call__(self, rows: Mapping[str, np.ndarray], seed: int) -> ChargedStageResult: ...


@dataclass(frozen=True, slots=True)
class CoupledHistories:
    """All rounds of one coupled batch, plus per-history ledger closure.

    ``exits``, ``charged`` and ``interactions`` are the photon-queue tables
    concatenated over rounds with a ``round`` column; ``charged["fate"]`` is
    ``launched``, ``local`` (outside the transport stack), ``positron`` or
    ``unresolved``. ``ledger`` holds :data:`LEDGER_TERMS` and ``incident_eV``
    per sorted unique ``history_id``.
    """

    exits: Mapping[str, np.ndarray]
    charged: Mapping[str, np.ndarray]
    interactions: Mapping[str, np.ndarray]
    ledger: Mapping[str, np.ndarray]
    rounds: int


def stage_seed(seed: int, round_index: int) -> int:
    """Charged-stage seed of one coupling round, under its own salt.

    Validation: photon-charged-coupling
    """
    with np.errstate(over="ignore"):
        x = np.uint64(seed) ^ _STAGE_SALT
        return int(_mix(x + _GOLDEN * (np.uint64(round_index) + np.uint64(1))))


def _in_stack(position, layers, width_ang, height_ang) -> np.ndarray:
    """Points inside a non-empty stack layer and the common footprint."""
    z = position[:, 2]
    inside = np.zeros(z.size, dtype=bool)
    for top, bottom, composition in layers:
        if composition:
            inside |= (z >= top) & (z < bottom)
    for axis, size in ((0, width_ang), (1, height_ang)):
        if size is not None:
            inside &= np.abs(position[:, axis]) < size / 2
    return inside


class _Ledger:
    """Per-history term lists, summed with compensated summation at the end."""

    def __init__(self, histories: np.ndarray, incident: np.ndarray) -> None:
        self.histories = histories
        self.incident = incident
        self.terms = {name: [[] for _ in histories] for name in LEDGER_TERMS}

    def add(self, name: str, history_id, values) -> None:
        rows = np.searchsorted(self.histories, np.asarray(history_id))
        for row, value in zip(np.atleast_1d(rows), np.atleast_1d(values), strict=True):
            self.terms[name][int(row)].append(float(value))

    def photon_round(self, histories: PhotonHistories) -> None:
        ledger = histories.ledger
        for name in _PHOTON_TERMS:
            self.add(name, ledger["history_id"], ledger[name])

    def finish(self) -> dict[str, np.ndarray]:
        out = {"history_id": self.histories, "incident_eV": self.incident}
        for name in LEDGER_TERMS:
            out[name] = np.array([fsum(values) for values in self.terms[name]])
        unresolved = out["unresolved_eV"] + out["charged_unresolved_eV"]
        out["complete"] = unresolved == 0.0
        return out


def coupled_ledger_residual_eV(ledger: Mapping[str, np.ndarray]) -> np.ndarray:
    """Incident minus every coupled terminal term; zero up to roundoff.

    Validation: photon-charged-coupling
    """
    return ledger["incident_eV"] - sum(ledger[name] for name in LEDGER_TERMS)


def _split_charged(charged, threshold_eV, layers, footprint, *, positrons=False):
    """Fates of one round's charged rows: launch, local, positron.

    Validation: photon-charged-coupling
    """
    position = charged["position_ang"].reshape(-1, 3)
    electron = charged["species"] == ELECTRON
    above = charged["kinetic_eV"] > threshold_eV
    inside = _in_stack(position, layers, footprint["width_ang"], footprint["height_ang"])
    fate = np.where(electron, np.where(inside & above, "launched", "local"), "positron")
    if positrons:
        fate[~electron & inside] = "launched"
    return fate.astype("<U10")


def _book_stage(ledger, rows, result: ChargedStageResult, cutoff_eV):
    """Book a stage's per-row split; return the returned photon batch.

    Validation: photon-charged-coupling, photon-secondary-feedback
    """
    weight, history = rows["weight"], rows["history_id"]
    if result.rest_energy_eV is not None:
        from .pair_production import PAIR_THRESHOLD_EV

        expected = np.where(rows["species"] == POSITRON, PAIR_THRESHOLD_EV, 0.0)
        if not np.array_equal(result.rest_energy_eV, expected):
            raise ValueError(
                "charged stage must account for exactly one rest reservoir per positron"
            )
        ledger.add("positron_rest_eV", history, -weight * result.rest_energy_eV)
    ledger.add("charged_escaped_eV", history, weight * result.escaped_eV)
    ledger.add("charged_deposited_eV", history, weight * result.deposited_eV)
    ledger.add("charged_binding_eV", history, weight * result.binding_eV)
    photons = result.photons
    parent = np.asarray(photons["parent"], dtype=np.int64)
    energy = np.asarray(photons["energy_eV"], dtype=float)
    soft = energy <= cutoff_eV
    ledger.add("radiated_local_eV", history[parent[soft]], weight[parent[soft]] * energy[soft])
    keep = ~soft
    keys = np.array(
        [
            daughter_stream_key(rows["key"][p], o)
            for p, o in zip(parent[keep], np.asarray(photons["ordinal"])[keep], strict=True)
        ],
        dtype=np.uint64,
    )
    p = parent[keep]
    return {
        "position_ang": np.asarray(photons["position_ang"], dtype=float)[keep].reshape(-1, 3),
        "direction": np.asarray(photons["direction"], dtype=float)[keep].reshape(-1, 3),
        "energy_eV": energy[keep],
        "weight": weight[p],
        "history_id": history[p],
        "source_kind": rows["source_kind"][p],
        "stream_key": keys,
    }


def _concat(tables: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate([t[name] for t in tables]) for name in tables[0]}


def _tagged(table: Mapping[str, np.ndarray], round_index: int) -> dict[str, np.ndarray]:
    size = next(iter(table.values())).shape[0]
    return dict(table, round=np.full(size, round_index, dtype=np.int64))


def track_coupled_histories(
    position_ang,
    direction,
    energy_eV,
    weight,
    history_id,
    event_id,
    layers: Sequence,
    *,
    seed: int,
    photon_cutoff_eV: float,
    charged_stage: ChargedStage,
    width_ang=None,
    height_ang=None,
    max_rounds: int = 64,
    **photon_options,
) -> CoupledHistories:
    """Photon histories whose photon-born electrons radiate back into the queue.

    Round 0 tracks the source photons with ``electron_cutoff_eV`` set to the
    stage's ``threshold_eV``, so electrons at or below it deposit in the
    photon queue. Each later round hands the previous round's electrons above
    the threshold and inside a non-empty stack layer to ``charged_stage``;
    electrons born outside it (filter boxes) deposit locally. Electron-only
    stages keep positrons pending. A stage with ``transports_positrons=True``
    accepts in-stack positrons even below threshold and replaces their
    reserved rest energy by returned annihilation photons or escaped rest
    energy. Returned photons above
    ``photon_cutoff_eV`` restart in the photon queue with key
    ``daughter_stream_key(charged key, ordinal)``, their parent's weight,
    history and root ``source_kind``; softer ones deposit at their vertex.
    After ``max_rounds`` stages any remaining launch is unresolved.

    Per history: incident = photon terminal terms + charged escaped,
    deposited, binding, local, radiated-local, positron-pending and
    unresolved, to roundoff (:func:`coupled_ledger_residual_eV`).
    Source: PENELOPE-2024 Chapter 7 §7.1 p.283 secondary stack (scheduling order does not
    change physical means); the stage's own transport claims.
    Validation: photon-charged-coupling
    """
    if not isinstance(max_rounds, int) or max_rounds < 0:
        raise ValueError("max_rounds must be a non-negative integer")
    if "electron_cutoff_eV" in photon_options:
        raise ValueError("electron_cutoff_eV is the charged stage's threshold_eV")
    if getattr(charged_stage, "requires_stack_only", False) and photon_options.get("boxes"):
        raise ValueError("this charged feedback stage does not support filter boxes")
    threshold = float(charged_stage.threshold_eV)
    footprint = {"width_ang": width_ang, "height_ang": height_ang}
    options: dict[str, Any] = dict(
        photon_options, seed=seed, photon_cutoff_eV=photon_cutoff_eV,
        electron_cutoff_eV=threshold, **footprint,
    )  # fmt: skip
    batch = track_photon_histories(
        position_ang, direction, energy_eV, weight, history_id, event_id, layers, **options
    )
    ledger = _Ledger(batch.ledger["history_id"], batch.ledger["incident_eV"])
    tables = {"exits": [], "charged": [], "interactions": []}
    round_index = 0
    while True:
        ledger.photon_round(batch)
        charged = dict(batch.charged)
        fate = _split_charged(
            charged,
            threshold,
            layers,
            footprint,
            positrons=getattr(charged_stage, "transports_positrons", False),
        )
        launch = fate == "launched"
        if launch.any() and round_index >= max_rounds:
            fate[launch] = "unresolved"
            launch[:] = False
        energy = charged["weight"] * charged["kinetic_eV"]
        for name, label in (
            ("charged_local_eV", "local"),
            ("positron_pending_eV", "positron"),
            ("charged_unresolved_eV", "unresolved"),
        ):
            mask = fate == label
            ledger.add(name, charged["history_id"][mask], energy[mask])
        for name in ("exits", "interactions"):
            tables[name].append(_tagged(getattr(batch, name), round_index))
        tables["charged"].append(_tagged(dict(charged, fate=fate), round_index))
        if not launch.any():
            break
        rows = {name: value[launch] for name, value in charged.items()}
        result = charged_stage(rows, stage_seed(seed, round_index))
        returned = _book_stage(ledger, rows, result, photon_cutoff_eV)
        round_index += 1
        n = returned["energy_eV"].size
        batch = track_photon_histories(
            returned["position_ang"], returned["direction"], returned["energy_eV"],
            returned["weight"], returned["history_id"], np.arange(n), layers,
            **dict(options, source_kind=returned["source_kind"],
                   stream_key=returned["stream_key"]),
        )  # fmt: skip
    return CoupledHistories(
        _concat(tables["exits"]),
        _concat(tables["charged"]),
        _concat(tables["interactions"]),
        ledger.finish(),
        round_index,
    )


@dataclass(frozen=True)
class CascadeChargedStage:
    """Shell soft/hard cascade from explicit photon-born electron launches.

    ``transport`` holds :func:`simulate_trajectories` keywords other than
    the beam, ``Ne``, ``E0_keV`` and seed; it must select
    ``inelastic_model='shell-soft-hard'`` and the coupled
    ``radiative_model='bremslib-soft-hard'`` so radiative losses are returned
    as photons, and leaves pair production and positrons to the photon queue.
    """

    transport: Mapping
    threshold_eV: float

    def __call__(self, rows: Mapping[str, np.ndarray], seed: int) -> ChargedStageResult:
        """Validation: photon-charged-coupling"""
        from .api import simulate_trajectories
        from .secondaries import LaunchState, secondary_energy_balance, transport_secondary_cascade
        from .secondaries import _radiative_keys as radiative_keys

        n = int(rows["kinetic_eV"].size)
        with np.errstate(over="ignore"):
            keys = _mix(np.asarray(rows["key"], dtype=np.uint64) ^ _STAGE_SALT)
        launch = LaunchState(
            np.asarray(rows["position_ang"], dtype=float).reshape(n, 3),
            np.asarray(rows["direction"], dtype=float).reshape(n, 3),
            np.asarray(rows["kinetic_eV"], dtype=float) * 1e-3,
            np.zeros(n),
            np.zeros(n),
            keys,
            radiative_keys(keys),
        )
        kw = _transport_kwargs(self.transport, seed, self.threshold_eV)
        result = transport_secondary_cascade(simulate_trajectories, kw, launch=launch)
        terms = secondary_energy_balance(result, per_history=True)
        photons = _hard_photons(result)
        return ChargedStageResult(
            terms["escaped_keV"] * 1e3,
            terms["deposited_keV"] * 1e3,
            terms["binding_reserved_keV"] * 1e3,
            terms["radiated_keV"] * 1e3,
            photons,
        )


def cascade_charged_stage(layers: Sequence, *, width_ang=None, height_ang=None, **transport):
    """Build a :class:`CascadeChargedStage` on the photon queue's own stack.

    ``layers`` must be the contiguous transport stack starting at z=0 (the
    photon navigator's gaps are not transportable); the footprint converts
    from Å to the transport's millimetres.

    Validation: photon-charged-coupling
    """
    for name in ("pair_production_model", "positron_transport", "seed", "Ne", "E0_keV"):
        if name in transport:
            raise ValueError(f"{name} is owned by the photon coupling, not the charged stage")
    if transport.get("radiative_model") != "bremslib-soft-hard":
        raise ValueError("the charged stage needs radiative_model='bremslib-soft-hard'")
    threshold = transport.get("secondary_threshold_eV")
    if threshold is None:
        raise ValueError("the charged stage needs secondary_threshold_eV")
    footprint = {
        "crystal_width_mm": None if width_ang is None else width_ang / _ANG_PER_MM,
        "crystal_height_mm": None if height_ang is None else height_ang / _ANG_PER_MM,
    }
    stack = dict(transport, layers=list(layers), thickness_ang=float(layers[-1][1]), **footprint)
    return CascadeChargedStage(stack, float(threshold))


def _transport_kwargs(transport: Mapping, seed: int, threshold_eV: float) -> dict:
    """Full ``simulate_trajectories`` keyword mapping for one stage call."""
    from .api import simulate_trajectories

    signature = inspect.signature(simulate_trajectories)
    kw = {
        name: p.default
        for name, p in signature.parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    kw.update(transport, seed=int(seed), Ne=1, E0_keV=threshold_eV * 1e-3)
    return kw


def _hard_photons(result) -> dict[str, np.ndarray]:
    """Hard radiative events at their row endpoints, owner and emission ordinal.

    Validation: photon-charged-coupling
    """
    from .secondaries import _host

    energy = _host(result["hard_radiative_k_eV"])
    rows = np.flatnonzero(energy > 0.0)
    owner = _host(result["electron_id"])[rows].astype(np.int64)
    track = _host(result["track_id"])[rows]
    order = np.lexsort(
        (_host(result["substep_id"])[rows], _host(result["flight_id"])[rows], track, owner)
    )
    rows, owner = rows[order], owner[order]
    first = np.ones(rows.size, dtype=bool)
    first[1:] = owner[1:] != owner[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(rows.size), 0))
    length = _host(result["L_ang"])[rows]
    position = _host(result["r_mid"])[rows] + 0.5 * length[:, None] * _host(result["v_hat"])[rows]
    return {
        "position_ang": position.reshape(-1, 3),
        "direction": _host(result["hard_radiative_direction"])[rows].reshape(-1, 3),
        "energy_eV": energy[rows],
        "parent": owner,
        "ordinal": np.arange(rows.size) - start,
    }
