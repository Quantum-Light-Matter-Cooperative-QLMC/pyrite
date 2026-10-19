"""Host reference photon histories with ordinary detector hits (#341).

Chains the standalone first-flight navigator and interaction samplers into a
lineage-keyed photon queue. Charged daughters are returned as handoff rows for
the charged launch boundary; they are not transported here. Filter plates and
detectors enter as sample-frame boxes and apertures converted from lab poses.
No production caller: source adapters, identity and device execution are
later slices. Coordinates are sample-frame Å, energies eV.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from math import fsum
from typing import NamedTuple

import numpy as np

from ..._planar_geometry import PlanarPose
from ...materials import CATALOG, MediumSpec
from ...materials._composition import resolve_composition
from ...materials.photon_cross_sections import EPDL_E_MAX_EV, EPDL_E_MIN_EV, _atomic_number
from ..eadl_relaxation import EADLRelaxation, load_eadl_relaxation
from .pair_production import PAIR_THRESHOLD_EV, _mix, sample_pair
from .photoelectric import sample_photoelectric
from .photon_navigation import (
    PHOTON_CHANNEL_ORDER,
    PhotonBox,
    scene_first_interactions,
    validate_photon_regions,
)
from .photon_scattering import sample_compton, sample_rayleigh

ELECTRON, POSITRON = 0, 1
#: Charged-row ``channel`` for relaxation (Auger/Coster–Kronig) electrons.
RELAXATION_CHANNEL = len(PHOTON_CHANNEL_ORDER)

_PHOTON_HISTORY_SALT = np.uint64(0x5BE0CD19137E2179)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_ANG_PER_MM = 1.0e7
_ONE = np.uint64(1)

_LEDGER_TERMS = (
    "escaped_eV",
    "detected_eV",
    "deposited_eV",
    "terminal_binding_eV",
    "transition_defect_eV",
    "charged_kinetic_eV",
    "positron_rest_eV",
    "unresolved_eV",
)


@dataclass(frozen=True, slots=True)
class PhotonAperture:
    """Absorbing rectangular aperture outside modeled material, sample frame [Å].

    ``x_axis`` and ``y_axis`` are orthonormal; the accepting face looks along
    ``normal = x_axis × y_axis``, so a hit requires ``direction · normal > 0``.
    The closed rectangle is ``|u| <= half_width_ang``, ``|v| <= half_height_ang``.
    The tracker refuses an aperture that enters any modeled body.
    """

    center_ang: tuple[float, float, float]
    x_axis: tuple[float, float, float]
    y_axis: tuple[float, float, float]
    half_width_ang: float
    half_height_ang: float

    def __post_init__(self) -> None:
        vectors = np.array([self.center_ang, self.x_axis, self.y_axis], dtype=float)
        if vectors.shape != (3, 3) or not np.isfinite(vectors).all():
            raise ValueError("aperture center and axes must be finite 3-vectors")
        basis = vectors[1:]
        if not np.allclose(basis @ basis.T, np.eye(2), rtol=0.0, atol=1e-12):
            raise ValueError("aperture axes must be orthonormal")
        sizes = np.array([self.half_width_ang, self.half_height_ang], dtype=float)
        if not np.isfinite(sizes).all() or np.any(sizes <= 0.0):
            raise ValueError("aperture half sizes must be finite and positive")

    @classmethod
    def from_pose(cls, pose: PlanarPose, size_mm, sample_to_lab) -> "PhotonAperture":
        """Lab-frame detector pose [mm] as a sample-frame aperture [Å].

        ``sample_to_lab`` is ``R`` with ``v_lab = R v_sample`` (``geometry.sample_to_lab_R``);
        the lab origin is the sample entrance origin, so positions rotate
        without translation. The pose's downstream ``+z`` becomes ``normal``.
        """
        rotation = _rotation(sample_to_lab)
        width, height = (float(v) for v in size_mm)
        return cls(
            tuple(rotation.T @ np.asarray(pose.center_mm) * _ANG_PER_MM),
            tuple(rotation.T @ np.asarray(pose.x_axis)),
            tuple(rotation.T @ np.asarray(pose.y_axis)),
            width * _ANG_PER_MM / 2,
            height * _ANG_PER_MM / 2,
        )

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.x_axis, self.y_axis)

    def region(self):
        """Zero-thickness rectangle for :func:`validate_photon_regions`."""
        axes = np.array([self.x_axis, self.y_axis, self.normal])
        half = np.array([self.half_width_ang, self.half_height_ang, 0.0])
        return np.asarray(self.center_ang, dtype=float), axes, half

    def intersect(self, origins: np.ndarray, directions: np.ndarray):
        """Forward hit distance [Å] (inf on a miss) and local ``(u, v)`` [Å]."""
        normal = self.normal
        center = np.asarray(self.center_ang, dtype=float)
        facing = directions @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = np.where(facing > 0.0, ((center - origins) @ normal) / facing, np.inf)
        distance = np.where(distance > 0.0, distance, np.inf)
        finite = np.isfinite(distance)
        local = np.full((origins.shape[0], 2), np.nan)
        offset = origins[finite] + distance[finite, None] * directions[finite] - center
        local[finite] = offset @ np.array([self.x_axis, self.y_axis]).T
        inside = (np.abs(local[:, 0]) <= self.half_width_ang) & (
            np.abs(local[:, 1]) <= self.half_height_ang
        )
        return np.where(inside, distance, np.inf), np.where(inside[:, None], local, np.nan)


def _rotation(sample_to_lab) -> np.ndarray:
    rotation = np.asarray(sample_to_lab, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(
        rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12
    ):
        raise ValueError("sample_to_lab must be a 3x3 rotation matrix")
    return rotation


def photon_box_from_pose(
    pose: PlanarPose, size_mm, thickness_mm: float, material: str | MediumSpec, sample_to_lab
) -> PhotonBox:
    """Lab-frame finite plate (e.g. ``FilterPlate`` fields) as a sample-frame box.

    Composition resolves through the material catalog [atoms/Å³]; sizes are
    full widths [mm]. Rotation convention as :meth:`PhotonAperture.from_pose`.
    Validation: photon-scene-navigation
    """
    rotation = _rotation(sample_to_lab)
    width, height = (float(v) for v in size_mm)
    axes = np.array([pose.x_axis, pose.y_axis, pose.normal], dtype=float) @ rotation
    return PhotonBox(
        tuple(rotation.T @ np.asarray(pose.center_mm) * _ANG_PER_MM),
        tuple(tuple(axis) for axis in axes),
        (width * _ANG_PER_MM / 2, height * _ANG_PER_MM / 2, float(thickness_mm) * _ANG_PER_MM / 2),
        tuple(resolve_composition(material, catalog=CATALOG)),
    )


@dataclass(frozen=True, slots=True)
class PhotonHistories:
    """Terminal records of one batch; each mapping holds equal-length arrays.

    ``exits``: escaped photons at their last interaction point (or source)
    with ``detected`` and aperture ``local_ang``. ``charged``: electron and
    positron handoff rows with lineage ``key`` and producing ``channel``.
    ``interactions``: every sampled collision. ``ledger``: weighted energy
    terms per sorted unique ``history_id``; ``escaped_eV`` excludes
    ``detected_eV``. See :func:`ledger_residual_eV`.
    """

    exits: Mapping[str, np.ndarray]
    charged: Mapping[str, np.ndarray]
    interactions: Mapping[str, np.ndarray]
    ledger: Mapping[str, np.ndarray]


def photon_stream_key(seed, history_id, event_id):
    """Root photon stream key from (seed, primary history, source event).

    Independent of batch position, queue order and the electron/pair salts.
    Validation: photon-history-queue
    """
    with np.errstate(over="ignore"):
        x = np.uint64(seed) ^ _PHOTON_HISTORY_SALT
        x = _mix(x + _GOLDEN * (np.asarray(history_id, dtype=np.uint64) + _ONE))
        return _mix(x + _GOLDEN * (np.asarray(event_id, dtype=np.uint64) + _ONE))


def daughter_stream_key(parent_key, ordinal):
    """Key of an interaction's ``ordinal``-th emitted particle (photon or charged).

    Validation: photon-history-queue
    """
    with np.errstate(over="ignore"):
        return _mix(np.uint64(parent_key) + _GOLDEN * (np.uint64(ordinal) + _ONE))


class _Photon(NamedTuple):
    """One pending flight; ``row`` indexes the sorted unique history ids."""

    position: np.ndarray
    direction: np.ndarray
    energy: float
    row: int
    weight: float
    key: np.uint64
    kind: int
    generation: int


@dataclass(slots=True)
class _Queue:
    """Pending photons plus terminal record lists for one batch."""

    histories: np.ndarray
    photon_cutoff_eV: float
    electron_cutoff_eV: float
    pending: list[_Photon] = field(default_factory=list)
    exits: list[tuple] = field(default_factory=list)
    charged: list[tuple] = field(default_factory=list)
    interactions: list[tuple] = field(default_factory=list)
    terms: dict[str, list[list[float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.terms = {name: [[] for _ in self.histories] for name in _LEDGER_TERMS}

    def book(self, name: str, row: int, energy_eV: float) -> None:
        self.terms[name][row].append(energy_eV)

    def photon(self, parent: _Photon, ordinal: int, position, direction, energy_eV) -> None:
        """Queue a daughter photon, or deposit it at or below the photon cutoff."""
        if energy_eV <= self.photon_cutoff_eV:
            self.book("deposited_eV", parent.row, parent.weight * energy_eV)
            return
        self.pending.append(
            parent._replace(
                position=position,
                direction=np.asarray(direction, dtype=float),
                energy=float(energy_eV),
                key=daughter_stream_key(parent.key, ordinal),
                generation=parent.generation + 1,
            )
        )

    def particle(self, parent: _Photon, ordinal, species, channel, position, direction, kinetic_eV):
        """Hand a charged daughter off; deposit sub-cutoff electrons locally."""
        energy = parent.weight * kinetic_eV
        if species == ELECTRON and kinetic_eV <= self.electron_cutoff_eV:
            self.book("deposited_eV", parent.row, energy)
            return
        self.book("charged_kinetic_eV", parent.row, energy)
        self.charged.append(
            (species, channel, position, direction, kinetic_eV, parent.weight, parent.row,
             daughter_stream_key(parent.key, ordinal), parent.kind, parent.generation + 1)
        )  # fmt: skip


def _isotropic(draws: Iterator[float]) -> np.ndarray:
    """Uniform direction on the sphere from two uniforms: cosθ=1-2u, φ=2πu."""
    cosine = 1.0 - 2.0 * next(draws)
    azimuth = 2.0 * np.pi * next(draws)
    sine = np.sqrt(max(0.0, 1.0 - cosine * cosine))
    return np.array([sine * np.cos(azimuth), sine * np.sin(azimuth), cosine])


def _photoabsorb(queue, parent: _Photon, channel, element, relaxation, rng, position) -> None:
    """Photoelectron, isotropic relaxation particles and binding remainder."""
    cutoff = min(queue.photon_cutoff_eV, queue.electron_cutoff_eV)
    draws = iter(rng.random, None)
    event = sample_photoelectric(
        element, parent.energy, parent.direction, relaxation, cutoff, draws
    )
    queue.particle(
        parent, 0, ELECTRON, channel, position, event.electron_direction,
        event.electron_energy_eV,
    )  # fmt: skip
    for ordinal, transition in enumerate(event.relaxation.transitions, start=1):
        emitted = _isotropic(draws)
        if transition.kind == "photon":
            queue.photon(parent, ordinal, position, emitted, transition.energy_eV)
        else:
            queue.particle(
                parent, ordinal, ELECTRON, RELAXATION_CHANNEL, position, emitted,
                transition.energy_eV,
            )  # fmt: skip
    budget = event.relaxation.energy_budget
    queue.book("terminal_binding_eV", parent.row, parent.weight * budget.terminal_binding_eV)
    queue.book(
        "transition_defect_eV", parent.row, parent.weight * budget.transition_energy_defect_eV
    )


def _interact(queue, parent: _Photon, channel, element, relaxation, rng, position) -> None:
    """Dispatch one sampled collision to its conditional final-state sampler.

    Daughter ordinals: scattered photon 0 then recoil 1; photoelectron 0 then
    relaxation particles in cascade order; pair electron 0, positron 1.
    """
    energy, direction = parent.energy, parent.direction
    name = PHOTON_CHANNEL_ORDER[channel]
    if name == "photoelectric":
        _photoabsorb(queue, parent, channel, element, relaxation, rng, position)
    elif name in ("coherent", "incoherent"):
        sampler = sample_rayleigh if name == "coherent" else sample_compton
        event = sampler(element, energy, direction, iter(rng.random, None))
        queue.photon(parent, 0, position, event.photon_direction, event.photon_energy_eV)
        if name == "incoherent":
            queue.particle(
                parent, 1, ELECTRON, channel, position, event.electron_direction,
                event.electron_energy_eV,
            )  # fmt: skip
    else:
        pair = sample_pair(energy, _atomic_number(element), direction, rng)
        queue.book("positron_rest_eV", parent.row, parent.weight * PAIR_THRESHOLD_EV)
        queue.particle(parent, 0, ELECTRON, channel, position, pair.electron_direction,
                       pair.electron_eV)  # fmt: skip
        queue.particle(parent, 1, POSITRON, channel, position, pair.positron_direction,
                       pair.positron_eV)  # fmt: skip


def _validate_batch(position_ang, direction, energy_eV, weight, history_id, event_id):
    position = np.asarray(position_ang, dtype=float)
    direction = np.asarray(direction, dtype=float)
    energy, weight = np.asarray(energy_eV, dtype=float), np.asarray(weight, dtype=float)
    history, event = np.asarray(history_id), np.asarray(event_id)
    n = energy.shape[0] if energy.ndim == 1 else -1
    if position.shape != (n, 3) or direction.shape != (n, 3):
        raise ValueError("positions and directions must have shape (N, 3) matching energies (N,)")
    if any(a.shape != (n,) for a in (weight, history, event)):
        raise ValueError("weights, history and event ids must have shape (N,)")
    if not (np.isfinite(position).all() and np.isfinite(direction).all()):
        raise ValueError("photon positions and directions must be finite")
    if not np.allclose(np.linalg.norm(direction, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("photon directions must be unit vectors")
    if not np.isfinite(weight).all() or np.any(weight <= 0.0):
        raise ValueError("photon weights must be finite and positive")
    if not np.isfinite(energy).all() or np.any(energy > EPDL_E_MAX_EV):
        raise ValueError("photon energy is outside the EPDL2025 attenuation domain")
    return position, direction, energy, weight, *_validate_ids(history, event)


def _validate_ids(history: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative integer ids whose (history, event) pairs key unique streams."""
    if history.size and not (
        np.issubdtype(history.dtype, np.integer) and np.issubdtype(event.dtype, np.integer)
    ):
        raise ValueError("history and event ids must be integers")
    if history.size and (history.min() < 0 or event.min() < 0):
        raise ValueError("history and event ids must be non-negative")
    if len({(int(h), int(e)) for h, e in zip(history, event, strict=True)}) != history.size:
        raise ValueError("(history_id, event_id) pairs must be unique")
    return history.astype(np.int64), event.astype(np.int64)


def _validate_options(energy, photon_cutoff_eV, electron_cutoff_eV, max_generations, source_kind):
    """Cutoffs, generation cap and integer source kinds; returns the kinds."""
    if not np.isfinite(photon_cutoff_eV) or photon_cutoff_eV < EPDL_E_MIN_EV:
        raise ValueError("photon cutoff must be finite and at least the EPDL minimum (1 eV)")
    if not np.isfinite(electron_cutoff_eV) or electron_cutoff_eV <= 0.0:
        raise ValueError("electron cutoff must be finite and positive")
    if np.any(energy <= photon_cutoff_eV):
        raise ValueError("source photon energies must exceed the photon cutoff")
    if not isinstance(max_generations, int) or max_generations < 1:
        raise ValueError("max_generations must be a positive integer")
    kinds = np.zeros(energy.shape, np.int64) if source_kind is None else np.asarray(source_kind)
    if kinds.shape != energy.shape or not np.issubdtype(kinds.dtype, np.integer):
        raise ValueError("source_kind must be an integer array of shape (N,)")
    return kinds


def _flight(queue, active: list[_Photon], layers, boxes, footprint, aperture, rngs):
    """Advance one generation: record hits/escapes, return collisions and points."""
    origins = np.array([p.position for p in active], dtype=float).reshape(-1, 3)
    directions = np.array([p.direction for p in active], dtype=float).reshape(-1, 3)
    energies = np.array([p.energy for p in active], dtype=float)
    uniforms = np.array([rng.random(2) for rng in rngs]).reshape(-1, 2)
    flight = scene_first_interactions(
        origins, directions, energies, uniforms, layers, boxes, **footprint
    )
    hit_distance = np.full(len(active), np.inf)
    local = np.full((len(active), 2), np.nan)
    if aperture is not None and active:
        hit_distance, local = aperture.intersect(origins, directions)
    detected = hit_distance < flight["distance_ang"]
    collided = ~detected & np.isfinite(flight["distance_ang"])
    for i, p in enumerate(active):
        if collided[i]:
            continue
        queue.book("detected_eV" if detected[i] else "escaped_eV", p.row, p.weight * p.energy)
        queue.exits.append(
            (p.position, p.direction, p.energy, p.weight, p.row, p.key, p.kind, p.generation,
             detected[i], local[i])
        )  # fmt: skip
    step = np.where(collided, flight["distance_ang"], 0.0)
    return flight, collided, origins + step[:, None] * directions


def track_photon_histories(
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
    electron_cutoff_eV: float,
    source_kind=None,
    aperture: PhotonAperture | None = None,
    boxes: Sequence[PhotonBox] = (),
    width_ang=None,
    height_ang=None,
    max_generations: int = 1000,
    relaxation: Callable[[str], EADLRelaxation] = load_eadl_relaxation,
) -> PhotonHistories:
    """Transport analog photons to escape, detection, absorption or cutoff.

    Each flight samples one EPDL first collision (``photon_first_interactions``)
    from its own Philox stream keyed by lineage, then the selected conditional
    sampler: Rayleigh, Compton (free electron, no vacancy), EPDL subshell
    photoabsorption with a sampled EADL cascade, or PENELOPE pair kinematics
    for nuclear and electron-field pairs (triplet simulated as a pair).
    Relaxation photons and electrons are emitted isotropically. Daughter
    photons at or below ``photon_cutoff_eV`` and electrons at or below
    ``electron_cutoff_eV`` deposit locally; positrons are always handed off
    with ``2 m_e c²`` booked as pending rest energy. Relaxation stops at
    shells bound at or below the smaller cutoff; their binding stays terminal.

    ``boxes`` (e.g. filter plates from :func:`photon_box_from_pose`) join
    the same navigation, so photons scatter, fluoresce and re-enter across
    target, vacuum gaps and filters; regions index ``layers`` then ``boxes``.
    The aperture is an absorbing boundary: a photon is detected when its
    forward aperture distance is shorter than its sampled collision distance.
    Overlapping bodies, or an aperture entering material, raise. A photon
    with no further collision escapes; straight lines cross each convex
    region once, so no later re-entry exists. Photons whose lineage
    reaches ``max_generations`` interactions are unresolved, never absorbed.

    Per history (weighted, eV): incident = escaped + detected + deposited +
    terminal binding + transition defect + charged kinetic + positron rest +
    unresolved, to floating-point roundoff. Order, chunking and batch position
    do not change any history: draws depend only on (seed, history, event,
    daughter ordinals). Source photons need ``photon_cutoff_eV < E <= 100 GeV``.
    Source: PENELOPE-2024 §1.4 and §2.4 (photon histories, pair kinematics),
    ENDF-102 File 28; component samplers' own ledger claims.
    Validation: photon-history-queue, photon-scene-navigation
    """
    position, direction, energy, weight, history, event = _validate_batch(
        position_ang, direction, energy_eV, weight, history_id, event_id
    )
    kinds = _validate_options(
        energy, photon_cutoff_eV, electron_cutoff_eV, max_generations, source_kind
    )
    footprint = {"width_ang": width_ang, "height_ang": height_ang}
    surfaces = () if aperture is None else (aperture.region(),)
    validate_photon_regions(layers, boxes, surfaces=surfaces, **footprint)
    compositions = [layer[2] for layer in layers] + [box.composition for box in boxes]
    histories, rows = np.unique(history, return_inverse=True)
    queue = _Queue(histories, float(photon_cutoff_eV), float(electron_cutoff_eV))
    keys = photon_stream_key(seed, history, event)
    queue.pending = [
        _Photon(position[i], direction[i], float(energy[i]), int(rows[i]), float(weight[i]),
                keys[i], int(kinds[i]), 0)
        for i in range(energy.size)
    ]  # fmt: skip
    cache: dict[str, EADLRelaxation] = {}
    while queue.pending:
        active, queue.pending = queue.pending, []
        for p in active:
            if p.generation >= max_generations:
                queue.book("unresolved_eV", p.row, p.weight * p.energy)
        active = [p for p in active if p.generation < max_generations]
        rngs = [np.random.Generator(np.random.Philox(key=int(p.key))) for p in active]
        flight, collided, points = _flight(queue, active, layers, boxes, footprint, aperture, rngs)
        for i in np.flatnonzero(collided):
            parent = active[i]
            layer, element_index = int(flight["layer"][i]), int(flight["element"][i])
            element = compositions[layer][element_index][0]
            channel = int(flight["channel"][i])
            queue.interactions.append(
                (points[i], parent.energy, parent.row, parent.key, parent.generation, layer,
                 element_index, channel)
            )  # fmt: skip
            if PHOTON_CHANNEL_ORDER[channel] == "photoelectric" and element not in cache:
                cache[element] = relaxation(element)
            _interact(queue, parent, channel, element, cache.get(element), rngs[i], points[i])
    return _assemble(queue, rows, energy * weight)


def _columns(records: list[tuple], names: Sequence[str], shapes: Mapping[str, tuple]):
    """Transpose record tuples into named arrays, keeping empty shapes."""
    columns = list(zip(*records, strict=True)) if records else [()] * len(names)
    out = {}
    for name, values in zip(names, columns, strict=True):
        array = np.array(values)
        out[name] = array.reshape((len(records), *shapes.get(name, ())))
    return out


def _assemble(queue: _Queue, rows, incident) -> PhotonHistories:
    vector = {"position_ang": (3,), "direction": (3,), "local_ang": (2,)}
    exits = _columns(
        queue.exits,
        ("position_ang", "direction", "energy_eV", "weight", "row", "key", "source_kind",
         "generation", "detected", "local_ang"),
        vector,
    )  # fmt: skip
    charged = _columns(
        queue.charged,
        ("species", "channel", "position_ang", "direction", "kinetic_eV", "weight", "row",
         "key", "source_kind", "generation"),
        vector,
    )  # fmt: skip
    interactions = _columns(
        queue.interactions,
        ("position_ang", "energy_eV", "row", "key", "generation", "layer", "element", "channel"),
        vector,
    )
    for table in (exits, charged, interactions):
        table["history_id"] = queue.histories[table.pop("row").astype(np.int64)]
        table["key"] = table["key"].astype(np.uint64)
    ledger = {"history_id": queue.histories}
    ledger["incident_eV"] = np.array(
        [fsum(incident[rows == r]) for r in range(queue.histories.size)]
    )
    for name in _LEDGER_TERMS:
        ledger[name] = np.array([fsum(values) for values in queue.terms[name]])
    ledger["complete"] = ledger["unresolved_eV"] == 0.0
    return PhotonHistories(exits, charged, interactions, ledger)


def ledger_residual_eV(ledger: Mapping[str, np.ndarray]) -> np.ndarray:
    """Incident minus all terminal terms per history; zero up to roundoff.

    Validation: photon-history-queue
    """
    return ledger["incident_eV"] - sum(ledger[name] for name in _LEDGER_TERMS)


def detected_spectrum(histories: PhotonHistories, edges_eV, n_primaries: int) -> np.ndarray:
    """Weighted aperture hits per primary per eV in each energy bin.

    This is an aperture-integrated count density, not a per-steradian
    spectrum; compare it with directional results only through a derived
    solid-angle estimator. ``n_primaries`` is the fixed population denominator.
    Validation: photon-history-queue
    """
    edges = np.asarray(edges_eV, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy edges must be strictly increasing")
    if n_primaries <= 0:
        raise ValueError("n_primaries must be positive")
    exits = histories.exits
    hit = exits["detected"].astype(bool)
    counts, _ = np.histogram(exits["energy_eV"][hit], edges, weights=exits["weight"][hit])
    return counts / (n_primaries * np.diff(edges))
