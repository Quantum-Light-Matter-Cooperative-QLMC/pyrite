"""Planar boundary crossings of transported tracks, for transition radiation.

Transition radiation (TR) is emitted where a charge crosses a change of
dielectric response (#103). This module reduces ``energy_model="midpoint"``
transport rows to the physical crossings of the planar stack's ``z`` faces: the
entrance through ``z = 0``, every ``LAYER_BOUNDARY`` between layers of
different composition, and the ``EXIT_TOP``/``EXIT_BOTTOM`` faces. It derives
crossings from the row-event contract in
``docs/physics/beam-transport/transport-outputs.md`` and draws no random
numbers, so the records are a function of the transport result alone.

Only physical interfaces are recorded. ``SUBSTEP`` (and ``DELTA``) rows are
integration nodes and never close a row at a surface; a boundary between two
layers of equal composition has no dielectric contrast and is dropped and
counted. Lateral ``EXIT_SIDE`` faces are not planar-stack interfaces and are
counted, not recorded. Grooved runs and frozen rows are rejected.

:func:`crossing_amplitudes` evaluates each crossing's vacuum amplitude through
the stack (:func:`~pyrite.montecarlo.spectrum.transition_radiation.interface_transition_radiation_amplitude`)
and :func:`crossing_transition_radiation` sums them per history, so the entry,
exit and inner interfaces of one history interfere with their transport phases.
Histories add incoherently by default; ``coherence="bunch"`` keeps the
cross-history terms (coherent TR, CTR) through the distinct-pair bunch
estimator of the coherent-emission path.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...materials.crystal import M_E_EV
from ..transport.events import (
    EVENT_EXIT_BOTTOM,
    EVENT_EXIT_SIDE,
    EVENT_EXIT_TOP,
    EVENT_GROOVE_SURFACE,
    EVENT_LAYER_BOUNDARY,
)
from .transition_radiation import interface_transition_radiation_amplitude

__all__ = [
    "VACUUM",
    "BoundaryCrossings",
    "boundary_crossings",
    "crossing_amplitudes",
    "crossing_transition_radiation",
]

#: ``layer_before``/``layer_after`` code of the vacuum outside either face.
VACUUM = -1
#: Identity marker of the run-case TR component (#103). v1: per-history or
#: bunch crossing superposition, Chantler layer permittivity, Ne_brem population.
TRANSITION_MODEL = "planar-stack-crossings-v1"
_LAUNCH_POSITRON = 3  # transport.secondaries.LAUNCH_POSITRON, without its import cost
_ENTRY_ATOL_ANG = 1e-5  # the interface nudge allowed by check_segment_event_contract


@dataclass(frozen=True)
class BoundaryCrossings:
    """One row per physical crossing of a planar stack face, in track and time order.

    Attributes
    ----------
    electron_id, track_id
        Primary history and transported track (equal without a cascade).
    charge
        Track charge in units of ``e``: ``-1`` electrons, ``+1`` positrons.
    position_ang
        Crossing point ``(n, 3)`` in the slab frame [Å], ``z`` exactly on the
        interface plane (transport's sub-Å interface nudge removed).
    direction
        Unit velocity direction ``(n, 3)`` through the face. Transport keeps
        direction across a layer boundary, so it is the same on both sides.
    kinetic_energy_keV
        Kinetic energy at the crossing [keV], continuous across the face.
    time_ang
        Bunch arrival offset plus transport age at the crossing, ``t0 + t``
        [Å, ``c = 1``].
    interface
        Index of the plane ``z_interfaces_ang[interface]``.
    normal
        Unit face normal ``(n, 3)``, ``+z``: from the entrance face into the stack.
    layer_before, layer_after
        Media on the incoming and outgoing side: layer index or :data:`VACUUM`.
    z_interfaces_ang
        Plane depths ``(n_layers + 1,)`` [Å], entrance face first.
    n_matched_dropped
        Layer boundaries between equal compositions, not recorded.
    n_side_exits
        ``EXIT_SIDE`` rows, not recorded.
    """

    electron_id: np.ndarray
    track_id: np.ndarray
    charge: np.ndarray
    position_ang: np.ndarray
    direction: np.ndarray
    kinetic_energy_keV: np.ndarray
    time_ang: np.ndarray
    interface: np.ndarray
    normal: np.ndarray
    layer_before: np.ndarray
    layer_after: np.ndarray
    z_interfaces_ang: np.ndarray
    n_matched_dropped: int
    n_side_exits: int

    def __len__(self) -> int:
        return int(self.interface.size)


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _same_medium(a, b) -> bool:
    try:
        return bool(a == b)
    except ValueError:  # array-valued compositions
        return bool(np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object)))


def _stack_planes(segments, layers):
    if layers is None:
        return np.array([0.0, float(segments["thickness_ang"])]), [None]
    tops = np.array([float(top) for top, _, _ in layers])
    bots = np.array([float(bot) for _, bot, _ in layers])
    if tops.size == 0 or tops[0] != 0.0 or np.any(tops[1:] != bots[:-1]) or np.any(bots <= tops):
        raise ValueError("layers must be contiguous [(z_top, z_bot, composition), ...] from z = 0")
    return np.concatenate([tops[:1], bots]), [comp for _, _, comp in layers]


def _charges(segments, track):
    tracks = segments.get("secondary_tracks")
    if tracks is None or "launch_kind" not in tracks:
        return np.full(track.size, -1, dtype=np.int8)
    kind = np.empty(int(np.max(_host(tracks["track_id"]))) + 1, dtype=np.int8)
    kind[_host(tracks["track_id"])] = _host(tracks["launch_kind"])
    return np.where(kind[track] == _LAUNCH_POSITRON, 1, -1).astype(np.int8)


def boundary_crossings(
    segments, layers: Sequence[tuple[float, float, object]] | None = None
) -> BoundaryCrossings:
    """Physical planar-face crossings of a midpoint transport result.

    Parameters
    ----------
    segments
        :func:`pyrite.montecarlo.transport.simulate_trajectories` result with
        ``energy_model="midpoint"``, on host or device.
    layers
        The ``layers`` stack passed to transport, ``[(z_top, z_bot,
        composition), ...]``; compositions compare with ``==`` to find
        contrast-free boundaries. ``None`` is transport's single layer over
        ``[0, thickness_ang]``.

    Returns
    -------
    BoundaryCrossings
        Entry of each primary through ``z = 0``, every layer boundary with a
        change of composition, and every top or bottom exit.

    Validation: tr-crossing-superposition
    """
    if segments.get("event_kind") is None:
        raise ValueError("boundary crossings need midpoint rows (energy_model='midpoint')")
    kind = _host(segments["event_kind"]).astype(np.int8, copy=False)
    vacuum = segments.get("vacuum_elec_id")
    if np.any(kind == EVENT_GROOVE_SURFACE) or (vacuum is not None and _host(vacuum).size):
        raise ValueError("grooved entrance faces are not planar-stack interfaces")
    planes, media = _stack_planes(segments, layers)
    n_layers = planes.size - 1

    electron = _host(segments["electron_id"]).astype(np.int64, copy=False)
    track = segments.get("track_id")
    track = electron if track is None else _host(track).astype(np.int64, copy=False)
    order = np.lexsort(
        (_host(segments["substep_id"]), _host(segments["flight_id"]), track), axis=-1
    )
    kind, electron, track = kind[order], electron[order], track[order]
    r_mid = _host(segments["r_mid"])[order]
    v_hat = _host(segments["v_hat"])[order]
    length = _host(segments["L_ang"])[order]
    layer = _host(segments["layer"]).astype(np.int64, copy=False)[order]
    t0 = _host(segments["t0_ang"])[order]
    generation = segments.get("generation")
    primary = (
        np.ones(kind.size, dtype=bool) if generation is None else _host(generation)[order] == 0
    )

    # Entry: each primary track's first row starts on the entrance face.
    first = np.ones(kind.size, dtype=bool)
    first[1:] = track[1:] != track[:-1]
    entry = first & primary
    start = r_mid - 0.5 * length[:, None] * v_hat
    if np.any(np.abs(start[entry, 2]) > _ENTRY_ATOL_ANG) or np.any(layer[entry] != 0):
        raise ValueError("a primary track does not start on the entrance face z = 0")

    down = v_hat[:, 2] > 0.0
    boundary = kind == EVENT_LAYER_BOUNDARY
    after = np.where(down, layer + 1, layer - 1)
    matched = np.zeros(kind.size, dtype=bool)
    for i in np.flatnonzero(boundary):
        matched[i] = _same_medium(media[layer[i]], media[after[i]])
    top, bottom = kind == EVENT_EXIT_TOP, kind == EVENT_EXIT_BOTTOM
    face = (boundary & ~matched) | top | bottom

    index = np.concatenate([np.flatnonzero(entry), np.flatnonzero(face)])
    is_entry = np.arange(index.size) < np.count_nonzero(entry)
    position = np.where(
        is_entry[:, None],
        start[index],
        r_mid[index] + 0.5 * length[index, None] * v_hat[index],
    )
    interface = np.where(is_entry, 0, np.where(down[index], layer[index] + 1, layer[index]))
    position[:, 2] = planes[interface]
    energy = np.where(
        is_entry,
        _host(segments["E_start_keV"])[order][index],
        _host(segments["E_end_keV"])[order][index],
    )
    age = np.where(
        is_entry,
        _host(segments["t_start_ang"])[order][index],
        _host(segments["t_end_ang"])[order][index],
    )
    before = np.where(is_entry, VACUUM, layer[index])
    beyond = np.where(top[index] | bottom[index], VACUUM, after[index])
    beyond = np.where(is_entry, 0, beyond)
    if np.any((beyond < VACUUM) | (beyond >= n_layers)):
        raise ValueError("a layer boundary leads outside the stack")

    ranked = np.lexsort((age, track[index]))
    index, position, interface = index[ranked], position[ranked], interface[ranked]
    count = index.size
    return BoundaryCrossings(
        electron_id=electron[index],
        track_id=track[index],
        charge=_charges(segments, track[index]),
        position_ang=position,
        direction=v_hat[index],
        kinetic_energy_keV=energy[ranked],
        time_ang=t0[index] + age[ranked],
        interface=interface.astype(np.int64),
        normal=np.broadcast_to(np.array([0.0, 0.0, 1.0]), (count, 3)).copy(),
        layer_before=before[ranked].astype(np.int64),
        layer_after=beyond[ranked].astype(np.int64),
        z_interfaces_ang=planes,
        n_matched_dropped=int(np.count_nonzero(matched)),
        n_side_exits=int(np.count_nonzero(kind == EVENT_EXIT_SIDE)),
    )


_FLIP = np.array([1.0, -1.0, -1.0])  # rotation by pi about x: the entrance face on top
#: Fused CUDA kernel for CUDA CuPy arrays (``transition_radiation_jit_kernel``);
#: a test patches it off to run the CuPy array route on the device. ROCm/HIP
#: CuPy always takes the array route.
_USE_JIT_TRANSITION = True


def crossing_amplitudes(
    crossings: BoundaryCrossings, photon_energy_eV, direction, layer_eps, *, xp=np
):
    """Vacuum TR amplitude of every crossing, in the slab frame [1/eV].

    Parameters
    ----------
    crossings
        :func:`boundary_crossings` records.
    photon_energy_eV, direction
        Photon energies and unit slab-frame emission directions, broadcast to a
        shape ``s``. All directions leave through one face: ``n_z < 0`` the
        entrance face (backward), ``n_z > 0`` the exit face (forward).
    layer_eps
        One permittivity per transport layer (scalar or broadcasting against
        the energies), e.g. :func:`~pyrite.montecarlo.spectrum.transition_radiation.medium_dielectric`.
        Vacuum surrounds the stack.

    Returns
    -------
    xp.ndarray
        ``(*s, K, 3)`` complex slab-frame vectors with the crossings' charge
        signs, in the convention of
        :func:`~pyrite.montecarlo.spectrum.transition_radiation.interface_transition_radiation_amplitude`:
        far-field origin at the transverse origin of the emitting face, time
        the crossings' ``t0 + age``, ``d^2N/dE dOmega = E n_z^2 |sum a|^2 / (16 pi^4)``.

    Validation: tr-crossing-superposition
    """
    frame = _emission_frame(crossings, direction, layer_eps)
    amplitude = interface_transition_radiation_amplitude(
        photon_energy_eV,
        frame.direction,
        frame.layers,
        1.0,
        frame.velocity,
        crossings.time_ang,
        frame.position,
        frame.face,
        xp=xp,
    )
    scale = xp.asarray(crossings.charge.astype(float)[:, None] * frame.flip)
    return amplitude * scale


@dataclass(frozen=True)
class _EmissionFrame:
    """Crossings in the stack frame of the emitting face (that face on top)."""

    direction: np.ndarray
    layers: list
    velocity: np.ndarray
    position: np.ndarray
    face: np.ndarray
    flip: np.ndarray


def _emission_frame(crossings: BoundaryCrossings, direction, layer_eps) -> _EmissionFrame:
    n = np.asarray(direction, dtype=float)
    if not (np.all(n[..., 2] > 0.0) or np.all(n[..., 2] < 0.0)):
        raise ValueError("directions must all leave through one face (one sign of n_z)")
    thickness = np.diff(crossings.z_interfaces_ang)
    if len(layer_eps) != thickness.size:
        raise ValueError("layer_eps needs one permittivity per transport layer")
    layers = list(zip(layer_eps, thickness, strict=True))
    forward = bool(n.reshape(-1, 3)[0, 2] > 0.0)
    flip = np.ones(3) if forward else _FLIP
    face = crossings.interface if forward else thickness.size - crossings.interface
    gamma = 1.0 + crossings.kinetic_energy_keV * 1e3 / M_E_EV
    velocity = np.sqrt(1.0 - 1.0 / gamma**2)[:, None] * crossings.direction
    return _EmissionFrame(
        n * flip,
        layers if forward else layers[::-1],
        velocity * flip,
        crossings.position_ang * flip,
        np.asarray(face, dtype=np.int64),
        flip,
    )


def crossing_transition_radiation(
    crossings: BoundaryCrossings,
    photon_energy_eV,
    direction,
    layer_eps,
    *,
    n_electrons: int,
    coherence: str = "history",
    physical_electrons: float | None = None,
    form_factor=None,
    max_elements: int = 1 << 22,
    xp=np,
):
    """TR spectral-angular yield of transported histories [photons/eV/sr per electron].

    ``coherence="history"`` adds the amplitudes of every crossing of one
    primary history (all its tracks) before squaring, so entry/exit and
    multilayer interference follow from the transported crossing times and
    points; histories add incoherently. ``coherence="crossing"`` squares each
    crossing alone: the independent-interface (dephased) limit, not the
    default thin-film model. ``n_electrons`` is the incident count (``Ne``),
    and ``max_elements`` bounds one chunk's ``len(s) * crossings`` (a single
    history above it is evaluated alone). CUDA CuPy arrays take the fused
    kernel of ``transition_radiation_jit_kernel``, which keeps no per-crossing
    arrays and needs no chunks.

    ``coherence="bunch"`` adds coherent transition radiation (CTR) between
    histories. With ``S_e`` the summed field of history ``e``, ``G = sum |S_e|^2``
    and ``P = |sum S_e|^2`` over the ``M = n_electrons`` sampled histories, the
    yield per electron of a bunch of ``N = physical_electrons`` is
    ``[G + (N - 1)/(M - 1) F (P - G)] / M`` (the distinct-pair estimator of
    ``coherent_population.pair_scale``). The sampled arrival offsets and
    transverse positions already sit in the crossing phases, so the default
    ``F = 1``; pass ``form_factor`` (an array broadcast against the result, or
    ``form_factor(E)``) only for histories transported without those offsets,
    e.g. the Gaussian ``exp(-(E sigma_t / hbar c)^2)``. ``N = None`` takes the
    sampled histories as the bunch (``N = M``, the plain coherent sum ``P / M``).
    The estimate is signed: an undersampled excess can give negative bins,
    which warn and are returned unclipped.

    Validation: tr-crossing-superposition, tr-coherent-bunch
    """
    if coherence not in ("history", "crossing", "bunch"):
        raise ValueError("coherence must be 'history', 'crossing' or 'bunch'")
    if coherence != "bunch" and (physical_electrons is not None or form_factor is not None):
        raise ValueError("physical_electrons and form_factor need coherence='bunch'")
    energy = np.asarray(photon_energy_eV, dtype=float)
    n = np.asarray(direction, dtype=float)
    shape = np.broadcast_shapes(energy.shape, n.shape[:-1])
    omega_nz2 = xp.asarray(np.broadcast_to(energy * n[..., 2] ** 2, shape))
    order = np.argsort(crossings.electron_id, kind="stable")
    history = crossings.electron_id[order]
    starts = np.flatnonzero(np.r_[True, history[1:] != history[:-1]])[: history.size]
    if history.size and _fused_kernel(xp):
        total, field = _device_sums(
            _take(crossings, order), starts, energy, n, layer_eps, coherence, shape
        )
    else:
        total, field = _chunked_sums(
            crossings, order, starts, energy, n, layer_eps, coherence, shape, max_elements, xp
        )
    if coherence == "bunch":
        total = _bunch_power(
            total, field, energy, shape, n_electrons, physical_electrons, form_factor, xp
        )
    return omega_nz2 * total / (16.0 * np.pi**4 * int(n_electrons))


def _fused_kernel(xp) -> bool:
    """CUDA CuPy (not ROCm/HIP) with the fused kernel enabled."""
    if not _USE_JIT_TRANSITION or getattr(xp, "__name__", "") != "cupy":
        return False
    return not getattr(xp.cuda.runtime, "is_hip", False)


def _device_sums(crossings, starts, energy, n, layer_eps, coherence, shape):
    """Per-history power and bunch field of sorted crossings, fused on the CUDA device."""
    import cupy

    from .transition_radiation import _checked_kinematics, interface_response
    from .transition_radiation_jit_kernel import history_sums

    frame = _emission_frame(crossings, n, layer_eps)
    _checked_kinematics(energy, frame.direction, frame.velocity)
    stack = interface_response(energy, frame.direction, frame.layers, 1.0, xp=cupy)
    count = crossings.interface.size
    bounds = np.arange(count + 1) if coherence == "crossing" else np.r_[starts, count]
    power, field = history_sums(
        stack, frame.velocity, crossings.time_ang, frame.position, frame.face,
        crossings.charge.astype(float), bounds,
    )  # fmt: skip
    field = field * cupy.asarray(frame.flip)
    return power.reshape(shape), field.reshape((*shape, 3))


def _chunked_sums(crossings, order, starts, energy, n, layer_eps, coherence, shape, budget, xp):
    """Per-history power and bunch field over chunks of whole histories, on ``xp`` arrays."""
    total = xp.zeros(shape, dtype=float)
    field = xp.zeros((*shape, 3), dtype=complex)
    history = crossings.electron_id[order]
    per_chunk = max(1, int(budget) // max(1, int(np.prod(shape, dtype=int))))
    history_end = np.r_[starts[1:], history.size]
    lo = 0
    while lo < starts.size:
        first = starts[lo]
        # Whole histories whose crossings end within the budget; one history
        # larger than the budget is taken alone, never split.
        hi = int(np.searchsorted(history_end, first + per_chunk, side="right"))
        hi = max(hi, lo + 1)
        last = history_end[hi - 1]
        amplitude = crossing_amplitudes(
            _take(crossings, order[first:last]), energy, n, layer_eps, xp=xp
        )
        if coherence != "crossing":
            cumulative = xp.cumsum(amplitude, axis=-2)
            ends = xp.asarray(np.r_[starts[lo + 1 : hi], last] - first - 1)
            summed = cumulative[..., ends, :]
            summed[..., 1:, :] -= cumulative[..., ends[:-1], :]
            amplitude = summed
        total += xp.sum(xp.abs(amplitude) ** 2, axis=(-2, -1))
        if coherence == "bunch":
            field += xp.sum(amplitude, axis=-2)
        lo = hi
    return total, field


def _bunch_power(grouped, field, energy, shape, n_electrons, physical_electrons, form_factor, xp):
    """Distinct-pair bunch power ``G + (N - 1)/(M - 1) F (P - G)`` before division by ``M``.

    Source: the ensemble power per physical electron ``E|S|^2 + (N-1) F |E S|^2``
    of ``coherent_population.pair_scale``, with ``G/M`` and ``(P-G)/(M(M-1))``
    estimating its two moments. One physical electron leaves ``G``; identical
    in-phase histories give ``N G``.

    Validation: tr-coherent-bunch
    """
    from .coherent_population import CoherentSamplingWarning, pair_scale

    population = int(n_electrons) if physical_electrons is None else physical_electrons
    scale = pair_scale(population, int(n_electrons))
    if form_factor is None:
        factor = 1.0
    else:
        factor = form_factor(energy) if callable(form_factor) else form_factor
        factor = xp.asarray(np.broadcast_to(np.asarray(factor, dtype=float), shape))
    flat = xp.sum(xp.abs(field) ** 2, axis=-1)
    power = grouped + scale * factor * (flat - grouped)
    negative = int(xp.count_nonzero(power < 0.0))
    if negative:
        warnings.warn(
            CoherentSamplingWarning(
                f"CTR pair estimate is negative in {negative} of {power.size} bins: "
                "the sampled histories do not resolve the bunch excess; returned unclipped"
            ),
            stacklevel=3,
        )
    return power


def _take(crossings: BoundaryCrossings, index) -> BoundaryCrossings:
    fields = {
        name: getattr(crossings, name)[index]
        for name in (
            "electron_id", "track_id", "charge", "position_ang", "direction",
            "kinetic_energy_keV", "time_ang", "interface", "normal", "layer_before",
            "layer_after",
        )
    }  # fmt: skip
    return BoundaryCrossings(
        **fields,
        z_interfaces_ang=crossings.z_interfaces_ang,
        n_matched_dropped=crossings.n_matched_dropped,
        n_side_exits=crossings.n_side_exits,
    )
