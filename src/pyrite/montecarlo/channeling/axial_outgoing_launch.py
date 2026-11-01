"""Physical axial outgoing instrument: fibre events to correlated lab launches (#883).

Research CPU reference for the selected entrance-zone architecture. Each
incident either enters the retained subspace P(kappa_0) or is outgoing at the
surface (the entrance amplitude's complement on the outgoing box), then follows
:func:`.unravel_fiber_outgoing`. Represented-outgoing and classified tail
events are *physical outgoing* only under a declared high-energy matching cut
(:class:`AxialMatchingDeclaration`); out-of-box remainders stay unresolved
numerical weight and are never launched. A launch carries incident ID, depth,
lab position, kinetic energy, unit lab direction, clock and the unchanged
native stream key, all from one joint draw per incident.

Not supplied: the depth budget and omitted return flux (#884), the photon
source (#885), native ``LaunchState`` conversion or admission (#886). Standard
axial runs remain refused.

Validation: cr-axial-zone-handoff-physical
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from .axial_fiber_unraveling import FiberChannels, FiberEvent, _Engine, unravel_fiber_outgoing
from .axial_phase_space import sample_phase_space
from .axial_zone_handoff import (
    _M_E_EV,
    crystal_to_lab_frame,
    momentum_inv_ang,
    on_shell_direction,
)

_HANDOFF_DOMAIN = 0x6230883
_HBARC_EV_ANG = 1973.269804


@dataclass(frozen=True)
class AxialIncident:
    """Classical per-incident inputs from the native beam sample.

    ``impact_ang`` is the crystal-frame transverse impact point relative to
    the entrance origin, which must be a lattice point of the cell coordinates
    used by the spectrum and strings. Clocks are native ``t_ang``/``t0_ang``
    (Å, c=1); ``stream_key`` is the unchanged native primary key.
    """

    incident_id: int
    impact_ang: tuple[float, float]
    clock_at_surface_ang: float
    t0_ang: float
    stream_key: int


@dataclass(frozen=True)
class AxialZoneKinematics:
    """Zone frame and declared continuous-slowing approximation.

    ``frame_lab`` columns are lab unit vectors of crystal x, y, z;
    ``entrance_origin_lab_ang`` is the lab point of the crystal origin.
    The zone loses energy at the constant mean stopping power
    ``stopping_eV_per_ang`` (CSDA, no straggling); cases whose fractional
    loss over the zone exceeds ``max_energy_loss_fraction`` refuse, because
    the quantum zone propagates at the entrance energy.
    """

    entrance_origin_lab_ang: tuple[float, float, float]
    frame_lab: np.ndarray
    kinetic_energy_eV: float
    stopping_eV_per_ang: float
    max_angle_rad: float = 0.1
    max_energy_loss_fraction: float = 1e-3


@dataclass(frozen=True)
class AxialMatchingDeclaration:
    """Declared matching cut under which box outgoing counts as physical.

    The research rule (#623, SLURM 1651-1656) places the cut within
    ``window_eV`` of ``target_cut_eV >= U_max + (U_max - U_min)``; energies
    share the spectrum's reference. Every cut the unraveling selects is
    checked against it. A declaration is not calibration evidence.
    """

    target_cut_eV: float
    potential_max_eV: float
    potential_min_eV: float
    window_eV: float = 2.0

    def __post_init__(self) -> None:
        values = (self.target_cut_eV, self.potential_max_eV, self.potential_min_eV, self.window_eV)
        if not all(np.isfinite(v) for v in values) or self.window_eV < 0:
            raise ValueError("matching declaration needs finite energies and window")
        if self.potential_max_eV < self.potential_min_eV:
            raise ValueError("potential_max_eV is below potential_min_eV")
        floor = 2 * self.potential_max_eV - self.potential_min_eV
        if self.target_cut_eV < floor:
            raise ValueError("matching target is barrier adjacent: below U_max + (U_max - U_min)")


@dataclass(frozen=True)
class AxialLaunch:
    r"""One physical outgoing launch (research record, not a ``LaunchState``).

    ``source`` is ``"entry"`` (surface complement), ``"outgoing"`` (box,
    Weyl measured) or ``"tail"`` (out-of-box string output, exact momentum).
    ``r_ang``/``v_hat`` are lab frame; ``k_perp_inv_ang`` is the crystal-frame
    transverse wavevector. With $\sin\theta_b$ = box reach / $p(E_1)$,
    ``lateral_bound_ang`` $= z\tan\theta_b$ bounds the unresolved lattice
    image (retained drift) and ``clock_bound_ang`` $= z(\sec\theta_b-1)/\beta_1$
    the omitted transverse path-length delay.
    """

    incident_id: int
    source: str
    depth_ang: float
    r_ang: np.ndarray
    v_hat: np.ndarray
    E_keV: float
    t_ang: float
    t0_ang: float
    stream_key: int
    k_perp_inv_ang: tuple[float, float]
    lateral_bound_ang: float
    clock_bound_ang: float


@dataclass(frozen=True)
class AxialHandoffRecord:
    """Terminal fate of one incident: ``retained``, ``physical_outgoing`` or ``unresolved``."""

    incident_id: int
    fate: str
    depth_ang: float
    launch: AxialLaunch | None = None
    event: FiberEvent | None = None


def handoff_stream(seed: int, incident_id: int) -> np.random.Generator:
    """Version-1 entry-split stream for one incident, disjoint from jump/measurement domains.

    Validation: cr-axial-zone-handoff-physical
    """
    for name, value in (("seed", seed), ("incident_id", incident_id)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    key = np.random.SeedSequence((int(seed), _HANDOFF_DOMAIN, 1, int(incident_id)))
    return np.random.Generator(np.random.Philox(key))


def zone_energy_and_clock(zone: AxialZoneKinematics, depth_ang: float) -> tuple[float, float]:
    r"""CSDA kinetic energy [eV] and axial flight time [Å, c=1] at ``depth_ang``.

    With $E(z)=E_0-Sz$ and $\beta=pc/(E+mc^2)$,
    $\Delta t=\int_0^z dz'/\beta=(p_0c-p_1c)/S
    =z\,(E_0+E_1+2mc^2)/(p_0c+p_1c)$, written without cancellation; $S=0$
    gives $z/\beta_0$.
    Validation: cr-axial-zone-handoff-physical
    """
    e0 = float(zone.kinetic_energy_eV)
    e1 = e0 - float(zone.stopping_eV_per_ang) * float(depth_ang)
    if e1 <= 0:
        raise ValueError("zone stopping leaves no kinetic energy")
    pc0, pc1 = (np.sqrt(e * (e + 2 * _M_E_EV)) for e in (e0, e1))
    return e1, float(depth_ang) * (e0 + e1 + 2 * _M_E_EV) / float(pc0 + pc1)


def _check_zone(zone: AxialZoneKinematics, max_depth_ang: float) -> np.ndarray:
    frame = crystal_to_lab_frame(zone.frame_lab)
    origin = np.asarray(zone.entrance_origin_lab_ang, dtype=float)
    numbers = (zone.kinetic_energy_eV, zone.stopping_eV_per_ang, zone.max_angle_rad)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(numbers)):
        raise ValueError("zone origin, energy, stopping and angle must be finite")
    if zone.kinetic_energy_eV <= 0 or zone.stopping_eV_per_ang < 0:
        raise ValueError("zone needs positive energy and nonnegative stopping")
    if not 0 < zone.max_angle_rad < np.pi / 2:
        raise ValueError("max_angle_rad outside (0, pi/2)")
    loss = zone.stopping_eV_per_ang * max_depth_ang / zone.kinetic_energy_eV
    if not loss <= zone.max_energy_loss_fraction:
        raise ValueError("zone energy loss exceeds max_energy_loss_fraction; refuse before work")
    return frame


class _Mapper:
    """Event-to-lab kinematics for one zone and outgoing box."""

    def __init__(self, engine, zone, frame, channels: FiberChannels):
        self.zone, self.frame = zone, frame
        self.origin = np.asarray(zone.entrance_origin_lab_ang, dtype=float)
        self.g, self.box = engine.g, engine.box
        self.n = np.array(engine.box_shape)
        self.cell = 2 * np.pi / self.g
        self.strings = list(channels.tds_strings)
        # Any retained or box state at a folded carrier has |k_perp| < box reach.
        self.reach = float(np.hypot(*((self.box + 1) * self.g)))

    def box_kinematics(self, carrier, position_index, momentum_index):
        """In-cell position x_a = a L / N and wavevector kappa + m g (FFT order m)."""
        a, b = np.array(position_index), np.array(momentum_index)
        order = np.where(b <= self.box, b, b - self.n)
        return a * self.cell / self.n, np.asarray(carrier) + order * self.g

    def launch(self, incident: AxialIncident, source, depth, in_cell, k_perp) -> AxialLaunch:
        energy, flight = zone_energy_and_clock(self.zone, depth)
        sin_b = self.reach / momentum_inv_ang(energy)
        if sin_b >= 1.0:
            raise ValueError("outgoing box reach exceeds the on-shell momentum")
        cos_b = np.sqrt(1.0 - sin_b * sin_b)
        beta = momentum_inv_ang(energy) * _HBARC_EV_ANG / (energy + _M_E_EV)
        direction = on_shell_direction(k_perp, energy, self.zone.max_angle_rad)
        impact = np.asarray(incident.impact_ang, dtype=float)
        lateral = np.floor(impact / self.cell) * self.cell + np.asarray(in_cell, dtype=float)
        point = np.array([lateral[0], lateral[1], depth])
        return AxialLaunch(
            incident_id=incident.incident_id,
            source=source,
            depth_ang=float(depth),
            r_ang=self.origin + self.frame @ point,
            v_hat=self.frame @ direction,
            E_keV=energy / 1000.0,
            t_ang=float(incident.clock_at_surface_ang) + flight,
            t0_ang=float(incident.t0_ang),
            stream_key=int(incident.stream_key),
            k_perp_inv_ang=(float(k_perp[0]), float(k_perp[1])),
            lateral_bound_ang=float(depth) * sin_b / cos_b,
            clock_bound_ang=float(depth) * (1.0 / cos_b - 1.0) / beta,
        )

    def event_launch(self, incident: AxialIncident, event: FiberEvent) -> AxialLaunch:
        if event.kind == "tail":
            index, shift = event.tail_string, event.tail_displacement_ang
            if index is None or shift is None or event.tail_momentum_inv_ang is None:
                raise ValueError("tail event lacks its string, displacement or momentum")
            in_cell = np.add(self.strings[index].position_ang, shift)
            k_perp = np.asarray(event.tail_momentum_inv_ang)
        else:
            in_cell, k_perp = self.box_kinematics(
                event.carrier_inv_ang, event.position_index, event.momentum_index
            )
        return self.launch(incident, event.kind, event.depth_ang, in_cell, k_perp)


def _checked_cut(channels: FiberChannels, matching: AxialMatchingDeclaration):
    def cut(energies):
        value = float(channels.cut(energies))
        if abs(value - matching.target_cut_eV) > matching.window_eV:
            raise ValueError("selected cut lies outside the declared matching window")
        return value

    return cut


def _check_tail_reach(channels, engine, zone, matching) -> None:
    """Every classified tail must be above every admitted cut: floor + U_min >= target + window."""
    if channels.tail_q_max_inv_ang is None or not channels.tds_strings:
        return
    k_min = float(np.min((engine.box + 0.5) * engine.g))
    total = zone.kinetic_energy_eV + _M_E_EV
    kinetic = (_HBARC_EV_ANG * k_min) ** 2 / (2 * total)
    if kinetic + matching.potential_min_eV < matching.target_cut_eV + matching.window_eV:
        raise ValueError("outgoing box too small: tail outputs may lie below the matching cut")


def _validate_incidents(incidents: Sequence[AxialIncident]) -> None:
    ids = [i.incident_id for i in incidents]
    if len(set(ids)) != len(ids):
        raise ValueError("incident IDs must be unique")
    for i in incidents:
        values = (*i.impact_ang, i.clock_at_surface_ang, i.t0_ang)
        if len(i.impact_ang) != 2 or not np.all(np.isfinite(values)):
            raise ValueError("incident impact and clocks must be finite")
        if i.clock_at_surface_ang < i.t0_ang:
            raise ValueError("surface clock predates incident clock")
        key = i.stream_key
        if isinstance(key, bool) or not isinstance(key, Integral) or not 0 <= key < 2**64:
            raise ValueError("stream_key must be a 64-bit nonnegative integer")


def sample_physical_handoff(
    channels: FiberChannels,
    entry_orders: np.ndarray,
    *,
    carrier_inv_ang: tuple[float, float] = (0.0, 0.0),
    hbar_v_eV_ang: float,
    window,
    max_depth_ang: float,
    incidents: Sequence[AxialIncident],
    zone: AxialZoneKinematics,
    matching: AxialMatchingDeclaration,
    seed: int = 0,
    search_step_ang: float = 100.0,
    depth_tolerance_ang: float = 1e-6,
) -> list[AxialHandoffRecord]:
    r"""Sample the correlated physical outgoing instrument, one record per incident.

    Entry: with $c=P(\kappa_0)\psi_0$ and $w=\lVert c\rVert^2$ the incident
    enters with probability $w$ (one ``handoff_stream`` draw); otherwise the
    surface complement $f=B\psi_0-c$ is Weyl measured on the same stream and
    launched at depth 0. Entered incidents follow
    :func:`.unravel_fiber_outgoing`, so the joint law of depth, carrier,
    order and window position is the fibre model's. Box outgoing maps to
    in-cell position $x_a=aL/N$ (window peak at grid index 0) and
    $\mathbf k_\perp=\kappa+m\mathbf g$; tails use $\kappa+\mathbf Q$ at the
    struck string $\mathbf s+\mathbf u$. Position adds the lattice cell
    containing the incident's impact point; lab position, direction
    (:func:`.on_shell_direction`), CSDA energy and clock
    (:func:`zone_energy_and_clock`) follow deterministically, so no
    marginal is resampled. The unit-weight fates sum to the incident count
    at every depth.

    Validation: cr-axial-zone-handoff-physical
    """
    frame = _check_zone(zone, max_depth_ang)
    _validate_incidents(incidents)
    checked = replace(channels, cut=_checked_cut(channels, matching))
    engine = _Engine(checked, hbar_v_eV_ang)
    _check_tail_reach(checked, engine, zone, matching)
    entry = np.asarray(entry_orders, dtype=complex)
    if entry.shape != (engine.k,) or not np.all(np.isfinite(entry)):
        raise ValueError("entry_orders must be finite on the order window")
    if not np.isclose(np.vdot(entry, entry).real, 1.0, rtol=1e-12, atol=0):
        raise ValueError("entry_orders must be a unit incident state")
    g = np.asarray(window, dtype=complex)
    if g.shape != engine.box_shape or not np.isclose(np.vdot(g, g).real, 1.0):
        raise ValueError("window must be a unit-norm array on the outgoing box shape")
    kappa = (float(carrier_inv_ang[0]), float(carrier_inv_ang[1]))
    if np.any(np.abs(kappa) > engine.g / 2) or np.any(np.isclose(kappa, engine.g / 2)):
        raise ValueError("carrier_inv_ang must lie in the half-open zone [-g/2, g/2)")
    fiber = engine.fiber(kappa)
    c = fiber.vectors.conj().T @ entry
    entered_weight = float(np.vdot(c, c).real)
    complement = np.zeros(int(np.prod(engine.box_shape)), dtype=complex)
    complement[engine.window_in_box] = entry - fiber.vectors @ c
    mapper = _Mapper(engine, zone, frame, checked)
    records: dict[int, AxialHandoffRecord] = {}
    entered = []
    for incident in incidents:
        rng = handoff_stream(seed, incident.incident_id)
        if rng.random() < entered_weight:
            entered.append(incident)
            continue
        layout = np.zeros(engine.box_shape, dtype=complex)
        nx, ny = (np.arange(-b, b + 1) for b in engine.box)
        layout[np.ix_(nx % engine.box_shape[0], ny % engine.box_shape[1])] = complement.reshape(
            engine.box_shape
        )
        o = sample_phase_space(np.fft.ifft2(layout, norm="ortho"), g, rng)
        in_cell, k_perp = mapper.box_kinematics(kappa, o.position_index, o.momentum_index)
        launch = mapper.launch(incident, "entry", 0.0, in_cell, k_perp)
        records[incident.incident_id] = AxialHandoffRecord(
            incident.incident_id, "physical_outgoing", 0.0, launch
        )
    if entered:
        events = unravel_fiber_outgoing(
            checked,
            entry,
            carrier_inv_ang=kappa,
            hbar_v_eV_ang=hbar_v_eV_ang,
            window=g,
            max_depth_ang=max_depth_ang,
            incident_ids=[i.incident_id for i in entered],
            seed=seed,
            search_step_ang=search_step_ang,
            depth_tolerance_ang=depth_tolerance_ang,
        )
        for incident, event in zip(entered, events, strict=True):
            fate = {"outgoing": "physical_outgoing", "tail": "physical_outgoing"}.get(
                event.kind, event.kind
            )
            launch = mapper.event_launch(incident, event) if fate == "physical_outgoing" else None
            records[incident.incident_id] = AxialHandoffRecord(
                incident.incident_id, fate, event.depth_ang, launch, event
            )
    return [records[i.incident_id] for i in incidents]


def handoff_balance(records: Sequence[AxialHandoffRecord], depths_ang) -> dict[str, np.ndarray]:
    """Incident-normalized retained / physical outgoing / unresolved fractions by depth.

    A terminal fate at depth ``d`` counts from ``d`` on; retained records never
    terminate. Fractions sum to one at every depth (unit-weight sampling).
    Validation: cr-axial-zone-handoff-physical
    """
    depths = np.asarray(depths_ang, dtype=float)
    n = len(records)
    if not n:
        raise ValueError("no records")
    out = {}
    for fate in ("physical_outgoing", "unresolved"):
        terminal = np.array([r.depth_ang for r in records if r.fate == fate])
        out[fate] = (terminal[None, :] <= depths[:, None]).sum(axis=1) / n
    out["retained"] = 1.0 - out["physical_outgoing"] - out["unresolved"]
    return out
