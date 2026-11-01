"""Explicit host charged continuation through disjoint photon-scene bodies.

Material stages retain escaped phase space; this scheduler flies those rows
through vacuum to the next body. Production preparation owns catalog material
resolution and pins; this owner accepts explicit body-local resources.
"""

from dataclasses import dataclass, replace

import numpy as np

from .pair_production import PAIR_THRESHOLD_EV
from .photon_charged_escape import join_charged_exits
from .photon_coupling import ChargedStageResult, stage_seed
from .photon_feedback import _charged_rows, _join_emissions, cascade_feedback_stage
from .photon_histories import POSITRON
from .photon_navigation import (
    _box_intervals,
    _layer_intervals,
    validate_photon_regions,
)

_PER_LAYER = ("stopping_tables", "inelastic_materials", "elastic_tables")


@dataclass(frozen=True)
class _Region:
    stage: object
    origin: np.ndarray
    axes: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def local_rows(self, rows):
        return {
            **rows,
            # The navigator already selected a nonempty material interval.
            # Roundoff in a rotated boundary transform must not make that
            # entry miss the kernel's exact local footprint admission.
            "position_ang": np.clip(
                (rows["position_ang"] - self.origin) @ self.axes.T, self.lower, self.upper
            ),
            "direction": rows["direction"] @ self.axes.T,
        }

    def world_rows(self, rows):
        return {
            **rows,
            "position_ang": rows["position_ang"] @ self.axes + self.origin,
            "direction": rows["direction"] @ self.axes,
        }


@dataclass(frozen=True)
class SceneFeedbackStage:
    """Bounded charged queue over target layers, vacuum gaps and filter boxes.

    Every boundary exit is a continuation, including escaping shell and Auger
    daughters. The nearest positive material interval owns the next launch;
    vacuum changes only position. Material visits use fresh lineage-keyed
    streams, not replayed flight draws. The unchanged single-stack stage is
    still available separately. Exceeding the crossing cap raises.

    CPU/CUDA selection is explicit: CUDA navigation uses the shared binary64
    clipping equations on device and every material visit uses the CUDA charged
    core. Lineage, phase-space transforms, scheduling and ledgers remain host
    controlled. CUDA admission never falls back to CPU, even for empty batches.

    Source: PENELOPE-2024 §1.4.4 interface continuation and Chapter 7 §7.1
    secondary stack. Assumptions: disjoint analytic bodies, no fields or
    interactions in vacuum, inherited condensed charged transport. Energies
    are eV, positions Å. A miss escapes; zero gap gives a direct interface.
    Validation: photon-charged-reentry
    """

    layers: tuple
    boxes: tuple
    regions: tuple
    material_indices: tuple
    width_ang: float | None
    height_ang: float | None
    max_crossings: int = 64
    backend: str = "cpu"
    transports_positrons = True

    @property
    def photon_cutoff_eV(self):
        return self.regions[0].stage.photon_cutoff_eV

    @property
    def threshold_eV(self):
        return self.regions[0].stage.threshold_eV

    def validate_geometry(self, layers, boxes, width_ang, height_ang):
        if (
            tuple(layers) != self.layers
            or tuple(boxes) != self.boxes
            or width_ang != self.width_ang
            or height_ang != self.height_ang
        ):
            raise ValueError("charged and photon stages must use the same scene geometry")

    def _next(self, rows, previous=None):
        xp = np
        if self.backend == "cuda":
            from ..._backend import select_cupy_backend

            xp = select_cupy_backend().xp
        position, direction = xp.asarray(rows["position_ang"]), xp.asarray(rows["direction"])
        lo, hi = _layer_intervals(
            position, direction, self.layers, self.width_ang, self.height_ang, xp=xp
        )
        blo, bhi = _box_intervals(position, direction, self.boxes, xp=xp)
        lo, hi = xp.column_stack([lo, blo]), xp.column_stack([hi, bhi])
        indices = xp.asarray(self.material_indices)
        distance = xp.where(hi[:, indices] > lo[:, indices], lo[:, indices], xp.inf)
        if previous is not None:
            previous = xp.asarray(previous)
            valid = previous >= 0
            distance[xp.flatnonzero(valid), previous[valid]] = xp.inf
        chosen = xp.argmin(distance, axis=1)
        distance = distance[xp.arange(len(chosen)), chosen]
        from .secondaries import _host

        return _host(chosen), _host(distance)

    def contains(self, rows):
        """Material ownership includes an entering boundary, excludes an outgoing one."""
        _, distance = self._next(rows)
        return distance == 0.0

    def __call__(self, rows, seed):
        """Substitute each material visit's escape with its next material history.

        Per input: T + 2mc² I(positron) = final escape + deposit + binding +
        photons. Intermediate escape and rest reservoirs cancel once, without
        touching the original source electron. Validation: photon-charged-reentry
        """
        pending = _charged_rows(rows)
        n = len(pending["kinetic_eV"])
        rest = np.where(pending["species"] == POSITRON, PAIR_THRESHOLD_EV, 0.0)
        owners, totals, emitted, final_exits = np.arange(n), np.zeros((4, n)), [], []
        previous = np.full(n, -1)
        for visit in range(self.max_crossings + 1):
            if not len(owners):
                return ChargedStageResult(
                    *totals, _join_emissions(emitted), rest, join_charged_exits(final_exits)
                )
            region, distance = self._next(pending, previous)
            miss = ~np.isfinite(distance)
            escaped = pending["kinetic_eV"] + np.where(
                pending["species"] == POSITRON, PAIR_THRESHOLD_EV, 0.0
            )
            np.add.at(totals[0], owners[miss], escaped[miss])
            final_exits.append({**{k: v[miss] for k, v in pending.items()}, "parent": owners[miss]})
            if np.all(miss):
                return ChargedStageResult(
                    *totals, _join_emissions(emitted), rest, join_charged_exits(final_exits)
                )
            if visit == self.max_crossings:
                raise RuntimeError("charged scene feedback exceeded max_crossings")
            next_parts, next_owners, next_regions = [], [], []
            for index, body in enumerate(self.regions):
                select = np.flatnonzero(~miss & (region == index))
                if not len(select):
                    continue
                part = {k: v[select].copy() for k, v in pending.items()}
                part["position_ang"] += distance[select, None] * part["direction"]
                local = body.local_rows(part)
                result = body.stage(local, stage_seed(seed, visit))
                _check_visit_balance(result, local)
                owner = owners[select]
                for term, name in enumerate(("deposited_eV", "binding_eV", "radiated_eV"), 1):
                    np.add.at(totals[term], owner, getattr(result, name))
                photons = body.world_rows(result.photons)
                emitted.append({**photons, "parent": owner[photons["parent"]]})
                exits = body.world_rows(result.escaped_particles)
                next_owners.append(owner[exits["parent"]])
                next_parts.append(exits)
                next_regions.append(np.full(len(exits["parent"]), index))
            pending = {k: np.concatenate([p[k] for p in next_parts]) for k in pending}
            owners = np.concatenate(next_owners)
            previous = np.concatenate(next_regions)
        raise AssertionError("unreachable charged queue state")


def _check_visit_balance(result: ChargedStageResult, rows):
    """Reject dropped launches or exits before substituting a material visit.

    Roundoff allowance matches the inherited charged feedback ledger checks.
    Validation: photon-charged-reentry
    """
    n = len(rows["kinetic_eV"])
    incident = rows["kinetic_eV"] + np.where(rows["species"] == POSITRON, PAIR_THRESHOLD_EV, 0.0)
    total = result.escaped_eV + result.deposited_eV + result.binding_eV + result.radiated_eV
    if not np.allclose(total, incident, rtol=2e-12, atol=1e-8):
        raise RuntimeError("charged material visit failed energy closure")
    exits = result.escaped_particles
    if exits is None:
        raise RuntimeError("charged material visit did not retain escaped phase space")
    budget = exits["kinetic_eV"] + np.where(exits["species"] == POSITRON, PAIR_THRESHOLD_EV, 0.0)
    escaped = np.bincount(exits["parent"], weights=budget, minlength=n)
    if not np.allclose(escaped, result.escaped_eV, rtol=2e-12, atol=1e-8):
        raise RuntimeError("charged material escape phase space failed energy closure")


def scene_feedback_stage(
    layers,
    *,
    boxes=(),
    photon_cutoff_eV,
    width_ang=None,
    height_ang=None,
    max_crossings=64,
    backend="cpu",
    **transport,
):
    """Build explicit reference stages for every non-vacuum slab and box.

    Per-layer table/material lists cover all target layers (including empty
    ones), followed by all boxes. Each material body gets its own local
    contiguous stage, preserving its composition and table ownership. No
    field acts in gaps. Production filter feedback uses this stage after
    resolving each catalog material and pinning its electron/positron tables.
    Validation: photon-charged-reentry
    """
    _admit_scene_backend(backend, transport)
    for name in ("layers", "thickness_ang", "crystal_width_mm", "crystal_height_mm", "groove"):
        if transport.get(name) is not None:
            raise ValueError(f"{name} is owned by the charged scene geometry")
    if isinstance(transport.get("inelastic_materials"), str):
        transport["inelastic_materials"] = [transport["inelastic_materials"]]
    validate_photon_regions(layers, boxes, width_ang=width_ang, height_ang=height_ang)
    if type(max_crossings) is not int or max_crossings < 1:
        raise ValueError("max_crossings must be a positive integer")
    count = len(layers) + len(boxes)
    for name in _PER_LAYER:
        if transport.get(name) is not None and len(transport[name]) != count:
            raise ValueError(f"{name} must cover target layers followed by boxes")
    specs = []
    for top, bottom, composition in layers:
        specs.append(
            (np.array([0.0, 0.0, top]), np.eye(3), bottom - top, width_ang, height_ang, composition)
        )
    for box in boxes:
        axes, half = np.asarray(box.axes), np.asarray(box.half_extent_ang)
        origin = np.asarray(box.center_ang) - half[2] * axes[2]
        specs.append((origin, axes, 2 * half[2], 2 * half[0], 2 * half[1], box.composition))
    regions, indices = [], []
    for index, (origin, axes, thickness, width, height, composition) in enumerate(specs):
        if not composition:
            continue
        kw = dict(transport)
        for name in _PER_LAYER:
            if kw.get(name) is not None:
                kw[name] = [kw[name][index]]
        stage = cascade_feedback_stage(
            [(0.0, thickness, composition)],
            photon_cutoff_eV=photon_cutoff_eV,
            width_ang=width,
            height_ang=height,
            **kw,
        )
        sizes = [stage.base.transport[f"crystal_{axis}_mm"] for axis in ("width", "height")]
        half = [np.inf if size is None else size * 1e7 / 2 for size in sizes]
        regions.append(
            _Region(
                replace(stage, retain_exits=True),
                origin,
                axes,
                np.array([-half[0], -half[1], 0.0]),
                np.array([half[0], half[1], thickness]),
            )
        )
        indices.append(index)
    if not regions:
        raise ValueError("charged scene feedback requires at least one material body")
    return SceneFeedbackStage(
        tuple(layers),
        tuple(boxes),
        tuple(regions),
        tuple(indices),
        width_ang,
        height_ang,
        max_crossings,
        backend,
    )


def _admit_scene_backend(backend, transport):
    """Require the requested device and matching charged core before geometry setup."""
    if backend not in ("cpu", "cuda"):
        raise ValueError("charged scene backend must be cpu or cuda")
    if backend == "cuda":
        from ..._backend import select_cupy_backend

        select_cupy_backend()  # explicit admission, including an empty input population
        transport["transport_core"] = "cuda"
    elif transport.get("transport_core") == "cuda":
        raise ValueError("CUDA charged core requires backend='cuda'")
