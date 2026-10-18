"""Transition radiation integrated through planar-detector acceptance and energy channels.

TR from a thin film or a foil stack carries formation-zone and multilayer
fringes much narrower than a broad logarithmic energy grid, and angular
structure on the ``1/gamma`` scale (#103). This module integrates
:func:`~pyrite.montecarlo.spectrum.transition_radiation_crossings.crossing_transition_radiation`
over each detector pixel's solid angle and each energy channel by successive
refinement, instead of sampling the density on a fixed grid.

Quadrature: composite trapezoid nodes inside every channel (shared channel
edges) and a midpoint grid of ``s x s`` point sub-pixels per pixel, each with
its point solid angle ``A cos(obliquity) / r^2``. The energy node count and
``s`` double, one axis at a time, toward the axis whose last doubling changed
the table most, until the error estimate is within ``rtol`` of the table's peak
or a node limit is reached. Both rules are second order, so the estimate is a
third of the larger last change: asymptotic, not a bound, and it can miss a
fringe that no level has resolved yet.

The source is a point at the lab origin, which is the far-field origin of the
crossing amplitudes; ``tilt_polar_rad``/``tilt_azim_rad`` orient the slab as in
transport. Photons are incident on the detector plane: filters and detector
response are read-time factors applied by the caller.

Validation: tr-detector-integration
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..._planar_geometry import PixelGrid, PlanarPose, planar_rays
from ..geometry import directions_to_sample_frame
from .transition_radiation_crossings import BoundaryCrossings, crossing_transition_radiation

__all__ = ["DetectedTransitionRadiation", "detected_transition_radiation"]


@dataclass(frozen=True)
class DetectedTransitionRadiation:
    """TR photons per incident electron in each pixel and energy channel.

    Attributes
    ----------
    photons
        ``(ny, nx, n_channel)`` incident photons per electron; ``(1, 1, n)``
        for an unpixelated plane.
    energy_edges_eV
        Channel edges ``(n_channel + 1,)`` [eV].
    error_estimate
        Asymptotic absolute error of ``photons`` (a third of the larger change
        over the last refinement of either axis), same shape.
    energy_intervals
        Trapezoid intervals per channel at the final level.
    angular_subdivisions
        Sub-pixels per pixel side at the final level.
    converged
        ``max(error_estimate) <= rtol * max(photons)``.
    """

    photons: np.ndarray
    energy_edges_eV: np.ndarray
    error_estimate: np.ndarray
    energy_intervals: int
    angular_subdivisions: int
    converged: bool


def _sub_rays(pose, pixels, size_mm, s):
    """Point sub-pixel rays ``(ny, nx, s, s)`` of a pixel grid or a bare plane.

    Validation: tr-detector-integration
    """
    if pixels is None:
        ny, nx, pitch = 1, 1, (size_mm[1], size_mm[0])
    else:
        (ny, nx), pitch = pixels.shape, pixels.pitch_mm
    fine = PixelGrid(shape=(ny * s, nx * s), pitch_mm=(pitch[0] / s, pitch[1] / s))
    rays = planar_rays(pose, pixels=fine)
    directions = rays.directions_lab.reshape(ny, s, nx, s, 3).transpose(0, 2, 1, 3, 4)
    weight = rays.solid_angle_sr.reshape(ny, s, nx, s).transpose(0, 2, 1, 3)
    return directions.reshape(ny, nx, s * s, 3), weight.reshape(ny, nx, s * s)


def _channel_nodes(edges, m):
    """Trapezoid nodes and per-channel weights ``(n_channel, n_node)``.

    Validation: tr-detector-integration
    """
    fraction = np.arange(m + 1) / m
    nodes = edges[:-1, None] + np.diff(edges)[:, None] * fraction[None]
    weights = np.full(m + 1, 1.0)
    weights[[0, -1]] = 0.5
    width = np.diff(edges)[:, None] / m
    return nodes, width * weights[None]


def _channel_table(
    crossings, layer_eps, pose, pixels, size_mm, edges, m, s, tilt, n_electrons, coherence, xp
):
    """Photons ``(ny, nx, n_channel)`` with ``m`` trapezoid intervals and ``s x s`` sub-pixels.

    Validation: tr-detector-integration
    """
    directions, solid_angle = _sub_rays(pose, pixels, size_mm, s)
    shape = directions.shape[:-1]
    n_slab = directions_to_sample_frame(directions.reshape(-1, 3), *tilt)
    if not (np.all(n_slab[:, 2] > 0.0) or np.all(n_slab[:, 2] < 0.0)):
        raise ValueError("the detector plane must lie beyond one slab face")
    nodes, weights = _channel_nodes(edges, m)
    energy = np.unique(nodes)[:, None]
    density = crossing_transition_radiation(
        crossings,
        energy,
        n_slab[None],
        list(layer_eps(energy)),
        n_electrons=n_electrons,
        coherence=coherence,
        xp=xp,
    )
    density = density.get() if hasattr(density, "get") else np.asarray(density)
    per_pixel = (density.reshape(energy.size, *shape) * solid_angle[None]).sum(-1)
    index = np.searchsorted(energy[:, 0], nodes)
    return np.einsum("cn,cnyx->yxc", weights, per_pixel[index])


def detected_transition_radiation(
    crossings: BoundaryCrossings,
    layer_eps: Callable[[np.ndarray], Sequence],
    pose: PlanarPose,
    energy_edges_eV,
    *,
    n_electrons: int,
    pixels: PixelGrid | None = None,
    size_mm: tuple[float, float] | None = None,
    tilt_polar_rad: float = 0.0,
    tilt_azim_rad: float = 0.0,
    coherence: str = "history",
    rtol: float = 1e-3,
    initial_intervals: int = 4,
    max_energy_intervals: int = 256,
    max_angular_subdivisions: int = 16,
    xp=np,
) -> DetectedTransitionRadiation:
    """Integrate TR over a planar detector's pixels and energy channels.

    Parameters
    ----------
    crossings
        :func:`~pyrite.montecarlo.spectrum.transition_radiation_crossings.boundary_crossings`
        records of the transported run.
    layer_eps
        ``layer_eps(E)`` returns one permittivity per transport layer at photon
        energies ``E`` [eV] (called with shape ``(n, 1)``), e.g.
        ``lambda E: [medium_dielectric(m, E) for m in materials]``.
    pose, pixels, size_mm
        Detector plane as in :class:`pyrite._planar_geometry.PlanarPose`;
        ``pixels`` for a pixel grid, else ``size_mm`` ``(width_x, height_y)``.
        The whole plane must lie beyond one slab face.
    energy_edges_eV
        Increasing channel edges [eV], all positive.
    n_electrons, coherence
        As for ``crossing_transition_radiation``.
    rtol, initial_intervals, max_energy_intervals, max_angular_subdivisions
        Convergence target relative to the table peak, the starting trapezoid
        intervals per channel, and the refinement limits.

    Validation: tr-detector-integration
    """
    edges = np.asarray(energy_edges_eV, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or edges[0] <= 0.0 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("energy_edges_eV must be increasing positive channel edges")
    if pixels is None and size_mm is None:
        raise ValueError("an unpixelated plane needs size_mm")

    def table(m, s):
        return _channel_table(
            crossings, layer_eps, pose, pixels, size_mm, edges, m, s,
            (tilt_polar_rad, tilt_azim_rad), n_electrons, coherence, xp,
        )  # fmt: skip

    level = {"energy": int(initial_intervals), "angle": 1}
    limit = {"energy": int(max_energy_intervals), "angle": int(max_angular_subdivisions)}
    current = table(level["energy"], level["angle"])
    # Last measured change per axis (inf until measured). Every open axis is
    # measured once; afterwards only the one with the larger change refines,
    # and the other keeps its last change.
    delta = {name: np.full(current.shape, np.inf) for name in level}
    while True:
        open_axes = [name for name in level if 2 * level[name] <= limit[name]]
        if not open_axes:
            break
        todo = [name for name in open_axes if np.isinf(delta[name]).all()]
        todo = todo or [max(open_axes, key=lambda name: float(np.max(delta[name])))]
        refined = {}
        for name in todo:
            trial = dict(level, **{name: 2 * level[name]})
            refined[name] = table(trial["energy"], trial["angle"])
            delta[name] = np.abs(refined[name] - current)
        axis = max(refined, key=lambda name: float(np.max(delta[name])))
        current, level[axis] = refined[axis], 2 * level[axis]
        # Both rules are O(h^2): a doubling leaves about a third of its change.
        if np.max(np.maximum(delta["energy"], delta["angle"])) / 3.0 <= rtol * np.max(current):
            break
    error = np.maximum(delta["energy"], delta["angle"]) / 3.0
    return DetectedTransitionRadiation(
        photons=current,
        energy_edges_eV=edges,
        error_estimate=error,
        energy_intervals=level["energy"],
        angular_subdivisions=level["angle"],
        converged=bool(np.max(error) <= rtol * np.max(np.abs(current))),
    )
