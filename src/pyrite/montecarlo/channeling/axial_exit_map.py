"""Coherent/absorptive axial exit-angle maps by single-cell Bloch reduction (#1216).

A plane wave entering an axial crystal at tilt ``theta`` (transverse momentum
``k_t = p theta``) obeys the paraxial equation of :mod:`.realspace` with the
complex optical potential ``U(r) - i W(r)``. Both ``U`` and ``W`` are periodic
with the projected crystal cell (lattice vectors ``L_x``, ``L_y``), so the
transverse Hamiltonian commutes with cell translations and the state stays in
the Bloch sector of its entry carrier:

``psi(r, z) = e^{i k_t . r} phi(r, z)``, ``phi`` periodic with **one** cell.

Expanding ``phi = sum_G c_G(z) e^{i G . r}`` over the cell's reciprocal lattice
``G = (2 pi m / L_x, 2 pi n / L_y)``, the exit wave at depth ``z`` is a comb of
plane waves at ``theta_G = (k_t + G) / p``, each carrying ``|c_G(z)|^2`` of the
incident electron (transverse momentum is conserved across the exit face; Shul'ga
& Shulga, Phys. Lett. B 769, 141 (2017), Eqs. 13--14, in two dimensions). A
``c_x x c_y`` supercell adds the orders ``G / (c_x, c_y)`` but, by the same
symmetry, leaves them empty, so one cell carries every populated order; the
supercell's angular pitch is not a resolution limit (Dabagov & Ognev, Nucl.
Instrum. Methods B 30, 185 (1988), split-step channeling with an absorptive
thermal-scattering potential).

**Divergence.** A beam of independent electrons with Gaussian angular spread is
an incoherent sum over entry tilts. :func:`axial_incidence_nodes` lays a square
midpoint grid over ``+-4 sigma`` per axis (the planar
:func:`~.exit_angular.incidence_nodes` convention), keeps the nodes inside the
``4 sigma`` disc and weights them by the normalized 2D Gaussian. The comb of
each tilt is then shifted by that tilt, filling the map continuously once the
node spacing ``8 sigma / n`` is below the pixel.

**Absorption.** ``W = (hbar v / 2) C(r, r)`` is the no-jump loss of
:func:`~.axial_dissipator.axial_dissipator` on the one-cell grid. Flux removed by
``W`` was scattered incoherently (TDS, electronic excitation) and is **not** in
the map; the map is the coherent exit distribution and ``surviving`` its norm
(``1`` when ``W = 0``). Incoherent exit maps (MCWF jumps) break the cell
periodicity and are out of scope.

**Binning.** Order weights are deposited bilinearly on a square pixel grid of
free pitch, centred on the axis. Bilinear deposition preserves each order's
total weight and mean angle when its four neighbours lie in the window; weight
falling outside is dropped, so ``map.sum() <= surviving``.

Assumptions: continuum ``U`` and local ``W`` of the profile (first Born per
atom, Einstein displacements), transverse kinetic mass ``gamma m``, small
angles, no energy loss, no surface refraction, no detector response.
Lengths in angstroms, energies in eV, angles in radians, in the profile's
transverse frame ``(axes[0], axes[1])``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from ..._units import reject_quantity
from ...materials.channeling.axial import AxialProfile
from .exit_angular import DIVERGENCE_SPAN
from .realspace import RealSpaceGrid, SplitStepPropagator, _cupy
from .states import momentum_inv_ang

#: Midpoint nodes per axis of the square divergence grid (``~pi (n/2)^2`` kept).
DEFAULT_DIVERGENCE_NODES = 30
#: Tilts propagated per batch.
DEFAULT_TILT_BATCH = 128


def axial_incidence_nodes(
    tilt_rad: tuple[float, float],
    divergence_rms_rad: float,
    n_per_axis: int = DEFAULT_DIVERGENCE_NODES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(tilts (n, 2), weights (n,))`` for an isotropic Gaussian divergence.

    Uniform midpoint nodes over ``+-4 sigma`` per axis (as
    :func:`~.exit_angular.incidence_nodes`) restricted to the ``4 sigma`` disc,
    with normalized weights ``exp(-|theta - tilt|^2 / 2 sigma^2)``. A zero
    divergence returns the single node ``tilt``.

    Validation: cr-axial-exit-angular
    """
    reject_quantity(tilt_rad, field="axial_incidence_nodes.tilt_rad", unit="rad")
    reject_quantity(
        divergence_rms_rad, field="axial_incidence_nodes.divergence_rms_rad", unit="rad"
    )
    centre = np.asarray(tilt_rad, dtype=float)
    if centre.shape != (2,) or not np.all(np.isfinite(centre)):
        raise ValueError("tilt_rad needs two finite angles")
    sigma = float(divergence_rms_rad)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("divergence_rms_rad must be finite and non-negative")
    if sigma == 0.0:
        return centre[None, :].copy(), np.ones(1)
    if not isinstance(n_per_axis, Integral) or n_per_axis < 2:
        raise ValueError("n_per_axis must be an integer >= 2 for a divergent beam")
    half = DIVERGENCE_SPAN * sigma
    step = 2.0 * half / n_per_axis
    axis = -half + step * (np.arange(n_per_axis) + 0.5)
    dx, dy = (a.ravel() for a in np.meshgrid(axis, axis, indexing="ij"))
    r2 = dx**2 + dy**2
    keep = r2 <= half**2
    tilts = centre + np.stack([dx[keep], dy[keep]], axis=1)
    weights = np.exp(-0.5 * r2[keep] / sigma**2)
    return tilts, weights / weights.sum()


@dataclass(frozen=True)
class AxialExitMap:
    """Coherent exit-angle maps per incident electron.

    Attributes
    ----------
    depths_ang
        Exit depths ``z`` [angstrom].
    theta_x_rad, theta_y_rad
        Pixel centres along the profile frame's ``x`` and ``y`` [rad].
    probability
        Coherent exit probability per pixel, shape ``(depth, n_x, n_y)``.
    surviving
        Coherent norm at each depth (all orders, before windowing).
    tilts_rad, tilt_weights
        Divergence quadrature nodes ``(n, 2)`` [rad] and their weights.
    pixel_rad
        Pixel pitch [rad].
    axes
        Transverse frame ``(x, y, z)`` of the profile (rows).

    Validation: cr-axial-exit-angular
    """

    depths_ang: np.ndarray
    theta_x_rad: np.ndarray
    theta_y_rad: np.ndarray
    probability: np.ndarray
    surviving: np.ndarray
    tilts_rad: np.ndarray
    tilt_weights: np.ndarray
    pixel_rad: float
    axes: np.ndarray

    @property
    def captured(self) -> np.ndarray:
        """Probability inside the angular window per depth (``<= surviving``)."""
        return self.probability.sum(axis=(1, 2))

    @property
    def density_per_rad2(self) -> np.ndarray:
        """``probability / pixel_rad^2`` [1/rad^2]."""
        return self.probability / self.pixel_rad**2


def _validate(kinetic_energy_eV, depths, pixel_rad, half_range_rad, step_ang, shape, batch):
    for name, value in (
        ("kinetic_energy_eV", kinetic_energy_eV),
        ("pixel_rad", pixel_rad),
        ("half_range_rad", half_range_rad),
        ("step_ang", step_ang),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if depths.ndim != 1 or not depths.size or not np.all(np.isfinite(depths)):
        raise ValueError("depths_ang must be a nonempty finite sequence")
    if np.any(depths < 0.0) or np.any(np.diff(depths) <= 0.0):
        raise ValueError("depths_ang must be non-negative and increasing")
    if len(shape) != 2 or any(not isinstance(s, Integral) or s < 2 for s in shape):
        raise ValueError("shape needs two integers >= 2")
    if not isinstance(batch, Integral) or batch < 1:
        raise ValueError("batch must be a positive integer")


def _deposit(xp, angle_x, angle_y, power, origin, pixel, n_pix):
    """Bilinear deposit of ``power`` at ``(angle_x, angle_y)``; outside is dropped.

    Validation: cr-axial-exit-angular
    """
    fx = ((angle_x - origin) / pixel).ravel()
    fy = ((angle_y - origin) / pixel).ravel()
    value = power.ravel()
    ix, iy = xp.floor(fx), xp.floor(fy)
    tx, ty = fx - ix, fy - iy
    ix, iy = ix.astype(np.int64), iy.astype(np.int64)
    total = xp.zeros(n_pix * n_pix)
    for sx, wx in ((0, 1.0 - tx), (1, tx)):
        for sy, wy in ((0, 1.0 - ty), (1, ty)):
            jx, jy = ix + sx, iy + sy
            inside = (jx >= 0) & (jx < n_pix) & (jy >= 0) & (jy < n_pix)
            total += xp.bincount(
                (jx * n_pix + jy)[inside],
                weights=(value * wx * wy)[inside],
                minlength=n_pix * n_pix,
            )
    return total.reshape(n_pix, n_pix)


def coherent_axial_exit_map(
    profile: AxialProfile,
    kinetic_energy_eV: float,
    depths_ang: float | Iterable[float],
    *,
    pixel_rad: float,
    half_range_rad: float,
    tilt_rad: tuple[float, float] = (0.0, 0.0),
    divergence_rms_rad: float = 0.0,
    mechanisms: tuple[str, ...] = ("tds", "electronic"),
    shape: tuple[int, int] = (32, 32),
    step_ang: float = 5.0,
    n_divergence: int = DEFAULT_DIVERGENCE_NODES,
    backend: str = "cpu",
    batch: int = DEFAULT_TILT_BATCH,
) -> AxialExitMap:
    """Return coherent/absorptive exit-angle maps of a plane-wave beam on one cell.

    Each divergence node (:func:`axial_incidence_nodes`) is a plane wave of
    unit norm per cell, propagated on the one-cell grid of ``shape`` samples
    with ``U`` of ``profile`` and ``W`` of ``mechanisms`` (``()`` for none) by
    a batched :class:`~.realspace.SplitStepPropagator`. At each depth the
    order weights ``|c_G|^2`` sit at ``tilt + G / p`` and are binned bilinearly
    on pixels of pitch ``pixel_rad`` with centres ``k pixel_rad`` for
    ``|k pixel_rad| <= half_range_rad``. ``tilt_rad`` and the map share the
    profile frame. ``backend="cuda"`` keeps states and binning on the device.

    The scope (single-cell Bloch reduction, absorption as coherent loss, no
    incoherent redistribution) is described in the module docstring.

    Validation: cr-axial-exit-angular
    """
    for name, value, unit in (
        ("kinetic_energy_eV", kinetic_energy_eV, "eV"),
        ("depths_ang", depths_ang, "angstrom"),
        ("pixel_rad", pixel_rad, "rad"),
        ("half_range_rad", half_range_rad, "rad"),
        ("step_ang", step_ang, "angstrom"),
    ):
        reject_quantity(value, field=f"coherent_axial_exit_map.{name}", unit=unit)
    depths = np.atleast_1d(np.asarray(depths_ang, dtype=float))
    _validate(kinetic_energy_eV, depths, pixel_rad, half_range_rad, step_ang, shape, batch)
    tilts, weights = axial_incidence_nodes(tilt_rad, divergence_rms_rad, n_divergence)
    grid = RealSpaceGrid(profile.period_ang, tuple(shape))
    potential = profile.on_grid((1, 1), grid.shape)
    absorptive = None
    if mechanisms:
        from .axial_dissipator import axial_dissipator

        absorptive = axial_dissipator(
            profile, kinetic_energy_eV, (1, 1), grid.shape, mechanisms=tuple(mechanisms)
        ).absorptive_eV
    p = momentum_inv_ang(kinetic_energy_eV)
    n_half = int(np.floor(half_range_rad / pixel_rad + 1e-9))
    centres = pixel_rad * np.arange(-n_half, n_half + 1)
    n_pix = centres.size
    count = grid.shape[0] * grid.shape[1]
    # |c_G|^2 = A |FFT(phi)_G|^2 / N^2 (Parseval with norm = (A/N) sum |phi|^2).
    scale = grid.area_ang2 / count**2
    plane_wave = np.full(grid.shape, 1.0 / np.sqrt(grid.area_ang2), dtype=complex)
    if backend not in ("cpu", "cuda"):
        raise ValueError("backend must be 'cpu' or 'cuda'")
    xp: Any = np if backend == "cpu" else _cupy()
    maps = xp.zeros((depths.size, n_pix, n_pix))
    surviving = xp.zeros(depths.size)
    for start in range(0, len(tilts), batch):
        chunk = tilts[start : start + batch]
        solver = SplitStepPropagator(
            grid,
            kinetic_energy_eV,
            potential_eV=potential,
            absorptive_eV=absorptive,
            tilt_rad=chunk,
            backend=backend,
        )
        w = xp.asarray(weights[start : start + batch])[:, None, None]
        angle_x = xp.asarray(chunk[:, 0, None, None] + grid.kx[None, :, None] / p)
        angle_y = xp.asarray(chunk[:, 1, None, None] + grid.ky[None, None, :] / p)
        angle_x, angle_y = xp.broadcast_arrays(angle_x, angle_y)
        trajectory = solver.trajectory(plane_wave, depths, step_ang=step_ang)
        for i, (_, phi) in enumerate(trajectory):
            power = w * scale * xp.abs(xp.fft.fft2(phi)) ** 2
            surviving[i] += power.sum()
            maps[i] += _deposit(xp, angle_x, angle_y, power, centres[0], pixel_rad, n_pix)
    if xp is not np:
        maps, surviving = xp.asnumpy(maps), xp.asnumpy(surviving)
    return AxialExitMap(
        depths_ang=depths,
        theta_x_rad=centres,
        theta_y_rad=centres.copy(),
        probability=np.asarray(maps),
        surviving=np.asarray(surviving),
        tilts_rad=tilts,
        tilt_weights=weights,
        pixel_rad=float(pixel_rad),
        axes=np.asarray(profile.axes),
    )


__all__ = [
    "DEFAULT_DIVERGENCE_NODES",
    "AxialExitMap",
    "axial_incidence_nodes",
    "coherent_axial_exit_map",
]
