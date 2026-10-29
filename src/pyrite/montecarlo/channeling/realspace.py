"""Real-space split-step propagation of 2D transverse states (experimental, #508).

The axial tier's transverse state ``psi(x, y)`` obeys the paraxial equation
(longitudinal depth ``z`` as time; ``hbar v = hbar c beta``)

``i hbar v d psi / dz = [-(hbar^2 / 2 gamma m) nabla_perp^2 + U(r) - i W(r)] psi``,

where ``W >= 0`` is an optional absorptive potential (the no-jump part of
the Lindblad evolution). An eigenbasis of the 2D string potential has
``N ~ 10^3`` states, so precomputed ``N^2 x n_q^2`` jump tables are
infeasible (proposal §9); the state lives instead on a periodic real-space
grid, where the potential is a pointwise phase and the kinetic term is
diagonal in momentum. Strang splitting

``psi(z + dz) = K(dz / 2) V(dz) K(dz / 2) psi(z) + O(dz^3)``

with ``K = exp(-i hbar |k + k_t|^2 dz / 2 gamma m v)`` and
``V = exp(-i (U - i W) dz / hbar v)`` is unitary for ``W = 0`` and
second-order accurate (Strang, SIAM J. Numer. Anal. 5, 506 (1968)).

**Grid.** A ``L_x x L_y`` supercell sampled at ``n_x x n_y`` points, periodic.
States are stored as ``psi(r) = exp(i k_t . r) phi(r)`` with periodic ``phi``:
the mean carrier tilt ``k_t = p theta`` is exact for any angle, so a beam
tilted far from the axis needs no extra sampling. An isolated entry state is
represented by its periodization, ``phi(r) = (1/A) sum_k psi~(k) exp(i k . r)``
on the grid momenta (Poisson summation), which equals ``psi`` when the probe
fits inside the cell.

Lengths in angstroms, energies in eV, angles in radians.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from ..._units import reject_quantity
from ...materials.channeling.transverse import HBAR2_2M_EV_ANG2, longitudinal_kinematics
from ...materials.crystal import HBARC_EV_ANG
from .states import EntryState, MixedState, TransverseState, momentum_inv_ang


@dataclass(frozen=True)
class RealSpaceGrid:
    """Periodic transverse grid.

    Attributes
    ----------
    extent_ang
        Supercell size ``(L_x, L_y)`` [angstrom].
    shape
        Samples ``(n_x, n_y)``; point ``(i, j)`` is at ``(i L_x / n_x, j L_y / n_y)``.

    Validation: cr-axial-realspace-propagation
    """

    extent_ang: tuple[float, float]
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        reject_quantity(self.extent_ang, field="RealSpaceGrid.extent_ang", unit="angstrom")
        if min(self.shape) < 2 or min(self.extent_ang) <= 0.0:
            raise ValueError("grid needs positive extent and at least 2 samples per axis")

    @property
    def area_ang2(self) -> float:
        """Supercell area [angstrom^2]."""
        return self.extent_ang[0] * self.extent_ang[1]

    @property
    def cell_area_ang2(self) -> float:
        """Area per sample [angstrom^2]."""
        return self.area_ang2 / (self.shape[0] * self.shape[1])

    @property
    def x(self) -> np.ndarray:
        """Sample ``x`` coordinates [angstrom]."""
        return np.arange(self.shape[0]) * (self.extent_ang[0] / self.shape[0])

    @property
    def y(self) -> np.ndarray:
        """Sample ``y`` coordinates [angstrom]."""
        return np.arange(self.shape[1]) * (self.extent_ang[1] / self.shape[1])

    @property
    def kx(self) -> np.ndarray:
        """Grid momenta along ``x`` in FFT order [1/angstrom]."""
        return 2.0 * np.pi * np.fft.fftfreq(self.shape[0], self.extent_ang[0] / self.shape[0])

    @property
    def ky(self) -> np.ndarray:
        """Grid momenta along ``y`` in FFT order [1/angstrom]."""
        return 2.0 * np.pi * np.fft.fftfreq(self.shape[1], self.extent_ang[1] / self.shape[1])

    def norm(self, phi: np.ndarray) -> float:
        """Return ``int |psi|^2 dA`` over the supercell."""
        return float(np.sum(np.abs(phi) ** 2) * self.cell_area_ang2)

    def sample(self, state: TransverseState) -> np.ndarray:
        """Return the periodized ``phi`` of a pure 2D state (carrier tilt excluded).

        Momenta above the grid's Nyquist limit are dropped; compare
        :meth:`norm` with 1 to see the lost weight.

        Validation: cr-axial-realspace-propagation
        """
        if not isinstance(state, TransverseState) or state.dimension != 2:
            raise ValueError("the real-space backend takes a pure 2D TransverseState")
        kx = np.fft.fftshift(self.kx)
        ky = np.fft.fftshift(self.ky)
        amplitude = np.fft.ifftshift(state.amplitude(kx, ky))
        return np.fft.ifft2(amplitude) * (self.shape[0] * self.shape[1] / self.area_ang2)

    def ensemble(self, state: EntryState) -> list[tuple[float, np.ndarray]]:
        """Return ``(weight, phi)`` pairs: one for a pure state, one per mixture component."""
        if isinstance(state, MixedState):
            return [(float(w), self.sample(s)) for w, s in state.components]
        return [(1.0, self.sample(state))]


class SplitStepPropagator:
    """Strang split-step propagator on a :class:`RealSpaceGrid`.

    Parameters
    ----------
    grid
        Periodic transverse grid.
    kinetic_energy_eV
        Electron kinetic energy [eV].
    potential_eV
        Real ``U`` on the grid [eV]; ``None`` is free space.
    absorptive_eV
        Optional absorptive potential ``W >= 0`` on the grid [eV].
    tilt_rad
        Mean carrier tilt ``(theta_x, theta_y)`` relative to ``z`` [rad].

    Inputs are numeric-only; Pint quantities (including dimensionless tilts)
    raise ``TypeError``.

    Validation: cr-axial-realspace-propagation
    """

    def __init__(
        self,
        grid: RealSpaceGrid,
        kinetic_energy_eV: float,
        *,
        potential_eV: np.ndarray | None = None,
        absorptive_eV: np.ndarray | None = None,
        tilt_rad: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        for name, value, unit in (
            ("kinetic_energy_eV", kinetic_energy_eV, "eV"),
            ("potential_eV", potential_eV, "eV"),
            ("absorptive_eV", absorptive_eV, "eV"),
            ("tilt_rad", tilt_rad, "rad"),
        ):
            reject_quantity(value, field=f"SplitStepPropagator.{name}", unit=unit)
        if not np.isfinite(kinetic_energy_eV) or kinetic_energy_eV <= 0.0:
            raise ValueError("kinetic_energy_eV must be finite and positive")
        self.grid = grid
        self.gamma, beta = longitudinal_kinematics(float(kinetic_energy_eV))
        self.hbar_v = HBARC_EV_ANG * beta
        p = momentum_inv_ang(kinetic_energy_eV)
        self.tilt_inv_ang = (p * float(tilt_rad[0]), p * float(tilt_rad[1]))
        kx = grid.kx[:, None] + self.tilt_inv_ang[0]
        ky = grid.ky[None, :] + self.tilt_inv_ang[1]
        self._kinetic_eV = HBAR2_2M_EV_ANG2 / self.gamma * (kx**2 + ky**2)
        u = np.zeros(grid.shape) if potential_eV is None else np.asarray(potential_eV, float)
        w = np.zeros(grid.shape) if absorptive_eV is None else np.asarray(absorptive_eV, float)
        if u.shape != grid.shape or w.shape != grid.shape:
            raise ValueError(f"potentials must have the grid shape {grid.shape}")
        if np.any(w < 0.0):
            raise ValueError("absorptive potential must be non-negative")
        self._complex_potential = u - 1j * w
        self._step: float | None = None

    def _factors(self, step: float) -> None:
        if self._step == step:
            return
        self._half_kinetic = np.exp(-0.5j * step * self._kinetic_eV / self.hbar_v)
        self._potential = np.exp(-1j * step * self._complex_potential / self.hbar_v)
        self._step = step

    def evolve(self, phi: np.ndarray, depth_ang: float, *, step_ang: float) -> np.ndarray:
        """Return ``phi`` after ``depth_ang``; the last step is shortened to land exactly."""
        for _, out in self.trajectory(phi, [depth_ang], step_ang=step_ang):
            return out
        return np.array(phi, dtype=complex)

    def trajectory(
        self, phi: np.ndarray, depths_ang: Iterable[float], *, step_ang: float
    ) -> Iterator[tuple[float, np.ndarray]]:
        """Yield ``(z, phi(z))`` at each requested depth (ascending, >= 0)."""
        if step_ang <= 0.0:
            raise ValueError("step_ang must be positive")
        state = np.fft.fft2(np.asarray(phi, dtype=complex))
        z = 0.0
        for target in depths_ang:
            target = float(target)
            if target < z - 1e-12 * max(1.0, z):
                raise ValueError("depths must be ascending and non-negative")
            n = int(np.ceil((target - z) / step_ang - 1e-12))
            if n > 0:
                step = (target - z) / n
                self._factors(step)
                for _ in range(n):
                    state *= self._half_kinetic
                    state = np.fft.fft2(self._potential * np.fft.ifft2(state))
                    state *= self._half_kinetic
                z = target
            yield z, np.fft.ifft2(state)


__all__ = ["RealSpaceGrid", "SplitStepPropagator"]
