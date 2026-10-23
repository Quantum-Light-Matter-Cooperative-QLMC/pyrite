"""Bounded ensemble reductions; no individual-state purity averaging.

Validation: cr-quantum-ensemble-metrics
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2, longitudinal_kinematics

RECORDING_BUDGET_BYTES = 128 * 1024**2


@dataclass(frozen=True)
class EnsembleMetrics:
    """Survivor-conditioned empirical ensemble metrics at recorded depths.

    Purity and squared off-diagonal Frobenius coherence are dimensionless.
    Energy mean and standard deviation are eV. Zero trace gives NaN, because
    the conditional ensemble is undefined. Probe SEM excludes MC sampling
    error and finite-ensemble bias. Coherence depends on the named basis.
    """

    purity: np.ndarray
    coherence_squared: np.ndarray
    energy_mean_eV: np.ndarray
    energy_std_eV: np.ndarray
    purity_probe_sem: np.ndarray
    basis: str
    probes: int = 0
    electrons: int | None = None


def _energy_spread(first, second):
    """Clamp cancellation roundoff only; inconsistent moments remain visible."""
    variance = np.asarray(second) - np.asarray(first) ** 2
    tolerance = 64 * np.finfo(float).eps * np.maximum(1.0, np.abs(second) + np.asarray(first) ** 2)
    if np.any(variance < -tolerance):
        raise ValueError("transverse-energy variance is negative beyond roundoff")
    return np.sqrt(np.maximum(variance, 0))


def density_metrics(rho, energies) -> np.ndarray:
    """Exact reductions of orthogonal Bloch blocks, including their weights."""
    diagonal = np.diagonal(rho, axis1=-2, axis2=-1).real
    trace = diagonal.sum()
    if trace <= 0:
        return np.full(4, np.nan)
    square = np.sum(np.abs(rho) ** 2)
    mean = np.sum(diagonal * energies) / trace
    second = np.sum(diagonal * energies**2) / trace
    return np.array(
        [
            square / trace**2,
            (square - np.sum(diagonal**2)) / trace**2,
            mean,
            _energy_spread(mean, second),
        ]
    )


class AxialMetricsObservable:
    """Density-action sketches and exact H moments on a finite physical grid.

    Fixed independent complex Rademacher probes use their own seed. Storage is
    O(depths * probes * grid), never O(depths * electrons * grid). Device
    callbacks return device arrays and retain constants on that device.
    """

    cuda_compatible = True

    def __init__(self, grid, potential_eV, energy_eV, depths: int, probes: int = 32):
        if any(
            isinstance(v, bool) or not isinstance(v, Integral) or v < minimum
            for v, minimum in ((depths, 1), (probes, 2))
        ):
            raise ValueError("depths must be a positive integer and probes an integer >= 2")
        potential = np.asarray(potential_eV)
        if (
            potential.shape != grid.shape
            or np.iscomplexobj(potential)
            or not np.all(np.isfinite(potential))
        ):
            raise ValueError("potential must be finite real values of the grid shape")
        if not np.isfinite(energy_eV) or energy_eV <= 0:
            raise ValueError("kinetic energy must be finite and positive")
        self.grid = grid
        self.probes = probes
        size = int(np.prod(grid.shape))
        self.width = probes * size + 2
        # Declare one output copy; CUDA admission reserves three simultaneous
        # copies for device sums, transferred per-depth values and final stack.
        self.recording_bytes = depths * self.width * np.dtype(complex).itemsize
        self.scratch_bytes = (3 * probes + 8) * size * np.dtype(complex).itemsize
        if depths * self.width * np.dtype(complex).itemsize > RECORDING_BUDGET_BYTES:
            raise ValueError("ensemble metric recording exceeds 128 MiB budget")
        rng = np.random.default_rng(620)
        self._host_probes = (1j ** rng.integers(0, 4, (probes, *grid.shape))).astype(complex)
        self._host_potential = np.asarray(potential_eV)
        self._factor = HBAR2_2M_EV_ANG2 / longitudinal_kinematics(energy_eV)[0]
        self._device = None

    def __call__(self, phi, tilt):
        xp = np
        if hasattr(phi, "__cuda_array_interface__"):
            import cupy as xp
        if self._device is None or self._device[0] is not xp:
            self._device = (
                xp,
                xp.asarray(self._host_probes),
                xp.asarray(self._host_potential),
                xp.asarray(self.grid.kx[:, None]),
                xp.asarray(self.grid.ky[None, :]),
                xp.asarray(self.grid.x[:, None]),
                xp.asarray(self.grid.y[None, :]),
            )
        _, probes, potential, kx, ky, x, y = self._device
        state = phi * xp.exp(1j * (tilt[0] * x + tilt[1] * y)) * np.sqrt(self.grid.cell_area_ang2)
        overlaps = xp.sum(state.conj()[None] * probes, axis=(-2, -1))
        actions = state[None] * overlaps[:, None, None]
        kinetic = self._factor * ((kx + tilt[0]) ** 2 + (ky + tilt[1]) ** 2)
        hphi = xp.fft.ifft2(kinetic * xp.fft.fft2(phi)) + potential * phi
        first = xp.sum(phi.conj() * hphi).real * self.grid.cell_area_ang2
        second = xp.sum(xp.abs(hphi) ** 2) * self.grid.cell_area_ang2
        return xp.concatenate((actions.ravel(), xp.stack((first, second))))

    def sum_batch(self, states, tilts):
        """Sum sketches without allocating an electron-by-probe-by-grid tensor."""
        xp = np
        if hasattr(states, "__cuda_array_interface__"):
            import cupy as xp
        # Initialize the same cached constants as the scalar entry point.
        self(states[0], tilts[0])
        assert self._device is not None
        _, probes, potential, kx, ky, x, y = self._device
        q = xp.asarray(tilts)[:, :, None, None]
        physical = states * xp.exp(1j * (q[:, 0] * x + q[:, 1] * y))
        physical = physical.reshape(len(states), -1) * np.sqrt(self.grid.cell_area_ang2)
        overlaps = physical.conj() @ probes.reshape(self.probes, -1).T
        actions = overlaps.T @ physical
        kinetic = self._factor * ((kx + q[:, 0]) ** 2 + (ky + q[:, 1]) ** 2)
        hphi = xp.fft.ifft2(kinetic * xp.fft.fft2(states, axes=(-2, -1)), axes=(-2, -1))
        hphi += potential * states
        first = xp.sum(states.conj() * hphi).real * self.grid.cell_area_ang2
        second = xp.sum(xp.abs(hphi) ** 2) * self.grid.cell_area_ang2
        return xp.concatenate((actions.ravel(), xp.stack((first, second))))

    def finish(self, values, survival, density, *, electrons=None) -> EnsembleMetrics:
        """Reduce stored ensemble density actions; raw estimates may exceed bounds."""
        actions = values[:, :-2].reshape(len(values), self.probes, -1)
        squares = np.sum(np.abs(actions) ** 2, axis=-1)
        denominator = np.where(survival > 0, survival**2, np.nan)
        purity = squares.mean(axis=1) / denominator
        sem = squares.std(axis=1, ddof=1) / np.sqrt(self.probes) / denominator
        diagonal_square = np.sum((density * self.grid.cell_area_ang2) ** 2, axis=(1, 2))
        mean = values[:, -2].real / np.where(survival > 0, survival, np.nan)
        second = values[:, -1].real / np.where(survival > 0, survival, np.nan)
        return EnsembleMetrics(
            purity,
            purity - diagonal_square / denominator,
            mean,
            _energy_spread(mean, second),
            sem,
            "physical position grid",
            self.probes,
            electrons,
        )
