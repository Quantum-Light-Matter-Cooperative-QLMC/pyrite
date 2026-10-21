"""Bounded global finite-grid on-axis photon-source research reference.

No physical axial dissipator, full-Hilbert convergence, recoil, transport
handoff or production CR/CBS source partition is implied by this reference.

Validation: cr-axial-qrt-source-normalization
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from pyrite.materials.channeling.transverse import HBAR2_2M_EV_ANG2, longitudinal_kinematics
from pyrite.materials.crystal import ALPHA_FS, HBARC_EV_ANG, M_E_EV

from .quantum_regression import RegressionCorrelation, spectral_quadratic_form
from .realspace import RealSpaceGrid


@dataclass(frozen=True)
class GlobalAxialSource:
    """Complete represented eigensystem and dimensionless x/y lowering currents.

    Eigenvector columns are orthonormal in the discrete position basis
    ``u=sqrt(cell_area)*phi``. All represented global states are retained,
    including unbound states. ``currents[axis, final, initial]`` keeps positive
    Bohr frequencies only, with no arbitrary splitting of degenerate levels.
    User-supplied density and GKSL jumps for quantum regression must use this
    eigenbasis; density retains incident normalization and may be coherent.
    """

    energies_eV: np.ndarray
    eigenvectors: np.ndarray
    currents: np.ndarray
    hbar_v_eV_ang: float
    kinetic_energy_eV: float
    grid: RealSpaceGrid
    working_bytes: int
    bohr_tolerance_eV: float

    def state_amplitudes(self, phi: np.ndarray) -> np.ndarray:
        """Convert a sampled envelope to global energy amplitudes, without renormalizing."""
        values = np.asarray(phi, dtype=complex)
        if values.shape != self.grid.shape or not np.all(np.isfinite(values)):
            raise ValueError("phi must have finite values of the represented grid shape")
        return self.eigenvectors.conj().T @ (values.ravel() * np.sqrt(self.grid.cell_area_ang2))


def _admit_grid(
    grid,
    potential_eV,
    kinetic_energy_eV,
    tilt_inv_ang,
    max_hilbert_dimension,
    max_working_bytes,
    bohr_tolerance_eV,
):
    """Admit a small represented problem before allocating any dense transform."""
    for name, value in (
        ("max_hilbert_dimension", max_hilbert_dimension),
        ("max_working_bytes", max_working_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    size = int(np.prod(grid.shape))
    estimate = 16 * 32 * size**2
    if size > max_hilbert_dimension or estimate > max_working_bytes:
        raise ValueError("global source exceeds finite-Hilbert dimension/working-memory budget")
    potential = np.asarray(potential_eV)
    if (
        potential.shape != grid.shape
        or np.iscomplexobj(potential)
        or not np.all(np.isfinite(potential))
    ):
        raise ValueError("potential_eV must be finite real values of the grid shape")
    tilt = np.asarray(tilt_inv_ang, dtype=float)
    if tilt.shape != (2,) or not np.all(np.isfinite(tilt)):
        raise ValueError("tilt_inv_ang needs two finite values")
    if not np.isfinite(kinetic_energy_eV) or kinetic_energy_eV <= 0:
        raise ValueError("kinetic_energy_eV must be finite and positive")
    if not np.isfinite(bohr_tolerance_eV) or bohr_tolerance_eV < 0:
        raise ValueError("bohr_tolerance_eV must be finite and nonnegative")
    return potential, tilt, size, estimate


def global_axial_source(
    grid: RealSpaceGrid,
    potential_eV: np.ndarray,
    kinetic_energy_eV: float,
    *,
    tilt_inv_ang: tuple[float, float] = (0.0, 0.0),
    bohr_tolerance_eV: float = 1e-10,
    max_hilbert_dimension: int = 16,
    max_working_bytes: int = 64 * 1024**2,
) -> GlobalAxialSource:
    r"""Build all represented global levels and on-axis positive-frequency currents.

    In the orthonormal finite position basis, the periodic FFT Hamiltonian is
    $H=F\,\mathrm{diag}[\hbar^2|k+k_t|^2/(2\gamma m)]F^\dagger+U$.
    Dimensionless photon currents are the global energy-basis matrices of
    $v_{x,y}/c=\hbar(k_{x,y}+k_{t,x,y})/(\gamma mc)$, retaining matrix
    elements from higher to lower energy only (RWA). Gaps at or below the
    explicit absolute ``bohr_tolerance_eV`` are excluded. Degenerate multiplet
    blocks retain every inter-multiplet current element and density coherence.

    This finite periodic reference includes represented downward free/free
    transitions. It cannot be combined with production CBS by assuming a
    nonoverlapping source split. Off-axis momentum translation and physical
    scattering-sector closure are outside its scope. Admission precedes dense
    Fourier/eigenproblem allocations; this is not a large-grid solver.

    Validation: cr-axial-qrt-source-normalization
    """
    potential, tilt, size, estimate = _admit_grid(
        grid,
        potential_eV,
        kinetic_energy_eV,
        tilt_inv_ang,
        max_hilbert_dimension,
        max_working_bytes,
        bohr_tolerance_eV,
    )
    gamma, beta = longitudinal_kinematics(kinetic_energy_eV)
    if beta <= 0:
        raise ValueError("kinetic energy is below numerical velocity resolution")
    kx, ky = np.meshgrid(grid.kx, grid.ky, indexing="ij")
    momenta = np.stack((kx.ravel(), ky.ravel())) + tilt[:, None]
    # Columns of F are coordinate samples of orthonormal periodic plane waves.
    fourier = np.fft.ifft2(np.eye(size).reshape(size, *grid.shape), axes=(-2, -1), norm="ortho")
    fourier = fourier.reshape(size, size).T
    kinetic = HBAR2_2M_EV_ANG2 / gamma * np.sum(momenta**2, axis=0)
    hamiltonian = (fourier * kinetic) @ fourier.conj().T + np.diag(potential.ravel())
    energies, vectors = np.linalg.eigh(hamiltonian)
    plane_vectors = fourier.conj().T @ vectors
    currents = np.stack(
        [
            plane_vectors.conj().T
            @ (component[:, None] * plane_vectors)
            * HBARC_EV_ANG
            / (gamma * M_E_EV)
            for component in momenta
        ]
    )
    positive = energies[None, :] - energies[:, None] > bohr_tolerance_eV
    currents *= positive[None]
    return GlobalAxialSource(
        energies,
        vectors,
        currents,
        HBARC_EV_ANG * beta,
        kinetic_energy_eV,
        grid,
        estimate,
        bohr_tolerance_eV,
    )


def on_axis_photon_spectrum(
    correlation: RegressionCorrelation,
    photon_energy_eV: np.ndarray,
    kinetic_energy_eV: float,
    *,
    observation_angle_rad: float = 0.0,
    max_working_bytes: int = 64 * 1024**2,
) -> np.ndarray:
    r"""Return on-axis photons/(eV sr incident electron) from velocity/c currents.

    For $D=1-\beta$, sum the two orthogonal transverse polarization intensities:
    $d^2N/(dE\,d\Omega)=\alpha E\sum_\lambda Q_\lambda(ED/\hbar v)
    /(4\pi^2(\hbar v)^2)$. $Q$ is the finite-depth two-time current integral,
    with incident density normalization inherited from ``correlation``.
    Evaluate D as ``1/(gamma**2*(1+beta))`` to avoid high-gamma cancellation.
    The caller must generate the correlation with the same physical
    ``hbar_v=HBARC_EV_ANG*beta`` and kinetic energy: supplied correlation
    arrays do not carry sufficient metadata to infer or verify that match.
    There is no synthetic broadening or survivor renormalization. The current
    convention must be dimensionless velocity/c, not dipole length or charge.

    Exactly two x/y current components are required. Off-axis observations
    are rejected because recoil-phase operators change the Bloch sector.
    Abrupt finite windows have ultraviolet tails; this quadrature requires
    depth/energy-window convergence and is not an all-frequency sum rule.

    Validation: cr-axial-qrt-source-normalization
    """
    if not np.isfinite(observation_angle_rad) or observation_angle_rad != 0:
        raise ValueError("only on-axis observation is supported; off-axis changes Bloch sector")
    energies = np.atleast_1d(np.asarray(photon_energy_eV, dtype=float))
    if (
        energies.ndim != 1
        or not energies.size
        or not np.all(np.isfinite(energies))
        or np.any(energies <= 0)
    ):
        raise ValueError(
            "photon_energy_eV must be a nonempty positive finite one-dimensional array"
        )
    if not np.isfinite(kinetic_energy_eV) or kinetic_energy_eV <= 0:
        raise ValueError("kinetic_energy_eV must be finite and positive")
    if correlation.values.shape[2:] != (2, 2):
        raise ValueError("photon normalization requires exactly two transverse current components")
    if (
        isinstance(max_working_bytes, bool)
        or not isinstance(max_working_bytes, Integral)
        or max_working_bytes < 1
    ):
        raise ValueError("max_working_bytes must be a positive integer")
    overhead = 8 * energies.nbytes
    if overhead >= max_working_bytes:
        raise ValueError("photon normalization exceeds working-memory budget")
    gamma, beta = longitudinal_kinematics(kinetic_energy_eV)
    if beta <= 0:
        raise ValueError("kinetic energy is below numerical velocity resolution")
    hbar_v = HBARC_EV_ANG * beta
    doppler = 1 / (gamma**2 * (1 + beta))
    phase = energies * doppler / hbar_v
    output = np.zeros(energies.size)
    for component in range(2):
        # Restrict to one polarization at a time: unobserved orthogonal
        # polarizations add intensities, never coherent x+y amplitudes.
        single = RegressionCorrelation(
            correlation.depths_ang,
            correlation.values[:, :, component : component + 1, component : component + 1],
            correlation.trace,
            correlation.working_bytes,
        )
        output += spectral_quadratic_form(
            single, phase, max_working_bytes=max_working_bytes - overhead
        )
    return ALPHA_FS * energies * output / (4 * np.pi**2 * hbar_v**2)
