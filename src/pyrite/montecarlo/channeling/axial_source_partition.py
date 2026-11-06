"""Exactly-once ownership of finite-grid axial photon transitions (#885).

Research reference: it partitions the represented lowering currents of
:class:`~.axial_qrt_source.GlobalAxialSource`, quantifies what a truncated
emitting subspace omits and maps an on-axis ray to its slab exit path. It does
not establish a material source, a physical dissipator, off-axis recoil or
standard-run admission (``cr-axial-persisted-channel`` stays unverified).

Ownership follows ADR-0018: a transition belongs to the axial source when its
final state is at or below the declared boundary (bound→bound and free→bound
capture). Free→free transitions are coherent-bremsstrahlung-like and are never
scored by the axial source. Lowering currents have no bound→free block.

Units: energies in eV, lengths in Å, spectra in photons/(eV sr incident electron).

Validation: cr-axial-source-partition
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from pyrite.materials.channeling.transverse import longitudinal_kinematics
from pyrite.materials.crystal import ALPHA_FS

from .axial_qrt_source import GlobalAxialSource, on_axis_photon_spectrum
from .quantum_regression import regression_correlation


@dataclass(frozen=True)
class AxialCurrentPartition:
    """Complementary current blocks; ``axial + free_free`` equals the input currents.

    Validation: cr-axial-source-partition
    """

    axial: np.ndarray
    free_free: np.ndarray
    bound: np.ndarray
    boundary_eV: float


@dataclass(frozen=True)
class AxialPartitionSpectra:
    """Separately computed yields whose defect is the coherent cross term.

    ``total`` uses the unsplit current, so ``interference = total - axial -
    free_free`` is a measured signed quantity rather than an assumed zero. The
    blocks have orthogonal final subspaces, so Hamiltonian evolution gives zero;
    only jumps carrying final-state coherence between the blocks make it nonzero.
    Adding ``axial`` to a native free→free score omits ``interference``.

    Validation: cr-axial-source-partition
    """

    total: np.ndarray
    axial: np.ndarray
    free_free: np.ndarray
    interference: np.ndarray


@dataclass(frozen=True)
class AxialSubspaceClosure:
    """Secular on-axis yield per depth of a retained subspace and of the full grid.

    Rates are photons/(sr Å incident electron). ``omitted_rate`` is the exact
    represented remainder, including transitions from unretained initial states.
    The bound covers the represented grid only, not grid or sector refinement.

    Validation: cr-axial-source-partition
    """

    full_rate: float
    retained_rate: float
    omitted_rate: float
    omitted_fraction: float
    retained_population: float


def partition_axial_currents(
    source: GlobalAxialSource, boundary_eV: float
) -> AxialCurrentPartition:
    """Split lowering currents by whether their final state is bound.

    ``boundary_eV`` is a caller-declared energy on the same scale as
    ``source.energies_eV``; choosing a physical barrier is outside this function.
    A boundary inside a degenerate multiplet has no basis-independent meaning
    and is refused, since a rotation within the multiplet would move weight
    between owners.

    Validation: cr-axial-source-partition
    """
    if not isfinite(boundary_eV):
        raise ValueError("boundary_eV must be finite")
    energies = source.energies_eV
    bound = energies <= boundary_eV
    if bound.any() and (~bound).any():
        gap = float(energies[~bound].min() - energies[bound].max())
        if gap <= source.bohr_tolerance_eV:
            raise ValueError("boundary_eV splits a degenerate multiplet")
    axial = source.currents * bound[None, :, None]
    return AxialCurrentPartition(axial, source.currents - axial, bound, float(boundary_eV))


def partition_on_axis_spectra(
    source: GlobalAxialSource,
    partition: AxialCurrentPartition,
    initial_density: np.ndarray,
    jumps_per_sqrt_ang,
    depths_ang,
    photon_energy_eV,
    *,
    absorption_inv_ang=None,
    exit_path_ang=None,
    max_working_bytes: int = 64 * 1024**2,
) -> AxialPartitionSpectra:
    """Compute total, owned and free→free spectra from three independent correlations.

    All blocks evolve under the same full GKSL model, so a jump between owners
    stays in the propagation. ``initial_density`` and jumps use the global
    eigenbasis of ``source``.

    Validation: cr-axial-source-partition
    """
    hamiltonian = np.diag(source.energies_eV)

    def spectrum(currents: np.ndarray) -> np.ndarray:
        correlation = regression_correlation(
            hamiltonian,
            jumps_per_sqrt_ang,
            initial_density,
            currents,
            depths_ang,
            hbar_v_eV_ang=source.hbar_v_eV_ang,
            max_working_bytes=max_working_bytes,
        )
        return on_axis_photon_spectrum(
            correlation,
            photon_energy_eV,
            source.kinetic_energy_eV,
            absorption_inv_ang=absorption_inv_ang,
            exit_path_ang=exit_path_ang,
            max_working_bytes=max_working_bytes,
        )

    total = spectrum(source.currents)
    axial = spectrum(partition.axial)
    free_free = spectrum(partition.free_free)
    return AxialPartitionSpectra(total, axial, free_free, total - axial - free_free)


def _secular_rate(source: GlobalAxialSource, density: np.ndarray, currents: np.ndarray) -> float:
    energies = source.energies_eV
    same_level = np.abs(energies[:, None] - energies[None, :]) <= source.bohr_tolerance_eV
    secular = density * same_level
    gaps = energies[None, :] - energies[:, None]
    weighted = np.einsum("afi,fi,afj,ij->", currents, gaps, currents.conj(), secular, optimize=True)
    gamma, beta = longitudinal_kinematics(source.kinetic_energy_eV)
    doppler = 1 / (gamma**2 * (1 + beta))
    # Golden-rule directional rate alpha E0 |A|^2 / (2 pi hbar_v D) with E0 = gap / D.
    return float(weighted.real) * ALPHA_FS / (2 * np.pi * source.hbar_v_eV_ang * doppler**2)


def emitting_subspace_closure(
    source: GlobalAxialSource, density: np.ndarray, retained: np.ndarray
) -> AxialSubspaceClosure:
    """Quantify the secular on-axis yield omitted by a truncated emitting subspace.

    ``density`` is the incident-normalized eigenbasis density at one depth and
    ``retained`` is a boolean mask over global eigenstates. Coherence between
    nondegenerate levels averages out of the long-path secular rate and is
    dropped; coherence inside a multiplet is kept, so a retained set must hold
    whole multiplets to stay basis independent.

    Validation: cr-axial-source-partition
    """
    rho = np.asarray(density, dtype=complex)
    keep = np.asarray(retained)
    size = source.energies_eV.size
    if rho.shape != (size, size) or not np.all(np.isfinite(rho)):
        raise ValueError("density must be a finite eigenbasis matrix of the source dimension")
    if keep.shape != (size,) or keep.dtype != bool:
        raise ValueError("retained must be a boolean mask over global eigenstates")
    energies = source.energies_eV
    same_level = np.abs(energies[:, None] - energies[None, :]) <= source.bohr_tolerance_eV
    if np.any(same_level & (keep[:, None] != keep[None, :])):
        raise ValueError("retained subspace splits a degenerate multiplet")
    full = _secular_rate(source, rho, source.currents)
    pair = keep[:, None] & keep[None, :]
    retained_rate = _secular_rate(source, rho * pair, source.currents * pair[None])
    omitted = max(full - retained_rate, 0.0)
    return AxialSubspaceClosure(
        full,
        retained_rate,
        omitted,
        omitted / full if full > 0 else 0.0,
        float(np.sum(np.diag(rho)[keep]).real),
    )


def on_axis_exit_path(
    depths_ang,
    axis_lab,
    ray_lab,
    surface_normal,
    thickness_ang: float,
    *,
    angular_tolerance_rad: float = 1e-9,
) -> np.ndarray:
    r"""Photon path to the exit face of a laterally infinite slab, in Å.

    Axial depth $z$ sits at normal depth $z\cos\alpha$ for
    $\cos\alpha=\hat a\cdot\hat n$, with $\hat n$ the inward normal. A forward
    photon along the axis then travels $T/\cos\alpha-z$. Off-axis rays are
    refused, because recoil changes the Bloch sector
    (``cr-axial-qrt-source-normalization``). Side-face exit is not represented.

    Validation: cr-axial-source-partition
    """
    z = np.asarray(depths_ang, dtype=float)
    axis, ray, normal = (
        np.asarray(vector, dtype=float) for vector in (axis_lab, ray_lab, surface_normal)
    )
    for vector in (axis, ray, normal):
        if vector.shape != (3,) or not np.all(np.isfinite(vector)) or not np.any(vector):
            raise ValueError("axis, ray and normal must be finite nonzero 3-vectors")
    axis, ray, normal = (vector / np.linalg.norm(vector) for vector in (axis, ray, normal))
    if np.linalg.norm(np.cross(axis, ray)) > angular_tolerance_rad or axis @ ray <= 0:
        raise ValueError("only on-axis observation is supported; off-axis changes Bloch sector")
    cosine = float(axis @ normal)
    if cosine <= 0:
        raise ValueError("axis must point into the slab along the inward surface normal")
    if not isfinite(thickness_ang) or thickness_ang <= 0:
        raise ValueError("thickness_ang must be finite and positive")
    if z.ndim != 1 or not np.all(np.isfinite(z)) or np.any(z < 0):
        raise ValueError("depths_ang must be finite, nonnegative and one-dimensional")
    exit_depth = thickness_ang / cosine
    if np.any(z > exit_depth * (1 + 1e-12)):
        raise ValueError("depth lies beyond the slab exit face")
    return np.maximum(exit_depth - z, 0.0)
