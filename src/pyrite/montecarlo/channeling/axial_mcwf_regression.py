"""Bounded four-positive-map regression of a supplied earlier pure state.

This research estimator preserves raw complex sampling noise. It is not a
complete sampled-parent history kernel, photon spectrum, physical escape model
or proof of a material solver's block-linear Lindblad closure.

Validation: cr-mcwf-regression-estimator
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .axial_mcwf import AxialMCWF
from .mcwf import _MCWF_STREAM_SALT, _SM64_GOLDEN
from .states import from_array

_BRANCH_PHASES = (1.0, -1.0, 1j, -1j)
_BRANCH_COEFFICIENTS = (1.0, -1.0, 1j, -1j)
_AUX_VERSION_DOMAIN = 0x9 << 60


def _integer(name, value, minimum, exclusive_maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not minimum <= value < exclusive_maximum
    ):
        raise ValueError(f"{name} must be an integer in [{minimum}, {exclusive_maximum})")
    return int(value)


@dataclass(frozen=True)
class AuxiliaryAddress:
    """Version-1 auxiliary family; fixed-root addresses are injective.

    Primary electron indices must be below 2**63. Auxiliary domain/version
    bits are 0x9 in the high nibble; fields are parent20, earlier-depth10,
    inserted-component4, replica10, branch2, local-electron14 bits. Root seeds
    select the existing MCWF seed family; no cross-root injectivity is claimed.
    """

    root_seed: int
    parent_id: int
    earlier_depth_index: int
    insertion_component: int

    def __post_init__(self):
        for name, upper in (
            ("root_seed", 2**64),
            ("parent_id", 2**20),
            ("earlier_depth_index", 2**10),
            ("insertion_component", 16),
        ):
            _integer(name, getattr(self, name), 0, upper)

    def seed(self, replica: int, branch: int) -> int:
        """Seed whose actual local MCWF keys equal addressed root-family keys.

        If A is the packed branch's base global electron index, the seed is
        ``((root XOR salt) + golden*A) mod 2**64 XOR salt``. Thus local electron
        e uses exactly ``mcwf_stream_key(root, A+e)``. Multiplication by odd
        golden and the existing bijective mixer preserve fixed-root uniqueness.
        """
        replica = _integer("replica", replica, 0, 1024)
        branch = _integer("branch", branch, 0, 4)
        base = (
            _AUX_VERSION_DOMAIN
            | int(self.parent_id) << 40
            | int(self.earlier_depth_index) << 30
            | int(self.insertion_component) << 26
            | replica << 16
            | branch << 14
        )
        shifted = (
            (int(self.root_seed) ^ int(_MCWF_STREAM_SALT)) + int(_SM64_GOLDEN) * base
        ) % 2**64
        return shifted ^ int(_MCWF_STREAM_SALT)


@dataclass(frozen=True)
class MCWFRegressionEstimate:
    """Conditional complex correlation and independent combined-replica errors.

    ``replicas[r,...]`` contains the full four-branch sum. SEMs and real/imag
    covariance of the mean are conditional on the supplied earlier parent;
    they exclude primary-parent sampling uncertainty. Keep these joint replicas
    when contracting multiple depths/components: pointwise SEMs alone discard
    covariance. No PSD projection, clipping or survivor conditioning is applied.
    """

    values: np.ndarray
    sem_real: np.ndarray
    sem_imag: np.ndarray
    covariance_real_imag: np.ndarray
    replicas: np.ndarray
    auxiliary_trajectories: int
    address: AuxiliaryAddress
    insertion_scale: float


def _summarize(replicas, trajectories, address, insertion_scale):
    count = replicas.shape[0]
    centered = replicas - replicas.mean(axis=0)
    covariance = np.sum(centered.real * centered.imag, axis=0) / ((count - 1) * count)
    return MCWFRegressionEstimate(
        replicas.mean(axis=0),
        replicas.real.std(axis=0, ddof=1) / np.sqrt(count),
        replicas.imag.std(axis=0, ddof=1) / np.sqrt(count),
        covariance,
        replicas,
        trajectories,
        address,
        insertion_scale,
    )


def _admit(phi, area, record_shape, replicas, auxiliaries, max_trajectories, max_bytes):
    replicas = _integer("n_replicas", replicas, 2, 1025)
    auxiliaries = _integer("n_auxiliary", auxiliaries, 1, 2**14 + 1)
    _integer("max_auxiliary_trajectories", max_trajectories, 1, 2**63)
    _integer("max_working_bytes", max_bytes, 1, 2**63)
    if not record_shape or any(
        isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in record_shape
    ):
        raise ValueError("record_shape must contain positive integer dimensions")
    cells = 1
    for width in record_shape:
        cells *= int(width)
    if 4 * replicas * auxiliaries > max_trajectories:
        raise ValueError("four-map estimator exceeds auxiliary trajectory budget")
    estimate = 16 * ((replicas + 12) * cells + 4 * phi.size)
    if estimate > max_bytes:
        raise ValueError("four-map estimator exceeds recording/working-memory budget")
    if not np.isfinite(area) or area <= 0 or not phi.size or not np.all(np.isfinite(phi)):
        raise ValueError("parent values must be nonempty finite and cell_area must be positive")
    return replicas, auxiliaries


def four_map_regression(
    phi: np.ndarray,
    apply_insertion: Callable[[np.ndarray], np.ndarray],
    evolve: Callable[[np.ndarray, int, int], np.ndarray],
    *,
    record_shape: tuple[int, ...],
    address: AuxiliaryAddress,
    cell_area: float = 1.0,
    n_replicas: int = 8,
    n_auxiliary: int = 8,
    max_auxiliary_trajectories: int = 4096,
    max_working_bytes: int = 64 * 1024**2,
) -> MCWFRegressionEstimate:
    r"""Estimate Tr[J Phi(|phi><phi| B†)] using four positive auxiliary maps.

    ``apply_insertion(phi)`` returns B phi in the same inner product
    ``cell_area * sum(abs(phi)**2)``. The supplied parent may be subnormalized
    (or zero for an already-lost incident parent). ``evolve(unit_chi, seed, N)``
    must return an INCIDENT-normalized future expectation array of
    ``record_shape`` from N auxiliary trajectories of the SAME linear map.
    Removed auxiliaries contribute zero; do not renormalize their survivors.

    For b=||Bphi||/||phi|| and chi_s=phi+s Bphi/b, s=(1,-1,i,-i), multiply the returned normalized-state
    expectations by ||chi_s||², combine with (1,-1,i,-i)/4 WITHIN each replica,
    and outer factor b, then compute uncertainty across combined replicas.
    Balancing avoids subtracting nearly identical states for tiny currents.
    An identically zero Bphi returns exact zero without auxiliary maps. A zero-norm branch is
    skipped without consuming a random stream. Addresses and all budgets are
    checked before insertion/map calls. This conditional-parent helper does not
    average or reweight an incident parent ensemble automatically.

    Validation: cr-mcwf-regression-estimator
    """
    parent = np.asarray(phi, dtype=complex)
    if not isinstance(address, AuxiliaryAddress):
        raise ValueError("address must be a validated AuxiliaryAddress")
    n_replicas, n_auxiliary = _admit(
        parent,
        cell_area,
        record_shape,
        n_replicas,
        n_auxiliary,
        max_auxiliary_trajectories,
        max_working_bytes,
    )
    replicas = np.zeros((n_replicas, *record_shape), dtype=complex)
    if np.sum(abs(parent) ** 2) * cell_area == 0:
        return _summarize(replicas, 0, address, 0.0)
    inserted = np.asarray(apply_insertion(parent), dtype=complex)
    if inserted.shape != parent.shape or not np.all(np.isfinite(inserted)):
        raise ValueError("insertion must return finite values of the parent shape")
    parent_norm = float(np.sum(abs(parent) ** 2) * cell_area)
    inserted_norm = float(np.sum(abs(inserted) ** 2) * cell_area)
    if inserted_norm == 0:
        return _summarize(replicas, 0, address, 0.0)
    scale = np.sqrt(inserted_norm / parent_norm)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("inserted state scale must be finite and positive")
    inserted = inserted / scale
    trajectories = 0
    for replica in range(n_replicas):
        for branch, (phase, coefficient) in enumerate(
            zip(_BRANCH_PHASES, _BRANCH_COEFFICIENTS, strict=True)
        ):
            chi = parent + phase * inserted
            norm = float(np.sum(abs(chi) ** 2) * cell_area)
            if norm == 0:
                continue
            if not np.isfinite(norm):
                raise ValueError("auxiliary norm is not finite")
            future = np.asarray(
                evolve(chi / np.sqrt(norm), address.seed(replica, branch), n_auxiliary),
                dtype=complex,
            )
            if future.shape != record_shape or not np.all(np.isfinite(future)):
                raise ValueError(
                    "map must return finite incident-normalized expectations of record_shape"
                )
            replicas[replica] += coefficient * scale * norm * future / 4
            trajectories += n_auxiliary
    return _summarize(replicas, trajectories, address, float(scale))


def axial_mcwf_regression(
    solver: AxialMCWF,
    phi: np.ndarray,
    apply_insertion: Callable[[np.ndarray, tuple[float, float]], np.ndarray],
    current_expectation: Callable[[np.ndarray, tuple[float, float]], np.ndarray],
    future_depths_ang: Sequence[float],
    *,
    address: AuxiliaryAddress,
    tilt_inv_ang: tuple[float, float] = (0.0, 0.0),
    current_shape: tuple[int, ...] = (1,),
    step_ang: float = 5.0,
    n_replicas: int = 8,
    n_auxiliary: int = 8,
    batch_size: int = 128,
    max_auxiliary_trajectories: int = 4096,
    max_working_bytes: int = 64 * 1024**2,
) -> MCWFRegressionEstimate:
    """Use actual CPU AxialMCWF futures at the supplied parent's current carrier.

    Depths are relative to the earlier supplied parent, strictly increasing
    nonnegative multiples of step_ang. Array-state reconstruction is verified
    before invoking the solver. The insertion sees the starting carrier in Å⁻¹;
    future current callbacks receive the carrier updated by electronic jumps.
    This adapter supports the supplied material solver, without independently
    establishing that solver's Markov/block-linear physical closure. CUDA is
    explicitly unsupported here. No primary history or primary RNG is consumed.

    Validation: cr-mcwf-regression-estimator
    """
    if solver.backend != "cpu":
        raise ValueError("auxiliary regression adapter currently supports CPU only")
    _integer("batch_size", batch_size, 1, 2**31)
    depths = np.asarray(future_depths_ang, dtype=float)
    tilt = np.asarray(tilt_inv_ang, dtype=float)
    if (
        depths.ndim != 1
        or not depths.size
        or not np.all(np.isfinite(depths))
        or np.any(depths < 0)
        or np.any(np.diff(depths) <= 0)
    ):
        raise ValueError("future depths must be nonempty, finite, increasing and nonnegative")
    if (
        not np.isfinite(step_ang)
        or step_ang <= 0
        or not np.allclose(depths / step_ang, np.rint(depths / step_ang), rtol=0, atol=1e-9)
    ):
        raise ValueError("future depths must be multiples of positive step_ang")
    if tilt.shape != (2,) or not np.all(np.isfinite(tilt)):
        raise ValueError("tilt_inv_ang needs two finite values")
    parent = np.asarray(phi, dtype=complex)
    if parent.shape != solver.grid.shape:
        raise ValueError("parent must have the solver grid shape")
    n_replicas, n_auxiliary = _admit(
        parent,
        solver.grid.cell_area_ang2,
        (len(depths), *current_shape),
        n_replicas,
        n_auxiliary,
        max_auxiliary_trajectories,
        max_working_bytes,
    )
    scratch = 16 * parent.size * (8 * min(batch_size, n_auxiliary) + 8)
    if scratch >= max_working_bytes:
        raise ValueError("auxiliary solver exceeds working-memory budget")
    tilt_pair = (float(tilt[0]), float(tilt[1]))
    spacing = tuple(e / n for e, n in zip(solver.grid.extent_ang, solver.grid.shape, strict=True))

    def evolve(unit_chi, seed, count):
        entry = from_array(
            unit_chi,
            spacing_ang=spacing,
            origin_ang=(0.0, 0.0),
            kinetic_energy_eV=solver.kinetic_energy_eV,
        )
        if not np.allclose(solver.grid.sample(entry), unit_chi, rtol=2e-12, atol=2e-13):
            raise ValueError("array-state reconstruction changed the auxiliary grid state")
        result = solver.run(
            entry,
            tuple(float(depth) for depth in depths),
            count,
            seed=seed,
            step_ang=step_ang,
            tilt_rad=(tilt_pair[0] / solver.momentum, tilt_pair[1] / solver.momentum),
            observable=current_expectation,
            batch_size=batch_size,
        )
        return result.observables

    return four_map_regression(
        parent,
        lambda state: apply_insertion(state, tilt_pair),
        evolve,
        record_shape=(len(depths), *current_shape),
        address=address,
        cell_area=solver.grid.cell_area_ang2,
        n_replicas=n_replicas,
        n_auxiliary=n_auxiliary,
        max_auxiliary_trajectories=max_auxiliary_trajectories,
        max_working_bytes=max_working_bytes - scratch,
    )
