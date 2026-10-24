"""Diagnostic bridge from portable transverse source specs to planar density evolution.

This is *not* a production transport or channeling-radiation source adapter.
A sampled quantum wavepacket needs a physically consistent entry/handoff
sampling law (including off-band momenta) and identical live/rescored CR
sources before it may appear in a :class:`~pyrite.montecarlo.case.Case`.

The projection uses exactly the established #508 Bloch basis and the #506
static planar solver, without creating cases, writing results, or consuming
primary-electron Monte Carlo random numbers. ``incidence_rad`` must be the
**effective refracted** angle to the planes, not the laboratory beam tilt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .quantum import QuantumSetup
from .quantum_state import TransverseStateSpec

if TYPE_CHECKING:
    from .montecarlo.channeling.density import DensityMatrixResult


def _readonly(value: np.ndarray) -> np.ndarray:
    """Keep consumers from mutating stored projection arrays through NumPy views."""
    arr = np.ascontiguousarray(value)
    return np.frombuffer(arr.tobytes(), dtype=arr.dtype).reshape(arr.shape)


def _positive(name: str, value: float, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (number < 0 if zero else number <= 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} and finite")
    return number


def _setup_supported(setup: QuantumSetup) -> None:
    if not isinstance(setup, QuantumSetup):
        raise TypeError("setup must be a QuantumSetup")
    if setup.region.kind == "axial":
        # #624 central admission is shared with all production consumers.
        # On main it is exported by axial_spec; on the #624 continuation it
        # delegates to the common _channeling_admission contract. Do not
        # construct planar kernels or allocate a quantum grid first.
        from .campaign.axial_spec import refuse_unverified_axial_run

        refuse_unverified_axial_run()
    if (
        setup.region.kind != "planar"
        or setup.dynamics.kind != "channeling"
        or setup.numerics.representation != "bloch"
        or setup.numerics.method != "density-matrix"
    ):
        raise ValueError(
            "planar source diagnostics require a planar channeling region and "
            "Bloch density-matrix dynamics; axial/other solvers are not supported"
        )


def _check_source_energy(spec: TransverseStateSpec, kinetic_energy_eV: float) -> None:
    """Do not reinterpret energy-calibrated apertures or samples at another energy."""
    if spec.kind in ("aperture_probe", "sampled"):
        energy = dict(spec.parameters).get("kinetic_energy_eV")
        if energy is not None and not math.isclose(
            float(energy), kinetic_energy_eV, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError(
                f"{spec.kind} state is calibrated for {energy:g} eV, "
                f"not {kinetic_energy_eV:g} eV"
            )
    for _, child in spec.terms:
        _check_source_energy(child, kinetic_energy_eV)


@dataclass(frozen=True)
class PlanarStateProjection:
    """Read-only diagnostic Bloch entry population for one source and solver.

    ``plane_density`` uses plane-wave orders; ``subspace_density`` is the
    retained eigenbasis density. ``free_population`` is in-band unretained
    eigenpopulation; ``outside_population`` is momentum outside the basis.
    Both are *entry* flux, not evolved dechanneling flux. The total must be 1
    within the projector's numerical tolerance.

    :attr:`initial` produces the precise initial-state argument accepted by
    :meth:`DensityMatrixSolver.propagate`, with a writable copy of the density.
    """

    state_digest: str
    crystal: str
    plane_hkl: tuple[int, int, int]
    kinetic_energy_eV: float
    temperature_K: float
    margin_eV: float
    incidence_rad: float
    impact: str
    setup: QuantumSetup
    plane_density: np.ndarray
    subspace_density: np.ndarray
    free_population: np.ndarray
    outside_population: float

    @property
    def retained_population(self) -> float:
        return float(np.einsum("rii->", self.subspace_density).real)

    @property
    def entry_population(self) -> float:
        return float(self.free_population.sum()) + self.outside_population

    @property
    def total_population(self) -> float:
        return self.retained_population + self.entry_population

    @property
    def initial(self) -> tuple[np.ndarray, float]:
        """Return an independent density and the complete nonretained entry flux."""
        return self.subspace_density.copy(), self.entry_population


def _solver_for(projection: PlanarStateProjection):
    """Reuse expensive cached kernels, never mutate a cached production solver."""
    from .montecarlo.channeling.density import DensityMatrixSolver
    from .montecarlo.channeling.handoff import zone_solver

    profile, cached = zone_solver(
        projection.crystal, projection.plane_hkl, projection.temperature_K,
        projection.kinetic_energy_eV, projection.setup.numerics.basis_size,
        projection.setup.numerics.bloch_cells, projection.margin_eV,
    )
    return DensityMatrixSolver(
        profile, projection.kinetic_energy_eV, cached.kernel,
        margin_eV=projection.margin_eV,
    )


def project_planar_source(
    state: TransverseStateSpec,
    *,
    setup: QuantumSetup,
    crystal: str,
    kinetic_energy_eV: float,
    incidence_rad: float = 0.0,
    impact: str = "averaged",
    temperature_K: float = 293.15,
    margin_eV: float | None = None,
    base_dir: str | Path | None = None,
) -> PlanarStateProjection:
    """Project a portable source into the *existing* planar Bloch density solver.

    The angle is signed and must already include the surface refraction
    computed by :func:`pyrite.materials.channeling.entry.plane_incidence`
    for an actual Scene. Neither the beam-frame orientation nor an arbitrary
    crystal alignment is inferred from the source specification.

    ``impact='averaged'`` averages the probe position over a lattice period;
    ``'fixed'`` preserves coherent inter-order phase. A 2-D OAM beam has its
    in-plane momentum traced out and the planar solver cannot resolve OAM.
    """
    _setup_supported(setup)
    if not isinstance(state, TransverseStateSpec):
        raise TypeError("state must be a TransverseStateSpec")
    if not isinstance(crystal, str) or not crystal.strip():
        raise ValueError("crystal must be a nonempty catalog crystal identifier")
    energy = _positive("kinetic_energy_eV", kinetic_energy_eV)
    temperature = _positive("temperature_K", temperature_K)
    from .montecarlo.channeling.density import DECHANNELING_MARGIN_EV

    margin = (_positive("margin_eV", margin_eV)
              if margin_eV is not None else float(DECHANNELING_MARGIN_EV))
    if (
        isinstance(incidence_rad, bool)
        or not isinstance(incidence_rad, (int, float))
        or not math.isfinite(incidence_rad)
        or abs(incidence_rad) >= math.pi / 2
    ):
        raise ValueError("incidence_rad must be finite and strictly between -pi/2 and pi/2")
    if impact not in ("averaged", "fixed"):
        raise ValueError("impact must be 'fixed' or 'averaged'")
    _check_source_energy(state, energy)
    from .montecarlo.channeling.states import planar_entry_density

    plane = setup.region.plane_hkl
    assert plane is not None
    metadata = PlanarStateProjection(
        state_digest=state.digest, crystal=crystal, plane_hkl=plane,
        kinetic_energy_eV=energy, temperature_K=temperature,
        margin_eV=margin, incidence_rad=float(incidence_rad), impact=impact,
        setup=setup, plane_density=np.empty((0,)), subspace_density=np.empty((0,)),
        free_population=np.empty((0,)), outside_population=0.0,
    )
    solver = _solver_for(metadata)
    runtime = state.to_runtime(base_dir=base_dir)
    wave_density, outside = planar_entry_density(
        solver, runtime, incidence_rad=float(incidence_rad), impact=impact
    )
    rho, free, outside = solver.entry_from_plane(wave_density, outside)
    projected = PlanarStateProjection(
        state_digest=metadata.state_digest, crystal=crystal, plane_hkl=plane,
        kinetic_energy_eV=energy, temperature_K=temperature,
        margin_eV=margin, incidence_rad=float(incidence_rad), impact=impact,
        setup=setup, plane_density=_readonly(wave_density),
        subspace_density=_readonly(rho), free_population=_readonly(free),
        outside_population=float(outside),
    )
    if (
        not np.all(np.isfinite(rho))
        or not np.all(np.isfinite(free))
        or not math.isfinite(projected.total_population)
        or not math.isclose(projected.total_population, 1.0, rel_tol=0, abs_tol=2e-3)
    ):
        raise ValueError("quantum entry population is not normalized; check quadrature and grid")
    return projected


def propagate_planar_source(
    projection: PlanarStateProjection,
    *,
    depth_ang: float,
    stopping_eV_per_ang: float = 0.0,
    record_every: int = 1,
    record_metrics: bool = False,
) -> DensityMatrixResult:
    """Evolve a projected state diagnostically, without Monte Carlo handoff.

    The single-particle propagation omits native electron trajectories,
    their launch distributions, detector scoring and spectrum persistence.
    No implicit band-edge exit is treated as a physical dechanneling event.
    """
    if not isinstance(projection, PlanarStateProjection):
        raise TypeError("projection must be a PlanarStateProjection")
    depth = _positive("depth_ang", depth_ang)
    stopping = _positive("stopping_eV_per_ang", stopping_eV_per_ang, zero=True)
    cap = projection.setup.region.zone_max_ang
    if cap is not None and depth > cap:
        raise ValueError("depth_ang exceeds the quantum region's physical zone_max_ang")
    if type(record_every) is not int or record_every <= 0:
        raise ValueError("record_every must be a positive integer")
    if not isinstance(record_metrics, bool):
        raise TypeError("record_metrics must be a bool")
    solver = _solver_for(projection)
    return solver.propagate(
        depth, projection.setup.numerics.step_ang,
        initial=projection.initial,
        stopping_eV_per_ang=stopping,
        record_every=record_every,
        record_metrics=record_metrics,
    )
