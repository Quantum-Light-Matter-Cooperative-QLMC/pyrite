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

from ._units import normalize_unit
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
                f"{spec.kind} state is calibrated for {energy:g} eV, not {kinetic_energy_eV:g} eV"
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
    source_dimension: int = 1

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
        projection.crystal,
        projection.plane_hkl,
        projection.temperature_K,
        projection.kinetic_energy_eV,
        projection.setup.numerics.basis_size,
        projection.setup.numerics.bloch_cells,
        projection.margin_eV,
    )
    return DensityMatrixSolver(
        profile,
        projection.kinetic_energy_eV,
        cached.kernel,
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
    kinetic_energy_eV = normalize_unit(
        kinetic_energy_eV,
        "eV",
        field="project_planar_source.kinetic_energy_eV",
        expected="energy",
    )
    temperature_K = normalize_unit(
        temperature_K,
        "kelvin",
        field="project_planar_source.temperature_K",
        expected="temperature",
    )
    incidence_rad = normalize_unit(
        incidence_rad,
        "radian",
        field="project_planar_source.incidence_rad",
        expected="angle",
        kind="angle",
    )
    energy = _positive("kinetic_energy_eV", kinetic_energy_eV)
    temperature = _positive("temperature_K", temperature_K)
    from .montecarlo.channeling.density import DECHANNELING_MARGIN_EV

    margin = (
        _positive(
            "margin_eV",
            normalize_unit(
                margin_eV,
                "eV",
                field="project_planar_source.margin_eV",
                expected="energy",
            ),
        )
        if margin_eV is not None
        else float(DECHANNELING_MARGIN_EV)
    )
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
        state_digest=state.digest,
        crystal=crystal,
        plane_hkl=plane,
        kinetic_energy_eV=energy,
        temperature_K=temperature,
        margin_eV=margin,
        incidence_rad=float(incidence_rad),
        impact=impact,
        setup=setup,
        plane_density=np.empty((0,)),
        subspace_density=np.empty((0,)),
        free_population=np.empty((0,)),
        outside_population=0.0,
        source_dimension=state.dimension,
    )
    solver = _solver_for(metadata)
    runtime = state.to_runtime(base_dir=base_dir)
    wave_density, outside = planar_entry_density(
        solver, runtime, incidence_rad=float(incidence_rad), impact=impact
    )
    rho, free, outside = solver.entry_from_plane(wave_density, outside)
    projected = PlanarStateProjection(
        state_digest=metadata.state_digest,
        crystal=crystal,
        plane_hkl=plane,
        kinetic_energy_eV=energy,
        temperature_K=temperature,
        margin_eV=margin,
        incidence_rad=float(incidence_rad),
        impact=impact,
        setup=setup,
        plane_density=_readonly(wave_density),
        subspace_density=_readonly(rho),
        free_population=_readonly(free),
        outside_population=float(outside),
        source_dimension=state.dimension,
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
    depth = _positive(
        "depth_ang",
        normalize_unit(
            depth_ang, "angstrom", field="propagate_planar_source.depth_ang", expected="length"
        ),
    )
    stopping = _positive(
        "stopping_eV_per_ang",
        normalize_unit(
            stopping_eV_per_ang,
            "eV / angstrom",
            field="propagate_planar_source.stopping_eV_per_ang",
            expected="stopping power",
        ),
        zero=True,
    )
    cap = projection.setup.region.zone_max_ang
    if cap is not None and depth > cap:
        raise ValueError("depth_ang exceeds the quantum region's physical zone_max_ang")
    if type(record_every) is not int or record_every <= 0:
        raise ValueError("record_every must be a positive integer")
    if not isinstance(record_metrics, bool):
        raise TypeError("record_metrics must be a bool")
    solver = _solver_for(projection)
    return solver.propagate(
        depth,
        projection.setup.numerics.step_ang,
        initial=projection.initial,
        stopping_eV_per_ang=stopping,
        record_every=record_every,
        record_metrics=record_metrics,
    )


def _score_unit_inputs(
    divergence_y_rad,
    energy_spread_eV,
    aperture_rad,
    theta_rad,
    phi_rad,
    path_limits_ang,
    surface_normal,
    layers,
):
    """Convert diagnostic fields before entering the channeling scorer."""
    divergence_y_rad = normalize_unit(
        divergence_y_rad,
        "radian",
        field="score_planar_source.divergence_y_rad",
        expected="angle",
        kind="angle",
    )
    energy_spread_eV = normalize_unit(
        energy_spread_eV, "eV", field="score_planar_source.energy_spread_eV", expected="energy"
    )
    aperture_rad = normalize_unit(
        aperture_rad,
        "radian",
        field="score_planar_source.aperture_rad",
        expected="angle",
        kind="angle",
    )
    theta_rad = normalize_unit(
        theta_rad, "radian", field="score_planar_source.theta_rad", expected="angle", kind="angle"
    )
    phi_rad = normalize_unit(
        phi_rad, "radian", field="score_planar_source.phi_rad", expected="angle", kind="angle"
    )
    if path_limits_ang is not None:
        path_limits_ang = normalize_unit(
            path_limits_ang,
            "angstrom",
            field="score_planar_source.path_limits_ang",
            expected="length",
        )
    if surface_normal is not None:
        surface_normal = normalize_unit(
            surface_normal,
            "dimensionless",
            field="score_planar_source.surface_normal",
            expected="direction",
        )
    if layers is not None:
        layers = tuple(
            (
                normalize_unit(
                    top,
                    "angstrom",
                    field=f"score_planar_source.layers[{i}].top",
                    expected="length",
                ),
                normalize_unit(
                    bottom,
                    "angstrom",
                    field=f"score_planar_source.layers[{i}].bottom",
                    expected="length",
                ),
                composition,
            )
            for i, (top, bottom, composition) in enumerate(layers)
        )
    return (
        divergence_y_rad,
        energy_spread_eV,
        aperture_rad,
        theta_rad,
        phi_rad,
        path_limits_ang,
        surface_normal,
        layers,
    )


def score_planar_source(
    projection: PlanarStateProjection,
    *,
    depth_ang: float,
    photon_energy_eV,
    theta_rad,
    phi_rad=0.0,
    sample_ang: float = 500.0,
    divergence_y_rad: float | None = None,
    energy_spread_eV: float = 0.0,
    aperture_rad: float = 0.0,
    absorption: bool = True,
    layers=None,
    include_capture: bool = True,
    surface_normal=None,
    path_limits_ang=None,
):
    """Score diagnostic planar channeling radiation from a *projected* source.

    Reuse the same immutable Bloch entry density as
    :func:`propagate_planar_source`, without reprojecting the original
    wavefunction. This is an incident-normalized **diagnostic** spectrum, not
    a production electron transport or persisted CR implementation.

    ``theta_rad`` / ``phi_rad`` are expressed in the beam/plane frame (x toward
    the plane normal). The projected state's normal momentum spread is already
    present in its density; ``divergence_y_rad`` only broadens directions *in*
    the plane. ``path_limits_ang`` may account for independently specified
    entrance misses and face/cutoff geometry, not dechanneling (already in rho).
    Do not count an external classical normal divergence again. For a 2-D
    source, an explicit in-plane broadening estimate is mandatory: tracing over
    ky removes its distribution, so the Gaussian-width approximation must be
    supplied by the caller rather than silently set to zero. It is not an OAM
    emission model.
    """
    if not isinstance(projection, PlanarStateProjection):
        raise TypeError("projection must be a PlanarStateProjection")
    depth = _positive(
        "depth_ang",
        normalize_unit(
            depth_ang, "angstrom", field="score_planar_source.depth_ang", expected="length"
        ),
    )
    cap = projection.setup.region.zone_max_ang
    if cap is not None and depth > cap:
        raise ValueError("depth_ang exceeds the quantum region's physical zone_max_ang")
    sample = _positive(
        "sample_ang",
        normalize_unit(
            sample_ang, "angstrom", field="score_planar_source.sample_ang", expected="length"
        ),
    )
    if divergence_y_rad is None:
        if projection.source_dimension == 2:
            raise ValueError(
                "a 2-D state requires explicit divergence_y_rad; the planar trace "
                "loses its in-plane momentum distribution"
            )
        divergence_y_rad = 0.0
    (
        divergence_y_rad,
        energy_spread_eV,
        aperture_rad,
        theta_rad,
        phi_rad,
        path_limits_ang,
        surface_normal,
        layers,
    ) = _score_unit_inputs(
        divergence_y_rad,
        energy_spread_eV,
        aperture_rad,
        theta_rad,
        phi_rad,
        path_limits_ang,
        surface_normal,
        layers,
    )
    for name, value in (
        ("divergence_y_rad", divergence_y_rad),
        ("energy_spread_eV", energy_spread_eV),
        ("aperture_rad", aperture_rad),
    ):
        _positive(name, value, zero=True)
    if not isinstance(absorption, bool) or not isinstance(include_capture, bool):
        raise TypeError("absorption and include_capture must be bool")
    energy = np.asarray(
        normalize_unit(
            photon_energy_eV,
            "eV",
            field="score_planar_source.photon_energy_eV",
            expected="energy",
        ),
        dtype=float,
    )
    if (
        energy.ndim != 1
        or energy.size < 2
        or not np.all(np.isfinite(energy))
        or np.any(energy <= 0.0)
        or np.any(np.diff(energy) <= 0.0)
    ):
        raise ValueError("photon_energy_eV must be a finite increasing positive 1-D grid")
    from .montecarlo.channeling.emission import channeling_spectrum

    solver = _solver_for(projection)
    return channeling_spectrum(
        solver,
        depth,
        energy,
        theta_rad,
        phi_rad,
        entry_initial=projection.initial,
        dz_ang=projection.setup.numerics.step_ang,
        sample_ang=sample,
        divergence_y_rad=float(divergence_y_rad),
        energy_spread_eV=float(energy_spread_eV),
        aperture_rad=float(aperture_rad),
        absorption=absorption,
        layers=layers,
        include_capture=include_capture,
        surface_normal=surface_normal,
        path_limits_ang=path_limits_ang,
    )
