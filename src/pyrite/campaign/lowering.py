from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .._energy_bins import EnergyBins
from .._numerics import Numerics, Physics
from ..instrument import PlanarDetector
from ..montecarlo import Case
from ..observations.spec import Observation
from ..quantum import QuantumSetup
from .channeling_cases import resolve_channeling
from .resolution import UNSET, ResolvedSimulation, _Unset, resolve_models, resolve_simulation
from .transition_cases import resolve_shell_ionization, resolve_transition_radiation

if TYPE_CHECKING:
    from .model import Scene, Sweep
from .sweep import SourceDirection, build_cases
from .sweep import Sweep as LegacySweep


def scene_physics(scene: Scene, numerics: Numerics) -> Physics:
    """Compatibility accessor for the shared model-resolution boundary."""
    return resolve_models(scene, numerics)[0]


def build_case(
    scene: Scene,
    numerics: Numerics | None = None,
    *,
    physics: Physics | None = None,
    quantum: QuantumSetup | None | _Unset = UNSET,
    observation: Observation | None = None,
) -> Case:
    """Resolve and lower one scene to the existing typed transport input.

    Explicit physics/numerics override legacy scene settings. Omitted quantum
    inherits ``Scene.quantum``; explicit ``None`` disables the quantum setup.
    """
    return lower_simulation(
        resolve_simulation(
            scene, numerics=numerics, physics=physics, quantum=quantum, observation=observation
        )
    )


def lower_simulation(request: ResolvedSimulation) -> Case:
    """Lower effective settings without repeating defaults or quantum adaptation."""
    scene, physics, numerics = request.scene, request.physics, request.numerics
    convergence = numerics.cxr_reflections
    scalar_detector = (
        scene.detector.scalar_detector()
        if isinstance(scene.detector, PlanarDetector)
        else (SourceDirection() if scene.detector is None else scene.detector)
    )
    target = scene.target_spec
    legacy = LegacySweep(
        material=target.material,
        beam=(
            replace(scene.beam, state=None).to_spec()
            if scene.beam.state is not None
            else scene.beam.to_spec()
        ),
        target=target,
        detector=scalar_detector,
        energy_bins=numerics.energy_bins or EnergyBins(),
        n_families=convergence.n_families,
        max_reflections=convergence.max_reflections,
        spec_chunk=numerics.cxr_chunk,
        brem_chunk=numerics.brem_chunk,
        mosaic_route=convergence.mosaic_route,
        mosaic_nodes=convergence.mosaic_nodes,
    )
    cases = build_cases(
        legacy,
        n_electrons=numerics.cxr_trials,
        n_electrons_brem=numerics.brem_trials,
        coherent_emission=physics.coherent_emission,
        straggling=physics.straggling,
        energy_model=numerics.energy_model,
        max_dE_frac=numerics.max_dE_frac,
        inelastic_model=physics.inelastic_model,
        inelastic_cutoff_eV=physics.inelastic_cutoff_eV,
        secondary_threshold_eV=physics.secondary_threshold_eV,
        elastic_model=physics.elastic_model,
        bremsstrahlung_model=physics.bremsstrahlung_model,
        radiative_model=physics.radiative_model,
        radiative_cutoff_eV=physics.radiative_cutoff_eV,
        pair_production_model=physics.pair_production_model,
        positron_transport=physics.positron_transport,
        atomic_electron_deflection=physics.atomic_electron_deflection,
        precision=numerics.precision,
    )
    if len(cases) != 1:  # Scene rejects every implicit multi-value field.
        raise RuntimeError(f"one Scene lowered to {len(cases)} cases")
    # Opt-in channeling zone (#506) and TR component (#103): "off" is unchanged.
    cases = resolve_shell_ionization(resolve_transition_radiation(cases, physics), physics)
    case = resolve_channeling(cases, physics, numerics)[0]
    if scene.beam.state is not None:
        from ..montecarlo.channeling.planar_injection import source_record

        case = replace(case, channeling_source=source_record(scene.beam.state))
    if numerics.seed is not None:
        case = replace(case, seed=numerics.seed)
    from .photon_cases import resolve_photon_transport

    return resolve_photon_transport(scene, case, numerics)


def build_sweep_cases(
    sweep: Sweep,
    numerics: Numerics | None = None,
    *,
    physics: Physics | None = None,
    quantum: QuantumSetup | None | _Unset = UNSET,
    observation: Observation | None = None,
    check_kinematic_validity: bool = True,
) -> list[Case]:
    """Lower a public sweep; preserve exact D7 expansion for converted sweeps."""
    if sweep._legacy_source is not None:
        old_sweep, settings = sweep._legacy_source
        request = resolve_simulation(
            sweep.base,
            numerics=numerics,
            physics=physics,
            quantum=quantum,
            observation=observation,
        )
        physics, resolved_numerics = request.physics, request.numerics
        if physics.photon_transport is not None:
            raise ValueError(
                "photon transport currently requires a scalar Scene; sweeps are unsupported"
            )
        if observation is not None:
            for _, point in sweep.expand():
                resolve_simulation(
                    point,
                    numerics=request.scene.numerics,
                    physics=request.scene.physics,
                    quantum=request.scene.quantum,
                    observation=observation,
                )
        if observation is not None:
            detector = request.scene.detector
            old_sweep = replace(
                old_sweep,
                detector=(
                    detector.scalar_detector() if isinstance(detector, PlanarDetector) else detector
                ),
            )
        if numerics is not None:
            convergence = resolved_numerics.cxr_reflections
            old_sweep = replace(
                old_sweep,
                energy_bins=resolved_numerics.energy_bins or EnergyBins(),
                n_families=convergence.n_families,
                max_reflections=convergence.max_reflections,
                mosaic_nodes=convergence.mosaic_nodes,
                mosaic_route=convergence.mosaic_route,
                spec_chunk=resolved_numerics.cxr_chunk,
                brem_chunk=resolved_numerics.brem_chunk,
            )
        cases = build_cases(
            old_sweep,
            n_electrons=resolved_numerics.cxr_trials,
            n_electrons_brem=resolved_numerics.brem_trials,
            coherent_emission=physics.coherent_emission,
            straggling=physics.straggling,
            energy_model=resolved_numerics.energy_model,
            max_dE_frac=resolved_numerics.max_dE_frac,
            inelastic_model=physics.inelastic_model,
            inelastic_cutoff_eV=physics.inelastic_cutoff_eV,
            secondary_threshold_eV=physics.secondary_threshold_eV,
            elastic_model=physics.elastic_model,
            bremsstrahlung_model=physics.bremsstrahlung_model,
            radiative_model=physics.radiative_model,
            radiative_cutoff_eV=physics.radiative_cutoff_eV,
            temporal_profile=getattr(settings, "temporal_profile", False),
            pair_production_model=physics.pair_production_model,
            positron_transport=physics.positron_transport,
            atomic_electron_deflection=physics.atomic_electron_deflection,
            precision=resolved_numerics.precision,
            check_kinematic_validity=check_kinematic_validity,
        )
        cases = resolve_shell_ionization(resolve_transition_radiation(cases, physics), physics)
        cases = resolve_channeling(cases, physics, resolved_numerics)
        if resolved_numerics.seed is not None:
            cases = [
                replace(case, seed=resolved_numerics.seed + index)
                for index, case in enumerate(cases)
            ]
        return cases
    if not check_kinematic_validity:
        raise ValueError("check_kinematic_validity=False requires a configured legacy sweep")
    overrides = {}
    seed_start = numerics.seed if numerics is not None else sweep.base.numerics.seed
    if numerics is not None:
        overrides["numerics"] = numerics
    if physics is not None:
        overrides["physics"] = physics
    if quantum is not UNSET:
        overrides["quantum"] = quantum
    if overrides:
        sweep = replace(sweep, base=replace(sweep.base, **overrides))
    cases = []
    for index, (label, scene) in enumerate(sweep.expand()):
        if scene.physics.photon_transport is not None:
            raise ValueError(
                "photon transport currently requires a scalar Scene; sweeps are unsupported"
            )
        case = build_case(scene, observation=observation)
        if label:
            case = replace(case, name=f"{case.name} {label}")
        cases.append(replace(case, seed=index + 1 if seed_start is None else seed_start + index))
    return cases
