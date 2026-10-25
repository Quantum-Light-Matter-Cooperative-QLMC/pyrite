from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .._energy_bins import EnergyBins
from .._numerics import Numerics, Physics
from ..instrument import PlanarDetector
from ..montecarlo import Case
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
) -> Case:
    """Resolve and lower one scene to the existing typed transport input.

    Explicit physics/numerics override legacy scene settings. Omitted quantum
    inherits ``Scene.quantum``; explicit ``None`` disables the quantum setup.
    """
    return lower_simulation(
        resolve_simulation(scene, numerics=numerics, physics=physics, quantum=quantum)
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
    from .photon_cases import resolve_photon_transport

    return resolve_photon_transport(scene, case, numerics)


def build_sweep_cases(sweep: Sweep, numerics: Numerics | None = None) -> list[Case]:
    """Lower a public sweep; preserve exact D7 expansion for converted sweeps."""
    if sweep._legacy_source is not None:
        old_sweep, settings = sweep._legacy_source
        physics, resolved_numerics = resolve_models(sweep.base, sweep.base.numerics)
        cases = build_cases(
            old_sweep,
            n_electrons=settings.n_electrons,
            n_electrons_brem=settings.n_electrons_brem,
            coherent_emission=settings.coherent_emission,
            straggling=getattr(settings, "straggling", False),
            energy_model=getattr(settings, "energy_model", "frozen"),
            max_dE_frac=getattr(settings, "max_dE_frac", 0.0),
            inelastic_model=getattr(settings, "inelastic_model", "auto"),
            inelastic_cutoff_eV=getattr(settings, "inelastic_cutoff_eV", None),
            secondary_threshold_eV=getattr(settings, "secondary_threshold_eV", None),
            elastic_model=getattr(settings, "elastic_model", "elsepa"),
            bremsstrahlung_model=getattr(settings, "bremsstrahlung_model", "auto"),
            radiative_model=getattr(settings, "radiative_model", "uncoupled"),
            radiative_cutoff_eV=getattr(settings, "radiative_cutoff_eV", None),
            temporal_profile=getattr(settings, "temporal_profile", False),
            pair_production_model=getattr(settings, "pair_production_model", None),
            positron_transport=getattr(settings, "positron_transport", False),
            atomic_electron_deflection=getattr(settings, "atomic_electron_deflection", "kawrakow"),
            precision=getattr(settings, "precision", None),
        )
        cases = resolve_shell_ionization(resolve_transition_radiation(cases, physics), physics)
        return resolve_channeling(cases, physics, resolved_numerics)
    if numerics is not None:
        sweep = replace(sweep, base=replace(sweep.base, numerics=numerics))
    cases = []
    for index, (label, scene) in enumerate(sweep.expand()):
        if scene.physics.photon_transport is not None:
            raise ValueError(
                "photon transport currently requires a scalar Scene; sweeps are unsupported"
            )
        case = build_case(scene)
        if label:
            case = replace(case, name=f"{case.name} {label}")
        cases.append(replace(case, seed=index + 1))
    return cases
