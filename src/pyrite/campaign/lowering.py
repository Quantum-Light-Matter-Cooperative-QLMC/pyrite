from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .._energy_bins import EnergyBins
from .._numerics import Numerics, Physics, resolve_physics
from ..instrument import PlanarDetector
from ..montecarlo import Case
from .channeling_cases import resolve_channeling
from .transition_cases import resolve_shell_ionization, resolve_transition_radiation

if TYPE_CHECKING:
    from .model import Scene, Sweep
from .sweep import SourceDirection, build_cases
from .sweep import Sweep as LegacySweep


def scene_physics(scene: Scene, numerics: Numerics) -> Physics:
    """Resolve the scene's model policy."""
    physics = resolve_physics(scene.physics, numerics)
    if scene.quantum is not None:
        physics, numerics = scene.quantum.lower(physics, numerics)
    physics.validate_transport(numerics.energy_model)
    if scene.beam.state is not None:
        from ..montecarlo.channeling.planar_injection import source_record
        from ..montecarlo.transverse import TransverseDistribution

        if physics.channeling_model != "planar":
            raise ValueError("Beam.state production requires planar channeling")
        if isinstance(scene.beam.transverse, TransverseDistribution):
            raise ValueError("Beam.state cannot combine with classical Twiss slopes")
        if scene.beam.energy_spread_frac not in (None, 0.0):
            raise ValueError("Beam.state requires a monoenergetic source")
        source_record(scene.beam.state)  # fail before materials and case construction
    if physics.channeling_model == "axial":
        # Refuse before material lookup, any case construction, or output writes.
        # Physical source and handoff cannot be inferred from an FFT sink (#623).
        from .axial_spec import refuse_unverified_axial_run

        refuse_unverified_axial_run()
    return physics


def build_case(scene: Scene, numerics: Numerics | None = None) -> Case:
    """Lower one resolved scene to the existing typed transport input."""
    numerics = scene.numerics if numerics is None else numerics
    physics = scene_physics(scene, numerics)
    if scene.quantum is not None:
        _, numerics = scene.quantum.lower(scene.physics, numerics)
    convergence = numerics.cxr_reflections
    if numerics.precision is not None and isinstance(scene.detector, PlanarDetector):
        raise ValueError("adaptive precision does not support physical detectors; use fixed counts")
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
    return case


def build_sweep_cases(sweep: Sweep, numerics: Numerics | None = None) -> list[Case]:
    """Lower a public sweep; preserve exact D7 expansion for converted sweeps."""
    if sweep._legacy_source is not None:
        old_sweep, settings = sweep._legacy_source
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
        from .legacy import numerics_from_legacy, physics_from_legacy

        physics = physics_from_legacy(settings)
        cases = resolve_shell_ionization(resolve_transition_radiation(cases, physics), physics)
        return resolve_channeling(cases, physics, numerics_from_legacy(old_sweep, settings))
    if numerics is not None:
        sweep = replace(sweep, base=replace(sweep.base, numerics=numerics))
    cases = []
    for index, (label, scene) in enumerate(sweep.expand()):
        case = build_case(scene)
        if label:
            case = replace(case, name=f"{case.name} {label}")
        cases.append(replace(case, seed=index + 1))
    return cases
