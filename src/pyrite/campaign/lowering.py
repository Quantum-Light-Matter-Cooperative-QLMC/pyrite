from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from .._energy_bins import EnergyBins
from .._mixed_elastic_policy import refuse_mixed_cxr
from .._numerics import Numerics, Physics
from .._sampling import case_budget
from ..instrument import PlanarDetector
from ..montecarlo import Case
from ..observations.spec import Observation
from ..quantum import Channeling
from .channeling_cases import resolve_channeling
from .resolution import UNSET, ResolvedSimulation, _Unset, resolve_models, resolve_simulation
from .transition_cases import (
    resolve_shell_ionization,
    resolve_transition_radiation,
    resolve_transport_mode,
)

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
    channeling: Channeling | None | _Unset = UNSET,
    observation: Observation | None = None,
) -> Case:
    """Resolve and lower one scene to the existing typed transport input.

    Explicit physics/numerics override legacy scene settings. Omitted channeling
    inherits ``Scene.channeling``; explicit ``None`` disables the channeling setup.
    """
    return lower_simulation(
        resolve_simulation(
            scene,
            numerics=numerics,
            physics=physics,
            channeling=channeling,
            observation=observation,
        )
    )


def lower_simulation(request: ResolvedSimulation) -> Case:
    """Lower effective settings without repeating defaults or model validation."""
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
    n_electrons, n_electrons_brem, precision = case_budget(numerics.sampling)
    cases = build_cases(
        legacy,
        n_electrons=n_electrons,
        n_electrons_brem=n_electrons_brem,
        coherent_emission=physics.coherent_emission,
        temporal_profile=numerics.temporal_profile,
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
        precision=precision,
    )
    if len(cases) != 1:  # Scene rejects every implicit multi-value field.
        raise RuntimeError(f"one Scene lowered to {len(cases)} cases")
    # Opt-in channeling zone (#506) and TR component (#103): "off" is unchanged.
    cases = resolve_shell_ionization(resolve_transition_radiation(cases, physics), physics)
    cases = resolve_transport_mode(cases, physics)
    case = resolve_channeling(cases, scene.channeling)[0]
    if scene.beam.state is not None:
        from ..montecarlo.channeling.planar_injection import source_record

        case = replace(case, channeling_source=source_record(scene.beam.state))
    case = _with_run_options([case], numerics)[0]
    from .photon_cases import resolve_photon_transport

    return resolve_photon_transport(scene, case, numerics)


def _with_run_options(cases: list[Case], numerics: Numerics) -> list[Case]:
    """Apply the explicit seed stream and opt-in line roulette to lowered cases.

    Replacing re-runs case validation, which refuses non-additive line routes.
    """
    if numerics.seed is not None:
        cases = [replace(case, seed=numerics.seed + index) for index, case in enumerate(cases)]
    if numerics.line_roulette is not None:
        record = numerics.line_roulette.to_dict()
        cases = [replace(case, line_roulette=record) for case in cases]
    return cases


def _kept_brem_band(old_sweep: LegacySweep, energy_bins: EnergyBins) -> bool:
    """True while the replacement bins still carry the catalog's own brem band."""
    new, old = energy_bins.brem, old_sweep.energy_bins.brem
    if not old_sweep.brem_band or new is None or old is None:
        return old_sweep.brem_band and new is old
    return bool(np.array_equal(new, old))


def build_sweep_cases(
    sweep: Sweep,
    numerics: Numerics | None = None,
    *,
    physics: Physics | None = None,
    channeling: Channeling | None | _Unset = UNSET,
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
            channeling=channeling,
            observation=observation,
        )
        physics, resolved_numerics = request.physics, request.numerics
        refuse_mixed_cxr({"transport_mode": physics.transport_mode})  # one physics, every case
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
                    channeling=request.scene.channeling,
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
            energy_bins = resolved_numerics.energy_bins or EnergyBins()
            old_sweep = replace(
                old_sweep,
                energy_bins=energy_bins,
                # A replacement brem grid is explicit; only the catalog's own
                # band keeps stopping at each case's beam energy (#815).
                brem_band=_kept_brem_band(old_sweep, energy_bins),
                n_families=convergence.n_families,
                max_reflections=convergence.max_reflections,
                mosaic_nodes=convergence.mosaic_nodes,
                mosaic_route=convergence.mosaic_route,
                spec_chunk=resolved_numerics.cxr_chunk,
                brem_chunk=resolved_numerics.brem_chunk,
            )
        n_electrons, n_electrons_brem, precision = case_budget(resolved_numerics.sampling)
        cases = build_cases(
            old_sweep,
            n_electrons=n_electrons,
            n_electrons_brem=n_electrons_brem,
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
            precision=precision,
            check_kinematic_validity=check_kinematic_validity,
        )
        cases = resolve_shell_ionization(resolve_transition_radiation(cases, physics), physics)
        cases = resolve_transport_mode(cases, physics)
        cases = resolve_channeling(cases, request.scene.channeling)
        return _with_run_options(cases, resolved_numerics)
    if not check_kinematic_validity:
        raise ValueError("check_kinematic_validity=False requires a configured legacy sweep")
    overrides = {}
    seed_start = numerics.seed if numerics is not None else sweep.base.numerics.seed
    if numerics is not None:
        overrides["numerics"] = numerics
    if physics is not None:
        overrides["physics"] = physics
    if channeling is not UNSET:
        overrides["channeling"] = channeling
    if overrides:
        sweep = replace(sweep, base=replace(sweep.base, **overrides))
    cases = []
    for index, (label, scene) in enumerate(sweep.expand()):
        if scene.physics.photon_transport is not None:
            raise ValueError(
                "photon transport currently requires a scalar Scene; sweeps are unsupported"
            )
        case = build_case(scene, observation=observation)
        refuse_mixed_cxr(case)
        if label:
            case = replace(case, name=f"{case.name} {label}")
        cases.append(replace(case, seed=index + 1 if seed_start is None else seed_start + index))
    return cases
