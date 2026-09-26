from dataclasses import replace
from typing import TYPE_CHECKING

from ..instrument import PlanarDetector
from ..montecarlo import Case

if TYPE_CHECKING:
    from .model import Numerics, Scene, Sweep
from .sweep import Sweep as LegacySweep
from .sweep import build_cases


def build_case(scene: Scene, numerics: Numerics) -> Case:
    """Lower one resolved scene to the existing typed transport input."""
    convergence = numerics.convergence
    scalar_detector = (
        scene.detector.scalar_detector()
        if isinstance(scene.detector, PlanarDetector)
        else scene.detector
    )
    legacy = LegacySweep(
        material=scene.target.material,
        beam=scene.beam,
        target=scene.target,
        detector=scalar_detector,
        n_families=convergence.n_families,
        max_reflections=convergence.max_reflections,
        spec_chunk=numerics.spec_chunk,
        brem_chunk=numerics.brem_chunk,
        mosaic_route=convergence.mosaic_route,
        mosaic_nodes=convergence.mosaic_nodes,
    )
    cases = build_cases(
        legacy,
        n_electrons=numerics.n_electrons,
        n_electrons_brem=numerics.n_electrons_brem,
        coherent_emission=scene.emission in {"coherent", "both"},
        straggling=numerics.straggling,
        energy_model=numerics.energy_model,
        max_dE_frac=numerics.max_dE_frac,
        inelastic_model=numerics.inelastic_model,
        inelastic_cutoff_eV=numerics.inelastic_cutoff_eV,
        elastic_model=numerics.elastic_model,
        bremsstrahlung_model=numerics.bremsstrahlung_model,
    )
    if len(cases) != 1:  # Scene rejects every implicit multi-value field.
        raise RuntimeError(f"one Scene lowered to {len(cases)} cases")
    return cases[0]


def build_sweep_cases(sweep: Sweep, numerics: Numerics | None = None) -> list[Case]:
    """Lower a public sweep; preserve exact D7 expansion for converted sweeps."""
    if sweep._legacy_source is not None:
        old_sweep, settings = sweep._legacy_source
        return build_cases(
            old_sweep,
            n_electrons=settings.n_electrons,
            n_electrons_brem=settings.n_electrons_brem,
            coherent_emission=settings.coherent_emission,
            straggling=getattr(settings, "straggling", False),
            energy_model=getattr(settings, "energy_model", "frozen"),
            max_dE_frac=getattr(settings, "max_dE_frac", 0.0),
            inelastic_model=getattr(settings, "inelastic_model", "continuous"),
            inelastic_cutoff_eV=getattr(settings, "inelastic_cutoff_eV", None),
            elastic_model=getattr(settings, "elastic_model", "elsepa"),
            bremsstrahlung_model=getattr(settings, "bremsstrahlung_model", "eedl"),
        )
    resolved = Numerics() if numerics is None else numerics
    cases = []
    for index, (label, scene) in enumerate(sweep.expand()):
        case = build_case(scene, resolved)
        if label:
            case = replace(case, name=f"{case.name} {label}")
        cases.append(replace(case, seed=index + 1))
    return cases
