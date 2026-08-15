"""Public Python simulation entry points."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from . import __version__
from .campaign.model import (
    BremSource,
    EmissionMode,
    Numerics,
    Scene,
    Sweep,
    XrayDispersion,
)
from .campaign.profiles import case_content_key
from .campaign.sweep import Sweep as LegacySweep
from .campaign.sweep import build_cases
from .detectors import Detector
from .instrument import FilterPlate, PixelScorer, PlanarDetector
from .montecarlo import Case, run_case
from .montecarlo._backend import BACKEND
from .results.model import Result


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
        xray_dispersion=scene.xray_dispersion,
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
            xray_dispersion=settings.xray_dispersion,
        )
    resolved = Numerics() if numerics is None else numerics
    cases = []
    for index, (label, scene) in enumerate(sweep.expand()):
        case = build_case(scene, resolved)
        if label:
            case = replace(case, name=f"{case.name} {label}")
        cases.append(replace(case, seed=index + 1))
    return cases


def build_legacy_cases(old_sweep: Any, settings: Any) -> list[Case]:
    """Internal scan/blaze bridge onto the public sweep lowering path."""
    from .campaign.legacy import adapt_legacy_sweep

    return build_sweep_cases(adapt_legacy_sweep(old_sweep, settings))


def simulate(
    beam: Any,
    target: Any,
    detector: Detector | PlanarDetector,
    *,
    numerics: Numerics | None = None,
    emission: EmissionMode = "incoherent",
    xray_dispersion: XrayDispersion = "vacuum",
    brem_source: BremSource = "mc",
    filters: tuple[FilterPlate, ...] = (),
    pixel_scorer: PixelScorer | None = None,
) -> Result:
    """Simulate one scene without reading or writing a checkpoint.

    This function only composes the established case builder and Monte Carlo
    runner. The returned arrays are intrinsic; score them through the detector
    explicitly when detected units are wanted.
    """
    resolved_numerics = Numerics() if numerics is None else numerics
    if not isinstance(resolved_numerics, Numerics):
        raise TypeError("numerics must be a Numerics")
    scene = Scene(
        beam=beam,
        target=target,
        detector=detector,
        filters=filters,
        pixel_scorer=pixel_scorer,
        emission=emission,
        xray_dispersion=xray_dispersion,
        brem_source=brem_source,
    )
    if scene.filters or scene.pixel_scorer is not None:
        raise NotImplementedError("positioned filters and pixel scoring require the spatial runner")
    if resolved_numerics.backend not in {"auto", BACKEND.name}:
        raise ValueError(
            f"Numerics.backend={resolved_numerics.backend!r} does not match the active "
            f"backend {BACKEND.name!r}; select the backend before importing pyrite"
        )
    case = build_case(scene, resolved_numerics)
    output = run_case(case, transport_core=resolved_numerics.transport_core)
    background_energy = output.get("E_grid_brem", output["E_grid"])
    background = output.get("brem_wide", output["brem"])
    if scene.brem_source != "mc":
        background = np.zeros_like(background)
    coherent = output.get("spec_coherent")
    spectrum = coherent if scene.emission == "coherent" else output["spec"]
    return Result(
        energy_eV=np.asarray(output["E_grid"]),
        spectrum=np.asarray(spectrum),
        background_energy_eV=np.asarray(background_energy),
        background=np.asarray(background),
        case=case,
        provenance={
            "scene": scene,
            "numerics": resolved_numerics,
            "identity_digest": case_content_key(case),
            "backend": BACKEND.name,
            "device": BACKEND.device,
            "versions": {"pyrite": __version__, "numpy": np.__version__},
        },
        coherent_spectrum=(None if coherent is None else np.asarray(coherent)),
    )


__all__ = ["build_case", "build_legacy_cases", "build_sweep_cases", "simulate"]
