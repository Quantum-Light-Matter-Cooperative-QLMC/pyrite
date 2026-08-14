"""Public Python simulation entry points."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import __version__
from .campaign.model import BremSource, EmissionMode, Numerics, Scene, XrayDispersion
from .campaign.profiles import case_content_key
from .campaign.sweep import Sweep as LegacySweep
from .campaign.sweep import build_cases
from .detectors import Detector
from .montecarlo import Case, run_case
from .montecarlo._backend import BACKEND
from .results.model import Result


def build_case(scene: Scene, numerics: Numerics) -> Case:
    """Lower one resolved scene to the existing typed transport input."""
    convergence = numerics.convergence
    legacy = LegacySweep(
        material=scene.target.material,
        beam=scene.beam,
        target=scene.target,
        detector=scene.detector,
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


def simulate(
    beam: Any,
    target: Any,
    detector: Detector,
    *,
    numerics: Numerics | None = None,
    emission: EmissionMode = "incoherent",
    xray_dispersion: XrayDispersion = "vacuum",
    brem_source: BremSource = "mc",
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
        emission=emission,
        xray_dispersion=xray_dispersion,
        brem_source=brem_source,
    )
    if resolved_numerics.backend not in {"auto", BACKEND.name}:
        raise ValueError(
            f"Numerics.backend={resolved_numerics.backend!r} does not match the active "
            f"backend {BACKEND.name!r}; select the backend before importing pyrite"
        )
    case = build_case(scene, resolved_numerics)
    output = run_case(case, transport_core=resolved_numerics.transport_core)
    background_energy = output.get("E_grid_brem", output["E_grid"])
    background = output.get("brem_wide", output["brem"])
    return Result(
        energy_eV=np.asarray(output["E_grid"]),
        spectrum=np.asarray(output["spec"]),
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
    )


__all__ = ["build_case", "simulate"]
