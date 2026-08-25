"""Public Python simulation entry points."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from . import __version__
from ._backend import BACKEND
from .campaign.model import (
    BremSource,
    EmissionMode,
    Numerics,
    Scene,
    Sweep,
)
from .campaign.profiles import case_content_key
from .campaign.sweep import Sweep as LegacySweep
from .campaign.sweep import build_cases
from .detectors import Detector
from .instrument import FilterPlate, PixelScorer, PlanarDetector
from .instrument.geometry import (
    angular_tiles,
    filter_path_lengths,
    planar_detector_rays,
)
from .materials import CATALOG, MediumSpec
from .materials.attenuation import linear_attenuation_inv_mm
from .montecarlo import Case, run_case
from .montecarlo.geometry import directions_to_sample_frame
from .montecarlo.runner import run_case_directions
from .montecarlo.transport import STOPPING_MODEL
from .results.model import PixelRayMap, Result, SpatialResult, SpectralFactors


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "__dict__"):
        return {
            key: _canonical_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    raise TypeError(f"cannot canonicalize observation value {type(value).__name__}")


def _array_identity(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _resolved_composition(material: str | MediumSpec) -> tuple[tuple[str, float], ...]:
    if isinstance(material, MediumSpec):
        return material.composition
    if material in CATALOG.media:
        return CATALOG.media[material].composition
    return CATALOG.crystals[material].composition


def _observation_provenance(
    source_digest: str,
    scene: Scene,
    scorer: PixelScorer,
    coefficients: tuple[np.ndarray, ...],
) -> tuple[str, dict[str, Any]]:
    detector = scene.detector
    assert isinstance(detector, PlanarDetector)
    detector_payload = {
        "pose": _canonical_value(detector.pose),
        "size_mm": list(detector.size_mm or ()),
        "pixels": None if detector.pixels is None else _canonical_value(detector.pixels),
        "response": (
            None
            if detector.response is None
            else {
                "type": f"{type(detector.response).__module__}.{type(detector.response).__qualname__}",
                "config": _canonical_value(detector.response),
            }
        ),
    }
    filters_payload = [
        {
            "composition": _canonical_value(_resolved_composition(plate.material)),
            "thickness_mm": plate.thickness_mm,
            "size_mm": list(plate.size_mm),
            "pose": _canonical_value(plate.pose),
        }
        for plate in scene.filters
    ]
    payload = {
        "schema": "pyrite.observation.v1",
        "source_identity_digest": source_digest,
        "detector": detector_payload,
        "filters": filters_payload,
        "angular_shape": list(scorer.angular_shape),
        "attenuation_arrays": [_array_identity(array) for array in coefficients],
        "approximations": {
            "emission_source": "point-source-target-reference",
            "pixel_sampling": "centre-ray",
            "filter_interactions": "primary-attenuation-only",
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _attenuation_matrix(filters: tuple[FilterPlate, ...], energy_eV: np.ndarray) -> np.ndarray:
    if not filters:
        return np.empty((0, energy_eV.size), dtype=float)
    return np.stack([linear_attenuation_inv_mm(plate.material, energy_eV) for plate in filters])


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
    brem_source: BremSource = "mc",
    filters: tuple[FilterPlate, ...] = (),
    pixel_scorer: PixelScorer | None = None,
) -> Result:
    """Simulate one scene without reading or writing a checkpoint.

    A scalar :class:`Detector` returns response-free source photon densities
    per incident electron per eV per sr. With a :class:`PlanarDetector`, the
    scalar arrays are filter-attenuated, solid-angle-weighted observation
    averages in the same per-sr units. Selected spatial spectra include pixel
    solid angle and are accepted flux per incident electron per eV. Detector
    response remains an explicit read-time operation.

    Parameters
    ----------
    beam
        Scalar :class:`~pyrite.Beam` description.
    target
        Scalar :class:`~pyrite.Slab` or :class:`~pyrite.Stack`.
    detector
        Scalar detector model or physical :class:`~pyrite.PlanarDetector`.
    numerics
        Sampling and execution controls. ``None`` uses :class:`~pyrite.Numerics`.
    emission
        ``"incoherent"``, ``"coherent"``, or ``"both"`` line policy.
    brem_source
        ``"mc"``, ``"external"``, or ``"none"`` background policy.
    filters
        Ordered filter plates; valid only with a physical planar detector.
    pixel_scorer
        Optional factorized spatial-scoring request for a pixelated detector.

    Returns
    -------
    Result
        In-memory spectra, resolved case, provenance, and optional spatial data.

    Raises
    ------
    TypeError
        If components or numerics have incompatible types.
    ValueError
        If scene geometry or requested backend is inconsistent.

    Notes
    -----
    This function performs no checkpoint I/O.
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
        brem_source=brem_source,
    )
    if resolved_numerics.backend not in {"auto", BACKEND.name}:
        raise ValueError(
            f"Numerics.backend={resolved_numerics.backend!r} does not match the active "
            f"backend {BACKEND.name!r}; select the backend before importing pyrite"
        )
    case = build_case(scene, resolved_numerics)
    if isinstance(scene.detector, PlanarDetector):
        return _simulate_planar(scene, resolved_numerics, case)
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
            "stopping_model": STOPPING_MODEL,
            "backend": BACKEND.name,
            "device": BACKEND.device,
            "versions": {"pyrite": __version__, "numpy": np.__version__},
        },
        coherent_spectrum=(None if coherent is None else np.asarray(coherent)),
    )


def _simulate_planar(scene: Scene, numerics: Numerics, case: Case) -> Result:
    detector = scene.detector
    assert isinstance(detector, PlanarDetector)
    scorer = PixelScorer() if scene.pixel_scorer is None else scene.pixel_scorer
    rays = planar_detector_rays(detector)
    tile_index, directions_lab = angular_tiles(rays, scorer.angular_shape)
    directions_sample = directions_to_sample_frame(
        directions_lab,
        np.deg2rad(float(case.get("tilt_deg", 0.0))),
        np.deg2rad(float(case.get("tilt_azim_deg", 0.0))),
    )
    output = run_case_directions(
        case,
        directions_sample,
        transport_core=numerics.transport_core,
    )
    energy = np.asarray(output["E_grid"])
    background_energy = np.asarray(output["E_grid_brem"])
    line_mu = _attenuation_matrix(scene.filters, energy)
    background_mu = _attenuation_matrix(scene.filters, background_energy)
    ray_map = PixelRayMap(
        tile_index=tile_index,
        solid_angle_sr=rays.solid_angle_sr,
        path_length_mm=filter_path_lengths(rays, scene.filters),
    )
    line = SpectralFactors(energy, np.asarray(output["spec_by_direction"]), line_mu)
    background_intrinsic = np.asarray(output["brem_wide_by_direction"])
    if scene.brem_source != "mc":
        background_intrinsic = np.zeros_like(background_intrinsic)
    background = SpectralFactors(background_energy, background_intrinsic, background_mu)
    coherent = (
        None
        if "spec_coherent_by_direction" not in output
        else SpectralFactors(
            energy,
            np.asarray(output["spec_coherent_by_direction"]),
            line_mu,
        )
    )
    spatial = SpatialResult(
        ray_map=ray_map,
        line=line,
        background=background,
        detector=detector,
        coherent_line=coherent,
    )
    selected_line = coherent if scene.emission == "coherent" else line
    selected_name = "coherent" if selected_line is coherent else "line"
    source_digest = case_content_key(case)
    observation_digest, observation = _observation_provenance(
        source_digest,
        scene,
        scorer,
        (line_mu, background_mu),
    )
    coherent_average = None if coherent is None else spatial.average_density("coherent")
    return Result(
        energy_eV=energy,
        spectrum=spatial.average_density(selected_name),
        background_energy_eV=background_energy,
        background=spatial.average_density("background"),
        case=case,
        provenance={
            "scene": scene,
            "numerics": numerics,
            "identity_digest": source_digest,
            "observation_identity_digest": observation_digest,
            "observation": observation,
            "stopping_model": STOPPING_MODEL,
            "backend": BACKEND.name,
            "device": BACKEND.device,
            "versions": {
                "pyrite": __version__,
                "numpy": np.__version__,
                "xraydb": _package_version("xraydb"),
            },
        },
        coherent_spectrum=coherent_average,
        spatial=spatial if scene.pixel_scorer is not None else None,
    )


__all__ = ["build_case", "build_legacy_cases", "build_sweep_cases", "simulate"]
