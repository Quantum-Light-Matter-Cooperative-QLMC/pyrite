"""Explicit, CPU-only Scene entry point for the #341 photon-continuum reference.

This is *not* the normal :func:`pyrite.simulate` result and must not be added to
``Result.brem``: the latter is per steradian while this scores ordinary hits
across a physical detector aperture. The baseline sweep/recompute/checkpoint
paths remain unchanged. A source batch is an expected-value estimate of the
BremsLib continuum, not a second physical shower on the same electrons.

Validation: photon-continuum-scene-handoff (experimental reference boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ._photon_support import MODEL as MODEL
from ._photon_support import (
    _feedback_table_markers as _feedback_table_markers,
)
from ._photon_support import (
    _identity as _identity,
)
from ._photon_support import (
    _sample_segments as _sample_segments,
)
from ._photon_support import (
    _scene_feedback as _scene_feedback,
)
from ._spec_defaults import UNSET, _Unset
from .campaign.resolution import resolve_simulation

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike
    from pathlib import Path

    from ._numerics import Numerics, Physics
    from .campaign.model import Scene
    from .montecarlo.case import Case
    from .montecarlo.runner.photon_emission import PhotonContinuumScore
    from .observations.spec import Observation
    from .quantum import QuantumSetup


@dataclass(frozen=True, slots=True)
class PhotonContinuumObservation:
    """A reference aperture tally with separate source and observation identity.

    ``score.total_per_primary_eV`` is photons/(electron eV), integrated across
    the physical aperture, **not** photons/(electron eV sr). It must not be
    combined with ``pyrite.Result.brem``. No default spectrum is replaced.
    ``provenance`` is a JSON-compatible dictionary (not a mutable physics
    selector), recording model, source-table pins and diagnostic limitations.
    """

    score: PhotonContinuumScore
    case: Case
    source_identity_digest: str
    observation_identity_digest: str
    provenance: Mapping[str, Any]

    def save(self, path: str | PathLike[str]) -> Path:
        """Save reduced aperture tallies and provenance to an atomic NPZ artifact.

        Reload with ``pyrite.photon_observation.load_photon_continuum``.
        Source rows, histories and the Case are excluded; this is not a
        production checkpoint or a transport restart.
        """
        from .photon_observation import save_photon_continuum

        return save_photon_continuum(self, path)


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return number


def _energy_edges(edges_eV) -> np.ndarray:
    edges = np.asarray(edges_eV, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all():
        raise ValueError("energy edges must be a finite one-dimensional array of length >= 2")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError("energy edges must be strictly increasing")
    return edges.copy()


def _scene_geometry(case, detector, plates):
    """One sample-frame conversion for detector, filters and target footprint."""
    from .montecarlo.geometry import sample_to_lab_R
    from .montecarlo.transport.photon_histories import PhotonAperture, photon_box_from_pose

    R = sample_to_lab_R(
        np.deg2rad(float(case.get("tilt_deg", 0.0))),
        np.deg2rad(float(case.get("tilt_azim_deg", 0.0))),
    )
    aperture = PhotonAperture.from_pose(detector.pose, detector.size_mm, R)
    boxes = tuple(
        photon_box_from_pose(plate.pose, plate.size_mm, plate.thickness_mm, plate.material, R)
        for plate in plates
    )
    layers = case.get("abs_layers")
    if layers is None:
        layers = [(0.0, float(case["thickness_ang"]), case["composition"])]
    width = case.get("crystal_width_mm")
    height = case.get("crystal_height_mm")
    if (width is None) != (height is None):
        raise ValueError("target footprint width and height must both be present or absent")
    return (
        layers,
        aperture,
        boxes,
        (
            None if width is None else float(width) * 1.0e7,
            None if height is None else float(height) * 1.0e7,
        ),
    )


def _preflight_scene(scene):
    """Reject unsupported *physical* and execution requests before transport."""
    from ._backend import xp
    from ._numerics import Numerics
    from .campaign.model import Scene
    from .instrument import PlanarDetector

    if not isinstance(scene, Scene):
        raise TypeError("photon reference requires a Scene")
    detector = scene.detector
    if not isinstance(detector, PlanarDetector):
        raise ValueError("photon ordinary hits require a physical PlanarDetector")
    if detector.pixels is not None or detector.response is not None or detector.scorer is not None:
        raise ValueError("photon reference supports an unpixelated, ideal detector only")
    if scene.acquisition is not None:
        raise ValueError("photon reference does not support acquisition response")
    if scene.physics is None or scene.physics.brem_source != "mc":
        raise ValueError("photon reference requires Physics(brem_source='mc')")
    if xp is not np:
        raise ValueError("photon reference is CPU-only (PYRITE_MC_BACKEND=cpu)")
    numerics = scene.numerics
    if not isinstance(numerics, Numerics):
        raise TypeError("scene.numerics must be Numerics")
    if numerics.precision is not None:
        raise ValueError("photon reference does not support adaptive precision")
    if numerics.transport_core not in ("auto", "lockstep", "per-electron"):
        raise ValueError("photon reference needs a CPU transport core")
    return detector, numerics


def _preflight(scene, n_samples, photon_cutoff_eV, electron_cutoff_eV, source_ceiling_eV):
    """Validate the common Scene boundary and continuous-source proposal."""
    from .materials.photon_cross_sections import EPDL_E_MAX_EV, EPDL_E_MIN_EV

    detector, numerics = _preflight_scene(scene)
    if type(n_samples) is not int or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    low = _positive("photon_cutoff_eV", photon_cutoff_eV)
    electron = _positive("electron_cutoff_eV", electron_cutoff_eV)
    high = _positive("source_ceiling_eV", source_ceiling_eV)
    if low < EPDL_E_MIN_EV or high > EPDL_E_MAX_EV or high <= low:
        raise ValueError("photon source energies must fit the EPDL domain above the cutoff")
    return detector, numerics, low, electron, high


def simulate_photon_continuum(
    scene: Scene,
    edges_eV,
    *,
    n_samples: int,
    photon_cutoff_eV: float = 1_000.0,
    electron_cutoff_eV: float = 1_000.0,
    source_ceiling_eV: float = 1_000_000.0,
    seed: int | None = None,
    charged_feedback: bool = False,
    physics: Physics | None = None,
    numerics: Numerics | None = None,
    quantum: QuantumSetup | None | _Unset = UNSET,
    observation: Observation | None = None,
) -> PhotonContinuumObservation:
    """Run existing primary electron transport once, then score tracked brem hits.

    Explicit opt-in, separate from ``simulate``, sweeps, replay and results.
    Supports CPU BremsLib expected-value sources, a planar **unpixelated**
    detector, finite filter plates, and analytic stack/rectangular geometry.
    Rejects grooves, channeling, non-BremsLib source models and unsupported
    backends. The default records charged handoffs. ``charged_feedback=True``
    follows photon-born electrons/positrons, charged-impact EADL relaxation,
    Auger descendants and annihilation photons in a contiguous target stack.
    It requires shell-soft-hard transport and currently refuses filter boxes.
    Neither mode is a validated complete MeV detector response or a
    coincidence/pulse-height model. No conditional/forced detector estimator.

    The result is *not* a per-steradian source spectrum and is not added to a
    physical hard-photon event histogram or to ``Result.brem``. This means
    enabling this entry point does not introduce a second cascade or an energy
    debit into the already sampled primary electrons. Original primary
    bremsstrahlung energy losses are those of the chosen coupled transport.

    Independent ``physics`` and ``numerics`` override legacy Scene settings.
    Omitted ``quantum`` inherits ``Scene.quantum``; explicit ``None`` disables that setup.
    ``observation`` binds through the shared resolver, subject to this API's
    ideal, unpixelated aperture restrictions. The input Scene is unchanged.

    Validation: photon-continuum-scene-handoff, photon-secondary-feedback.
    """
    request = resolve_simulation(
        scene, physics=physics, numerics=numerics, quantum=quantum, observation=observation
    )
    scene = request.scene
    detector, numerics, low, electron, high = _preflight(
        scene, n_samples, photon_cutoff_eV, electron_cutoff_eV, source_ceiling_eV
    )
    if type(charged_feedback) is not bool:
        raise ValueError("charged_feedback must be a bool")
    edges = _energy_edges(edges_eV)
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")

    from .campaign.lowering import lower_simulation
    from .campaign.profiles import case_content_key
    from .montecarlo import runner
    from .montecarlo.runner.case_tables import case_table_markers
    from .montecarlo.runner.photon_emission import score_bremslib_continuum

    case = lower_simulation(request)
    if case.get("bremsstrahlung_model") != "bremslib":
        raise ValueError("photon-continuum reference requires resolved BremsLib cross sections")
    if case.get("radiative_model") != "bremslib-soft-hard":
        raise ValueError("photon-continuum reference requires coupled radiative electron tracks")
    refused = {
        "groove_spacing_ang": "grooved entrances",
        "channeling": "channeling source histories",
        "transition_radiation": "transition-radiation source histories",
        "adaptive_precision": "adaptive primary populations",
        "pair_production_model": "physical-event pair conversion",
        "positron_transport": "positron transport and annihilation",
    }
    for name, reason in refused.items():
        if case.get(name) is not None:
            raise ValueError(f"photon-continuum reference does not support {reason}")
    if case.get("Ne_brem", 0) <= 0:
        raise ValueError("case must have positive Ne_brem")

    # Identity is resolved BEFORE tracking: a data or physical geometry change
    # cannot silently reuse a stale detector tally.
    layers, aperture, boxes, (width, height) = _scene_geometry(case, detector, scene.filter_plates)
    stage = None
    feedback_identity = None
    if charged_feedback:
        if boxes:
            raise ValueError(
                "charged feedback does not yet transport charged particles through filter boxes"
            )
        stage, feedback_identity = _scene_feedback(
            case, layers, width, height, numerics, low, electron
        )
    effective_seed = int(case["seed"]) if seed is None else seed
    source_digest = case_content_key(case, xsgen_tables=case_table_markers(case))
    observation_digest, identity = _identity(
        source_digest,
        aperture,
        boxes,
        edges,
        n_samples=n_samples,
        seed=effective_seed,
        photon_cutoff_eV=low,
        electron_cutoff_eV=electron,
        source_ceiling_eV=high,
        **({"charged_feedback": feedback_identity} if stage is not None else {}),
    )
    tables = runner._case_bremslib_tables(case)
    if tables is None:
        raise ValueError("resolved BremsLib tables are unavailable")
    transport = runner._transport_case(
        case, transport_core=numerics.transport_core, keep_segments_on_device=False
    )
    segments = _sample_segments(transport["segs"], int(case["Ne_brem"]))
    score = score_bremslib_continuum(
        segments,
        layers,
        tables,
        aperture,
        edges,
        n_primaries=int(case["Ne_brem"]),
        n_samples=n_samples,
        seed=effective_seed,
        photon_cutoff_eV=low,
        **({"electron_cutoff_eV": electron} if stage is None else {"charged_stage": stage}),
        source_ceiling_eV=high,
        boxes=boxes,
        width_ang=width,
        height_ang=height,
    )
    return PhotonContinuumObservation(
        score=score,
        case=case,
        source_identity_digest=source_digest,
        observation_identity_digest=observation_digest,
        provenance={
            **identity,
            "observation_identity_digest": observation_digest,
            "normalization": "photons / (incident electron eV), aperture-integrated",
            "origin": "physical PlanarDetector ordinary hits; no ballistic filter reapplication",
            "charged_descendants": "CPU electron/positron, Auger and annihilation feedback"
            if stage is not None
            else "handoff-only, not transported",
            "limitations": [
                "CPU-only experimental reference, not a validated MeV shower",
                *(
                    ["contiguous target stack only; no charged filter transport"]
                    if stage is not None
                    else ["no charged feedback or positron annihilation"]
                ),
                "unpolarized BremsLib sources; no PXR/CBS or primary characteristic",
                "no pixelation, response, acquisition or uncertainty estimates",
                "no source/physical-hard-photon histogram summation",
            ],
        },
    )


__all__ = ["MODEL", "PhotonContinuumObservation", "simulate_photon_continuum"]
