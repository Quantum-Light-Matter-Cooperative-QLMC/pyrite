"""Experimental physical-event Scene sources for ordinary photon aperture hits.

Separate from weighted continuum sampling and directional Result spectra.
Validation: photon-secondary-feedback (bounded CPU accounting rederived).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ._photon_support import EVENT_MODEL
from ._photon_support import _event_identity as _event_identity
from ._photon_support import _event_sources as _event_sources
from ._spec_defaults import UNSET, _Unset
from .campaign.resolution import resolve_simulation
from .photon_transport import (
    _energy_edges,
    _positive,
    _preflight_scene,
    _scene_feedback,
    _scene_geometry,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike
    from pathlib import Path

    from ._numerics import Numerics, Physics
    from .campaign.model import Scene
    from .montecarlo.case import Case
    from .montecarlo.runner.photon_emission import PhotonContinuumScore
    from .montecarlo.transport.photon_event_sources import PrimaryRelaxationSources
    from .observations.spec import Observation
    from .quantum import QuantumSetup


MODEL = EVENT_MODEL


@dataclass(frozen=True, slots=True)
class PhotonEventObservation:
    """One physical emission component scored over a detector aperture.

    The score is photons/(original incident electron eV), integrated over
    the aperture. ``relaxation`` retains the primary binding-reservoir budget
    for characteristic sources, including Auger feedback, or is None for
    recorded annihilation sources. The photon ledger covers tracked sources;
    sub-cutoff source energy remains on ``score.sources``. This experimental
    object has no production Result/checkpoint contract. ``save`` writes an
    explicit reduced analysis artifact, not source replay or a transport restart.
    """

    score: PhotonContinuumScore
    case: Case
    source_identity_digest: str
    observation_identity_digest: str
    provenance: Mapping[str, Any]
    relaxation: PrimaryRelaxationSources | None

    def save(self, path: str | PathLike[str]) -> Path:
        """Save reduced event tallies, provenance and primary relaxation budget.

        Reload with ``pyrite.photon_event_observation.load_photon_events``.
        Photon/source rows and the Case are excluded; no production checkpoint
        or replay contract is implied. Existing files are replaced atomically.
        """
        from .photon_event_observation import save_photon_events

        return save_photon_events(self, path)


def _event_case(case, source: str, electron_cutoff: float) -> int:
    """Require retained physical cascades with one unambiguous population."""
    required = {
        "bremsstrahlung_model": "bremslib",
        "radiative_model": "bremslib-soft-hard",
        "inelastic_model": "shell-soft-hard",
        "secondary_threshold_eV": electron_cutoff,
    }
    for name, value in required.items():
        if case.get(name) != value:
            raise ValueError(f"photon event reference requires {name}={value!r}")
    for name in ("groove_spacing_ang", "channeling", "transition_radiation", "adaptive_precision"):
        if case.get(name) is not None:
            raise ValueError(f"photon event reference does not support {name}")
    n = case.get("Ne")
    if type(n) is not int or n <= 0 or case.get("Ne_brem") != n:
        raise ValueError("photon event reference requires equal positive Ne and Ne_brem")
    if source == "annihilation":
        if case.get("pair_production_model") is None or not case.get("positron_transport"):
            raise ValueError("annihilation sources require pair conversion and positron transport")
    elif case.get("pair_production_model") is not None or case.get("positron_transport"):
        raise ValueError("characteristic Scene sources do not yet support primary pair conversion")
    return n


def simulate_photon_events(
    scene: Scene,
    edges_eV,
    *,
    source: Literal["characteristic", "annihilation"],
    photon_cutoff_eV: float = 1000.0,
    electron_cutoff_eV: float = 1000.0,
    seed: int | None = None,
    charged_feedback: bool = False,
    physics: Physics | None = None,
    numerics: Numerics | None = None,
    quantum: QuantumSetup | None | _Unset = UNSET,
    observation: Observation | None = None,
) -> PhotonEventObservation:
    """Transport one Scene once and score one physical photon emission component.

    CPU-only experimental diagnostic, separate from ``simulate`` and
    ``simulate_photon_continuum``. Requires the same ideal unpixelated physical
    detector and BremsLib-coupled shell transport, equal positive line/brem
    populations, and ``secondary_threshold_eV == electron_cutoff_eV``.
    Characteristic sources sample retained impact vacancies and follow Auger
    daughters. Annihilation sources reuse recorded photon energies/directions
    and ignore their old first-hit sink; primary pair/positron models must be
    explicitly selected. No weighted continuum or hard-brem tally is added.

    ``charged_feedback=True`` follows photon-born charged descendants too.
    Characteristic source generation always follows its Auger daughters.
    Either use of charged feedback requires a contiguous target stack without
    filter boxes; photon-only annihilation tracking supports photon filters.
    Neither primary charged transport nor these stages follow charged target
    re-entry. No pixel response, pulse-height, production identity or replay
    support is implied. Bounded secondary-feedback accounting is rederived;
    external full-track validation remains open.

    ``seed`` controls relaxation and photon tracking; original primary tracks
    retain the Scene seed. The output preserves source and observation digests
    separately; source selection, seed, cutoffs, geometry and feedback enter
    only the explicit observation identity.
    Independent ``physics`` and ``numerics`` override legacy Scene settings.
    Omitted ``quantum`` inherits ``Scene.quantum``; explicit ``None`` disables that setup.
    ``observation`` binds through the shared resolver, subject to this API's
    ideal, unpixelated aperture restrictions. The input Scene is unchanged.

    Validation: photon-secondary-feedback.
    """
    from .campaign.lowering import lower_simulation
    from .campaign.profiles import case_content_key
    from .materials.photon_cross_sections import EPDL_E_MAX_EV, EPDL_E_MIN_EV
    from .montecarlo import runner
    from .montecarlo.runner.case_tables import case_table_markers
    from .montecarlo.runner.photon_emission import score_photon_sources
    from .montecarlo.runner.transport_options import kernel_schedule

    if source not in ("characteristic", "annihilation"):
        raise ValueError("source must be 'characteristic' or 'annihilation'")
    if type(charged_feedback) is not bool:
        raise ValueError("charged_feedback must be a bool")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError("seed must be an unsigned 64-bit integer or None")
    request = resolve_simulation(
        scene, physics=physics, numerics=numerics, quantum=quantum, observation=observation
    )
    scene = request.scene
    detector, numerics = _preflight_scene(scene)
    low = _positive("photon_cutoff_eV", photon_cutoff_eV)
    electron = _positive("electron_cutoff_eV", electron_cutoff_eV)
    if not EPDL_E_MIN_EV <= low < EPDL_E_MAX_EV:
        raise ValueError("photon cutoff must lie inside the EPDL domain")
    edges = _energy_edges(edges_eV)
    case = lower_simulation(request)
    n = _event_case(case, source, electron)
    layers, aperture, boxes, (width, height) = _scene_geometry(case, detector, scene.filter_plates)
    effective_seed = int(case["seed"]) if seed is None else seed
    if not 0 <= effective_seed < 2**64:
        raise ValueError("resolved seed must be an unsigned 64-bit integer")
    stage = feedback_identity = None
    if source == "characteristic" or charged_feedback:
        if boxes:
            raise ValueError("event charged feedback does not support filter boxes")
        # Pair conversion belongs to the photon queue for these newly launched
        # particles, even when primary transport selected the older pair model.
        feedback_case = {
            k: v
            for k, v in case.items()
            if k not in ("pair_production_model", "positron_transport")
        }
        stage, feedback_identity = _scene_feedback(
            feedback_case, layers, width, height, numerics, low, electron
        )
    source_digest = case_content_key(case, xsgen_tables=case_table_markers(case))
    observation_digest, identity = _event_identity(
        source_digest,
        aperture,
        boxes,
        edges,
        source=source,
        n_primaries=n,
        seed=effective_seed,
        low=low,
        electron=electron,
        charged_feedback=charged_feedback,
        feedback_identity=feedback_identity,
    )
    with kernel_schedule(numerics):
        transport = runner._transport_case(
            case, transport_core=numerics.transport_core, keep_segments_on_device=False
        )
    sources, relaxation = _event_sources(transport["segs"], source, n, stage, effective_seed, low)
    score = score_photon_sources(
        sources,
        layers,
        aperture,
        edges,
        n_primaries=n,
        seed=effective_seed,
        photon_cutoff_eV=low,
        charged_stage=stage if charged_feedback else None,
        electron_cutoff_eV=None if charged_feedback else electron,
        boxes=boxes,
        width_ang=width,
        height_ang=height,
    )
    return PhotonEventObservation(
        score,
        case,
        source_digest,
        observation_digest,
        {
            **identity,
            "observation_identity_digest": observation_digest,
            "normalization": "photons / (incident electron eV), aperture-integrated",
            "limitations": [
                "experimental CPU event component; bounded feedback accounting rederived",
                "no charged target/filter re-entry or pulse-height response",
                "no weighted-continuum or hard-brem histogram summation",
                "no production Result/checkpoint/replay support",
            ],
        },
        relaxation,
    )


__all__ = ["PhotonEventObservation", "simulate_photon_events"]
