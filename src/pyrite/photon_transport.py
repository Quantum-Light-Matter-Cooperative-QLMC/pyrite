"""Explicit, CPU-only Scene entry point for the #341 photon-continuum reference.

This is *not* the normal :func:`pyrite.simulate` result and must not be added to
``Result.brem``: the latter is per steradian while this scores ordinary hits
across a physical detector aperture. The baseline sweep/recompute/checkpoint
paths remain unchanged. A source batch is an expected-value estimate of the
BremsLib continuum, not a second physical shower on the same electrons.

Validation: photon-continuum-scene-handoff (experimental reference boundary).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .campaign.model import Scene
    from .montecarlo.case import Case
    from .montecarlo.runner.photon_emission import PhotonContinuumScore


MODEL = "bremslib-continuum-photon-only-cpu.v1"
# Only these arrays describe each individual electron segment in this adapter.
# Other arrays in the transport dictionary include per-layer/per-electron tables
# and must NEVER be filtered using a heuristic based on a matching first axis.
_SEGMENT_KEYS = ("L_ang", "r_mid", "v_hat", "elec_id", "E_repr_keV", "E_keV", "layer")


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


def _sample_segments(segments, n_primaries: int) -> dict:
    """Keep only continuum primary electron rows, preserving their global IDs.

    Transport may share one population between the line and brem scorers:
    ``max(Ne, Ne_brem)`` can exceed ``Ne_brem``. Unselected line-only
    electrons must neither enter the weighted estimator nor its denominator.
    """
    owners = np.asarray(segments["elec_id"])
    if owners.ndim != 1 or not np.issubdtype(owners.dtype, np.integer):
        raise ValueError("transport electron ids must be a one-dimensional integer array")
    if owners.size and owners.min() < 0:
        raise ValueError("transport electron ids must be nonnegative")
    mask = owners < n_primaries
    result = {}
    for key in _SEGMENT_KEYS:
        value = segments.get(key)
        if value is not None:
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != owners.size:
                raise ValueError(
                    f"transport segment field {key!r} does not align with electron ids"
                )
            result[key] = array[mask]
    return result


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


def _identity(
    case_digest,
    aperture,
    boxes,
    edges,
    *,
    n_samples,
    seed,
    photon_cutoff_eV,
    electron_cutoff_eV,
    source_ceiling_eV,
):
    """Observation digest includes all source proposals and physical acceptance.

    The source digest is the existing case-content key, including its BremsLib
    table markers. The observation digest additionally includes the scattering
    data pins, physical geometry, scoring grid, and random proposal policy.
    """
    from .datasets import EADL, EPDL, EPDL_DIFFERENTIAL

    def vec(value):
        return np.asarray(value, dtype=float).tolist()

    payload = {
        "schema": "pyrite.photon-continuum-observation.v1",
        "model": MODEL,
        "source_identity_digest": case_digest,
        "data_sha256": {
            "epdl_integrated": EPDL.sha256,
            "epdl_differential": EPDL_DIFFERENTIAL.sha256,
            "eadl": EADL.sha256,
        },
        "proposal": {
            "n_samples": n_samples,
            "seed": seed,
            "photon_cutoff_eV": photon_cutoff_eV,
            "electron_cutoff_eV": electron_cutoff_eV,
            "source_ceiling_eV": source_ceiling_eV,
            "angular_proposal": "isotropic-4pi",
            "energy_proposal": "uniform-on-segment-support",
            "source_role": "weighted-expected-value-not-physical-event",
        },
        "aperture": {
            "center_ang": vec(aperture.center_ang),
            "x_axis": vec(aperture.x_axis),
            "y_axis": vec(aperture.y_axis),
            "half_width_ang": float(aperture.half_width_ang),
            "half_height_ang": float(aperture.half_height_ang),
        },
        "boxes": [
            {
                "center_ang": vec(box.center_ang),
                "axes": vec(box.axes),
                "half_extent_ang": vec(box.half_extent_ang),
                "composition": [[str(el), float(density)] for el, density in box.composition],
            }
            for box in boxes
        ],
        "energy_bin_edges_eV": np.asarray(edges, dtype=float).tolist(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return digest, payload


def _preflight(scene, n_samples, photon_cutoff_eV, electron_cutoff_eV, source_ceiling_eV):
    """Reject unsupported *physical* and execution requests before transport."""
    from ._backend import xp
    from ._numerics import Numerics
    from .campaign.model import Scene
    from .instrument import PlanarDetector
    from .materials.photon_cross_sections import EPDL_E_MAX_EV, EPDL_E_MIN_EV

    if not isinstance(scene, Scene):
        raise TypeError("simulate_photon_continuum requires a Scene")
    detector = scene.detector
    if not isinstance(detector, PlanarDetector):
        raise ValueError("photon-continuum ordinary hits require a physical PlanarDetector")
    if detector.pixels is not None or detector.response is not None or detector.scorer is not None:
        raise ValueError("photon-continuum reference supports an unpixelated, ideal detector only")
    if scene.acquisition is not None:
        raise ValueError("photon-continuum reference does not support acquisition response")
    if scene.physics is None or scene.physics.brem_source != "mc":
        raise ValueError("photon-continuum reference requires Physics(brem_source='mc')")
    if xp is not np:
        raise ValueError("photon-continuum reference is CPU-only (PYRITE_MC_BACKEND=cpu)")
    numerics = scene.numerics
    if not isinstance(numerics, Numerics):
        raise TypeError("scene.numerics must be Numerics")
    if numerics.precision is not None:
        raise ValueError("photon-continuum reference does not support adaptive precision")
    if numerics.transport_core not in ("auto", "lockstep", "per-electron"):
        raise ValueError("photon-continuum reference needs a CPU transport core")
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
) -> PhotonContinuumObservation:
    """Run existing primary electron transport once, then score tracked brem hits.

    Explicit opt-in, separate from ``simulate``, sweeps, replay and results.
    Supports CPU BremsLib expected-value sources, a planar **unpixelated**
    detector, finite filter plates, and analytic stack/rectangular geometry.
    Rejects grooves, channeling, non-BremsLib source models and unsupported
    backends. A photon-born charged particle's kinetic energy is handed off in
    the score's ledger, but **not transported** in this photon-only reference.
    Pair/positron feedback and coincidence/pulse-height response therefore
    remain outside its validity. No conditional/forced detector estimator.

    The result is *not* a per-steradian source spectrum and is not added to a
    physical hard-photon event histogram or to ``Result.brem``. This means
    enabling this entry point does not introduce a second cascade or an energy
    debit into the already sampled primary electrons. Original primary
    bremsstrahlung energy losses are those of the chosen coupled transport.

    Validation: photon-continuum-scene-handoff.
    """
    detector, numerics, low, electron, high = _preflight(
        scene, n_samples, photon_cutoff_eV, electron_cutoff_eV, source_ceiling_eV
    )
    edges = _energy_edges(edges_eV)
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer or None")

    from .campaign.lowering import build_case
    from .campaign.profiles import case_content_key
    from .montecarlo import runner
    from .montecarlo.runner.case_tables import case_table_markers
    from .montecarlo.runner.photon_emission import score_bremslib_continuum

    case = build_case(scene, numerics)
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
        electron_cutoff_eV=electron,
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
            "charged_descendants": "handoff-only, not transported",
            "limitations": [
                "CPU-only experimental reference, not a validated MeV shower",
                "no charged feedback or positron annihilation",
                "unpolarized BremsLib sources; no PXR/CBS or primary characteristic",
                "no pixelation, response, acquisition or uncertainty estimates",
                "no source/physical-hard-photon histogram summation",
            ],
        },
    )


__all__ = ["MODEL", "PhotonContinuumObservation", "simulate_photon_continuum"]
