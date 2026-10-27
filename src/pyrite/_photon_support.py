"""Shared photon observation identity and preparation, below campaign drivers."""

from __future__ import annotations

import hashlib
import json

import numpy as np

MODEL = "bremslib-continuum-photon-only-cpu.v2"
EVENT_MODEL = "physical-event-photon-cpu.v1"
CUDA_MODEL = "bremslib-continuum-photon-only-cuda.v1"
CUDA_EVENT_MODEL = "physical-event-photon-cuda.v1"
_RELAXATION_SALT = 0x341EADE1
# Explicit segment fields only: never select arbitrary metadata by its array length.
_SEGMENT_KEYS = ("L_ang", "r_mid", "v_hat", "elec_id", "E_repr_keV", "E_keV", "layer")


def _sample_segments(segments, n_primaries: int) -> dict:
    """Keep only continuum primary electron rows, preserving their global IDs.

    Transport may share one population between the line and brem scorers:
    ``max(Ne, Ne_brem)`` can exceed ``Ne_brem``. Unselected line-only
    electrons must neither enter the weighted estimator nor its denominator.
    """
    from .montecarlo.transport.secondaries import _host

    owners = _host(segments["elec_id"])
    if owners.ndim != 1 or not np.issubdtype(owners.dtype, np.integer):
        raise ValueError("transport electron ids must be a one-dimensional integer array")
    if owners.size and owners.min() < 0:
        raise ValueError("transport electron ids must be nonnegative")
    mask = owners < n_primaries
    result = {}
    for key in _SEGMENT_KEYS:
        value = segments.get(key)
        if value is not None:
            array = _host(value)
            if array.ndim == 0 or array.shape[0] != owners.size:
                raise ValueError(
                    f"transport segment field {key!r} does not align with electron ids"
                )
            result[key] = array[mask]
    return result


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
    charged_feedback=None,
    execution=None,
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
            "angular_proposal": "defensive-isotropic-forward-4pi",
            "energy_proposal": "defensive-log-uniform-on-segment-support",
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
    if charged_feedback is not None:
        payload["charged_feedback"] = charged_feedback
    if execution is not None:
        if execution.get("backend") != "cuda":
            raise ValueError("unsupported photon observation execution backend")
        payload["execution"] = dict(execution)
        payload["model"] = CUDA_MODEL
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return digest, payload


def _scene_feedback(
    case,
    layers,
    width,
    height,
    numerics,
    photon_cutoff,
    electron_cutoff,
    *,
    boxes=(),
    box_materials=None,
):
    """Prepare enabled-only feedback and its additional positron table provenance.

    Validation: photon-secondary-feedback
    """
    from .montecarlo.runner.case_tables import (
        _case_elastic_kwargs,
        _case_inelastic_kwargs,
        _case_radiative_kwargs,
        _case_stopping_tables,
    )
    from .montecarlo.transport.photon_feedback import cascade_feedback_stage

    if case.get("inelastic_model") != "shell-soft-hard":
        raise ValueError("charged feedback requires inelastic_model='shell-soft-hard'")
    if boxes:
        from ._photon_scene_feedback import prepare_feedback_scene

        return prepare_feedback_scene(
            case,
            layers,
            boxes,
            box_materials,
            width,
            height,
            numerics,
            photon_cutoff,
            electron_cutoff,
        )
    options = {
        **_case_inelastic_kwargs(case),
        **_case_elastic_kwargs(case),
        **_case_radiative_kwargs(case),
        "stopping_tables": _case_stopping_tables(case),
        "secondary_threshold_eV": electron_cutoff,
        "E_cut_keV": electron_cutoff * 1e-3,
        "transport_core": numerics.transport_core,
        "energy_model": case.get("energy_model", "frozen"),
        "straggling": bool(case.get("straggling", False)),
        "max_dE_frac": case.get("max_dE_frac", 0.0),
    }
    stage = cascade_feedback_stage(
        layers, width_ang=width, height_ang=height, photon_cutoff_eV=photon_cutoff, **options
    )
    return stage, {
        "model": "cpu-charged-photon-feedback-v1",
        "positron_tables": _feedback_table_markers(stage),
        "max_relaxation_rounds": stage.max_relaxation_rounds,
    }


def _feedback_table_markers(stage):
    """Resolve enabled-only positron provenance at the Scene orchestration boundary."""
    from .xsgen.elsepa.catalog import resolve_layer_tables
    from .xsgen.sbethe.catalog import resolve_catalog_table
    from .xsgen.store import identity_markers

    transport = stage.base.transport
    materials = transport["inelastic_materials"]
    if isinstance(materials, str):
        materials = [materials]
    records = [resolve_catalog_table(key, projectile="positron") for key in materials]
    if transport.get("elastic_model") == "elsepa":
        records.extend(
            table
            for layer in transport["layers"]
            for entry in resolve_layer_tables(layer[2], projectile="positron")
            for table in entry.tables
        )
    return identity_markers(records)


def _event_sources(segments, source, n_primaries, stage, seed, low):
    """Adapt exactly one retained emission component, with no continuum sum."""
    from .montecarlo.transport.photon_event_sources import (
        emission_sources,
        primary_annihilation_sources,
        primary_relaxation_sources,
    )
    from .montecarlo.transport.secondaries import secondary_stream_keys

    if source == "annihilation":
        return primary_annihilation_sources(segments, photon_cutoff_eV=low), None
    keys = secondary_stream_keys(
        seed,
        np.arange(n_primaries),
        np.zeros(n_primaries, dtype=np.int64),
        salt=_RELAXATION_SALT,
    )
    relaxation = primary_relaxation_sources(segments, keys, stage, seed=seed)
    sources = emission_sources(
        relaxation.photons, np.arange(n_primaries), photon_cutoff_eV=low, source_kind=1
    )
    return sources, relaxation


def _event_identity(
    source_digest,
    aperture,
    boxes,
    edges,
    *,
    source,
    n_primaries,
    seed,
    low,
    electron,
    charged_feedback,
    feedback_identity,
    execution=None,
):
    """Reuse canonical geometry/pins, replacing the continuous proposal contract."""
    from .materials.photon_cross_sections import EPDL_E_MAX_EV

    _, payload = _identity(
        source_digest,
        aperture,
        boxes,
        edges,
        n_samples=0,
        seed=seed,
        photon_cutoff_eV=low,
        electron_cutoff_eV=electron,
        source_ceiling_eV=EPDL_E_MAX_EV,
        execution=execution,
    )
    payload.update(
        schema="pyrite.photon-event-observation.v1",
        model=EVENT_MODEL if execution is None else CUDA_EVENT_MODEL,
        proposal={
            "component": source,
            "source_role": "physical-event",
            "n_primaries": n_primaries,
            "seed": seed,
            "photon_cutoff_eV": low,
            "electron_cutoff_eV": electron,
            "relaxation_stream": "secondary-stream-keys/341eade1.v1",
        },
        charged_feedback=feedback_identity if charged_feedback else None,
        relaxation_feedback=feedback_identity if source == "characteristic" else None,
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return digest, payload
