"""Production ordinary-hit boundary; one selected source, one primary pass.

The CPU reference models and their validation domains are unchanged. Directional
spectra remain separate from this aperture-integrated observable (ADR-0020).
"""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from ..._backend import xp
from ..._photon_config import CPU_MODEL, CUDA_MODEL, PhotonTransport
from ..._photon_support import (
    _event_identity,
    _event_sources,
    _identity,
    _sample_segments,
    _scene_feedback,
)
from ...materials.photon_cross_sections import EPDL_E_MAX_EV, EPDL_E_MIN_EV
from ..trajectories import case_digest
from ..transport.photon_histories import PhotonAperture, PhotonBox
from .case_tables import _case_bremslib_tables, case_table_markers
from .photon_emission import score_bremslib_continuum, score_photon_sources

MODEL = CPU_MODEL  # historical CPU record; CUDA records have their own enabled-only model


def _backend_for_core(transport_core):
    from ..._backend import BACKEND

    if xp is np:
        if transport_core not in ("auto", "lockstep", "per-electron"):
            raise ValueError("photon transport requires a CPU backend and CPU transport core")
        return "cpu"
    if BACKEND.name != "cuda" or transport_core not in (
        "auto",
        "lockstep",
        "per-electron",
        "cuda",
    ):
        raise ValueError(
            "photon transport requires a CPU or CUDA backend and an exact transport core"
        )
    return "cuda"


def _execution_identity(backend, stage):
    if backend == "cpu":
        return None
    return {
        "backend": "cuda",
        "navigation": "fused-binary64.v1",
        "conditional_samplers": "epdl-eadl-pair-explicit-draw.v1",
        "queue": "host-controlled-lineage-device-waves.v1",
        "rng": "numpy-philox4x64-lineage.v1",
        "ledger": "host-fsum.v1",
        "charged_transport_core": "cuda" if stage is not None else None,
    }


def cached_photon_aperture(case, transport):
    """Score once across the scalar and every directional view of one transport."""
    if case.get("photon_transport") is None:
        return None
    if "_photon_aperture" not in transport:
        transport["_photon_aperture"] = score_photon_aperture(case, transport)
    return transport["_photon_aperture"]


def validate_photon_source(case, selection: PhotonTransport, transport_core: str) -> int:
    """Refuse unsupported primary populations/models before transport or replay."""
    _backend_for_core(transport_core)
    for key in ("groove_spacing_ang", "channeling", "transition_radiation", "adaptive_precision"):
        if case.get(key) is not None:
            raise ValueError(f"photon transport does not support {key}")
    if (
        case.get("bremsstrahlung_model") != "bremslib"
        or case.get("radiative_model") != "bremslib-soft-hard"
    ):
        raise ValueError("photon transport requires resolved coupled BremsLib")
    if not EPDL_E_MIN_EV <= selection.photon_cutoff_eV < EPDL_E_MAX_EV:
        raise ValueError("photon cutoff must lie inside the EPDL domain")
    n = case.get("Ne_brem")
    if type(n) is not int or n <= 0:
        raise ValueError("photon transport requires positive Ne_brem")
    if selection.source == "brem":
        if selection.source_ceiling_eV > EPDL_E_MAX_EV:
            raise ValueError("photon source ceiling exceeds the EPDL domain")
    else:
        if case.get("Ne") != n:
            raise ValueError("photon event sources require equal positive Ne and Ne_brem")
        if (
            case.get("inelastic_model") != "shell-soft-hard"
            or case.get("secondary_threshold_eV") != selection.electron_cutoff_eV
        ):
            raise ValueError(
                "photon event sources require shell-soft-hard and matching secondary threshold"
            )
    if selection.source == "annihilation":
        if case.get("pair_production_model") is None or not case.get("positron_transport"):
            raise ValueError("annihilation sources require pair conversion and positron transport")
    elif case.get("pair_production_model") is not None or case.get("positron_transport"):
        raise ValueError("this photon source does not support primary pair conversion")
    return n


def prepare_source(
    case, selection, aperture, boxes, width, height, transport_core, *, box_materials=None
):
    """Prepare enabled resources before primary transport; never transports."""
    n = validate_photon_source(case, selection, transport_core)
    backend = _backend_for_core(transport_core)
    dimensions = tuple(
        None if case.get(key) is None else float(case[key]) * 1e7
        for key in ("crystal_width_mm", "crystal_height_mm")
    )
    if (width, height) != dimensions:
        raise ValueError("photon target footprint disagrees with the resolved source Case")
    layers = case.get("abs_layers")
    if layers is None:
        layers = [(0.0, float(case["thickness_ang"]), case["composition"])]
    stage = feedback_identity = None
    if selection.charged_feedback or selection.source == "characteristic":
        feedback_case = {
            key: value
            for key, value in case.items()
            if key not in ("pair_production_model", "positron_transport")
        }
        stage, feedback_identity = _scene_feedback(
            feedback_case,
            layers,
            width,
            height,
            SimpleNamespace(transport_core="cuda" if backend == "cuda" else transport_core),
            selection.photon_cutoff_eV,
            selection.electron_cutoff_eV,
            **({"boxes": boxes, "box_materials": box_materials} if boxes else {}),
        )
        if backend == "cuda" and feedback_identity["model"] == "cpu-charged-photon-feedback-v1":
            feedback_identity = {
                **feedback_identity,
                "model": "cuda-charged-photon-feedback-v1",
                "transport_core": "cuda",
            }
    tables = _case_bremslib_tables(case) if selection.source == "brem" else None
    if selection.source == "brem" and tables is None:
        raise ValueError("resolved BremsLib tables are unavailable")
    return dict(
        selection=selection,
        n=n,
        layers=layers,
        aperture=aperture,
        boxes=boxes,
        width=width,
        height=height,
        stage=stage,
        feedback_identity=feedback_identity,
        tables=tables,
        backend=backend,
        execution=_execution_identity(backend, stage),
    )


def observation_identity(case, source_digest, context):
    """Use the reference observation identity and original data/table pins."""
    config = context["selection"]
    common = dict(
        seed=int(case["seed"]) if config.seed is None else config.seed,
        photon_cutoff_eV=config.photon_cutoff_eV,
        electron_cutoff_eV=config.electron_cutoff_eV,
    )
    execution = {} if context["execution"] is None else {"execution": context["execution"]}
    args = (source_digest, context["aperture"], context["boxes"], config.edges_eV)
    if config.source == "brem":
        return _identity(
            *args,
            **common,
            **execution,
            n_samples=config.n_samples,
            source_ceiling_eV=config.source_ceiling_eV,
            **(
                {"charged_feedback": context["feedback_identity"]}
                if config.charged_feedback
                else {}
            ),
        )
    return _event_identity(
        *args,
        source=config.source,
        n_primaries=context["n"],
        seed=common["seed"],
        low=config.photon_cutoff_eV,
        electron=config.electron_cutoff_eV,
        charged_feedback=config.charged_feedback,
        feedback_identity=context["feedback_identity"],
        **execution,
    )


def photon_case_record(case, source_digest, context):
    """Build a data-only selection snapshot for Case, cache, capture and replay."""
    digest, identity = observation_identity(case, source_digest, context)
    return dict(
        model=CUDA_MODEL if context["backend"] == "cuda" else MODEL,
        selection=asdict(context["selection"]),
        source_identity_digest=source_digest,
        source_case_digest=case_digest(case),
        source_table_markers=case_table_markers(case),
        observation_identity_digest=digest,
        identity=identity,
        width_ang=context["width"],
        height_ang=context["height"],
    )


def prepare_photon_case(case, transport_core="auto"):
    """Check the original resolved source and pins before execution/replay."""
    record = case["photon_transport"]
    if record["model"] not in (MODEL, CUDA_MODEL):
        raise ValueError("unsupported photon transport model")
    backend = _backend_for_core(transport_core)
    expected_model = CUDA_MODEL if backend == "cuda" else MODEL
    if record["model"] != expected_model:
        raise ValueError("photon transport backend changed; lower the Scene again for this backend")
    source_case = {key: value for key, value in case.items() if key != "photon_transport"}
    if case_digest(source_case) != record["source_case_digest"]:
        raise ValueError("photon transport source Case changed; lower the Scene again")
    if case_table_markers(source_case) != record["source_table_markers"]:
        raise ValueError("photon transport source table pins changed")
    config = PhotonTransport(**record["selection"])
    identity = record["identity"]
    aperture = PhotonAperture(**identity["aperture"])
    boxes = tuple(PhotonBox(**box) for box in identity["boxes"])
    context = prepare_source(
        source_case,
        config,
        aperture,
        boxes,
        record["width_ang"],
        record["height_ang"],
        transport_core,
        box_materials=(
            identity.get("charged_feedback") or identity.get("relaxation_feedback") or {}
        ).get("box_materials"),
    )
    digest, expected = observation_identity(source_case, record["source_identity_digest"], context)
    if digest != record["observation_identity_digest"] or expected != identity:
        raise ValueError("photon transport identity or data pins changed")
    context.update(record=record, case=source_case)
    return context


def score_photon_aperture(case, transport):
    """Score retained primary rows without rerunning or debiting primary transport."""
    from ...photon_event_observation import SCHEMA as EVENT_SCHEMA
    from ...photon_event_observation import _event_arrays
    from ...photon_event_observation import _validate as validate_events
    from ...photon_observation import NORMALIZATION, SCHEMA, _checksum, _tally_payload, _validate

    context = prepare_photon_case(case)
    config, record = context["selection"], context["record"]
    seed = record["identity"]["proposal"]["seed"]
    common = dict(
        n_primaries=context["n"],
        seed=seed,
        photon_cutoff_eV=config.photon_cutoff_eV,
        boxes=context["boxes"],
        width_ang=context["width"],
        height_ang=context["height"],
        **(
            {"charged_stage": context["stage"]}
            if config.charged_feedback
            else {"electron_cutoff_eV": config.electron_cutoff_eV}
        ),
    )
    if context["backend"] == "cuda":
        common["backend"] = "cuda"
    geometry = (context["layers"], context["aperture"], config.edges_eV)
    relaxation = None
    if config.source == "brem":
        segments = _sample_segments(transport["segs"], context["n"])
        score = score_bremslib_continuum(
            segments,
            geometry[0],
            context["tables"],
            *geometry[1:],
            **common,
            n_samples=config.n_samples,
            source_ceiling_eV=config.source_ceiling_eV,
        )
    else:
        sources, relaxation = _event_sources(
            transport["segs"],
            config.source,
            context["n"],
            context["stage"],
            seed,
            config.photon_cutoff_eV,
        )
        score = score_photon_sources(sources, *geometry, **common)
    observation = SimpleNamespace(
        score=score,
        relaxation=relaxation,
        source_identity_digest=record["source_identity_digest"],
        observation_identity_digest=record["observation_identity_digest"],
        provenance={
            **record["identity"],
            "normalization": NORMALIZATION,
            "observation_identity_digest": record["observation_identity_digest"],
            "limitations": [
                (
                    "experimental CUDA kernels with host lineage/RNG and ledgers; full-track validation remains open"
                    if context["backend"] == "cuda"
                    else "experimental CPU reference; full-track validation remains open"
                ),
                "no primary charged scene re-entry, time-dependent scoring or pixel response",
            ],
        },
    )
    metadata, arrays = _tally_payload(
        observation, SCHEMA if config.source == "brem" else EVENT_SCHEMA
    )
    if config.source == "brem":
        _validate(metadata, arrays)
    else:
        metadata["source_component"] = config.source
        _event_arrays(observation, metadata, arrays)
        validate_events(metadata, arrays)
    metadata["content_sha256"] = _checksum(metadata, arrays, tuple(arrays))
    return dict(metadata=metadata, arrays=arrays)
