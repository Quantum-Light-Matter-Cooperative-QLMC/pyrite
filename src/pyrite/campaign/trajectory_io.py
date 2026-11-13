"""Save and load :class:`~pyrite.transport.Trajectories` as trajectory artifacts (#1162).

The file is the HDF5 trajectory artifact ``pyrite run --trajectories`` writes
(spectrum-phase inputs since schema 2), so both producers share one format and
one replay engine. The typed scene is not serialized: ``load_trajectories``
rebinds caller-supplied components and refuses unless they lower to the stored
case exactly. Without components the product scores only its recorded case.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .trajectory_product import Trajectories

if TYPE_CHECKING:
    from .._numerics import Physics
    from ..detectors import Detector
    from .beam import Beam
    from .geometry import Slab, Stack
    from .model import Numerics
    from .target import Target

__all__ = ["load_trajectories", "save_trajectories"]

#: Spectrum-phase inputs a production transport carries next to its segments.
_SPECTRUM_INPUTS = ("E_grid", "E_brem", "n_hat", "Ne_lines", "Ne_brem", "groove", "diagnostic_grid")


def save_trajectories(
    trajectories: Trajectories,
    path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    storage_policy: str = "compact",
) -> Path:
    """Write ``trajectories`` to one trajectory artifact and return its path.

    The artifact records the resolved case, every segment row, the
    spectrum-phase inputs, the physics provenance and the segment
    fingerprint that :func:`load_trajectories` verifies. Explicit
    ``E_cut_keV``/``max_steps`` transports are written too and stay not
    scorable after loading.

    Parameters
    ----------
    trajectories
        Product of :func:`~pyrite.transport.simulate_trajectories`.
    path
        Final ``.h5`` path; parent directories are created; written atomically.
    overwrite
        Replace an existing complete artifact. False raises
        :class:`~pyrite.montecarlo.trajectories.TrajectoryArtifactExistsError`.
    storage_policy
        ``"compact"`` (LZF + shuffle) or ``"throughput"`` (uncompressed).
    """
    from .._physics_provenance import resolved_physics_view
    from ..montecarlo.runner.adaptive import case_transport_core
    from ..montecarlo.runner.case_driver import electron_cutoffs
    from ..montecarlo.trajectories import write_trajectory_artifact

    if not isinstance(trajectories, Trajectories):
        raise TypeError("trajectories must be a pyrite.transport.Trajectories")
    case, transport, segments = trajectories.case, trajectories._transport, trajectories.segments
    inputs = (
        {key: transport[key] for key in _SPECTRUM_INPUTS}
        if all(key in transport for key in _SPECTRUM_INPUTS)
        else None
    )
    numerics = trajectories.numerics
    core = "auto" if numerics is None else numerics.transport_core
    settings: dict[str, Any] = {
        "transport_core": case_transport_core(case, core),
        "segments_on_device": not isinstance(segments["L_ang"], np.ndarray),
        "Ne_transport": int(max(case["Ne"], case["Ne_brem"])),
        "resolved_physics": resolved_physics_view(
            case, table_markers=dict(trajectories._xsgen_tables)
        ).to_json(),
        "segment_fingerprint": trajectories.fingerprint,
        "engine_overrides": dict(trajectories._engine_overrides),
    }
    if not trajectories._engine_overrides:
        settings["E_cut_by_electrons"] = electron_cutoffs(case)
    return write_trajectory_artifact(
        path,
        segments,
        case=case,
        settings=settings,
        provenance={"producer": "pyrite.storage.save_trajectories"},
        overwrite=overwrite,
        spectrum_inputs=inputs,
        storage_policy=storage_policy,
    )


def load_trajectories(
    path: str | os.PathLike[str],
    beam: Beam | None = None,
    target: Target | Slab | Stack | None = None,
    *,
    physics: Physics | None = None,
    numerics: Numerics | None = None,
    detector: Detector | None = None,
    stream: bool = False,
) -> Trajectories:
    """Reopen a trajectory artifact as a :class:`~pyrite.transport.Trajectories`.

    Reads files from :func:`save_trajectories` and ``pyrite run
    --trajectories``. Integrity (completeness, case digest, physics and table
    provenance, recorded spectrum inputs) is checked here; the segment
    fingerprint is checked before scoring.

    Parameters
    ----------
    path
        Artifact path.
    beam, target, physics, numerics, detector
        The components the transport was made from, as passed to
        ``simulate_trajectories``. Given ``beam`` and ``target``, they are
        lowered and must reproduce the stored case exactly; the result then
        scores any detector like a fresh transport. Omitted, the result has no
        typed scene and :func:`~pyrite.score_spectra` replays only the recorded
        case and detector.
    stream
        Leave segments on disk; scoring then reads them in bounded-memory
        electron blocks (recorded case only; incoherent, non-temporal spectra;
        equal to in-memory scoring up to floating-point summation order).

    Raises
    ------
    ScoringRefusedError
        Unsupported schema, incomplete or corrupt artifact (``field`` is
        ``"schema_version"`` or ``"artifact"``), or components that lower to a
        different case (``field`` names the first differing case field).
    """
    from ..montecarlo.runner.artifacts import _checked_case_and_inputs
    from ..montecarlo.trajectories import TrajectoryArtifactError, read_trajectory_artifact
    from .trajectory_product import ScoringRefusedError

    try:
        artifact = read_trajectory_artifact(path, load_transport=not stream)
        overrides = dict(artifact.settings.get("engine_overrides") or {})
        if overrides and artifact.spectrum_inputs is None:
            # Explicit engine controls: a transport record, never scorable.
            case, inputs = artifact.case, {}
        else:
            case, inputs = _checked_case_and_inputs(artifact, None)
    except TrajectoryArtifactError as error:
        field = "schema_version" if "schema_version" in str(error) else "artifact"
        raise ScoringRefusedError(field, str(error)) from error
    from ..montecarlo.runner.case_tables import case_table_markers

    segments = artifact.transport
    if not stream and int(np.asarray(segments["L_ang"]).size) != int(
        artifact.attrs["segment_count"]
    ):
        raise ScoringRefusedError("artifact", f"trajectory artifact {path} segments are truncated")
    scene = request_physics = request_numerics = None
    if beam is not None or target is not None:
        from .trajectories import trajectory_request

        if beam is None or target is None:
            raise TypeError("load_trajectories needs both beam and target to rebind a scene")
        request, lowered, request_numerics = trajectory_request(
            beam, target, physics, numerics, detector
        )
        _require_same_case(case, lowered, path)
        scene, request_physics = request.scene, request.physics
    return Trajectories(
        scene,
        request_physics,
        request_numerics,
        case,
        dict(inputs, segs=segments),
        _xsgen_tables=case_table_markers(case),
        _engine_overrides=overrides,
        _stored_fingerprint=artifact.settings.get("segment_fingerprint"),
        _source=os.fspath(artifact.path),
        _stream=stream,
    )


def _require_same_case(stored, lowered, path) -> None:
    from ..montecarlo.trajectories import case_digest
    from .trajectory_product import ScoringRefusedError, _same, _short

    if case_digest(stored) == case_digest(lowered):
        return
    for key in sorted(set(stored) | set(lowered)):
        if not _same(stored.get(key), lowered.get(key)):
            raise ScoringRefusedError(
                key,
                f"the components given to load_trajectories do not reproduce {path}: "
                f"{key} stored {_short(stored.get(key))}, lowered {_short(lowered.get(key))}",
            )
    raise ScoringRefusedError("case", f"the components do not reproduce the case of {path}")
