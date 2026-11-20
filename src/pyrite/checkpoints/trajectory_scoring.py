"""Score saved trajectory artifacts (#186, #1169).

The opt-in second stage of a run that captured transport with
``pyrite run --trajectories``: each schema-2 artifact's spectrum phase is
replayed in bounded electron blocks
(:func:`~pyrite.montecarlo.runner.stream_spectrum_from_artifact`) and stored as
an ordinary checkpoint record of the stem and dataset identity the capturing run
recorded. Nothing is transported. Records keep their source artifact's path and
SHA-256 under ``source_trajectory``. The shared per-case cache is neither read
nor written, as with ``pyrite run --no-cache``.

Artifacts that record no run identity (written by
:func:`pyrite.storage.save_trajectories`) have no checkpoint to join. They are
replayed through the same engine as
``score_spectra(load_trajectories(path, stream=True))`` and written to one
``.npz`` result file each (:func:`score_saved`).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .._util.hashing import file_sha256
from ..montecarlo.case import refuse_axial_channeling
from ..montecarlo.trajectories import (
    TrajectoryArtifact,
    TrajectoryArtifactError,
    artifact_spectrum_inputs,
    read_trajectory_artifact,
)

#: Records scored between checkpoint saves; an interrupted run resumes from the
#: last save.
SAVE_EVERY = 25


@dataclass
class StemPlan:
    """The artifacts one checkpoint stem is scored from."""

    stem: str
    dataset_identity: dict[str, Any]
    artifacts: list[TrajectoryArtifact] = field(default_factory=list)


@dataclass
class ScoringPlan:
    """Every input split by where its result goes."""

    stems: dict[str, StemPlan]
    saved: list[TrajectoryArtifact] = field(default_factory=list)


def _check_axial_admission(artifacts: Iterable[TrajectoryArtifact]) -> None:
    """Refuse unsupported inputs before hashing, cache reads, or checkpoint writes."""
    try:
        for artifact in artifacts:
            refuse_axial_channeling(artifact.case)
    except ValueError as error:
        raise TrajectoryArtifactError(str(error)) from error


def plan_scoring(paths: Iterable[Path]) -> ScoringPlan:
    """Check every artifact and split them by result destination.

    Artifacts recording a checkpoint stem and dataset identity group by stem
    (:class:`StemPlan`). Artifacts recording neither are ``saved``: scored to
    result files, after verifying the recorded segment fingerprint.

    Raises
    ------
    TrajectoryArtifactError
        An artifact cannot feed the spectrum phase, records only half a run
        identity, fails its fingerprint or explicit transport controls,
        disagrees with another artifact of its stem, or duplicates a case.
    """
    from ..montecarlo.runner.artifacts import _check_physics_provenance
    from ..montecarlo.runner.streamed_scoring import streaming_refusal

    plans: dict[str, StemPlan] = {}
    saved: list[TrajectoryArtifact] = []
    seen: dict[tuple[str, str, float], Path] = {}
    for path in paths:
        artifact = read_trajectory_artifact(path, load_transport=False)
        _check_axial_admission((artifact,))
        artifact_spectrum_inputs(artifact)
        _check_physics_provenance(artifact)
        # The engine's own gate: automatic reflection selection streams (#1281).
        if (reason := streaming_refusal(artifact.case)) is not None:
            raise TrajectoryArtifactError(
                f"trajectory artifact {path}: {reason} and cannot be scored in bounded blocks"
            )
        identity = artifact.provenance.get("dataset_identity")
        stem = artifact.provenance.get("checkpoint_stem")
        if not isinstance(identity, dict) and not stem:
            _check_saved(artifact)
            saved.append(artifact)
            continue
        if not isinstance(identity, dict) or not stem:
            raise TrajectoryArtifactError(
                f"trajectory artifact {path} records only part of a run identity "
                "(checkpoint stem and dataset identity); it cannot be scored"
            )
        plan = plans.setdefault(stem, StemPlan(stem, identity))
        if identity.get("parameter_sha256") != plan.dataset_identity.get("parameter_sha256"):
            raise TrajectoryArtifactError(
                f"trajectory artifact {path} was captured by a different run of stem "
                f"{stem!r} (parameter digest differs); score each run separately"
            )
        key = (stem, str(artifact.case["name"]), float(artifact.case["E0_keV"]))
        if key in seen:
            raise TrajectoryArtifactError(
                f"trajectory artifacts {seen[key]} and {path} hold the same case"
            )
        seen[key] = path
        plan.artifacts.append(artifact)
    return ScoringPlan(plans, saved)


def plan_stems(paths: Iterable[Path]) -> dict[str, StemPlan]:
    """Group run-captured artifacts by the checkpoint stem their run recorded.

    Raises
    ------
    TrajectoryArtifactError
        As :func:`plan_scoring`, or an artifact records no run identity.
    """
    plan = plan_scoring(paths)
    if plan.saved:
        raise TrajectoryArtifactError(
            f"trajectory artifact {plan.saved[0].path} records no checkpoint stem and dataset "
            "identity; only artifacts captured by `pyrite run --trajectories` can "
            "be scored into a checkpoint"
        )
    return plan.stems


def _check_saved(artifact: TrajectoryArtifact) -> None:
    """Refuse an unscorable or corrupt API-saved artifact before any write."""
    from ..campaign.trajectory_io import stored_segment_fingerprint

    for name, value in (artifact.settings.get("engine_overrides") or {}).items():
        raise TrajectoryArtifactError(
            f"trajectory artifact {artifact.path} was transported with explicit "
            f"{name}={value!r}; pyrite.simulate uses the production default, so "
            f"transport again with simulate_trajectories(..., {name}=None)"
        )
    recorded = artifact.settings.get("segment_fingerprint")
    if recorded is not None:
        actual = stored_segment_fingerprint(artifact.path)
        if actual != recorded:
            raise TrajectoryArtifactError(
                f"trajectory artifact {artifact.path} segments do not match their "
                f"recorded fingerprint ({actual} != {recorded}); it is corrupt"
            )


def check_targets(plans: Iterable[StemPlan], checkpoint_dir: str | os.PathLike[str]) -> None:
    """Refuse before any write when a target checkpoint belongs to another run."""
    for plan in plans:
        _check_target(plan, Path(checkpoint_dir) / plan.stem)


def _check_target(plan: StemPlan, checkpoint_path: Path) -> None:
    from .persistence import _checkpoint_exists, _manifest_path_for

    if not _checkpoint_exists(checkpoint_path):
        return
    existing = _existing_parameter_digest(_manifest_path_for(checkpoint_path))
    expected = plan.dataset_identity.get("parameter_sha256")
    if existing is not None and existing != expected:
        raise TrajectoryArtifactError(
            f"checkpoint {checkpoint_path} was written by a different run "
            f"(parameter digest {existing[:12]}, artifacts {str(expected)[:12]}); "
            "choose another --checkpoint-dir"
        )


@dataclass(frozen=True)
class StemSummary:
    """Outcome of scoring one stem."""

    stem: str
    checkpoint_path: Path
    scored: int
    kept: int


def score_stem(
    plan: StemPlan,
    checkpoint_dir: str | os.PathLike[str],
    *,
    max_segments: int,
    overwrite: bool = False,
    on_case: Callable[[TrajectoryArtifact, dict[str, Any]], None] | None = None,
) -> StemSummary:
    """Score ``plan``'s artifacts into ``<checkpoint_dir>/<stem>``.

    Existing records of the stem are kept; a case that already has one is
    skipped unless ``overwrite``. Records of cases without an artifact are
    never touched.

    Raises
    ------
    TrajectoryArtifactError
        The existing checkpoint belongs to a different run identity, or an
        artifact fails its spectrum-phase checks.
    """
    from ..montecarlo.runner import stream_spectrum_from_artifact
    from ..results import store_result
    from .persistence import (
        _checkpoint_components_save,
        _checkpoint_exists,
        _checkpoint_load,
        _manifest_save,
    )

    _check_axial_admission(plan.artifacts)
    checkpoint_path = Path(checkpoint_dir) / plan.stem
    _check_target(plan, checkpoint_path)
    results: dict = _checkpoint_load(checkpoint_path) if _checkpoint_exists(checkpoint_path) else {}
    todo = [
        artifact
        for artifact in plan.artifacts
        if overwrite or artifact.case["E0_keV"] not in results.get(artifact.case["name"], {})
    ]

    def _save():
        _checkpoint_components_save(checkpoint_path, results)
        _manifest_save(checkpoint_path, results, plan.dataset_identity)

    pending = 0
    for artifact in todo:
        digest = file_sha256(artifact.path)
        out = stream_spectrum_from_artifact(artifact.path, max_segments=max_segments)
        out["source_trajectory"] = {
            "path": os.fspath(artifact.path),
            "sha256": digest,
            "schema_version": artifact.schema_version,
        }
        store_result(results, artifact.case, out)
        if on_case is not None:
            on_case(artifact, out)
        pending += 1
        if pending >= SAVE_EVERY:
            _save()
            pending = 0
    if pending:
        _save()
    return StemSummary(
        stem=plan.stem,
        checkpoint_path=checkpoint_path,
        scored=len(todo),
        kept=len(plan.artifacts) - len(todo),
    )


#: Result components written to a saved-artifact ``.npz``, as ``Result`` attribute paths.
_RESULT_COMPONENTS = (
    "cxr.incoherent",
    "cxr.coherent",
    "characteristic",
    "channeling",
    "transition",
    "brem",
)


@dataclass(frozen=True)
class SavedSummary:
    """Outcome of scoring one API-saved artifact."""

    artifact: Path
    target: Path
    n_segments: int | None
    written: bool


def score_saved(
    artifact: TrajectoryArtifact,
    target: str | os.PathLike[str],
    *,
    max_segments: int,
    overwrite: bool = False,
) -> SavedSummary:
    """Score an API-saved ``artifact`` to the ``.npz`` file ``target``.

    The replay is ``score_spectra(load_trajectories(path, stream=True))`` with
    ``max_segments`` bounding each block. Each present component of the
    :class:`~pyrite.Result` is stored as ``<name>_photon_energy_eV`` and
    ``<name>_density`` (photons / incident electron / eV / sr; ``<name>`` is the
    ``Result`` path with ``.`` as ``_``, plus ``<name>_bin_edges_eV`` for
    bin-mean sampling), with a JSON ``metadata`` string carrying units, the
    case, the identity digest and the source artifact's path and SHA-256. An
    existing ``target`` is kept unless ``overwrite``; the write is atomic.
    """
    from ..campaign.trajectory_io import load_trajectories
    from ..scoring import _score_recorded

    target = Path(target)
    if target.exists() and not overwrite:
        return SavedSummary(artifact.path, target, None, written=False)
    result = _score_recorded(
        load_trajectories(artifact.path, stream=True), None, max_segments=max_segments
    )
    arrays: dict[str, Any] = {}
    components: dict[str, dict[str, str]] = {}
    for path in _RESULT_COMPONENTS:
        part: Any = result
        for attribute in path.split("."):
            part = getattr(part, attribute)
        if part is None:
            continue
        name = path.replace(".", "_")
        arrays[f"{name}_photon_energy_eV"] = np.asarray(part.photon_energy_eV)
        arrays[f"{name}_density"] = np.asarray(part.density)
        if part.bin_edges_eV is not None:
            arrays[f"{name}_bin_edges_eV"] = np.asarray(part.bin_edges_eV)
        components[name] = {"sampling": part.sampling, "normalization": part.normalization}
    metadata = {
        "format": "pyrite-trajectory-spectra",
        "units": {"photon_energy_eV": "eV", "density": "photons / incident electron / eV / sr"},
        "components": components,
        "case": {"name": artifact.case["name"], "E0_keV": float(artifact.case["E0_keV"])},
        "identity_digest": result.provenance.get("identity_digest"),
        "source_trajectory": {
            "path": os.fspath(artifact.path),
            "sha256": file_sha256(artifact.path),
            "schema_version": artifact.schema_version,
        },
    }
    arrays["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    try:
        with open(partial, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return SavedSummary(artifact.path, target, int(artifact.attrs["segment_count"]), written=True)


def _existing_parameter_digest(manifest_path: str) -> str | None:
    try:
        with open(manifest_path) as handle:
            identity = json.load(handle).get("dataset_identity") or {}
    except (OSError, ValueError, TypeError):
        return None
    return identity.get("parameter_sha256")
