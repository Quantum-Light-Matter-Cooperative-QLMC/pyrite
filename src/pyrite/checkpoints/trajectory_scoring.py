"""Score checkpoint records from saved trajectory artifacts (#186).

The opt-in second stage of a run that captured transport with
``pyrite run --trajectories``: each schema-2 artifact's spectrum phase is
replayed in bounded electron blocks
(:func:`~pyrite.montecarlo.runner.stream_spectrum_from_artifact`) and stored as
an ordinary checkpoint record of the stem and dataset identity the capturing run
recorded. Nothing is transported. Records keep their source artifact's path and
SHA-256 under ``source_trajectory``. The shared per-case cache is neither read
nor written, as with ``pyrite run --no-cache``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..montecarlo.trajectories import (
    TrajectoryArtifact,
    TrajectoryArtifactError,
    artifact_spectrum_inputs,
    read_trajectory_artifact,
)

#: Records scored between checkpoint saves; an interrupted run resumes from the
#: last save.
SAVE_EVERY = 25

_HASH_BLOCK_BYTES = 8 * 1024 * 1024


def discover_artifacts(paths: Iterable[str | os.PathLike[str]]) -> list[Path]:
    """Expand directories recursively to their ``*.h5`` files; keep explicit files."""
    found: list[Path] = []
    for path in map(Path, paths):
        if path.is_dir():
            found.extend(p for p in sorted(path.rglob("*.h5")) if p.is_file())
        else:
            found.append(path)
    return found


def file_sha256(path: str | os.PathLike[str]) -> str:
    """SHA-256 of a file's bytes, read in bounded blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class StemPlan:
    """The artifacts one checkpoint stem is scored from."""

    stem: str
    dataset_identity: dict[str, Any]
    artifacts: list[TrajectoryArtifact] = field(default_factory=list)


def plan_stems(paths: Iterable[Path]) -> dict[str, StemPlan]:
    """Group artifacts by the checkpoint stem their capturing run recorded.

    Raises
    ------
    TrajectoryArtifactError
        An artifact cannot feed the spectrum phase, records no run identity,
        disagrees with another artifact of its stem, or duplicates a case.
    """
    from ..montecarlo.runner.artifacts import STREAM_REFUSED_KEYS

    plans: dict[str, StemPlan] = {}
    seen: dict[tuple[str, str, float], Path] = {}
    for path in paths:
        artifact = read_trajectory_artifact(path, load_transport=False)
        artifact_spectrum_inputs(artifact)
        for key in STREAM_REFUSED_KEYS:
            if artifact.case.get(key):
                raise TrajectoryArtifactError(
                    f"trajectory artifact {path}: {key} needs every segment at once and "
                    "cannot be scored in bounded blocks"
                )
        identity = artifact.provenance.get("dataset_identity")
        stem = artifact.provenance.get("checkpoint_stem")
        if not isinstance(identity, dict) or not stem:
            raise TrajectoryArtifactError(
                f"trajectory artifact {path} records no checkpoint stem and dataset "
                "identity; only artifacts captured by `pyrite run --trajectories` can "
                "be scored into a checkpoint"
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
    return plans


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


def _existing_parameter_digest(manifest_path: str) -> str | None:
    try:
        with open(manifest_path) as handle:
            identity = json.load(handle).get("dataset_identity") or {}
    except OSError, ValueError, TypeError:
        return None
    return identity.get("parameter_sha256")
