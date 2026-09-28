"""Lazy discovery of the stored observations that belong to one checkpoint.

The observation store at ``observations/<stem>`` holds exactly one checkpoint
stem's observations. Each is named by its case: first from the ``case`` that
sweep production writes into the record's provenance, then from the
checkpoint's ``cases.json`` (case to content key, written only when the shared
cache is enabled), and otherwise by its source content key. Only the index and
the small JSON records are read; a true-spatial HDF5 object is opened by
:meth:`ObservationInventory.load` alone.
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from .store import ObservationStore, ObservationStoreError, StoredObservation

CASE_MANIFEST_SCHEMA = "cxr.case-manifest.v1"


@dataclass(frozen=True)
class ObservationEntry:
    """One stored observation of one checkpoint case.

    Parameters
    ----------
    case_name, E0_keV
        Checkpoint case the observation's source transport belongs to;
        ``case_name`` falls back to ``"source <key prefix>"`` and ``E0_keV`` to
        NaN when no case is recorded.
    source_identity_digest
        Content key shared by the case and the observation index.
    observation_digest
        Layered observation identity digest.
    response_type
        Detector-response class name, for example ``"Timepix3"``.
    event_semantics
        Recorded count semantics of the response.
    exposure_s
        Acquisition exposure in seconds.
    mode
        ``"expected"`` or ``"poisson"``.
    """

    case_name: str
    E0_keV: float
    source_identity_digest: str
    observation_digest: str
    response_type: str
    event_semantics: str
    exposure_s: float
    mode: str

    @property
    def label(self) -> str:
        """Short menu label: case, response, exposure, and count mode."""
        case = (
            self.case_name if math.isnan(self.E0_keV) else f"{self.case_name} @ {self.E0_keV:g} keV"
        )
        return (
            f"{case} -- {self.response_type}, "
            f"{self.exposure_s:g} s, {self.mode} ({self.observation_digest[:8]})"
        )


@dataclass(frozen=True)
class ObservationInventory:
    """Observations discovered for one checkpoint stem.

    ``problems`` lists unreadable index or record files instead of raising, so
    a partially transferred store still lists what it can.
    """

    stem: str
    store: ObservationStore
    entries: tuple[ObservationEntry, ...]
    problems: tuple[str, ...] = ()

    def load(self, observation_digest: str) -> StoredObservation:
        """Reopen one listed observation without transport."""
        return self.store.load(observation_digest)


def _case_manifest(checkpoint_dir: Path, stem: str) -> dict | None:
    for path in (checkpoint_dir / stem / "cases.json", checkpoint_dir / f"{stem}.cases.json"):
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest.get("schema") != CASE_MANIFEST_SCHEMA:
                raise ValueError(f"unknown case manifest schema in {path}")
            return manifest
    return None


def observation_inventory(
    stem: str,
    *,
    checkpoint_dir: str | os.PathLike[str] | None = None,
    observation_root: str | os.PathLike[str] | None = None,
) -> ObservationInventory:
    """List the stored observations of one checkpoint stem.

    Parameters
    ----------
    stem
        Checkpoint stem, for example a material or profile name.
    checkpoint_dir
        Checkpoint root; ``None`` uses the default workspace checkpoints.
    observation_root
        Observation root; ``None`` uses the sibling ``observations`` of
        ``checkpoint_dir``, where ``pyrite run`` writes them.

    Returns
    -------
    ObservationInventory
        Entries ordered by case label, then observation digest. Unreadable
        manifests, indices, or records become ``problems``.
    """
    if checkpoint_dir is None:
        from ..checkpoints.persistence import DEFAULT_CHECKPOINT_DIR

        checkpoint_dir = DEFAULT_CHECKPOINT_DIR
    root = Path(checkpoint_dir).resolve()
    store = ObservationStore(
        stem, root.parent / "observations" if observation_root is None else observation_root
    )
    if not store.index_path.exists():
        return ObservationInventory(stem, store, ())
    problems: list[str] = []
    try:
        sources = store.sources()
    except ObservationStoreError as exc:
        return ObservationInventory(stem, store, (), (str(exc),))
    manifest_cases: dict[str, tuple[str, float]] = {}
    try:
        manifest = _case_manifest(root, stem)
    except (OSError, ValueError) as exc:
        manifest = None
        problems.append(f"unreadable case manifest: {exc}")
    for case in (manifest or {}).get("cases", ()):
        manifest_cases.setdefault(case["content_key"], (str(case["name"]), float(case["E0_keV"])))
    entries = []
    for key in sources:
        for digest in store.digests(key):
            try:
                record = store.record(digest)
                layers = record["identity"]
                response = layers["response"]
                config = layers["acquisition"]["config"]
                recorded = record.get("provenance", {}).get("case")
                if recorded is not None:
                    name, energy = str(recorded["name"]), float(recorded["E0_keV"])
                else:
                    name, energy = manifest_cases.get(key, (f"source {key[:12]}", math.nan))
                entries.append(
                    ObservationEntry(
                        case_name=name,
                        E0_keV=energy,
                        source_identity_digest=key,
                        observation_digest=digest,
                        response_type=str(response["type"]).rsplit(".", 1)[-1],
                        event_semantics=str(response["event_semantics"]),
                        exposure_s=float(config["exposure_s"]),
                        mode=str(config["mode"]),
                    )
                )
            except (ObservationStoreError, KeyError, TypeError, ValueError) as exc:
                problems.append(f"observation {digest}: {exc}")
    entries.sort(
        key=lambda entry: (
            entry.case_name,
            math.inf if math.isnan(entry.E0_keV) else entry.E0_keV,
            entry.observation_digest,
        )
    )
    return ObservationInventory(stem, store, tuple(entries), tuple(problems))


__all__ = ["ObservationEntry", "ObservationInventory", "observation_inventory"]
