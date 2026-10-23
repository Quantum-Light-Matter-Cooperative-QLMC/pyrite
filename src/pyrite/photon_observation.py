"""Portable aperture tallies for the experimental photon-continuum reference.

This artifact stores reduced observations, not tracks or a transport restart.
It is separate from directional Result spectra and production checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .photon_transport import PhotonContinuumObservation

SCHEMA = "pyrite.photon-continuum-tally.v1"
NORMALIZATION = "photons / (incident electron eV), aperture-integrated"
_ARRAYS = (
    "edges_eV",
    "total_per_primary_eV",
    "direct_per_primary_eV",
    "redistributed_per_primary_eV",
)
_IDENTITY_KEYS = (
    "schema",
    "model",
    "source_identity_digest",
    "data_sha256",
    "proposal",
    "aperture",
    "boxes",
    "energy_bin_edges_eV",
)


@dataclass(frozen=True, slots=True)
class PhotonContinuumTally:
    """Reloaded aperture observation, without source rows, histories or Case.

    Spectra have units photons/(incident electron eV), integrated over the
    physical aperture. Counts include hits outside the displayed energy bins.
    Provenance retains the original data pins even when local tables change.
    """

    edges_eV: np.ndarray
    total_per_primary_eV: np.ndarray
    direct_per_primary_eV: np.ndarray
    redistributed_per_primary_eV: np.ndarray
    detected_per_primary: float
    unbinned_detected_per_primary: float
    n_primaries: int
    n_proposals: int
    charged_feedback: bool
    source_identity_digest: str
    observation_identity_digest: str
    provenance: dict[str, Any]


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _checksum(metadata: dict, arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256(_json(metadata).encode())
    for name in _ARRAYS:
        array = arrays[name]
        digest.update(_json([name, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_identity(metadata: dict, arrays: dict[str, np.ndarray]) -> None:
    """Check stored identity without consulting today's model or data pins."""
    provenance = metadata["provenance"]
    payload = {key: provenance[key] for key in _IDENTITY_KEYS}
    if "charged_feedback" in provenance:
        payload["charged_feedback"] = provenance["charged_feedback"]
    digest = hashlib.sha256(_json(payload).encode()).hexdigest()
    if (
        payload["schema"] != "pyrite.photon-continuum-observation.v1"
        or digest != metadata["observation_identity_digest"]
        or digest != provenance["observation_identity_digest"]
        or payload["source_identity_digest"] != metadata["source_identity_digest"]
    ):
        raise ValueError("photon observation identity mismatch")
    if (
        provenance["normalization"] != NORMALIZATION
        or not np.array_equal(payload["energy_bin_edges_eV"], arrays["edges_eV"])
        or payload["proposal"]["n_samples"] != metadata["n_proposals"]
        or ("charged_feedback" in payload) != metadata["charged_feedback"]
    ):
        raise ValueError("photon tally disagrees with observation provenance")


def _validate(metadata: dict, arrays: dict[str, np.ndarray]) -> None:
    if metadata["schema"] != SCHEMA:
        raise ValueError("unsupported photon tally schema")
    for key in ("n_primaries", "n_proposals"):
        if type(metadata[key]) is not int or metadata[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(metadata["charged_feedback"]) is not bool:
        raise ValueError("charged_feedback must be a bool")
    for array in arrays.values():
        if array.dtype.kind != "f" or array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError("photon tally arrays must be finite floating-point vectors")
    edges = arrays["edges_eV"]
    if len(edges) < 2 or np.any(edges < 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("photon tally edges must be nonnegative and strictly increasing")
    spectra = [arrays[key] for key in _ARRAYS[1:]]
    if any(array.shape != (len(edges) - 1,) or np.any(array < 0) for array in spectra):
        raise ValueError("photon tally spectra must be nonnegative and match the bins")
    if not np.allclose(spectra[0], spectra[1] + spectra[2], rtol=1e-12, atol=0):
        raise ValueError("photon tally categories do not partition the total")
    _validate_counts(metadata, edges, spectra[0])
    _validate_identity(metadata, arrays)


def _validate_counts(metadata: dict, edges: np.ndarray, total: np.ndarray) -> None:
    for key in ("detected_per_primary", "unbinned_detected_per_primary"):
        value = metadata[key]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    expected = float(np.sum(total * np.diff(edges))) + metadata["unbinned_detected_per_primary"]
    if not np.isclose(metadata["detected_per_primary"], expected, rtol=1e-12, atol=0):
        raise ValueError("photon tally counts disagree with the integrated spectrum")


def save_photon_continuum(
    observation: PhotonContinuumObservation, path: str | os.PathLike[str]
) -> Path:
    """Atomically save an experimental observation as NPZ arrays and JSON.

    Uses the exact path supplied, replacing an existing file after validation.
    Parent directories must exist. No pickle, transport or data fetch occurs.
    Only reduced tallies and provenance are stored; this is not replay input.
    """
    score = observation.score
    arrays = {name: np.array(getattr(score, name), copy=True) for name in _ARRAYS}
    metadata = {
        "schema": SCHEMA,
        "detected_per_primary": float(score.detected_per_primary),
        "unbinned_detected_per_primary": float(score.unbinned_detected_per_primary),
        "n_primaries": score.n_primaries,
        "n_proposals": score.n_proposals,
        "charged_feedback": score.charged_feedback,
        "source_identity_digest": observation.source_identity_digest,
        "observation_identity_digest": observation.observation_identity_digest,
        "provenance": dict(observation.provenance),
    }
    # Round-trip JSON before validation to detach mutable provenance references.
    metadata = json.loads(_json(metadata))
    _validate(metadata, arrays)
    metadata["content_sha256"] = _checksum(metadata, arrays)
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(
                stream,
                metadata=np.array(_json(metadata)),
                edges_eV=arrays["edges_eV"],
                total_per_primary_eV=arrays["total_per_primary_eV"],
                direct_per_primary_eV=arrays["direct_per_primary_eV"],
                redistributed_per_primary_eV=arrays["redistributed_per_primary_eV"],
            )
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def load_photon_continuum(path: str | os.PathLike[str]) -> PhotonContinuumTally:
    """Load and check an experimental aperture tally without rerunning physics.

    Rejects unknown schemas, altered contents and inconsistent normalization
    or identity. The checksum detects corruption, not malicious replacement.
    Stored source pins are preserved; local table versions are irrelevant.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {*_ARRAYS, "metadata"}:
                raise ValueError("unexpected photon tally archive members")
            metadata = json.loads(str(data["metadata"]))
            arrays = {name: np.array(data[name], copy=True) for name in _ARRAYS}
        checksum = metadata.pop("content_sha256")
        if checksum != _checksum(metadata, arrays):
            raise ValueError("photon tally content checksum mismatch")
        _validate(metadata, arrays)
        metadata.pop("schema")
        return PhotonContinuumTally(
            edges_eV=arrays["edges_eV"],
            total_per_primary_eV=arrays["total_per_primary_eV"],
            direct_per_primary_eV=arrays["direct_per_primary_eV"],
            redistributed_per_primary_eV=arrays["redistributed_per_primary_eV"],
            **metadata,
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid photon tally metadata") from exc


__all__ = ["PhotonContinuumTally", "save_photon_continuum", "load_photon_continuum"]
