"""Reduced physical photon event observations, without transport or source replay.

Stores aperture tallies and primary relaxation budgets with their original pins.
The provisional schema is separate from continuum tallies and production Results.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .photon_observation import (
    _ARRAYS,
    _IDENTITY_KEYS,
    NORMALIZATION,
    _atomic_save,
    _checksum,
    _json,
    _read_tally,
    _tally_payload,
    _validate_spectra,
)

if TYPE_CHECKING:
    from os import PathLike
    from pathlib import Path

    from .photon_events import PhotonEventObservation
    from .results.quantified_aperture import QuantifiedApertureSpectrum

SCHEMA = "pyrite.photon-event-tally.v1"
_LOCAL = "local_source_energy_eV"
_TRACKED = "tracked_source_energy_eV"
_BUDGET = ("reserved_eV", "escaped_eV", "deposited_eV", "binding_eV", "emitted_eV")
_REQUIRED = (*_ARRAYS, _LOCAL, _TRACKED)


@dataclass(frozen=True, slots=True)
class PhotonRelaxationBudget:
    """Per-original-primary energies in eV, excluding photon transport.

    Reserved binding equals escaped + deposited + signed binding remainder +
    emitted photon energy. The emitted term includes sub-cutoff photons.
    No original-primary energy debit or reconstructed photon history is stored.
    """

    reserved_eV: np.ndarray
    escaped_eV: np.ndarray
    deposited_eV: np.ndarray
    binding_eV: np.ndarray
    emitted_eV: np.ndarray


@dataclass(frozen=True, slots=True)
class PhotonEventTally:
    """Reloaded event aperture tally and optional primary relaxation budget.

    Spectra are photons/(original incident electron eV), aperture-integrated.
    ``n_proposals`` counts tracked input source photons, including zero for an
    empty source. ``local_source_energy_eV`` records omitted sub-cutoff source
    energy per original primary; it is not detector energy or a second deposit.
    For characteristic sources it is included in ``relaxation.emitted_eV``.
    Arrays, original identities and table pins load without today's data files.
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
    source_component: str
    local_source_energy_eV: np.ndarray
    tracked_source_energy_eV: np.ndarray
    relaxation: PhotonRelaxationBudget | None

    def quantify(self) -> QuantifiedApertureSpectrum:
        """Return an independent aperture histogram without altering event budgets."""
        from .results.quantified_aperture import QuantifiedApertureSpectrum

        return QuantifiedApertureSpectrum.from_tally(self)


def _per_primary_energy(parents, energy, n_primaries):
    parents, energy = np.asarray(parents), np.asarray(energy)
    if parents.ndim != 1 or not np.issubdtype(parents.dtype, np.integer):
        raise ValueError("source energy owners must be an integer vector")
    if energy.shape != parents.shape or not np.isfinite(energy).all() or np.any(energy < 0):
        raise ValueError("source energies must be finite, nonnegative and match their owners")
    if np.any((parents < 0) | (parents >= n_primaries)):
        raise ValueError("source energy owners must belong to the original primary population")
    return np.bincount(parents.astype(np.int64), weights=energy, minlength=n_primaries).astype(
        float, copy=False
    )


def _event_arrays(observation, metadata, arrays):
    score, provenance = observation.score, metadata["provenance"]
    n = score.n_primaries
    if type(n) is not int or n <= 0:
        raise ValueError("n_primaries must be a positive integer")
    source = score.sources
    if score.n_proposals != np.asarray(source.energy_eV).size:
        raise ValueError("event n_proposals must equal the tracked source row count")
    if source.photon_cutoff_eV != provenance["proposal"]["photon_cutoff_eV"]:
        raise ValueError("event source cutoff disagrees with observation provenance")
    arrays[_LOCAL] = _per_primary_energy(source.local_history_id, source.local_energy_eV, n)
    arrays[_TRACKED] = _per_primary_energy(
        source.history_id, np.asarray(source.energy_eV) * np.asarray(source.weight), n
    )
    relaxation = observation.relaxation
    if relaxation is not None:
        for name in _BUDGET[:-1]:
            arrays[name] = np.array(getattr(relaxation, name), copy=True)
        arrays["emitted_eV"] = _per_primary_energy(
            relaxation.photons["parent"], relaxation.photons["energy_eV"], n
        )


def _validate_identity(metadata, arrays):
    provenance = metadata["provenance"]
    payload = {
        key: provenance[key] for key in (*_IDENTITY_KEYS, "charged_feedback", "relaxation_feedback")
    }
    if "execution" in provenance:
        payload["execution"] = provenance["execution"]
    digest = hashlib.sha256(_json(payload).encode()).hexdigest()
    if (
        payload["schema"] != "pyrite.photon-event-observation.v1"
        or digest != metadata["observation_identity_digest"]
        or digest != provenance["observation_identity_digest"]
        or payload["source_identity_digest"] != metadata["source_identity_digest"]
    ):
        raise ValueError("photon event observation identity mismatch")
    proposal = payload["proposal"]
    if (
        provenance["normalization"] != NORMALIZATION
        or not np.array_equal(payload["energy_bin_edges_eV"], arrays["edges_eV"])
        or proposal["source_role"] != "physical-event"
        or proposal["component"] != metadata["source_component"]
        or type(proposal["n_primaries"]) is not int
        or proposal["n_primaries"] != metadata["n_primaries"]
        or (payload["charged_feedback"] is not None) != metadata["charged_feedback"]
        or (payload["relaxation_feedback"] is not None)
        != (metadata["source_component"] == "characteristic")
    ):
        raise ValueError("event tally disagrees with observation provenance")


def _validate_budget(metadata, arrays):
    component, n = metadata["source_component"], metadata["n_primaries"]
    expected = {*_REQUIRED, *(_BUDGET if component == "characteristic" else ())}
    if set(arrays) != expected:
        raise ValueError("event component and relaxation budget members disagree")
    for name in (_LOCAL, _TRACKED, *(_BUDGET if component == "characteristic" else ())):
        array = arrays[name]
        if array.dtype.kind != "f" or array.shape != (n,) or not np.isfinite(array).all():
            raise ValueError("event energy arrays must be finite floating-point primary vectors")
        if name != "binding_eV" and np.any(array < 0):
            raise ValueError("event energies except signed binding must be nonnegative")
    if (metadata["n_proposals"] == 0) != (not np.any(arrays[_TRACKED] > 0)):
        raise ValueError("source row count disagrees with tracked source energy")
    if component == "characteristic":
        if not np.allclose(
            arrays["emitted_eV"], arrays[_LOCAL] + arrays[_TRACKED], rtol=1e-12, atol=1e-9
        ):
            raise ValueError("relaxation emission disagrees with tracked and local source energy")
        accounted = (
            arrays["escaped_eV"]
            + arrays["deposited_eV"]
            + arrays["binding_eV"]
            + arrays["emitted_eV"]
        )
        if not np.allclose(arrays["reserved_eV"], accounted, rtol=1e-12, atol=1e-9):
            raise ValueError("primary relaxation budget does not close")
        if np.any(arrays[_LOCAL] > arrays["emitted_eV"] + 1e-9):
            raise ValueError("local source energy exceeds emitted relaxation energy")


def _validate(metadata, arrays):
    if metadata["schema"] != SCHEMA:
        raise ValueError("unsupported photon event tally schema")
    if metadata["source_component"] not in ("characteristic", "annihilation"):
        raise ValueError("unknown photon event source component")
    n, rows = metadata["n_primaries"], metadata["n_proposals"]
    if type(n) is not int or n <= 0:
        raise ValueError("n_primaries must be a positive integer")
    if type(rows) is not int or rows < 0:
        raise ValueError("event n_proposals must be a nonnegative integer")
    if type(metadata["charged_feedback"]) is not bool:
        raise ValueError("charged_feedback must be a bool")
    if rows == 0 and metadata["detected_per_primary"] != 0:
        raise ValueError("an empty event source cannot have detected photons")
    _validate_spectra(metadata, arrays)
    _validate_budget(metadata, arrays)
    _validate_identity(metadata, arrays)


def save_photon_events(observation: PhotonEventObservation, path: str | PathLike[str]) -> Path:
    """Atomically store a reduced physical-event aperture tally at the exact path.

    Retains primary relaxation budgets and sub-cutoff source energy, excluding
    photon rows, tracks, Case and replay inputs. Parent directory must exist.
    Validates the observation before replacing an existing file. No physics or
    data fetch occurs; the checksum detects corruption, not malicious changes.
    """
    metadata, arrays = _tally_payload(observation, SCHEMA)
    metadata["source_component"] = metadata["provenance"]["proposal"]["component"]
    _event_arrays(observation, metadata, arrays)
    _validate(metadata, arrays)
    names = (*_REQUIRED, *(name for name in _BUDGET if name in arrays))
    metadata["content_sha256"] = _checksum(metadata, arrays, names)
    return _atomic_save(path, metadata, arrays)


def load_photon_events(path: str | PathLike[str]) -> PhotonEventTally:
    """Reload event tallies and budgets without primary/photon transport or tables.

    Checks content bytes, observation identity, populations, aperture
    normalization and budget closure against the original provenance. Empty
    source batches remain valid with a nonzero original primary population.
    This reduced analysis artifact is not a simulation restart or source replay.
    """
    try:
        metadata, arrays = _read_tally(path, _REQUIRED, _BUDGET)
        _validate(metadata, arrays)
        metadata.pop("schema")
        relaxation = (
            PhotonRelaxationBudget(**{name: arrays[name] for name in _BUDGET})
            if metadata["source_component"] == "characteristic"
            else None
        )
        return PhotonEventTally(
            **{name: arrays[name] for name in _ARRAYS},
            **metadata,
            local_source_energy_eV=arrays[_LOCAL],
            tracked_source_energy_eV=arrays[_TRACKED],
            relaxation=relaxation,
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid photon event tally metadata") from exc


__all__ = ["PhotonEventTally", "PhotonRelaxationBudget", "save_photon_events", "load_photon_events"]
