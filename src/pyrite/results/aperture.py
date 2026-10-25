"""Typed aperture results and histogram views, separate from per-sr spectra."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from ..photon_event_observation import _BUDGET, PhotonEventTally, PhotonRelaxationBudget
from ..photon_event_observation import SCHEMA as EVENT_SCHEMA
from ..photon_event_observation import _validate as _validate_events
from ..photon_observation import _ARRAYS, PhotonContinuumTally, _checksum, _validate
from ..photon_observation import SCHEMA as CONTINUUM_SCHEMA

if TYPE_CHECKING:
    import polars as pl

ApertureResult = PhotonContinuumTally | PhotonEventTally


def aperture_from_record(record: Mapping[str, Any]) -> ApertureResult | None:
    """Read one production checkpoint record's checked aperture observable.

    The record's enabled selection and original source/observation digests
    must agree. Missing enabled observations raise; mode-off records return
    ``None``. This operation uses stored pins and performs no data fetch.
    """
    case = record.get("case", {})
    selection = case.get("photon_transport")
    payload = record.get("photon_aperture")
    if selection is None:
        if payload is not None:
            raise ValueError("photon aperture requires an enabled Case selection")
        return None
    if payload is None:
        raise ValueError("selected photon checkpoint is missing its aperture component")
    tally = aperture_from_payload(payload)
    if (
        tally.source_identity_digest != selection["source_identity_digest"]
        or tally.observation_identity_digest != selection["observation_identity_digest"]
    ):
        raise ValueError("photon aperture identity disagrees with its Case selection")
    return tally


@overload
def aperture_from_payload(payload: dict) -> ApertureResult: ...


@overload
def aperture_from_payload(payload: None) -> None: ...


def aperture_from_payload(payload: dict | None) -> ApertureResult | None:
    """Restore and check checkpoint aperture data without transport or table fetches."""
    if payload is None:
        return None
    metadata, arrays = dict(payload["metadata"]), payload["arrays"]
    checksum = metadata.pop("content_sha256")
    if checksum != _checksum(metadata, arrays, tuple(arrays)):
        raise ValueError("photon aperture content checksum mismatch")
    schema = metadata.get("schema")
    common = {name: arrays[name] for name in _ARRAYS}
    if schema == CONTINUUM_SCHEMA:
        _validate(metadata, arrays)
        metadata.pop("schema")
        return PhotonContinuumTally(**common, **metadata)
    if schema == EVENT_SCHEMA:
        _validate_events(metadata, arrays)
        metadata.pop("schema")
        relaxation = (
            PhotonRelaxationBudget(**{name: arrays[name] for name in _BUDGET})
            if metadata["source_component"] == "characteristic"
            else None
        )
        return PhotonEventTally(
            **common,
            **metadata,
            relaxation=relaxation,
            local_source_energy_eV=arrays["local_source_energy_eV"],
            tracked_source_energy_eV=arrays["tracked_source_energy_eV"],
        )
    raise ValueError("unsupported photon aperture schema")


def aperture_dataframe(tally: ApertureResult, *, current_nA: float | None = None) -> pl.DataFrame:
    """Histogram densities/counts over the physical aperture, with explicit units.

    Without ``current_nA`` densities are photons/(incident electron eV), and
    counts are photons/incident electron. Supplying a finite nonnegative mean
    beam current scales to photons/(s eV) and photons/s. No solid angle or
    proposal-count factor is applied. Unbinned detections remain on ``tally``;
    they are not assigned to displayed energy bins or interpolated onto nodes.
    """
    import polars as pl
    from scipy.constants import elementary_charge

    scale = 1.0
    density_units, count_units = "photons / (electron eV)", "photons / electron"
    if current_nA is not None:
        if isinstance(current_nA, bool) or not np.isfinite(current_nA) or current_nA < 0:
            raise ValueError("current_nA must be finite and nonnegative")
        scale = float(current_nA) * 1e-9 / elementary_charge
        density_units, count_units = "photons / (s eV)", "photons / s"
    widths = np.diff(tally.edges_eV)
    densities = {
        "total": scale * tally.total_per_primary_eV,
        "direct": scale * tally.direct_per_primary_eV,
        "redistributed": scale * tally.redistributed_per_primary_eV,
    }
    return pl.DataFrame(
        {
            "energy_low_eV": tally.edges_eV[:-1],
            "energy_high_eV": tally.edges_eV[1:],
            **densities,
            **{f"{name}_counts": values * widths for name, values in densities.items()},
            "density_units": [density_units] * widths.size,
            "count_units": [count_units] * widths.size,
        }
    )
