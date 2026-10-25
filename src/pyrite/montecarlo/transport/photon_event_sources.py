"""Physical characteristic/annihilation emission adapters for photon reference tracking.

These are alternative or distinct physical contributions, never a histogram
of the same hard bremsstrahlung already represented by a weighted continuous
source. No source attenuation or original-electron energy debit is applied.
Validation: photon-secondary-feedback
"""

from dataclasses import dataclass

import numpy as np

from .photon_feedback import FeedbackChargedStage, _join_emissions, impact_relaxation_emissions
from .photon_histories import ELECTRON
from .photon_sources import WeightedPhotonSources


@dataclass(frozen=True, slots=True)
class EventPhotonSources(WeightedPhotonSources):
    """Tracked physical photons plus weighted sub-cutoff energy by original history.

    Validation: photon-secondary-feedback
    """

    local_history_id: np.ndarray
    local_energy_eV: np.ndarray
    photon_cutoff_eV: float


def emission_sources(
    emissions, history_id, *, weight=None, source_kind=1, event_offset=0, photon_cutoff_eV=1000.0
):
    """Adapt unattenuated emission rows into the common photon queue source batch.

    ``parent`` indexes the supplied original-history array; each physical
    photon has weight one unless parent statistical weights are supplied.
    The caller's normalization remains the original incident population.
    Use disjoint ``event_offset`` ranges when combining batches under one
    seed/history domain. Input may be ``annihilation_emissions`` or a primary
    characteristic emission adapter's ``photons``. No first-hit sink fields
    are read, and physical hard-photon histograms are not merged implicitly.
    Supply the tracker's cutoff: photons at or below it are kept separately
    as ``local_energy_eV``, grouped and weighted by ``local_history_id``. They
    do not enter the tracker or detector tally; their energy is not discarded.
    Validation: photon-secondary-feedback
    """
    histories = np.asarray(history_id)
    parent = np.asarray(emissions["parent"])
    for name, indices in (("history_id", histories), ("emission parent", parent)):
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if np.any(indices < 0):
            raise ValueError(f"{name} must be nonnegative")
    if np.any(parent >= histories.size):
        raise ValueError("emission parent outside original history population")
    if type(event_offset) is not int or event_offset < 0:
        raise ValueError("event_offset must be a nonnegative integer")
    if (
        not isinstance(source_kind, (int, np.integer))
        or isinstance(source_kind, bool)
        or source_kind < 0
    ):
        raise ValueError("source_kind must be a nonnegative integer")
    weights = np.ones(histories.size) if weight is None else np.asarray(weight, dtype=float)
    if weights.shape != histories.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("source weights must be finite, positive and match history_id")
    if not np.isfinite(photon_cutoff_eV) or photon_cutoff_eV <= 0:
        raise ValueError("photon_cutoff_eV must be finite and positive")
    count = parent.size
    energy = np.asarray(emissions["energy_eV"], dtype=float).reshape(count)
    if not np.isfinite(energy).all() or np.any(energy <= 0):
        raise ValueError("emission energies must be finite and positive")
    keep = energy > photon_cutoff_eV
    n = np.count_nonzero(keep)
    unique, inverse = np.unique(histories, return_inverse=True)
    local = np.bincount(
        inverse[parent[~keep]],
        weights=weights[parent[~keep]] * energy[~keep],
        minlength=unique.size,
    )
    return EventPhotonSources(
        np.asarray(emissions["position_ang"], dtype=float).reshape(count, 3)[keep],
        np.asarray(emissions["direction"], dtype=float).reshape(count, 3)[keep],
        energy[keep],
        weights[parent[keep]],
        histories[parent[keep]],
        np.arange(event_offset, event_offset + count)[keep],
        np.full(n, source_kind, dtype=np.int64),
        np.full(n, -1, dtype=np.int64),
        unique,
        local,
        float(photon_cutoff_eV),
    )


@dataclass(frozen=True)
class PrimaryRelaxationSources:
    """Unattenuated primary characteristic/Auger-radiation sources and their budget.

    Per original history, reserved binding = emitted photon energy + escaped
    charged energy + deposited energy + binding remainder (including signed
    tabulation defects). The frozen primary-electron ledger is not modified.
    Validation: photon-secondary-feedback
    """

    photons: dict
    reserved_eV: np.ndarray
    escaped_eV: np.ndarray
    deposited_eV: np.ndarray
    binding_eV: np.ndarray


def primary_relaxation_sources(result, keys, stage: FeedbackChargedStage, *, seed):
    """Sample recorded primary charged-impact vacancies and follow their Auger cascade.

    Includes all supplied charged-track generations, preserving their root
    ``electron_id``. The caller selects the desired primary population before
    this boundary. Binding was already reserved by shell transport; emitted
    EADL particles release that reservoir once. Above-threshold Auger electrons
    use the same charged feedback stage and return their radiation. Lower
    energies deposit at the vacancy. Source: ENDF-102 MF=28 and the shell
    hard-transfer partition; no vacancies yields an empty source and zero
    binding debit. Validation: photon-secondary-feedback
    """
    from .secondaries import _host
    from .shell_transport import hard_event_energy_accounting

    keys = np.asarray(keys, dtype=np.uint64)
    n = keys.size
    acc = hard_event_energy_accounting(result)
    root = _host(result["electron_id"]).astype(np.int64)
    reserved = np.bincount(root[acc["row"]], weights=acc["binding_keV"] * 1e3, minlength=n)
    particles, released = impact_relaxation_emissions(
        result, keys, min(stage.photon_cutoff_eV, stage.threshold_eV)
    )
    light = particles["species"] == -1
    photons = [
        {
            name: particles[name][light]
            for name in ("position_ang", "direction", "energy_eV", "parent")
        }
    ]
    escaped, deposited, binding = np.zeros(n), np.zeros(n), reserved - released
    low = ~light & (particles["energy_eV"] <= stage.threshold_eV)
    np.add.at(deposited, particles["parent"][low], particles["energy_eV"][low])
    keep = ~light & ~low
    if keep.any():
        parent = particles["parent"][keep]
        rows = dict(
            position_ang=particles["position_ang"][keep],
            direction=particles["direction"][keep],
            kinetic_eV=particles["energy_eV"][keep],
            key=particles["key"][keep],
            species=np.full(np.count_nonzero(keep), ELECTRON),
        )
        charged = stage(rows, seed)
        np.add.at(escaped, parent, charged.escaped_eV)
        np.add.at(deposited, parent, charged.deposited_eV)
        np.add.at(binding, parent, charged.binding_eV)
        photons.append({**charged.photons, "parent": parent[charged.photons["parent"]]})
    return PrimaryRelaxationSources(_join_emissions(photons), reserved, escaped, deposited, binding)


def primary_annihilation_sources(result, *, weight=None, event_offset=0, photon_cutoff_eV=1000.0):
    """Replay original #295 annihilation emissions without their first-hit sink.

    ``result`` is a charged transport result with recorded pair/annihilation
    events. Both photons are retained, including those originally labelled
    absorbed. Their existing energy, direction and emission position are
    reused without another annihilation draw. ``weight``, if supplied, is
    indexed by original primary electron id, not pair event. This standalone
    source must not be added to a weighted continuum that represents the
    same physical bremsstrahlung-driven shower.

    Validation: photon-secondary-feedback
    """
    from .secondaries import _host

    pair = result.get("pair_production")
    if pair is None or "annihilation_photons" not in pair:
        raise ValueError("primary annihilation sources require recorded annihilation photons")
    raw = pair["annihilation_photons"]
    histories = _host(pair["events"]["electron_id"]).astype(np.int64)
    photons = dict(
        position_ang=_host(raw["origin"]),
        direction=_host(raw["direction"]),
        energy_eV=_host(raw["k_keV"]) * 1e3,
        parent=_host(raw["event"]).astype(np.int64),
    )
    weights = None if weight is None else np.asarray(weight, dtype=float)[histories]
    return emission_sources(
        photons,
        histories,
        weight=weights,
        source_kind=2,
        event_offset=event_offset,
        photon_cutoff_eV=photon_cutoff_eV,
    )
