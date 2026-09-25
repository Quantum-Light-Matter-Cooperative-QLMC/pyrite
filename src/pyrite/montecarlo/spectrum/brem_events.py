"""Post-transport scoring of explicit hard bremsstrahlung photons.

Validation: bremslib-radiative-event-spectrum
"""

from collections.abc import Mapping

import numpy as np

from ..._backend import BACKEND
from ..._grid_semantics import node_bin_edges_and_widths
from ...materials.attenuation import _mu_total_inv_ang, _normalize_composition
from ..transport.events import EVENT_CUTOFF, EVENT_HARD_RADIATIVE
from .brem import mc_brem_spectrum
from .brem_bremslib import (
    BremsLibBremsstrahlungTable,
    bremslib_segment_state,
    evaluate_bremslib,
    stage_bremslib_table,
)
from .lines import _escape_length, _observation_direction


def _cutoff_edge(E_grid_eV, cutoff_eV):
    edges, widths = node_bin_edges_and_widths(E_grid_eV)
    cutoff = float(cutoff_eV)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("hard photon cutoff must be positive and finite")
    match = np.flatnonzero(edges == cutoff)
    if match.size != 1 or match[0] == 0 or match[0] == edges.size - 1:
        raise ValueError("hard photon cutoff must equal an interior energy-bin edge")
    return edges, widths, int(match[0])


def _check_transport_partition(segments, cutoff_eV, bremslib_tables):
    radiative = segments.get("radiative")
    if radiative is None:
        return
    if radiative.get("model") != "bremslib-soft-hard" or float(
        radiative.get("cutoff_eV", -1.0)
    ) != float(cutoff_eV):
        raise ValueError("spectrum cutoff must match the coupled transport cutoff")
    identity = tuple(
        sorted(
            (element, table.atomic_number, table.key, table.digest)
            for element, table in bremslib_tables.items()
        )
    )
    if identity != radiative.get("bremslib_tables"):
        raise ValueError("spectrum BremsLib tables must match coupled transport tables")


def mc_soft_brem_spectrum(segments, E_grid_eV, *, cutoff_eV, bremslib_tables, **kwargs):
    """Track-length bremsstrahlung below a bin-aligned hard-photon cutoff.

    ``mc_brem_spectrum`` remains the uncoupled compatibility estimator. This
    wrapper is for the soft side of coupled scoring; add its result to
    :func:`mc_hard_brem_event_spectrum` on the same energy grid. The cutoff
    must be a bin edge, so no histogram bin straddles the soft/hard boundary.

    Validation: bremslib-radiative-event-spectrum
    """
    _, _, split = _cutoff_edge(E_grid_eV, cutoff_eV)
    _check_transport_partition(segments, cutoff_eV, bremslib_tables)
    if "cross_section_model" in kwargs:
        raise ValueError("coupled soft scoring always uses the BremsLib cross section")
    if kwargs.get("E_cut_keV") is not None:
        raise NotImplementedError(
            "coupled soft scoring cannot reclip electron tracks at a new cutoff"
        )
    uncoupled_view = dict(segments)
    uncoupled_view.pop("radiative", None)
    full = mc_brem_spectrum(
        uncoupled_view,
        E_grid_eV,
        cross_section_model="bremslib",
        bremslib_tables=bremslib_tables,
        **kwargs,
    )
    soft = np.asarray(full).copy()
    soft[split:] = 0.0
    return soft


def mc_hard_brem_event_spectrum(
    segments,
    E_grid_eV,
    *,
    cutoff_eV,
    bremslib_tables: Mapping[str, BremsLibBremsstrahlungTable],
    element=None,
    n_atoms_per_ang3=None,
    composition=None,
    theta_obs_rad=np.deg2rad(119.0),
    n_hat=None,
    electron_limit=None,
):
    """Bin sampled hard photons, weighting each by its BremsLib angular law.

    Each ``HARD_RADIATIVE`` row (or terminal ``CUTOFF`` row carrying a photon)
    supplies ``hard_radiative_k_eV`` and ``hard_radiative_Z``. Its pre-event
    electron energy is ``E_end_keV`` and
    emission point is the row endpoint. The event frequency was already
    sampled by transport: this scorer applies only the conditional
    ``DDCS/SDCS`` angle density and planar-slab escape transmission. It must
    not multiply by number density, path length, or cross section again.

    Returns photons per incident electron per eV per sr at the energy-grid
    nodes. Only a single planar slab is supported in this first scorer.

    Validation: bremslib-radiative-event-spectrum
    """
    edges, widths, split = _cutoff_edge(E_grid_eV, cutoff_eV)
    _check_transport_partition(segments, cutoff_eV, bremslib_tables)
    for field in (
        "event_kind",
        "hard_radiative_k_eV",
        "hard_radiative_Z",
        "E_end_keV",
        "r_mid",
        "v_hat",
        "L_ang",
        "Ne",
        "thickness_ang",
    ):
        if field not in segments:
            raise ValueError(f"hard-radiative event scoring requires {field}")
    if segments.get("n_layers", 1) != 1:
        raise NotImplementedError("hard-radiative event scoring supports one planar slab")
    if (
        segments.get("crystal_width_ang") is not None
        or segments.get("crystal_height_ang") is not None
    ):
        raise NotImplementedError(
            "hard-radiative event scoring does not support a finite footprint"
        )
    comp = _normalize_composition(element, n_atoms_per_ang3, composition)
    direction = np.asarray(_observation_direction(theta_obs_rad, n_hat), dtype=float)
    if abs(direction[2]) < 1e-12:
        raise ValueError("observation direction must cross a slab face")
    kind = np.asarray(BACKEND.to_cpu(segments["event_kind"]))
    n_rows = kind.size
    k_all = np.asarray(BACKEND.to_cpu(segments["hard_radiative_k_eV"]), dtype=float)
    z_all = np.asarray(BACKEND.to_cpu(segments["hard_radiative_Z"]), dtype=int)
    event = (kind == EVENT_HARD_RADIATIVE) | ((kind == EVENT_CUTOFF) & (k_all > 0.0))
    if k_all.shape != (n_rows,) or z_all.shape != (n_rows,):
        raise ValueError("hard-radiative event fields must align with segment rows")
    if np.any(k_all[~event] != 0.0) or np.any(z_all[~event] != 0):
        raise ValueError("only hard-radiative rows may carry photon data")
    Ne = int(segments["Ne"] if electron_limit is None else electron_limit)
    if Ne <= 0:
        raise ValueError("incident electron count must be positive")
    event_indices = np.flatnonzero(event)
    if electron_limit is not None:
        if "electron_id" not in segments:
            raise ValueError("electron_limit requires electron_id")
        ids = np.asarray(BACKEND.to_cpu(segments["electron_id"]))
        event_indices = event_indices[ids[event_indices] < Ne]
    spectrum = np.zeros(widths.size, dtype=float)
    if not event_indices.size:
        return spectrum
    k = k_all[event_indices]
    Z = z_all[event_indices]
    energy = np.asarray(BACKEND.to_cpu(segments["E_end_keV"]), dtype=float)[event_indices] * 1e3
    v = np.asarray(BACKEND.to_cpu(segments["v_hat"]), dtype=float)[event_indices]
    r = np.asarray(BACKEND.to_cpu(segments["r_mid"]), dtype=float)[event_indices]
    length = np.asarray(BACKEND.to_cpu(segments["L_ang"]), dtype=float)[event_indices]
    if np.any(~np.isfinite(k)) or np.any(k < cutoff_eV) or np.any(k > energy) or np.any(Z <= 0):
        raise ValueError("hard-radiative photon energy, cutoff, or atomic number is invalid")
    endpoint_z = r[:, 2] + 0.5 * length * v[:, 2]
    thickness = float(segments["thickness_ang"])
    if np.any(endpoint_z < -1e-8) or np.any(endpoint_z > thickness + 1e-8):
        raise ValueError("hard-radiative event endpoint lies outside the slab")
    path = _escape_length(endpoint_z, thickness, direction[2])
    mu = np.asarray(BACKEND.to_cpu(_mu_total_inv_ang(comp, k)), dtype=float)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    transmission = np.exp(-path * mu)
    tables_by_Z = {table.atomic_number: table for table in bremslib_tables.values()}
    staged_by_Z = {
        atomic_number: stage_bremslib_table(table) for atomic_number, table in tables_by_Z.items()
    }
    weights = np.empty(k.size, dtype=float)
    for i, (atomic_number, T, photon_eV, flight) in enumerate(zip(Z, energy, k, v, strict=True)):
        staged = staged_by_Z.get(int(atomic_number))
        if staged is None:
            raise ValueError(f"no BremsLib table for hard-radiative Z={atomic_number}")
        if (
            not staged.table.minimum_incident_energy_keV
            <= T / 1e3
            <= staged.table.maximum_incident_energy_keV
        ):
            raise ValueError("hard-radiative event is outside the BremsLib table")
        isotropic = bremslib_segment_state(staged, [T / 1e3])
        directional = bremslib_segment_state(staged, [T / 1e3], [float(flight @ direction)])
        sdcs = float(BACKEND.to_cpu(evaluate_bremslib(staged, isotropic, [photon_eV]))[0, 0])
        ddcs = float(BACKEND.to_cpu(evaluate_bremslib(staged, directional, [photon_eV]))[0, 0])
        if not np.isfinite(sdcs) or sdcs <= 0.0 or not np.isfinite(ddcs) or ddcs < 0.0:
            raise ValueError("hard-radiative event has an invalid BremsLib conditional angle")
        weights[i] = ddcs / sdcs
    spectrum = np.histogram(k, bins=edges, weights=weights * transmission)[0] / (Ne * widths)
    spectrum[:split] = 0.0
    return spectrum
