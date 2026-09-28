"""Post-transport bremsstrahlung scoring of coupled radiative tracks.

Validation: bremslib-coupled-expected-spectrum, bremslib-radiative-event-spectrum
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
    evaluate_bremslib_rows,
    stage_bremslib_table,
)
from .lines import _observation_direction
from .segment_escape import _escape_paths_at


def _cutoff_edge(E_grid_eV, cutoff_eV):
    """Bin edges, widths and the bin containing the soft/hard split."""
    edges, widths = node_bin_edges_and_widths(E_grid_eV)
    cutoff = float(cutoff_eV)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("hard photon cutoff must be positive and finite")
    if cutoff <= edges[0]:
        return edges, widths, 0
    if cutoff >= edges[-1]:
        return edges, widths, widths.size
    return edges, widths, int(np.searchsorted(edges, cutoff, side="right") - 1)


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


def _uncoupled_view(segments, cutoff_eV, bremslib_tables, kwargs):
    """Coupled rows as the track-length scorer's input, after partition checks."""
    _check_transport_partition(segments, cutoff_eV, bremslib_tables)
    if "cross_section_model" in kwargs:
        raise ValueError("coupled track-length scoring always uses the BremsLib cross section")
    if kwargs.get("E_cut_keV") is not None:
        raise NotImplementedError(
            "coupled track-length scoring cannot reclip electron tracks at a new cutoff"
        )
    view = dict(segments)
    view.pop("radiative", None)
    return view


def mc_coupled_brem_spectrum(segments, E_grid_eV, *, cutoff_eV, bremslib_tables, **kwargs):
    """Expected-value continuum on coupled tracks, soft and hard photons alike.

    Scores the full BremsLib DDCS by track length along the coupled electron
    trajectories. Hard photons change those trajectories (each sampled event
    removes its energy), but their expected count per path length is the same
    ``n dσ/dk`` the track-length scorer integrates, so the estimate matches the
    soft-plus-event sum :func:`mc_soft_brem_spectrum` +
    :func:`mc_hard_brem_event_spectrum` in expectation without the sparse
    per-photon histogram above ``cutoff_eV``.

    Validation: bremslib-coupled-expected-spectrum
    """
    view = _uncoupled_view(segments, cutoff_eV, bremslib_tables, kwargs)
    return mc_brem_spectrum(
        view, E_grid_eV, cross_section_model="bremslib", bremslib_tables=bremslib_tables, **kwargs
    )


def mc_soft_brem_spectrum(segments, E_grid_eV, *, cutoff_eV, bremslib_tables, **kwargs):
    """Track-length bremsstrahlung below the hard-photon cutoff.

    ``mc_brem_spectrum`` remains the uncoupled compatibility estimator. This
    wrapper is for the soft side of coupled scoring; add its result to
    :func:`mc_hard_brem_event_spectrum` on the same energy grid. A cutoff
    inside a bin contributes its below-cutoff fraction of that bin's width;
    one below the grid leaves no soft bin to score.

    Validation: bremslib-radiative-event-spectrum
    """
    edges, widths, split = _cutoff_edge(E_grid_eV, cutoff_eV)
    uncoupled_view = _uncoupled_view(segments, cutoff_eV, bremslib_tables, kwargs)
    if cutoff_eV <= edges[0]:
        return np.zeros(np.asarray(E_grid_eV).size, dtype=float)
    full = mc_brem_spectrum(
        uncoupled_view,
        E_grid_eV,
        cross_section_model="bremslib",
        bremslib_tables=bremslib_tables,
        **kwargs,
    )
    soft = np.asarray(full).copy()
    soft[split:] = 0.0
    if split < soft.size and edges[split] < cutoff_eV:
        soft[split] = full[split] * (cutoff_eV - edges[split]) / widths[split]
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
    layers=None,
):
    """Bin sampled hard photons, weighting each by its BremsLib angular law.

    Each ``HARD_RADIATIVE`` row (or terminal ``CUTOFF`` row carrying a photon)
    supplies ``hard_radiative_k_eV`` and ``hard_radiative_Z``. Its pre-event
    electron energy is ``E_end_keV`` and
    emission point is the row endpoint. The event frequency was already
    sampled by transport: this scorer applies only the conditional
    ``DDCS/SDCS`` angle density and escape transmission from the emission
    point. It must not multiply by number density, path length, or cross
    section again.

    Escape follows the point-emitter geometry of the track-length scorers: the
    laterally infinite slab, a finite rectangular footprint
    (``crystal_width_ang``/``crystal_height_ang``), and, with ``layers``, the
    summed optical depth of every ``(z_top, z_bot, composition)`` layer the
    photon crosses. The emitting element is the event's own ``Z``, so a
    layered stack is scored in one call. Grooved faces are not supported.

    Returns photons per incident electron per eV per sr at the energy-grid
    nodes.

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
    if layers is None and segments.get("n_layers", 1) != 1:
        raise ValueError("layered hard-radiative rows require the absorbing layers")
    if (segments.get("crystal_width_ang") is None) != (segments.get("crystal_height_ang") is None):
        raise ValueError("a finite footprint needs both crystal_width_ang and crystal_height_ang")
    layer_comps = (
        [_normalize_composition(element, n_atoms_per_ang3, composition)]
        if layers is None
        else [layer_comp for _, _, layer_comp in layers]
    )
    direction = np.asarray(_observation_direction(theta_obs_rad, n_hat), dtype=float)
    if segments.get("crystal_width_ang") is None and abs(direction[2]) < 1e-12:
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
    endpoint = r + 0.5 * length[:, None] * v
    thickness = float(segments["thickness_ang"])
    if np.any(endpoint[:, 2] < -1e-8) or np.any(endpoint[:, 2] > thickness + 1e-8):
        raise ValueError("hard-radiative event endpoint lies outside the slab")
    # Point emitter: per-layer escape path from the event endpoint, as the
    # track scorers use at their segment ends. Validation: segment-escape-average
    path = np.asarray(
        BACKEND.to_cpu(_escape_paths_at(endpoint, direction, segments, layers, None, np)),
        dtype=float,
    )
    tau = np.zeros(k.size, dtype=float)
    for index, layer_comp in enumerate(layer_comps):
        mu = np.asarray(BACKEND.to_cpu(_mu_total_inv_ang(layer_comp, k)), dtype=float)
        tau += path[:, index] * np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    transmission = np.exp(-tau)
    tables_by_Z = {table.atomic_number: table for table in bremslib_tables.values()}
    # One vectorized pass per emitting element: each event is evaluated at
    # its own (T, k, cos theta), the diagonal of the grid evaluation.
    weights = np.empty(k.size, dtype=float)
    for atomic_number in np.unique(Z):
        table = tables_by_Z.get(int(atomic_number))
        if table is None:
            raise ValueError(f"no BremsLib table for hard-radiative Z={atomic_number}")
        rows = np.flatnonzero(Z == atomic_number)
        T_keV = energy[rows] / 1e3
        if np.any(T_keV < table.minimum_incident_energy_keV) or np.any(
            T_keV > table.maximum_incident_energy_keV
        ):
            raise ValueError("hard-radiative event is outside the BremsLib table")
        staged = stage_bremslib_table(table)
        isotropic = bremslib_segment_state(staged, T_keV)
        directional = bremslib_segment_state(staged, T_keV, v[rows] @ direction)
        sdcs = np.asarray(
            BACKEND.to_cpu(evaluate_bremslib_rows(staged, isotropic, k[rows])), dtype=float
        )
        ddcs = np.asarray(
            BACKEND.to_cpu(evaluate_bremslib_rows(staged, directional, k[rows])), dtype=float
        )
        if np.any(~np.isfinite(sdcs) | (sdcs <= 0.0) | ~np.isfinite(ddcs) | (ddcs < 0.0)):
            raise ValueError("hard-radiative event has an invalid BremsLib conditional angle")
        weights[rows] = ddcs / sdcs
    spectrum = np.histogram(k, bins=edges, weights=weights * transmission)[0] / (Ne * widths)
    spectrum[:split] = 0.0
    return spectrum
