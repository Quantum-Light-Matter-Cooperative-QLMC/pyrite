"""Restricted, conditional planar quantum entrance and native handoff (#680).

The admitted source is a *one-dimensional Gaussian momentum mixture*, after
uniform averaging of its impact parameter.  Draw a source momentum from a
counter-addressed stream, condition the Lindblad zone on its Bloch momentum
bin, then draw dechanneling fates from that conditional zone.  An above-band
primary bypasses the zone.  This deliberately does not unravel arbitrary
coherent wavefunctions into classical position-momentum launch points.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from ...quantum_state import TransverseStateSpec

SOURCE_MODEL = "planar-gaussian-averaged-conditional-v1"
MAX_MOMENTUM_BINS = 32


def source_spec(record: object) -> TransverseStateSpec:
    """Validate a versioned physical-source record, including raw Case inputs."""
    if not isinstance(record, dict) or set(record) != {"model", "impact", "state"}:
        raise ValueError("channeling_source must contain model, impact, and state")
    if record["model"] != SOURCE_MODEL or record["impact"] != "averaged":
        raise ValueError("unsupported channeling_source model or impact averaging")
    spec = TransverseStateSpec.from_dict(record["state"])
    if spec.kind != "gaussian" or spec.dimension != 1:
        raise ValueError("production quantum source requires a one-dimensional Gaussian")
    params = dict(spec.parameters)
    if tuple(params["center_ang"]) != (0.0,):
        raise ValueError("production Gaussian must be centered; impact is uniformly averaged")
    return spec


def validate_source_case(case) -> None:
    """Reject incompatible direct Case inputs before work."""
    if case.get("channeling") is None:
        raise ValueError("channeling_source requires a planar channeling case")
    source_spec(case["channeling_source"])
    if case.get("transverse_distribution") is not None:
        raise ValueError("quantum entrance excludes correlated classical Twiss slopes")
    if case.get("gdf_source") is not None:
        raise ValueError("quantum entrance excludes GDF sources")
    if case.get("energy_spread_frac") not in (None, 0.0):
        raise ValueError("quantum entrance needs a monoenergetic primary beam")


def source_record(spec: TransverseStateSpec) -> dict[str, Any]:
    """Produce the canonical, distinct Case identity for the supported source."""
    if not isinstance(spec, TransverseStateSpec):
        raise TypeError("source must be a TransverseStateSpec")
    record = {"model": SOURCE_MODEL, "impact": "averaged", "state": spec.to_dict()}
    source_spec(record)
    return record


def fresh_source_solver(case):
    """Construct an entrance-energy solver using cached *static* kernel data.

    A shared zone solver may end propagation in a reprojected, lower-energy
    basis. Do not ever form a new entrance density in that mutable basis.
    """
    from .density import DensityMatrixSolver
    from .handoff import zone_solver

    record = case["channeling"]
    profile, cached = zone_solver(
        case["crystal"], tuple(record["plane_hkl"]), float(record["temperature_K"]),
        float(case["E0_keV"]) * 1e3, int(record["basis_size"]),
        int(record["bloch_cells"]), float(record["margin_eV"]),
    )
    return DensityMatrixSolver(
        profile, float(case["E0_keV"]) * 1e3, cached.kernel,
        margin_eV=float(record["margin_eV"]),
    )


def source_entry_initial(case, solver, incidence_rad: float):
    """Projected *unconditional* density for the zone and expected-value CR."""
    from .states import planar_entry_density

    spec = source_spec(case["channeling_source"])
    plane, outside = planar_entry_density(
        solver, spec.to_runtime(), incidence_rad=incidence_rad, impact="averaged"
    )
    rho, free, out = solver.entry_from_plane(plane, outside)
    return rho, float(np.sum(free) + out)


def source_momentum_bins(case, v_hat: np.ndarray, seed: int):
    """Pure, counter-addressed Gaussian momentum draws and launch directions.

    The independent child stream leaves the historical scattering and bunch
    keys alone.  Position remains an independently sampled macroscopic spot;
    uniform impact averaging intentionally removes the crystalline cell phase.
    """
    from ..transport.kinematics import child_stream_root, counter_normals
    from .handoff import case_plane_incidence, zone_solver
    from .states import momentum_inv_ang

    spec = source_spec(case["channeling_source"])
    waist = float(dict(spec.parameters)["waist_ang"])
    record = case["channeling"]
    incidence = case_plane_incidence(case)
    normal = np.asarray(incidence.plane_normal, dtype=float)
    _, solver = zone_solver(
        case["crystal"], tuple(record["plane_hkl"]), float(record["temperature_K"]),
        float(case["E0_keV"]) * 1e3, int(record["basis_size"]),
        int(record["bloch_cells"]), float(record["margin_eV"]),
    )
    step = solver.profile.g_inv_ang / solver.n_cells
    # Prevent an unexpectedly broad source from multiplying full zone solves.
    if math.ceil(12.0 / (waist * step)) + 3 > MAX_MOMENTUM_BINS:
        raise ValueError("Gaussian momentum span needs over 32 Bloch bins; enlarge waist")
    p = momentum_inv_ang(float(case["E0_keV"]) * 1e3)
    base = np.asarray(v_hat, dtype=float)
    cospsi = base @ normal
    if np.any(np.abs(cospsi) > 0.5):
        raise ValueError("planar source requires incidence less than 30 degrees to the planes")
    root = child_stream_root(seed, 7, 6)
    delta_k = counter_normals(root, base.shape[0], 1)[:, 0] / waist
    tilt = np.arcsin(np.clip(cospsi, -1.0, 1.0)) + delta_k / p
    tangent = base - cospsi[:, None] * normal[None, :]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    new_v = np.cos(tilt)[:, None] * tangent + np.sin(tilt)[:, None] * normal[None, :]
    if np.any(new_v[:, 2] <= 0.0):
        raise ValueError("sampled quantum entrance would point out of the target")
    # The grid uses l = r + n_cells*nu; its signed bin centre is l*g/n_cells.
    bins = np.floor((p * float(incidence.effective_rad) + delta_k) / step + 0.5).astype(int)
    return new_v, bins


def _bin_ensemble(case, overall, bin_index: int):
    """Unit-weight Bloch-bin entry; retains correlations between k and fate."""
    record = case["channeling"]
    solver = fresh_source_solver(case)
    residue = bin_index % solver.n_cells
    order = (bin_index - residue) // solver.n_cells
    if abs(order) > solver.max_order:
        return None  # off-band momentum is a native electron at entry
    plane = np.zeros((solver.n_cells, solver.orders.size, solver.orders.size), dtype=complex)
    plane[residue, order + solver.max_order, order + solver.max_order] = 1.0
    rho, free, outside = solver.entry_from_plane(plane)
    result = solver.propagate(
        overall.path_ang, float(record["dz_ang"]),
        initial=(rho, float(np.sum(free) + outside)),
        stopping_eV_per_ang=overall.stopping_keV_per_ang * 1e3,
    )
    return replace(
        overall, z_ang=np.asarray(result.z_ang),
        free_cumulative=np.asarray(result.free_flux + result.reprojection_flux),
        overflow_cumulative=np.asarray(result.overflow_flux),
        entry=float(result.entry_flux), bound=float(result.trace[-1]),
        balance_defect=float(result.balance_defect[-1]),
        energy_edges_eV=np.asarray(result.energy_edges_eV),
        free_by_energy=np.asarray(result.free_flux_by_energy),
        divergence_rad=0.0,
    )


def conditional_handoffs(case, overall, seed: int, electron_ids, bins):
    """Fate distribution conditioned on each sampled Bloch momentum bin.

    Unique occupied bins are solved once per case. For outside-band electrons
    the fate is ENTRY, *not* a fictitious FFT dechanneling event.
    """
    from .handoff import FATE_ENTRY, HandoffSample, sample_handoffs

    ids = np.asarray(electron_ids, dtype=np.int64)
    bins = np.asarray(bins, dtype=int)
    if ids.shape != bins.shape or ids.ndim != 1:
        raise ValueError("electron indices and Bloch bins must be matching vectors")
    if np.unique(bins).size > MAX_MOMENTUM_BINS:
        raise ValueError("source occupies too many Bloch bins for conditional propagation")
    fate = np.full(ids.size, FATE_ENTRY, dtype=np.int8)
    depth = np.zeros(ids.size)
    energy = np.zeros(ids.size)
    psi = np.zeros(ids.size)
    sign = np.ones(ids.size)
    for key in np.unique(bins):
        group = np.flatnonzero(bins == key)
        zone = _bin_ensemble(case, overall, int(key))
        if zone is None:
            continue
        sampled = sample_handoffs(zone, seed, ids[group])
        for out, values in ((fate, sampled.fate), (depth, sampled.z_ang),
                            (energy, sampled.transverse_eV), (psi, sampled.psi_rad),
                            (sign, sampled.sign)):
            out[group] = values
    return HandoffSample(fate=fate, z_ang=depth, transverse_eV=energy, psi_rad=psi, sign=sign)
