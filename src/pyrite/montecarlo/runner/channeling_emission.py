"""Persisted ensemble CR source on the native line grid (ADR-0018 P5).

The source uses the diagnostic constant-energy density solve and the same
profile, kernel, cutoff margin, incidence and resolved divergence as the
transport zone. It is an expected-value source; sampled dechanneling fates
never weight it again. The geometric mean-field approximation averages the
entrance paths' face/cutoff limits independently of the ensemble density.
Photon attenuation retains the diagnostic laterally infinite slab model;
finite side-face photon escape is not represented.

Validation: cr-persisted-channel
"""

from __future__ import annotations

import numpy as np

from ..channeling.emission import channeling_spectrum
from ..channeling.handoff import case_plane_incidence, zone_solver
from ..geometry import beam_frame_basis
from .channeling import _exit_lengths


def source_frame(beam_dir, plane_normal) -> np.ndarray:
    """Right-handed frame: z along beam, x projected toward plane normal.

    If the beam is normal to the planes, use an arbitrary transverse x;
    the entry state is then unchanneled. All vectors are in the slab frame.

    Validation: cr-persisted-channel
    """
    beam = np.asarray(beam_dir, dtype=float)
    beam = beam / np.linalg.norm(beam)
    normal = np.asarray(plane_normal, dtype=float)
    x = normal - float(normal @ beam) * beam
    norm = np.linalg.norm(x)
    x = x / norm if norm > 1e-12 else beam_frame_basis(beam)[:, 0]
    return np.column_stack((x, np.cross(beam, x), beam))


def case_channeling_spectrum(case, transport) -> np.ndarray:
    """Score CR in photons/eV/sr/incident electron on ``transport['E_grid']``.

    Reuse the P3 constant-energy approximation. The transport handoff uses
    continuous stopping; emission omits its small gamma drift. Finite faces,
    misses and the line-scoring energy cutoff censor the expected source.
    They censor electron paths only; photon absorption assumes an infinite
    lateral slab, so side-escaping photon geometries are outside this model.

    Validation: cr-persisted-channel
    """
    record = case["channeling"]
    segs = transport["segs"]
    info = segs["channeling"]
    count = int(transport["Ne_lines"])
    r = np.asarray(segs["initial_r_ang"])[:count]
    v = np.asarray(segs["initial_v_hat"])[:count]
    energy = np.asarray(segs["initial_E_keV"])[:count]
    _profile, solver = zone_solver(
        case["crystal"],
        tuple(record["plane_hkl"]),
        float(record["temperature_K"]),
        float(case["E0_keV"]) * 1e3,
        int(record["basis_size"]),
        int(record["bloch_cells"]),
        float(record["margin_eV"]),
    )
    frame = source_frame(transport["channeling_beam_dir"], case_plane_incidence(case).plane_normal)
    direction = np.asarray(transport["n_hat"]) @ frame
    theta = float(np.arctan2(np.hypot(direction[0], direction[1]), direction[2]))
    phi = float(np.arctan2(direction[1], direction[0]))
    width = case.get("crystal_width_mm")
    height = case.get("crystal_height_mm")
    width = None if width is None else float(width) * 1e7
    height = None if height is None else float(height) * 1e7
    limits, _ = _exit_lengths(r, v, float(case["thickness_ang"]), width, height)
    if width is not None and height is not None:
        missed = (np.abs(r[:, 0]) > width / 2) | (np.abs(r[:, 1]) > height / 2)
        limits[missed] = 0.0
    stopping = float(info["stopping_keV_per_ang"])
    if stopping > 0:
        limits = np.minimum(
            limits, np.maximum(energy - case.get("E_cut_lines_keV", 5.0), 0) / stopping
        )
    # In-plane divergence changes the observed Doppler angle; the normal
    # divergence already participates in the density-matrix entry state.
    in_plane = np.arctan2(v @ frame[:, 1], v @ frame[:, 2])
    result = channeling_spectrum(
        solver,
        float(info["path_ang"]),
        np.asarray(transport["E_grid"]),
        theta,
        phi,
        incidence_rad=float(info["incidence_rad"]),
        divergence_rad=float(info["divergence_rad"]),
        divergence_y_rad=float(np.std(in_plane)),
        energy_spread_eV=float(case.get("energy_spread_frac", 0.0) or 0.0)
        * float(case["E0_keV"])
        * 1e3,
        aperture_rad=float(case.get("dtheta_obs_rad", 0.0)),
        dz_ang=float(record["dz_ang"]),
        sample_ang=float(record["dz_ang"]),
        layers=[(0.0, float(case["thickness_ang"]), case["composition"])],
        surface_normal=frame[2],
        path_limits_ang=limits,
    )
    return result.spectrum
