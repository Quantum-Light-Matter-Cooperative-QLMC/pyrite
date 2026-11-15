"""Per-(reflection, orientation) resonance rows over a case's line pieces.

The shared kinematics behind :mod:`.cxr_seeds` (incoherent resonance
populations and kinematic windows) and :mod:`.coherent_windows` (coherent
windows, #350): one resonance and in-medium denominator per usable piece and
row, read exactly as the line kernels read them.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...materials.crystal import CRYSTALS, HBARC_EV_ANG, reciprocal_g_vector
from ..geometry import _mosaic_quadrature, _orientation_R
from ..transport import beta_from_keV

__all__ = ["MIN_RESONANCE_EV", "ReflectionRows", "reflection_rows"]

#: Kinematic resonances below this are dropped by the line kernels too.
MIN_RESONANCE_EV = 10.0


def _host(array):
    get = getattr(array, "get", None)
    return np.asarray(get() if get is not None else array)


@dataclass(frozen=True, slots=True)
class ReflectionRows:
    """Per-(reflection, orientation) resonances over one set of line pieces.

    ``pieces`` are the host line pieces (escape pieces when the case has a slab
    geometry) and ``index`` selects the usable ones, which every per-row array
    is aligned with. ``reflections`` holds ``(label, orientations)`` with one
    ``(mosaic_weight, radiating, resonance, denominator)`` entry per mosaic
    orientation; ``radiating`` masks the rows the line kernels keep.
    """

    pieces: Mapping[str, Any]
    index: np.ndarray
    beta: np.ndarray
    flight_time: np.ndarray
    weight: np.ndarray
    reflections: list


def reflection_rows(
    segments,
    n_hat,
    *,
    crystal: str,
    hkl_list: Iterable[Sequence[int]],
    beam_uvw=None,
    surface_hkl=None,
    azimuth_rad: float = 0.0,
    recip_miscut_rad=None,
    mosaic_fwhm_rad=None,
    mosaic_nodes: int = 1,
    mosaic_sampling: str = "quadrature",
    mosaic_seed: int | None = None,
    electron_limit: int | None = None,
    label_prefix: str = "",
    composition: Iterable[tuple[str, float]] | None = None,
    band_eV: tuple[float, float] | None = None,
    groove=None,
) -> ReflectionRows:
    """Resonance and in-medium denominator of every usable piece, per row.

    The shared kinematics behind :func:`resonance_populations` and the
    coherent window seeds; see :func:`kinematic_cxr_seeds` for the source
    equation and root solver.

    Under ``mosaic_sampling="history"`` each orientation entry is one draw
    of every piece's own history crystallite (weight ``1/mosaic_nodes``), the
    rows the per-history sampled route scores. Validation: mosaic-mc-sampled

    Validation: line-window-seeding, line-kinematics-precision
    """
    from .cxr._kernels import (
        _vacuum_doppler,
    )
    from .cxr._recoil import _line_recoil_terms, _recoil_line_allowed, _recoil_slope_factor
    from .segment_escape import segment_escape_gradient

    # Seeding is host work even when transport returned device arrays.
    segments = {
        key: _host(value) if hasattr(value, "shape") else value for key, value in segments.items()
    }
    if segments.get("r_mid") is not None and segments.get("thickness_ang") is not None:
        from .cxr._formation import expand_escape_pieces

        segments, _, _, _ = expand_escape_pieces(segments, n_hat, groove=groove)
    gradient = np.asarray(segment_escape_gradient(segments, n_hat, groove=groove), dtype=float)
    energy_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    energy = _host(segments[energy_field]).astype(float, copy=False)
    length = _host(segments["L_ang"]).astype(float, copy=False)
    direction = _host(segments["v_hat"]).astype(float, copy=False).reshape(-1, 3)
    index = np.arange(energy.size)
    history = mosaic_sampling == "history" and bool(mosaic_fwhm_rad)
    owner = _host(segments["elec_id"]).astype(np.int64, copy=False) if history else index
    if electron_limit is not None:
        line = _host(segments["elec_id"]) < int(electron_limit)
        energy, length, direction = energy[line], length[line], direction[line]
        gradient, index = gradient[line], index[line]
    beta = beta_from_keV(energy)
    velocity = beta[:, None] * direction
    gamma, vacuum = _vacuum_doppler(energy, beta, direction, n_hat, xp=np)
    parent_length = np.asarray(segments.get("line_parent_L_ang", segments["L_ang"]), dtype=float)
    fraction = np.asarray(
        segments.get("line_piece_fraction", np.ones_like(parent_length)), dtype=float
    )
    if electron_limit is not None:
        parent_length, fraction = parent_length[line], fraction[line]
    weight = (parent_length / beta) ** 2 * fraction
    usable = np.isfinite(weight) & (weight > 0.0) & np.isfinite(vacuum) & (vacuum > 0.0)
    velocity, vacuum, weight = velocity[usable], vacuum[usable], weight[usable]
    gamma = gamma[usable]
    n_vector = np.asarray(n_hat, dtype=float)
    gradient, index = gradient[usable], index[usable]
    owner = owner[index]
    v_dot_grad = np.sum(velocity * gradient, axis=1)
    grad2 = np.sum(gradient**2, axis=1)
    flight_time = parent_length[usable] / beta[usable]
    refractive = None
    if band_eV is not None:
        from ...materials.crystal import refractive_index
        from .cxr._kernels import _cxr_tabulation_grid

        # The kernels' table for an axis spanning band_eV (lines/_setup.py).
        start, stop = float(band_eV[0]), float(band_eV[1])
        pad = 0.2 * (stop - start)
        table_energy = _cxr_tabulation_grid(
            CRYSTALS[crystal], list(composition or ()), max(start - pad, 1.0), stop + pad
        )
        refractive = (
            1.0 - np.asarray(refractive_index(crystal, table_energy).real, dtype=float),
            table_energy,
        )
    if refractive is not None:
        from .cxr._kernels import _in_medium_kinematics

    lattice = CRYSTALS[crystal]["lattice"]
    rotation = _orientation_R(
        lattice, beam_uvw, azimuth_rad, recip_miscut_rad, surface_hkl=surface_hkl
    )
    sampled, orientations = _seed_orientations(
        history, owner, mosaic_seed, mosaic_fwhm_rad, mosaic_nodes
    )

    reflections = []
    for hkl in hkl_list:
        g_vector, _magnitude = reciprocal_g_vector(hkl, lattice)
        if rotation is not None:
            g_vector = rotation @ g_vector
        rows = []
        for mosaic_rotation, mosaic_weight in orientations:
            v_dot_g, n_dot_g, g2 = _orientation_terms(
                sampled, mosaic_rotation, g_vector, velocity, n_vector
            )
            # Exact recoil root and slope, as the kernels place each line.
            # Validation: line-recoil-in-medium-root
            r_num, r_den, inv_E = _line_recoil_terms(gamma, g2, n_dot_g)
            numerator = v_dot_g - r_num
            with np.errstate(divide="ignore", invalid="ignore"):
                if refractive is None:
                    denominator = vacuum + r_den
                else:
                    denominator, _ = _in_medium_kinematics(
                        vacuum + r_den, numerator, *refractive, v_dot_grad, grad2
                    )
                omega = numerator / denominator
                resonance = HBARC_EV_ANG * omega
                allowed = _recoil_line_allowed(denominator, omega, inv_E, gamma)
                denominator = denominator * _recoil_slope_factor(omega, inv_E)
            radiating = np.isfinite(resonance) & (resonance > MIN_RESONANCE_EV) & allowed
            rows.append((float(mosaic_weight), radiating, resonance, denominator))
        label = label_prefix + "(" + " ".join(str(int(h)) for h in hkl) + ")"
        reflections.append((label, rows))
    return ReflectionRows(
        pieces=segments,
        index=index,
        beta=beta[usable],
        flight_time=flight_time,
        weight=weight,
        reflections=reflections,
    )


def _seed_orientations(history, owner, seed, fwhm_rad, nodes):
    """``(sampled, [(orientation, weight)])``: GH rotations, or draw indices.

    Validation: mosaic-mc, mosaic-mc-sampled
    """
    if not history:
        return None, _mosaic_quadrature(fwhm_rad, nodes) or [(None, 1.0)]
    from .cxr._mosaic_sampled import sampled_mosaic

    sampled = sampled_mosaic(owner, seed, fwhm_rad, nodes)
    return sampled, [(draw, 1.0 / sampled.draws) for draw in range(sampled.draws)]


def _orientation_terms(sampled, orientation, g_vector, velocity, n_vector):
    """``(v.g, n.g, g.g)`` of one orientation entry over the usable pieces.

    Sampled entries rotate ``g`` by each piece's own history crystallite;
    ``|g|`` is rotation-invariant. Validation: mosaic-mc-sampled
    """
    if sampled is not None:
        g_row = sampled.orientation_rows(g_vector, n_vector, orientation)[0]
        g_row = g_row[sampled.row_history]
        return np.sum(velocity * g_row, axis=1), g_row @ n_vector, float(g_vector @ g_vector)
    g_row = g_vector if orientation is None else orientation @ g_vector
    return velocity @ g_row, float(n_vector @ g_row), float(g_row @ g_row)
