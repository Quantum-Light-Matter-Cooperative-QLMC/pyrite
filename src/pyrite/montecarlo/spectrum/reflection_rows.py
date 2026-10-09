"""Per-(reflection, orientation) resonance rows over a case's line pieces.

The shared kinematics behind :mod:`.line_seeds` (incoherent resonance
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
    electron_limit: int | None = None,
    label_prefix: str = "",
    composition: Iterable[tuple[str, float]] | None = None,
    band_eV: tuple[float, float] | None = None,
    groove=None,
) -> ReflectionRows:
    """Resonance and in-medium denominator of every usable piece, per row.

    The shared kinematics behind :func:`resonance_populations` and the
    coherent window seeds; see :func:`kinematic_line_seeds` for the source
    equation and root solver.

    Validation: line-window-seeding
    """
    from .segment_escape import segment_escape_gradient

    # Seeding is host work even when transport returned device arrays.
    segments = {
        key: _host(value) if hasattr(value, "shape") else value for key, value in segments.items()
    }
    if segments.get("r_mid") is not None and segments.get("thickness_ang") is not None:
        from .lines._formation import expand_escape_pieces

        segments, _, _, _ = expand_escape_pieces(segments, n_hat, groove=groove)
    gradient = np.asarray(segment_escape_gradient(segments, n_hat, groove=groove), dtype=float)
    energy_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    energy = _host(segments[energy_field]).astype(float, copy=False)
    length = _host(segments["L_ang"]).astype(float, copy=False)
    direction = _host(segments["v_hat"]).astype(float, copy=False).reshape(-1, 3)
    index = np.arange(energy.size)
    if electron_limit is not None:
        line = _host(segments["elec_id"]) < int(electron_limit)
        energy, length, direction = energy[line], length[line], direction[line]
        gradient, index = gradient[line], index[line]
    beta = beta_from_keV(energy)
    velocity = beta[:, None] * direction
    v_dot_n = velocity @ np.asarray(n_hat, dtype=float)
    parent_length = np.asarray(segments.get("line_parent_L_ang", segments["L_ang"]), dtype=float)
    fraction = np.asarray(
        segments.get("line_piece_fraction", np.ones_like(parent_length)), dtype=float
    )
    if electron_limit is not None:
        parent_length, fraction = parent_length[line], fraction[line]
    weight = (parent_length / beta) ** 2 * fraction
    usable = np.isfinite(weight) & (weight > 0.0) & np.isfinite(v_dot_n) & (v_dot_n < 1.0)
    velocity, v_dot_n, weight = velocity[usable], v_dot_n[usable], weight[usable]
    gradient, index = gradient[usable], index[usable]
    v_dot_grad = np.sum(velocity * gradient, axis=1)
    grad2 = np.sum(gradient**2, axis=1)
    flight_time = parent_length[usable] / beta[usable]
    refractive = None
    if band_eV is not None:
        from ...materials.crystal import refractive_index
        from .lines._kernels import _line_tabulation_grid

        # The kernels' table for an axis spanning band_eV (lines/_setup.py).
        start, stop = float(band_eV[0]), float(band_eV[1])
        pad = 0.2 * (stop - start)
        table_energy = _line_tabulation_grid(
            CRYSTALS[crystal], list(composition or ()), max(start - pad, 1.0), stop + pad
        )
        refractive = (
            np.asarray(refractive_index(crystal, table_energy).real, dtype=float),
            table_energy,
        )
    if refractive is not None:
        from .lines._kernels import _in_medium_kinematics

    lattice = CRYSTALS[crystal]["lattice"]
    rotation = _orientation_R(
        lattice, beam_uvw, azimuth_rad, recip_miscut_rad, surface_hkl=surface_hkl
    )
    orientations = _mosaic_quadrature(mosaic_fwhm_rad, mosaic_nodes) or [(None, 1.0)]

    reflections = []
    for hkl in hkl_list:
        g_vector, _magnitude = reciprocal_g_vector(hkl, lattice)
        if rotation is not None:
            g_vector = rotation @ g_vector
        rows = []
        for mosaic_rotation, mosaic_weight in orientations:
            g_row = g_vector if mosaic_rotation is None else mosaic_rotation @ g_vector
            v_dot_g = velocity @ g_row
            with np.errstate(divide="ignore", invalid="ignore"):
                if refractive is None:
                    denominator = 1.0 - v_dot_n
                else:
                    denominator, _ = _in_medium_kinematics(
                        v_dot_n, v_dot_g, *refractive, v_dot_grad, grad2
                    )
                resonance = HBARC_EV_ANG * v_dot_g / denominator
            radiating = np.isfinite(resonance) & (resonance > MIN_RESONANCE_EV)
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
