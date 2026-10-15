"""Laboratory entry geometry for planar channeling.

Maps a sample tilt and the lattice orientation of a cut crystal (issue #25)
onto the plane-local incidence angle consumed by
:func:`~pyrite.materials.channeling.transverse.entry_populations`, including
refraction at the entrance face by the mean inner potential.

Frames follow the transport convention: the sample ``+z`` axis is the
entrance-face normal pointing into the crystal, and a lattice rotation ``R``
maps crystal reciprocal vectors into the sample frame (``ResolvedCut.rotation``
or :func:`pyrite.materials.crystal._orientation_R`). The beam travels along lab
``+z``; tilting the face normal to ``(sin t cos a, sin t sin a, cos t)`` in the
lab puts the beam along ``R_t^T e_z`` in the sample frame, as in
:func:`pyrite.montecarlo.geometry.tilted_geometry`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..atomic import Z_TABLE, cromer_mann_f0
from ..crystal import (
    CRYSTALS,
    E2_EV_ANG,
    HBARC_EV_ANG,
    M_E_EV,
    _orientation_R,
    _rotation_between,
    reciprocal_g_vector,
)

# Small-g nodes for the slope of f0 at g = 0 [1/angstrom]; s = g/(4 pi) <= 1e-2.
_SLOPE_G = (0.05, 0.1)


def mean_inner_potential_eV(crystal: str) -> float:
    """Return the electron's mean potential energy ``U_0`` in the crystal [eV].

    The ``g = 0`` limit of the planar Fourier sum: with
    ``f0(g) = Z - Z <r^2> g^2 / 6 + O(g^4)`` (Ibers 1958),
    ``(Z - f0) / g^2 -> Z <r^2> / 6``, so
    ``U_0 = -(4 pi e^2 / V) sum_j Z_j <r^2>_j / 6``. It is negative: an
    electron gains kinetic energy ``-U_0`` on entering. This is the neutral,
    independent-atom value; bonding lowers measured values by several percent.
    The slope is taken by a two-node difference in ``g^2`` near zero, whose
    error is ``O(g^2 <r^4>)``, below 1e-4 relative here.

    Validation: cr-mean-inner-potential
    """
    info = CRYSTALS[crystal]
    g1, g2 = _SLOPE_G
    total = 0.0
    for element, _ in info["basis"]:
        f1, f2 = cromer_mann_f0(element, np.array([g1, g2]))
        slope = (f1 - f2) / (g2**2 - g1**2)  # Z <r^2> / 6
        if not 0.0 < slope < Z_TABLE[element] * 10.0:
            raise ValueError(f"unphysical form-factor slope for {element}")
        total += slope
    return float(-4.0 * np.pi * E2_EV_ANG * total / info["V_cell"])


@dataclass(frozen=True)
class PlaneIncidence:
    """Beam incidence on a channeling plane family, in the sample frame.

    Attributes
    ----------
    beam_dir
        Unit vacuum beam direction in the sample frame.
    plane_normal
        Unit reciprocal normal of the channeling planes in the sample frame;
        it fixes the sign of the plane-local ``x`` axis.
    geometric_rad
        Vacuum incidence angle ``asin(beam_dir . plane_normal)`` [rad].
    internal_rad
        Incidence angle of the refracted electron inside the crystal [rad].
    effective_rad
        Angle to pass to ``entry_populations``: ``p_in . plane_normal / p``,
        with ``p`` the vacuum momentum, so that ``k_x = p psi_eff / hbar`` is
        the refracted transverse wavevector.
    mean_inner_potential_eV
        ``U_0`` used for refraction (0 when refraction is disabled).
    """

    beam_dir: np.ndarray
    plane_normal: np.ndarray
    geometric_rad: float
    internal_rad: float
    effective_rad: float
    mean_inner_potential_eV: float


def _momentum_inv_ang(kinetic_energy_eV: float) -> float:
    return float(np.sqrt(kinetic_energy_eV * (kinetic_energy_eV + 2.0 * M_E_EV)) / HBARC_EV_ANG)


def plane_incidence(
    crystal: str,
    plane_hkl,
    kinetic_energy_eV: float,
    *,
    tilt_deg: float = 0.0,
    tilt_azim_deg: float = 0.0,
    rotation: np.ndarray | None = None,
    refraction: bool = True,
) -> PlaneIncidence:
    """Return the incidence of a tilted beam on a channeling plane family.

    The channeling plane normal is ``n = R g_hkl / |g_hkl|``. The vacuum beam
    direction is the tilt convention of ``tilted_geometry``. At the entrance
    face (normal ``+z``) the electron gains kinetic energy ``-U_0``; the
    tangential momentum is conserved and the normal component grows to keep
    the total momentum consistent with ``T - U_0``:
    ``p_in = p_par + sqrt(p_in^2 - p_par^2) z``. Its projection on ``n``
    is the transverse wavevector of the channeling problem. For planes
    perpendicular to the face (``n . z = 0``) refraction changes nothing.

    Parameters
    ----------
    crystal
        Catalog crystal key.
    plane_hkl
        Miller indices of the channeling plane family.
    kinetic_energy_eV
        Electron kinetic energy in vacuum [eV].
    tilt_deg, tilt_azim_deg
        Sample tilt of the face normal from the beam and its azimuth, as for
        ``Target``/``Scene`` (degrees).
    rotation
        Lattice rotation into the sample frame, e.g.
        ``resolve_crystal_cut(cut, uvw, hkl).rotation(lattice)``. ``None``
        uses the catalog crystal's own face orientation.
    refraction
        Apply mean-inner-potential refraction at the entrance face.

    Returns
    -------
    PlaneIncidence
        Vacuum, internal and effective incidence angles plus the frame vectors.

    Raises
    ------
    ValueError
        If the beam does not enter through the face (``beam_dir . z <= 0``).

    Validation: cr-plane-incidence
    """
    if rotation is None:
        from ..catalog import _get_default_catalog

        spec = _get_default_catalog().crystals[crystal]
        rotation = _orientation_R(spec.lattice, spec.beam_uvw, 0.0, surface_hkl=spec.surface_hkl)
    lattice = CRYSTALS[crystal]["lattice"]
    g_vec, g_mag = reciprocal_g_vector(plane_hkl, lattice)
    normal = g_vec / g_mag
    if rotation is not None:
        normal = np.asarray(rotation, dtype=float) @ normal
    t, a = np.deg2rad(tilt_deg), np.deg2rad(tilt_azim_deg)
    face_lab = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])
    beam = _rotation_between(np.array([0.0, 0.0, 1.0]), face_lab).T @ np.array([0.0, 0.0, 1.0])
    if beam[2] <= 0.0:
        raise ValueError("beam does not enter through the entrance face")

    u0 = mean_inner_potential_eV(crystal) if refraction else 0.0
    p_out = _momentum_inv_ang(kinetic_energy_eV)
    p_in = _momentum_inv_ang(kinetic_energy_eV - u0)
    p_par = p_out * beam[:2]
    p_vec = np.array([p_par[0], p_par[1], np.sqrt(p_in**2 - p_par @ p_par)])
    k_x = float(p_vec @ normal)
    return PlaneIncidence(
        beam_dir=beam,
        plane_normal=normal,
        geometric_rad=float(np.arcsin(np.clip(beam @ normal, -1.0, 1.0))),
        internal_rad=float(np.arcsin(np.clip(k_x / p_in, -1.0, 1.0))),
        effective_rad=k_x / p_out,
        mean_inner_potential_eV=u0,
    )
