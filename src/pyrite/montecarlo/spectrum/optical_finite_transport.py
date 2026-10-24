"""Opt-in entrance-plane transport handoff to finite PEC-screen TR/DR (#616).

This is a *source-only diagnostic*, not a coupled radiation/transport operator.
The transported primary electron's entry position, velocity, kinetic energy,
charge and arrival clock are used to drive an **independently specified** ideal
PEC disc/aperture at the entrance plane. The PEC is NOT inferred from the
transport composition (and is not added to transport geometry). In particular,
the model neither propagates electrons through the screen nor adds recoil,
multiple scattering or a material Fresnel factor. An optical or X-ray TR source
on the same boundary must NOT be added incoherently to this idealized result.

Only primary electrons entering at z=0 (slab +z), with incidence within
the caller-supplied angular bound, and backward radiation into upstream
vacuum (-z), are supported. Local PEC coordinates are obtained by a
proper pi rotation about x: (x, y, z) -> (x, -y, -z). Each primary history
must contribute *exactly one* entrance crossing. No secondary/side/back-face
crossing may be substituted to fill a missing primary history.

The transport clock is in Angstroms of light travel; screen arrival times are
seconds, measured against an explicit time origin in the same transport clock.
The source's existing physical-bunch and importance-sampling law remains the
sole authority for intensity weights (never use macro-charge as charge_c).

Validation: optical-finite-boundary-field
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .optical_finite_screen import _E_CHARGE_C, PECCircularScreen, backward_formation_length_ang

if TYPE_CHECKING:
    from .transition_radiation_crossings import BoundaryCrossings

__all__ = ["entrance_pec_screens_from_crossings"]

_ANG_TO_M = 1e-10
_C_M_S = 299_792_458.0
_M_E_KEV = 510.99895069


def entrance_pec_screens_from_crossings(
    crossings: BoundaryCrossings,
    *,
    n_histories: int,
    outer_radius_m: float,
    inner_radius_m: float = 0.0,
    centre_xy_ang: tuple[float, float] = (0.0, 0.0),
    time_origin_ang: float = 0.0,
    max_angle_rad: float = 1e-3,
    min_straight_after_ang: float | None = None,
    formation_energy_eV: float | None = None,
    formation_theta_max_rad: float | None = None,
    min_formation_lengths: float | None = None,
) -> tuple[PECCircularScreen, ...]:
    """Map actual primary entry states to oblique PEC diagnostic sources.

    The returned tuple is ordered by ``electron_id = 0, ..., M-1`` so
    ``OpticalBunchSource.history_weights`` retains exactly its intended index.
    The only modeled surface is the separate, idealized z=0 PEC screen;
    ``outer_radius_m``/``inner_radius_m`` describe its physical metal extent.
    ``centre_xy_ang`` is the metal centre in *transport slab coordinates*.
    The reference time is an **absolute transport clock** [Angstrom/c]; using
    the same origin for different fields is essential to coherent comparison.

    The interface must be the initial vacuum -> layer 0 passage, not a later
    backscatter, reverse recrossing, internal layer or secondary. Straight
    ``max_angle_rad`` bounds the incidence angle, with normal incidence as
    the conservative default. The exact uniformly moving-charge oblique
    Fourier field is used inside this bound; no paraxial beta substitution.

    ``min_straight_after_ang`` optionally checks a fixed post-entry straight
    path. ``min_formation_lengths`` checks that path against a multiple of
    the conservative kinematic backward formation scale over photon energies
    >= ``formation_energy_eV`` and the collection cone <=
    ``formation_theta_max_rad``. These optional checks are *diagnostic* and
    not a derivation of finite-trajectory radiation or a guarantee of PEC
    transmission without electron deflection. Transport before entry is
    presumed straight in upstream vacuum. No screen scattering or feedback.

    Validation: optical-finite-boundary-field
    """
    center = _check_handoff_options(
        n_histories=n_histories,
        centre_xy_ang=centre_xy_ang,
        time_origin_ang=time_origin_ang,
        max_angle_rad=max_angle_rad,
        min_straight_after_ang=min_straight_after_ang,
        formation_energy_eV=formation_energy_eV,
        formation_theta_max_rad=formation_theta_max_rad,
        min_formation_lengths=min_formation_lengths,
    )
    indices = _entrance_indices(crossings, n_histories)
    p, v, e, t = _entry_state(crossings, indices, n_histories, max_angle_rad)
    if min_straight_after_ang is not None or min_formation_lengths is not None:
        run = _straight_run(crossings, indices, min_straight_after_ang)
    # Integer electron IDs are already sorted. Local coordinates are the
    # proper 180-degree x-rotation that sends the incident +z into -z.
    offset_m = (p[:, :2] - center) * _ANG_TO_M
    gamma = 1.0 + e / _M_E_KEV
    beta = np.sqrt(1.0 - gamma**-2)
    t_s = (t - time_origin_ang) * _ANG_TO_M / _C_M_S
    screens = tuple(
        PECCircularScreen(
            outer_radius_m=outer_radius_m,
            inner_radius_m=inner_radius_m,
            electron_x_m=float(x),
            electron_y_m=float(-y),
            beta_x=float(b * direction[0]),
            beta_y=float(-b * direction[1]),
            beta_z=float(-b * direction[2]),
            charge_c=-_E_CHARGE_C,
            arrival_time_s=float(clock),
        )
        for (x, y), b, direction, clock in zip(offset_m, beta, v, t_s, strict=True)
    )
    if min_formation_lengths is not None:
        assert formation_energy_eV is not None and formation_theta_max_rad is not None
        required = np.array(
            [
                min_formation_lengths
                * backward_formation_length_ang(
                    formation_energy_eV,
                    screen,
                    theta_max_rad=formation_theta_max_rad,
                )
                for screen in screens
            ]
        )
        if np.any(run < required):
            raise ValueError(
                "transport straight-after run is shorter than the requested "
                "backward formation-scale multiple"
            )
    return screens


def _check_handoff_options(
    *,
    n_histories,
    centre_xy_ang,
    time_origin_ang,
    max_angle_rad,
    min_straight_after_ang,
    formation_energy_eV,
    formation_theta_max_rad,
    min_formation_lengths,
) -> np.ndarray:
    """Validate scalar handoff options; return the metal centre [Angstrom]."""
    if (
        not isinstance(n_histories, (int, np.integer))
        or isinstance(n_histories, (bool, np.bool_))
        or n_histories < 1
    ):
        raise ValueError("n_histories must be a positive integer")
    center = np.asarray(centre_xy_ang, dtype=float)
    if center.shape != (2,) or not np.all(np.isfinite(center)):
        raise ValueError("centre_xy_ang must contain two finite transport coordinates")
    if not np.isfinite(time_origin_ang):
        raise ValueError("time_origin_ang must be finite")
    if not np.isfinite(max_angle_rad) or not 0 < max_angle_rad < np.pi / 2:
        raise ValueError("max_angle_rad must be finite and between zero and pi/2")
    if min_formation_lengths is not None:
        if not np.isfinite(min_formation_lengths) or min_formation_lengths <= 0:
            raise ValueError("min_formation_lengths must be positive and finite")
        if formation_energy_eV is None or formation_theta_max_rad is None:
            raise ValueError("formation check needs photon energy and collection angle")
    if min_straight_after_ang is not None and (
        not np.isfinite(min_straight_after_ang) or min_straight_after_ang <= 0
    ):
        raise ValueError("min_straight_after_ang must be positive and finite")
    return center


def _entrance_indices(crossings: BoundaryCrossings, n_histories: int) -> np.ndarray:
    """Crossing-row indices of each primary's vacuum -> layer 0 entry, by ID."""
    # Select only entrance records, not all crosses at the front face. Using
    # side markers disambiguates an exit through the entry face from launch.
    face = np.asarray(crossings.interface)
    before = np.asarray(crossings.layer_before)
    after = np.asarray(crossings.layer_after)
    ids = np.asarray(crossings.electron_id)
    if any(a.shape != face.shape for a in (before, after, ids)) or face.ndim != 1:
        raise ValueError("crossings face/media/history columns must be equally sized 1-D arrays")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("electron_id must be an integer history index")
    mask = (face == 0) & (before == -1) & (after == 0)
    # Transport may have Ne > Ne_brem: the latter scores primary IDs 0..M-1.
    # Ignore higher-ID primary entries rather than declaring them duplicates.
    mask &= (ids >= 0) & (ids < n_histories)
    indices = np.flatnonzero(mask)
    selected = ids[indices]
    if indices.size != n_histories or not np.array_equal(np.sort(selected), np.arange(n_histories)):
        raise ValueError(
            "need exactly one primary entrance crossing per sampled history 0..n_histories-1"
        )
    return indices[np.argsort(selected, kind="stable")]


def _entry_state(crossings: BoundaryCrossings, indices, n_histories, max_angle_rad):
    """Validated entry position, direction, energy [keV] and clock [Angstrom]."""
    p = np.asarray(crossings.position_ang, dtype=float)[indices]
    v = np.asarray(crossings.direction, dtype=float)[indices]
    e = np.asarray(crossings.kinetic_energy_keV, dtype=float)[indices]
    t = np.asarray(crossings.time_ang, dtype=float)[indices]
    q = np.asarray(crossings.charge)[indices]
    if (
        p.shape != (n_histories, 3)
        or v.shape != (n_histories, 3)
        or e.shape != (n_histories,)
        or t.shape != (n_histories,)
        or not all(np.all(np.isfinite(x)) for x in (p, v, e, t))
    ):
        raise ValueError("entry positions, directions, energies and arrival times must be finite")
    planes = np.asarray(crossings.z_interfaces_ang, dtype=float)
    if (
        planes.ndim != 1
        or planes.size < 2
        or planes[0] != 0.0
        or not np.all(np.abs(p[:, 2]) <= 1e-5)
    ):
        raise ValueError("PEC handoff needs the physical z=0 entrance plane")
    if np.any(q != -1):
        raise ValueError(
            "the finite PEC transport diagnostic currently supports electron primaries only"
        )
    if np.any(e <= 0) or np.any(np.abs(np.linalg.norm(v, axis=1) - 1.0) > 1e-9):
        raise ValueError("entry energy must be positive and directions unit normalized")
    if np.any(v[:, 2] <= 0) or np.any(v[:, 2] < np.cos(max_angle_rad)):
        raise ValueError(
            "oblique or reverse-incident electron exceeds max_angle_rad (normal-incidence default)"
        )
    return p, v, e, t


def _straight_run(crossings: BoundaryCrossings, indices, min_straight_after_ang) -> np.ndarray:
    """Post-entry straight path [Angstrom], checked against the fixed bound."""
    straight = getattr(crossings, "straight_after_ang", None)
    if straight is None:
        raise ValueError("formation run data are missing; cannot check straight-after bound")
    run = np.asarray(straight, dtype=float)[indices]
    if np.any(np.isnan(run)) or np.any(run < 0):
        raise ValueError("transport straight-after run must be nonnegative")
    if min_straight_after_ang is not None and np.any(run < min_straight_after_ang):
        raise ValueError("transport straight-after run is shorter than min_straight_after_ang")
    return run
