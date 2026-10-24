"""Explicit opt-in case/transport adapter for finite-screen optical CTR/CDR.

Unlike the planar-stack optical transition component, this adapter scores an
*independently specified* circular ideal PEC source at the electron entrance.
It neither modifies particle transport nor enters the production dispatcher,
and it does not inherit the case X-ray photon-energy grid or detector model.
The output has forward-vacuum-cone integrated photons/(eV pulse), not a
finite-distance detector signal. Incoherent addition to planar TR on the same
physical interface would double-count the source.

Each adapter accepts an explicit NumPy/CuPy ``xp`` for source generation and
bunch scoring. Crossing extraction, geometry, source laws and formation guards
remain host control; no electron transport or material model is changed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..spectrum.optical_finite_bunch import (
    finite_screen_bunch_band,
    finite_screen_realization_field,
)
from ..spectrum.optical_finite_transport import entrance_pec_screens_from_crossings

__all__ = [
    "case_finite_screen_band",
    "case_finite_screen_realization_field",
    "case_finite_screen_plane_band",
]


def _case_screens(
    case,
    transport,
    source,
    *,
    outer_radius_m,
    inner_radius_m,
    centre_xy_ang,
    time_origin_ang,
    max_angle_rad,
    min_straight_after_ang,
    min_formation_lengths,
    formation_energy_eV,
    formation_theta_max_rad,
):
    from ..spectrum.transition_radiation_crossings import boundary_crossings
    from .transition_emission import case_layers

    count = int(transport["Ne_brem"])
    if count < 1:
        raise ValueError("Ne_brem must be positive for the finite-screen diagnostic")
    case_charge = case.get("bunch_charge_pc")
    if case_charge is not None and not np.isclose(
        float(case_charge), float(source.charge_pC), rtol=1e-12, atol=0.0
    ):
        raise ValueError("optical source charge disagrees with case bunch_charge_pc")
    crossings = boundary_crossings(transport["segs"], case_layers(case))
    # Unlike planar TR, every sampled history must have its own entrance.
    # The adapter never mistakes absent histories for sampled zero fields.
    return entrance_pec_screens_from_crossings(
        crossings,
        n_histories=count,
        outer_radius_m=outer_radius_m,
        inner_radius_m=inner_radius_m,
        centre_xy_ang=centre_xy_ang,
        time_origin_ang=time_origin_ang,
        max_angle_rad=max_angle_rad,
        min_straight_after_ang=min_straight_after_ang,
        min_formation_lengths=min_formation_lengths,
        formation_energy_eV=formation_energy_eV,
        formation_theta_max_rad=formation_theta_max_rad,
    )


def case_finite_screen_band(
    case,
    transport,
    energy_eV,
    *,
    source,
    outer_radius_m: float,
    inner_radius_m: float = 0.0,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    xp: Any = np,
    theta_max_rad: float,
    theta_min_rad: float = 0.0,
    centre_xy_ang=(0.0, 0.0),
    time_origin_ang: float = 0.0,
    max_angle_rad: float = 1e-3,
    min_straight_after_ang: float | None = None,
    min_formation_lengths: float | None = None,
):
    """Sample transported entrances into an independent ideal PEC diagnostic.

    Optical energies are explicit (no lowered X-ray continuum floor), and
    neither target optical permittivity nor a detector QE is inferred. This
    diagnostic does not account for screen scattering, other radiation,
    finite-distance optical collection, or a radiative energy budget.
    Source weights are indexed by the same ``Ne_brem`` primary histories.
    The bunch normalization is delegated to ``finite_screen_bunch_band``.
    """
    screens = _case_screens(
        case,
        transport,
        source,
        outer_radius_m=outer_radius_m,
        inner_radius_m=inner_radius_m,
        centre_xy_ang=centre_xy_ang,
        time_origin_ang=time_origin_ang,
        max_angle_rad=max_angle_rad,
        min_straight_after_ang=min_straight_after_ang,
        min_formation_lengths=min_formation_lengths,
        formation_energy_eV=(
            float(np.min(np.asarray(energy_eV, dtype=float)))
            if min_formation_lengths is not None
            else None
        ),
        formation_theta_max_rad=theta_max_rad,
    )
    return finite_screen_bunch_band(
        energy_eV,
        screens,
        source=source,
        nx=nx,
        ny=ny,
        dx_m=dx_m,
        dy_m=dy_m,
        xp=xp,
        theta_max_rad=theta_max_rad,
        theta_min_rad=theta_min_rad,
    )


def case_finite_screen_realization_field(
    case,
    transport,
    energy_eV: float,
    *,
    source,
    outer_radius_m: float,
    inner_radius_m: float = 0.0,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    xp: Any = np,
    centre_xy_ang=(0.0, 0.0),
    time_origin_ang: float = 0.0,
    max_angle_rad: float = 1e-3,
    min_straight_after_ang: float | None = None,
    min_formation_lengths: float | None = None,
):
    """Actual N-electron complex spectral source field (realizations only).

    The returned ``ScreenSpectralField`` uses V*s/m and can be propagated by
    the #616 Green/FFT operators. It is **not** the stochastic expectation of
    a macroparticle-sampled bunch; that law is refused by the underlying
    field sum. ``time_origin_ang`` sets the coherent global phase reference.
    """
    screens = _case_screens(
        case,
        transport,
        source,
        outer_radius_m=outer_radius_m,
        inner_radius_m=inner_radius_m,
        centre_xy_ang=centre_xy_ang,
        time_origin_ang=time_origin_ang,
        max_angle_rad=max_angle_rad,
        min_straight_after_ang=min_straight_after_ang,
        min_formation_lengths=min_formation_lengths,
        formation_energy_eV=(float(energy_eV) if min_formation_lengths is not None else None),
        formation_theta_max_rad=np.pi / 2 - 1e-8,
    )
    return finite_screen_realization_field(
        energy_eV,
        screens,
        source=source,
        nx=nx,
        ny=ny,
        dx_m=dx_m,
        dy_m=dy_m,
        xp=xp,
    )


def case_finite_screen_plane_band(
    case,
    transport,
    energy_eV,
    *,
    source,
    outer_radius_m: float,
    inner_radius_m: float = 0.0,
    nx: int,
    ny: int,
    dx_m: float,
    dy_m: float,
    xp: Any = np,
    plane_z_m: float,
    aperture,
    jones_sp=None,
    centre_xy_ang=(0.0, 0.0),
    time_origin_ang: float = 0.0,
    max_angle_rad: float = 1e-3,
    min_straight_after_ang: float | None = None,
    min_formation_lengths: float | None = None,
    formation_theta_max_rad: float | None = None,
):
    """Opt-in transported-primary finite-distance CTR/CDR detector-plane band.

    ``aperture`` must be an explicit ``PlaneAcceptance`` in local PEC vacuum
    coordinates (x, -y, -z relative to the transported entrance face). This
    scorer never infers a detector from the case, or applies material Fresnel
    factors, finite source corrections, or optical QE by default.

    The angular acceptance of a finite detector cannot be inferred solely
    from its centre: an extended radiator drives a distribution of paths.
    Therefore a formation check requires a *user-supplied conservative*
    ``formation_theta_max_rad``; it is not silently identified with an
    aperture centre ray. Source uses #615 physical-population normalization.
    """
    from ..spectrum.optical_finite_collection import (
        PlaneAcceptance,
        finite_screen_bunch_plane_band,
    )

    if not isinstance(aperture, PlaneAcceptance):
        raise TypeError("aperture must be an explicit PlaneAcceptance")
    energies = np.asarray(energy_eV, dtype=float)
    if (
        energies.ndim != 1
        or energies.size < 2
        or not np.all(np.isfinite(energies))
        or np.any(energies <= 0)
        or np.any(np.diff(energies) <= 0)
    ):
        raise ValueError("energy_eV must be a strictly increasing positive finite 1-D grid")
    if min_formation_lengths is not None and formation_theta_max_rad is None:
        raise ValueError("formation check needs explicit formation_theta_max_rad")
    screens = _case_screens(
        case,
        transport,
        source,
        outer_radius_m=outer_radius_m,
        inner_radius_m=inner_radius_m,
        centre_xy_ang=centre_xy_ang,
        time_origin_ang=time_origin_ang,
        max_angle_rad=max_angle_rad,
        min_straight_after_ang=min_straight_after_ang,
        min_formation_lengths=min_formation_lengths,
        formation_energy_eV=(float(energies[0]) if min_formation_lengths is not None else None),
        formation_theta_max_rad=formation_theta_max_rad,
    )
    return finite_screen_bunch_plane_band(
        energies,
        screens,
        source=source,
        nx=nx,
        ny=ny,
        dx_m=dx_m,
        dy_m=dy_m,
        xp=xp,
        plane_z_m=plane_z_m,
        aperture=aperture,
        jones_sp=jones_sp,
    )
