"""Opt-in two-beam PXR/DTR/CBS scoring of the production transport rows.

The existing TR component remains separate. The line slot is a signed
correction to it, so their sum is the squared joint Maxwell field, including
interference. This model is experimental and independently unverified.

Validation: dtr-transported-current, line-tr-phase-reconciliation
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...materials.crystal import CRYSTALS, M_E_EV, U_g, _orientation_R, chi_g, reciprocal_g_vector
from ..spectrum.dtr_history import TransportedCurrent, two_beam_current_fields
from ..spectrum.transition_radiation_crossings import (
    _charges,
    _host,
    _take,
    boundary_crossings,
    crossing_amplitudes,
)
from .transition_emission import case_layer_eps, case_layers


def validate_dtr_case(case) -> None:
    """Refuse unsupported two-beam configurations before starting transport."""
    radiators = case.get("layer_radiators")
    active = [case] if radiators is None else [r for r in radiators if r is not None]
    if len(active) != 1 or len(active[0]["hkl_list"]) != 1:
        raise ValueError(
            "two-beam transition radiation requires one crystal layer and one reflection"
        )
    if case["Ne"] != case["Ne_brem"]:
        raise ValueError(
            "two-beam transition radiation requires equal line and continuum trial counts"
        )
    for key in (
        "line_grid_policy",
        "line_quadrature",
        "secondary_threshold_eV",
    ):
        if case.get(key) is not None:
            raise ValueError(f"two-beam transition radiation does not support {key}")
    if case.get("mosaic_fwhm_rad") or case.get("mosaic_mc_fwhm_rad"):
        raise ValueError("two-beam transition radiation currently requires a perfect crystal")
    origin = np.asarray(case["transition_radiation"]["lattice_origin_ang"], float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("two-beam lattice_origin_ang requires three finite coordinates")


def _transport_current(segs, layers, crossings, count):
    """Time-parametrized finite currents and their exterior continuations."""
    electron = _host(segs["electron_id"]).astype(int)
    mask = electron < count
    length = _host(segs["L_ang"])[mask]
    direction = _host(segs["v_hat"])[mask]
    duration = _host(segs["t_end_ang"])[mask] - _host(segs["t_start_ang"])[mask]
    if np.any(duration <= 0):
        raise ValueError("two-beam scoring needs positive transport row durations")
    track = segs.get("track_id", segs["electron_id"])
    exterior = (crossings.layer_before == -1) | (crossings.layer_after == -1)
    boundary = _take(crossings, np.flatnonzero(exterior & (crossings.electron_id < count)))
    gamma = 1 + boundary.kinetic_energy_keV * 1e3 / M_E_EV
    return TransportedCurrent(
        position=_host(segs["r_mid"])[mask] - 0.5 * length[:, None] * direction,
        velocity=direction * (length / duration)[:, None],
        time=_host(segs["t0_ang"])[mask] + _host(segs["t_start_ang"])[mask],
        duration=duration,
        history=electron[mask],
        layer=_host(segs["layer"])[mask].astype(int),
        charge=_charges(segs, _host(track)[mask]),
        boundary_position=boundary.position_ang.copy(),
        boundary_velocity=boundary.direction * np.sqrt(1 - gamma**-2)[:, None],
        boundary_time=boundary.time_ang,
        boundary_history=boundary.electron_id,
        boundary_side=(boundary.interface == len(layers)).astype(int),
        boundary_charge=boundary.charge,
        histories=count,
    )


def _radiator(case):
    radiators = case.get("layer_radiators")
    if radiators is None:
        return 0, case
    return next((i, row) for i, row in enumerate(radiators) if row is not None)


def _row_parameters(case, energy):
    layer, radiator = _radiator(case)
    crystal, hkl = radiator["crystal"], tuple(radiator["hkl_list"][0])
    lattice = CRYSTALS[crystal]["lattice"]
    g, _ = reciprocal_g_vector(hkl, lattice)
    rotation = _orientation_R(
        lattice,
        radiator.get("beam_uvw"),
        radiator.get("azimuth_rad", case.get("azimuth_rad", 0.0)),
        radiator.get("recip_miscut_rad", case.get("recip_miscut_rad")),
        surface_hkl=radiator.get("surface_hkl"),
    )
    if rotation is not None:
        g = rotation @ g
    # Positive hkl is the production outgoing-transfer row. The Maxwell
    # convention uses the conjugated Friedel mate for its transfer harmonic.
    mate = tuple(-h for h in hkl)
    b = radiator["B_ang2"]
    return dict(
        crystal_layer=layer,
        reciprocal_vector_ang=g,
        chi_transfer=np.conj(chi_g(crystal, mate, energy, B_ang2=b, use_henke=True)),
        chi_reverse=np.conj(chi_g(crystal, hkl, energy, B_ang2=b, use_henke=True)),
        potential_transfer_over_m=np.conj(U_g(crystal, mate, float(energy[0]), B_ang2=b)) / M_E_EV,
        potential_reverse_over_m=np.conj(U_g(crystal, hkl, float(energy[0]), B_ang2=b)) / M_E_EV,
        lattice_origin_ang=np.asarray(case["transition_radiation"]["lattice_origin_ang"], float),
    )


def _emitting_frame(current, n, layers, params):
    if n[2] > 0:
        return n, layers, params, np.ones(3)
    flip, depth = np.array([1.0, -1.0, -1.0]), sum(d for _, d in layers)
    for value in (current.position, current.boundary_position):
        value *= flip
        value[:, 2] += depth
    current.velocity[:] *= flip
    current.boundary_velocity[:] *= flip
    current.layer[:] = len(layers) - 1 - current.layer
    current.boundary_side[:] = 1 - current.boundary_side
    params = {
        **params,
        "crystal_layer": len(layers) - 1 - params["crystal_layer"],
        "reciprocal_vector_ang": params["reciprocal_vector_ang"] * flip,
        "lattice_origin_ang": params["lattice_origin_ang"] * flip + np.array([0.0, 0.0, depth]),
    }
    return n * flip, layers[::-1], params, flip


def _power(fields, count, population=None):
    grouped = np.sum(np.abs(fields) ** 2, axis=(1, 2))
    if population is None:
        return grouped / count
    from ..spectrum.coherent_population import pair_scale, require_resolved_power

    flat = np.sum(np.abs(fields.sum(axis=1)) ** 2, axis=1)
    power = grouped + pair_scale(population, count) * (flat - grouped)
    require_resolved_power(
        power,
        incident_samples=count,
        physical_electrons=population,
        quantity="two-beam joint power",
    )
    return power / count


def case_dtr_spectrum(
    case: Any,
    segs: Any,
    energy_eV: Any,
    direction: Any,
    *,
    coherent: bool = False,
    count: int | None = None,
) -> np.ndarray:
    """Signed line correction; adding spec_transition gives the joint density.

    Per-history amplitudes include sampled position/time offsets. The
    coherent companion uses the same distinct-pair physical-population
    estimator as CTR; no analytic offset factor is applied a second time.
    Validation: dtr-transported-current, line-tr-phase-reconciliation
    """
    energy, n = np.asarray(energy_eV, float), np.asarray(direction, float)
    count = int(case["Ne"] if count is None else count)
    if n[2] == 0:
        raise ValueError("two-beam observation cannot graze the planar faces")
    layers = case_layers(case)
    crossings = boundary_crossings(segs, layers)
    if crossings.n_side_exits:
        raise ValueError("two-beam scoring does not support tracks exiting a side face")
    eps = case_layer_eps(case, energy)
    crossing = _take(crossings, np.flatnonzero(crossings.electron_id < count))
    tr = np.zeros((energy.size, count, 3), complex)
    amplitude = crossing_amplitudes(crossing, energy, n, eps)
    for i, history in enumerate(crossing.electron_id):
        tr[:, history] += amplitude[:, i]
    current = _transport_current(segs, layers, crossings, count)
    stack = [(e, row[1] - row[0]) for e, row in zip(eps, layers, strict=True)]
    n_frame, stack, params, flip = _emitting_frame(current, n, stack, _row_parameters(case, energy))
    coupled, bare = two_beam_current_fields(energy, n_frame, stack, current, **params)
    joint = tr + (coupled - bare) * flip
    population = None
    if coherent:
        from ..spectrum.coherent_population import physical_bunch_electrons

        population = physical_bunch_electrons(case)
        population = count if population is None else population
    scale = energy * n[2] ** 2 / (16 * np.pi**4)
    result = scale * (_power(joint, count, population) - _power(tr, count))
    if not np.all(np.isfinite(result)):
        raise ValueError("two-beam scoring produced a nonfinite density")
    return result
